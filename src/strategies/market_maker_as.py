#!/usr/bin/env python3
"""
Avellaneda-Stoikov Market Making Strategy
Micro market maker with inventory control and risk aversion
"""

import os
import sys
import json
import time
import math
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import redis
import numpy as np

# Try to import spread optimizer if available
try:
    from src.layers.layer4_execution.spread_optimizer import optimal_offset
except ImportError:

    def optimal_offset(symbol, side, price):
        """Fallback spread optimizer."""
        return price * (1 + 0.0001 if side.lower() == "buy" else 1 - 0.0001)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("market_maker_as")


class AvellanedaStoikovMarketMaker:
    """Avellaneda-Stoikov market making strategy implementation."""

    def __init__(self):
        """Initialize market maker."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Avellaneda-Stoikov parameters
        self.config = {
            # Core AS parameters
            "gamma": 0.1,  # Risk aversion parameter
            "k": 1.5,  # Order book intensity parameter
            "T": 5.0,  # Time horizon (seconds)
            # Market making config
            "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "tick_interval": 1.0,  # Update quotes every 1 second
            "inv_limit_pct": 0.03,  # 3% of equity per symbol
            "min_spread_bps": 2.0,  # Minimum spread to quote
            "max_spread_bps": 50.0,  # Maximum spread to quote
            "min_qty_usd": 10.0,  # Minimum quote size in USD
            "max_qty_usd": 1000.0,  # Maximum quote size in USD
            # Risk controls
            "vol_spike_threshold": 3.0,  # Cancel if σ > 3σ_10m
            "max_inventory_age": 300,  # Max inventory age in seconds
            "inventory_decay": 0.95,  # Inventory target decay factor
            "quote_refresh_interval": 2.0,  # Refresh quotes every 2s minimum
        }

        # Symbol configurations
        self.symbol_configs = {
            "BTCUSDT": {"tick_size": 0.01, "min_qty": 1e-6, "base_vol": 0.60},
            "ETHUSDT": {"tick_size": 0.01, "min_qty": 1e-6, "base_vol": 0.75},
            "SOLUSDT": {"tick_size": 0.001, "min_qty": 1e-3, "base_vol": 1.20},
        }

        # State tracking
        self.inventories = {
            symbol: 0.0 for symbol in self.config["symbols"]
        }  # Current inventory per symbol
        self.inventory_targets = {
            symbol: 0.0 for symbol in self.config["symbols"]
        }  # Target inventory
        self.active_quotes = {}  # symbol -> {"bid": quote_info, "ask": quote_info}
        self.last_quote_time = {symbol: 0.0 for symbol in self.config["symbols"]}
        self.fill_history = []
        self.total_fills = 0
        self.total_volume = 0.0

        # Prometheus metrics
        self.metrics = {
            "mm_quotes_live": 0,
            "mm_inventory": 0.0,
            "mm_fill_rate": 0.0,
            "mm_spread_avg": 0.0,
            "mm_pnl_usd": 0.0,
            "mm_total_volume": 0.0,
        }

        logger.info("🎯 Avellaneda-Stoikov Market Maker initialized")
        logger.info(f"   Symbols: {self.config['symbols']}")
        logger.info(f"   γ (risk aversion): {self.config['gamma']}")
        logger.info(f"   k (intensity): {self.config['k']}")
        logger.info(f"   T (horizon): {self.config['T']}s")
        logger.info(f"   Inventory limit: {self.config['inv_limit_pct']:.0%} equity")

    def get_market_data(self, symbol: str) -> Dict[str, float]:
        """Get current market data for symbol."""
        try:
            market_data = {}

            # Get current mid price
            mid_key = f"price:{symbol.lower()}:mid"
            mid_price = self.redis.get(mid_key)
            if mid_price:
                market_data["mid_price"] = float(mid_price)
            else:
                # Get from bid/ask or fallback to mock
                bid_key = f"price:{symbol.lower()}:bid"
                ask_key = f"price:{symbol.lower()}:ask"
                bid_price = self.redis.get(bid_key)
                ask_price = self.redis.get(ask_key)

                if bid_price and ask_price:
                    market_data["mid_price"] = (float(bid_price) + float(ask_price)) / 2
                else:
                    # Mock prices for demo
                    mock_mids = {
                        "BTCUSDT": 97600.0,
                        "ETHUSDT": 3515.0,
                        "SOLUSDT": 184.0,
                    }
                    market_data["mid_price"] = mock_mids.get(symbol, 100.0)

            # Get volatility estimate
            vol_key = f"volatility:{symbol.lower()}:1m"
            volatility = self.redis.get(vol_key)
            if volatility:
                market_data["volatility"] = float(volatility)
            else:
                # Use base volatility from config, scaled to 1-second
                base_vol = self.symbol_configs.get(symbol, {}).get("base_vol", 0.5)
                # Convert annual vol to 1-second vol
                market_data["volatility"] = base_vol / np.sqrt(365 * 24 * 3600)

            # Get current spread
            spread_key = f"spread:{symbol.lower()}:bps"
            spread_bps = self.redis.get(spread_key)
            if spread_bps:
                market_data["spread_bps"] = float(spread_bps)
            else:
                # Mock spreads
                mock_spreads = {"BTCUSDT": 3.0, "ETHUSDT": 4.0, "SOLUSDT": 8.0}
                market_data["spread_bps"] = mock_spreads.get(symbol, 5.0)

            # Get 10-minute volatility for spike detection
            vol_10m_key = f"volatility:{symbol.lower()}:10m"
            vol_10m = self.redis.get(vol_10m_key)
            market_data["vol_10m"] = (
                float(vol_10m) if vol_10m else market_data["volatility"] * 10
            )

            return market_data

        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {}

    def calculate_inventory_limit(self, symbol: str, price: float) -> float:
        """Calculate maximum inventory limit in base currency."""
        try:
            # Get current equity
            equity_usd = float(self.redis.get("risk:equity_usd") or 100000)

            # Calculate max inventory value in USD
            max_inventory_value = equity_usd * self.config["inv_limit_pct"]

            # Convert to base currency quantity
            max_inventory_qty = max_inventory_value / price

            return max_inventory_qty

        except Exception as e:
            logger.error(f"Error calculating inventory limit for {symbol}: {e}")
            return 0.1  # Fallback to 0.1 units

    def check_volatility_spike(self, symbol: str, market_data: Dict) -> bool:
        """Check if there's a volatility spike."""
        try:
            current_vol = market_data["volatility"]
            avg_vol_10m = market_data["vol_10m"]

            # Check if current vol is > 3x the 10-minute average
            vol_ratio = current_vol / avg_vol_10m if avg_vol_10m > 0 else 1.0

            is_spike = vol_ratio > self.config["vol_spike_threshold"]

            if is_spike:
                logger.warning(
                    f"⚡ Volatility spike detected for {symbol}: {vol_ratio:.1f}x average"
                )

            return is_spike

        except Exception as e:
            logger.error(f"Error checking volatility spike: {e}")
            return False

    def calculate_as_quotes(
        self, symbol: str, market_data: Dict
    ) -> Tuple[float, float, float]:
        """Calculate Avellaneda-Stoikov bid/ask prices and optimal size."""
        try:
            s = market_data["mid_price"]  # Current mid price
            sigma = market_data["volatility"]  # Volatility
            q = self.inventories[symbol]  # Current inventory

            # Get calibrated parameters from Redis (updated by online calibrator)
            gamma_redis = self.redis.get("mm:gamma")
            k_redis = self.redis.get("mm:k")

            gamma = (
                float(gamma_redis) if gamma_redis else self.config["gamma"]
            )  # Risk aversion
            k = float(k_redis) if k_redis else self.config["k"]  # Order book intensity
            T = self.config["T"]  # Time horizon

            # AS reservation price: r = s - q * γ * σ² * T
            reservation_price = s - q * gamma * (sigma**2) * T

            # AS half-spread: δ = (γ * σ² * T)/2 + (1/k) * ln(1 + γ/k)
            half_spread = (gamma * (sigma**2) * T) / 2.0 + (1.0 / k) * math.log(
                1.0 + gamma / k
            )

            # AS bid/ask prices
            bid_price = reservation_price - half_spread
            ask_price = reservation_price + half_spread

            # Inventory skewing: reduce ask when long, reduce bid when short
            inventory_skew = abs(q) * 0.1  # 10% skew per unit of inventory

            if q > 0:  # Long inventory - reduce ask size, increase bid size
                bid_size_multiplier = 1.0 + inventory_skew
                ask_size_multiplier = max(0.1, 1.0 - inventory_skew)
            else:  # Short inventory - reduce bid size, increase ask size
                bid_size_multiplier = max(0.1, 1.0 - inventory_skew)
                ask_size_multiplier = 1.0 + inventory_skew

            # Base quote size calculation
            mid_price = market_data["mid_price"]
            base_size_usd = min(
                self.config["max_qty_usd"], max(self.config["min_qty_usd"], 500.0)
            )
            base_size_qty = base_size_usd / mid_price

            # Apply inventory skewing to sizes
            bid_size = base_size_qty * bid_size_multiplier
            ask_size = base_size_qty * ask_size_multiplier

            # Get inventory limit
            inv_limit = self.calculate_inventory_limit(symbol, mid_price)

            # Adjust sizes based on inventory limits
            if q + bid_size > inv_limit:
                bid_size = max(0, inv_limit - q)

            if q - ask_size < -inv_limit:
                ask_size = max(0, q + inv_limit)

            logger.debug(
                f"{symbol} AS quotes: r={reservation_price:.2f}, δ={half_spread:.2f}, "
                f"bid={bid_price:.2f}@{bid_size:.6f}, ask={ask_price:.2f}@{ask_size:.6f}"
            )

            return bid_price, ask_price, bid_size, ask_size

        except Exception as e:
            logger.error(f"Error calculating AS quotes for {symbol}: {e}")
            # Fallback to simple spread
            mid = market_data["mid_price"]
            spread_bps = market_data.get("spread_bps", 5.0)
            half_spread_px = mid * spread_bps / 20000  # Half spread in price
            bid_price = mid - half_spread_px
            ask_price = mid + half_spread_px
            base_size = self.config["min_qty_usd"] / mid
            return bid_price, ask_price, base_size, base_size

    def should_quote_market(self, symbol: str, market_data: Dict) -> Tuple[bool, str]:
        """Check if we should quote in this market."""
        try:
            # Check minimum spread requirement
            spread_bps = market_data.get("spread_bps", 0)
            if spread_bps < self.config["min_spread_bps"]:
                return False, f"spread_too_tight_{spread_bps:.1f}bps"

            # Check volatility spike
            if self.check_volatility_spike(symbol, market_data):
                return False, "volatility_spike"

            # Check capital allocation
            capital_effective = float(self.redis.get("risk:capital_effective") or 0.4)
            if capital_effective < 0.1:
                return False, f"low_capital_{capital_effective:.1%}"

            # Check system mode
            mode = self.redis.get("mode")
            if mode == "halt":
                return False, "system_halt"

            # Check if quotes are stale (force refresh)
            last_quote = self.last_quote_time.get(symbol, 0)
            time_since_quote = time.time() - last_quote
            if time_since_quote > self.config["quote_refresh_interval"]:
                return True, "quote_refresh"

            # Check if inventory is too old (needs refresh)
            inventory_age = self.get_inventory_age(symbol)
            if inventory_age > self.config["max_inventory_age"]:
                return True, "inventory_stale"

            return True, "normal_quoting"

        except Exception as e:
            logger.error(f"Error checking quoting conditions: {e}")
            return False, "check_error"

    def get_inventory_age(self, symbol: str) -> float:
        """Get age of current inventory position."""
        try:
            inventory_timestamp = self.redis.get(
                f"inventory:{symbol.lower()}:timestamp"
            )
            if inventory_timestamp:
                return time.time() - float(inventory_timestamp)
            return 0.0
        except:
            return 0.0

    def place_quote(
        self, symbol: str, side: str, price: float, size: float
    ) -> Dict[str, Any]:
        """Place/update a quote."""
        try:
            quote_id = f"{symbol}_{side}_{int(time.time())}"

            # Round price to tick size
            tick_size = self.symbol_configs.get(symbol, {}).get("tick_size", 0.01)
            rounded_price = round(price / tick_size) * tick_size

            # Round size appropriately
            min_qty = self.symbol_configs.get(symbol, {}).get("min_qty", 1e-6)
            if size < min_qty:
                return {"success": False, "reason": "size_too_small"}

            # Simulate quote placement
            quote = {
                "quote_id": quote_id,
                "symbol": symbol,
                "side": side,
                "price": rounded_price,
                "size": size,
                "timestamp": time.time(),
                "status": "active",
            }

            # Store quote state
            if symbol not in self.active_quotes:
                self.active_quotes[symbol] = {}
            self.active_quotes[symbol][side] = quote

            # Store in Redis
            quote_key = f"mm:quote:{symbol.lower()}:{side}"
            self.redis.set(quote_key, json.dumps(quote, default=str))

            logger.debug(
                f"📋 {side.upper()} quote: {symbol} {rounded_price:.2f} @ {size:.6f}"
            )

            return {"success": True, "quote": quote}

        except Exception as e:
            logger.error(f"Error placing {side} quote for {symbol}: {e}")
            return {"success": False, "error": str(e)}

    def cancel_quotes(self, symbol: str, reason: str = "update") -> int:
        """Cancel existing quotes for symbol."""
        try:
            cancelled_count = 0

            if symbol in self.active_quotes:
                for side in ["bid", "ask"]:
                    if side in self.active_quotes[symbol]:
                        quote = self.active_quotes[symbol][side]
                        quote["status"] = "cancelled"
                        quote["cancel_reason"] = reason

                        # Remove from Redis
                        quote_key = f"mm:quote:{symbol.lower()}:{side}"
                        self.redis.delete(quote_key)

                        cancelled_count += 1

                # Clear active quotes
                self.active_quotes[symbol] = {}

            if cancelled_count > 0:
                logger.debug(
                    f"❌ Cancelled {cancelled_count} quotes for {symbol}: {reason}"
                )

            return cancelled_count

        except Exception as e:
            logger.error(f"Error cancelling quotes for {symbol}: {e}")
            return 0

    def simulate_fills(self, symbol: str) -> List[Dict[str, Any]]:
        """Simulate potential fills for active quotes (mock trading)."""
        try:
            fills = []

            if symbol not in self.active_quotes:
                return fills

            current_time = time.time()
            market_data = self.get_market_data(symbol)
            mid_price = market_data.get("mid_price", 0)

            # Simulate fills with some probability
            for side, quote in self.active_quotes[symbol].items():
                if quote["status"] != "active":
                    continue

                quote_price = quote["price"]
                quote_size = quote["size"]

                # Simple fill probability based on how aggressive the quote is
                if side == "bid":
                    aggressiveness = (
                        (quote_price - mid_price) / mid_price * 10000
                    )  # bps above mid
                else:
                    aggressiveness = (
                        (mid_price - quote_price) / mid_price * 10000
                    )  # bps below mid

                # Higher aggressiveness = higher fill probability
                fill_prob = min(
                    0.1, max(0.001, aggressiveness / 100)
                )  # 0.1% to 10% per tick

                if np.random.random() < fill_prob:
                    # Simulate partial or full fill
                    fill_size = quote_size * np.random.uniform(0.3, 1.0)  # 30-100% fill

                    fill = {
                        "fill_id": f"fill_{quote['quote_id']}_{int(current_time)}",
                        "symbol": symbol,
                        "side": side,
                        "price": quote_price,
                        "size": fill_size,
                        "timestamp": current_time,
                        "quote_id": quote["quote_id"],
                    }

                    fills.append(fill)

                    # Update inventory
                    if side == "bid":  # Buy fill
                        self.inventories[symbol] += fill_size
                    else:  # Sell fill
                        self.inventories[symbol] -= fill_size

                    # Update fill history
                    self.fill_history.append(fill)
                    self.total_fills += 1
                    self.total_volume += fill_size * quote_price

                    if len(self.fill_history) > 1000:
                        self.fill_history = self.fill_history[-500:]

                    # Update inventory timestamp
                    self.redis.set(
                        f"inventory:{symbol.lower()}:timestamp", current_time
                    )
                    self.redis.set(
                        f"inventory:{symbol.lower()}", self.inventories[symbol]
                    )

                    logger.info(
                        f"💰 Fill: {side.upper()} {fill_size:.6f} {symbol} @ ${quote_price:.2f}"
                    )

            return fills

        except Exception as e:
            logger.error(f"Error simulating fills for {symbol}: {e}")
            return []

    def update_inventory_target(self, symbol: str):
        """Update inventory target with decay."""
        try:
            current_target = self.inventory_targets[symbol]
            decay_factor = self.config["inventory_decay"]

            # Decay target towards zero
            new_target = current_target * decay_factor

            # Update if significant change
            if abs(new_target - current_target) > 1e-6:
                self.inventory_targets[symbol] = new_target

        except Exception as e:
            logger.error(f"Error updating inventory target for {symbol}: {e}")

    def tick_symbol(self, symbol: str) -> Dict[str, Any]:
        """Run market making tick for single symbol."""
        try:
            tick_start = time.time()

            # Get market data
            market_data = self.get_market_data(symbol)
            if not market_data:
                return {"symbol": symbol, "status": "no_market_data"}

            # Check if we should quote
            should_quote, quote_reason = self.should_quote_market(symbol, market_data)

            if not should_quote:
                # Cancel existing quotes
                cancelled = self.cancel_quotes(symbol, quote_reason)
                return {
                    "symbol": symbol,
                    "status": "not_quoting",
                    "reason": quote_reason,
                    "quotes_cancelled": cancelled,
                }

            # Simulate fills for existing quotes
            fills = self.simulate_fills(symbol)

            # Calculate new AS quotes
            bid_price, ask_price, bid_size, ask_size = self.calculate_as_quotes(
                symbol, market_data
            )

            # Cancel existing quotes first
            self.cancel_quotes(symbol, "quote_update")

            # Place new quotes
            quotes_placed = 0

            # Place bid if size is meaningful
            if bid_size > self.symbol_configs.get(symbol, {}).get("min_qty", 1e-6):
                bid_result = self.place_quote(symbol, "bid", bid_price, bid_size)
                if bid_result["success"]:
                    quotes_placed += 1

            # Place ask if size is meaningful
            if ask_size > self.symbol_configs.get(symbol, {}).get("min_qty", 1e-6):
                ask_result = self.place_quote(symbol, "ask", ask_price, ask_size)
                if ask_result["success"]:
                    quotes_placed += 1

            # Update inventory target
            self.update_inventory_target(symbol)

            # Update last quote time
            self.last_quote_time[symbol] = tick_start

            # Calculate current spread
            if quotes_placed == 2:
                current_spread_bps = (
                    (ask_price - bid_price) / ((ask_price + bid_price) / 2) * 10000
                )
            else:
                current_spread_bps = market_data.get("spread_bps", 0)

            tick_result = {
                "symbol": symbol,
                "status": "completed",
                "quotes_placed": quotes_placed,
                "fills": len(fills),
                "current_inventory": self.inventories[symbol],
                "inventory_target": self.inventory_targets[symbol],
                "spread_bps": current_spread_bps,
                "tick_duration": time.time() - tick_start,
            }

            return tick_result

        except Exception as e:
            logger.error(f"Error in tick for {symbol}: {e}")
            return {"symbol": symbol, "status": "error", "error": str(e)}

    def tick(self) -> Dict[str, Any]:
        """Main market making tick - process all symbols."""
        try:
            tick_start = time.time()

            symbol_results = {}
            total_quotes = 0
            total_fills = 0

            # Process each symbol
            for symbol in self.config["symbols"]:
                try:
                    result = self.tick_symbol(symbol)
                    symbol_results[symbol] = result

                    if result["status"] == "completed":
                        total_quotes += result.get("quotes_placed", 0)
                        total_fills += result.get("fills", 0)

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    symbol_results[symbol] = {"status": "error", "error": str(e)}

            # Update metrics
            self.metrics["mm_quotes_live"] = sum(
                len(quotes) for quotes in self.active_quotes.values()
            )
            self.metrics["mm_inventory"] = sum(
                abs(inv) for inv in self.inventories.values()
            )
            self.metrics["mm_fill_rate"] = self.total_fills / max(
                1, time.time() - tick_start
            )  # Rough approximation
            self.metrics["mm_total_volume"] = self.total_volume

            # Calculate average spread
            spreads = [
                result.get("spread_bps", 0)
                for result in symbol_results.values()
                if result.get("spread_bps", 0) > 0
            ]
            self.metrics["mm_spread_avg"] = np.mean(spreads) if spreads else 0

            # Update metrics in Redis
            for metric, value in self.metrics.items():
                self.redis.set(f"metric:{metric}", value)

            tick_duration = time.time() - tick_start

            # Log summary periodically
            if total_quotes > 0 or total_fills > 0:
                logger.info(
                    f"🎯 MM Tick: {total_quotes} quotes, {total_fills} fills, "
                    f"inventory: {self.metrics['mm_inventory']:.3f}"
                )

            return {
                "timestamp": tick_start,
                "status": "completed",
                "symbol_results": symbol_results,
                "total_quotes_placed": total_quotes,
                "total_fills": total_fills,
                "total_inventory": self.metrics["mm_inventory"],
                "tick_duration": tick_duration,
            }

        except Exception as e:
            logger.error(f"Error in market making tick: {e}")
            return {"timestamp": time.time(), "status": "error", "error": str(e)}

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive market maker status."""
        try:
            # Calculate P&L estimate (simplified)
            total_pnl = 0.0
            for fill in self.fill_history[-100:]:  # Last 100 fills
                # Rough P&L estimate based on spread capture
                if fill["side"] == "bid":
                    total_pnl -= fill["size"] * fill["price"]  # Bought - cash out
                else:
                    total_pnl += fill["size"] * fill["price"]  # Sold - cash in

            self.metrics["mm_pnl_usd"] = total_pnl

            status = {
                "strategy": "avellaneda_stoikov_market_maker",
                "timestamp": time.time(),
                "config": self.config,
                "inventories": self.inventories.copy(),
                "inventory_targets": self.inventory_targets.copy(),
                "active_quotes_count": sum(
                    len(quotes) for quotes in self.active_quotes.values()
                ),
                "active_quotes": {
                    symbol: {
                        side: {
                            "price": quote["price"],
                            "size": quote["size"],
                            "age": time.time() - quote["timestamp"],
                        }
                        for side, quote in quotes.items()
                    }
                    for symbol, quotes in self.active_quotes.items()
                },
                "metrics": self.metrics.copy(),
                "totals": {
                    "total_fills": self.total_fills,
                    "total_volume": self.total_volume,
                    "estimated_pnl": total_pnl,
                },
                "recent_fills": self.fill_history[-5:] if self.fill_history else [],
            }

            return status

        except Exception as e:
            return {
                "strategy": "avellaneda_stoikov_market_maker",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }

    def loop(self):
        """Main market making loop."""
        logger.info("🎯 Starting Avellaneda-Stoikov market making loop")

        try:
            while True:
                try:
                    # Run market making tick
                    result = self.tick()

                    if result["status"] == "completed":
                        total_quotes = result["total_quotes_placed"]
                        total_fills = result["total_fills"]

                        if total_quotes > 0 or total_fills > 0:
                            logger.debug(
                                f"📊 MM: {total_quotes} quotes, {total_fills} fills, "
                                f"inventory: {result['total_inventory']:.3f}"
                            )

                    # Sleep until next tick
                    time.sleep(self.config["tick_interval"])

                except Exception as e:
                    logger.error(f"Error in market making loop: {e}")
                    time.sleep(5)  # Wait longer on error

        except KeyboardInterrupt:
            logger.info("Market maker stopped by user")

            # Cancel all quotes on exit
            logger.info("Cancelling all active quotes...")
            for symbol in self.config["symbols"]:
                self.cancel_quotes(symbol, "shutdown")

        except Exception as e:
            logger.error(f"Fatal error in market making loop: {e}")


def main():
    """Main entry point for market maker."""
    import argparse

    parser = argparse.ArgumentParser(description="Avellaneda-Stoikov Market Maker")
    parser.add_argument("--run", action="store_true", help="Run market making loop")
    parser.add_argument(
        "--tick", action="store_true", help="Run single market making tick"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show market maker status"
    )
    parser.add_argument("--symbol", help="Process specific symbol only")

    args = parser.parse_args()

    # Create market maker
    mm = AvellanedaStoikovMarketMaker()

    if args.symbol and args.symbol not in mm.config["symbols"]:
        print(f"Error: {args.symbol} not in configured symbols: {mm.config['symbols']}")
        return

    if args.status:
        # Show status report
        status = mm.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.tick:
        # Run single tick
        if args.symbol:
            result = mm.tick_symbol(args.symbol)
        else:
            result = mm.tick()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.run:
        # Filter symbols if specified
        if args.symbol:
            mm.config["symbols"] = [args.symbol]
            mm.inventories = {args.symbol: 0.0}
            mm.inventory_targets = {args.symbol: 0.0}
            mm.last_quote_time = {args.symbol: 0.0}

        # Run market making loop
        mm.loop()
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
