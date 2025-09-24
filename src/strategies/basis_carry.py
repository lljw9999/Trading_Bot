#!/usr/bin/env python3
"""
Spot-Perp Basis Carry Strategy
Delta-neutral strategy to earn funding/basis carry with hedged legs
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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("basis_carry")


class SpotPerpBasisCarryStrategy:
    """Spot-perp basis carry strategy implementation."""

    def __init__(self):
        """Initialize basis carry strategy."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Strategy configuration
        self.config = {
            "symbols": ["BTC", "ETH", "SOL"],  # Assets to trade
            "max_gross_per_strategy": 0.20,  # 20% equity per strategy
            "entry_basis_threshold": -5.0,  # Enter when basis < -5 bps (cheap carry)
            "entry_funding_threshold": 0.10,  # Enter when funding > 10% annual (paid to be long)
            "exit_basis_threshold": 1.0,  # Exit when basis mean-reverts to 1 bps
            "stop_loss_pct": -0.006,  # Stop if PnL < -0.6%
            "hedge_divergence_limit": 15.0,  # Stop if hedge diverges > 15 bps
            "timeout_hours": 8,  # Exit after 8 hours regardless
            "tick_interval": 10,  # Check every 10 seconds
            "min_notional_usd": 100,  # Minimum trade size
            "max_notional_usd": 50000,  # Maximum trade size per leg
        }

        # Track active positions
        self.active_positions = {}  # symbol -> position_info
        self.position_history = []
        self.total_pnl = 0.0
        self.total_trades = 0
        self.win_trades = 0

        # Prometheus metrics
        self.metrics = {
            "basis_open_trades": 0,
            "basis_notional_usd": 0.0,
            "basis_pnl_usd": 0.0,
            "basis_win_rate": 0.0,
            "basis_total_trades": 0,
        }

        logger.info("📊 Spot-Perp Basis Carry Strategy initialized")
        logger.info(f"   Symbols: {self.config['symbols']}")
        logger.info(f"   Max gross: {self.config['max_gross_per_strategy']:.0%}")
        logger.info(
            f"   Entry thresholds: basis<{self.config['entry_basis_threshold']}bps, funding>{self.config['entry_funding_threshold']:.0%}"
        )
        logger.info(
            f"   Exit threshold: basis>{self.config['exit_basis_threshold']}bps"
        )

    def compute_basis_bps(self, spot_px: float, perp_px: float) -> float:
        """Compute basis in basis points (perp - spot) / spot * 10000."""
        try:
            if spot_px <= 0 or perp_px <= 0:
                return 0.0

            basis_bps = (perp_px - spot_px) / spot_px * 10000
            return basis_bps

        except Exception as e:
            logger.error(f"Error computing basis: {e}")
            return 0.0

    def get_market_data(self, symbol: str) -> Dict[str, float]:
        """Get current spot and perp prices, funding rate."""
        try:
            market_data = {}

            # Get spot price
            spot_key = f"price:{symbol.lower()}:spot"
            spot_price = self.redis.get(spot_key)
            if spot_price:
                market_data["spot_price"] = float(spot_price)
            else:
                # Mock spot prices for demo
                mock_spots = {"BTC": 97650.0, "ETH": 3520.0, "SOL": 185.5}
                market_data["spot_price"] = mock_spots.get(symbol, 100.0)

            # Get perp price
            perp_key = f"price:{symbol.lower()}:perp"
            perp_price = self.redis.get(perp_key)
            if perp_price:
                market_data["perp_price"] = float(perp_price)
            else:
                # Mock perp prices (slightly different from spot)
                spot_px = market_data["spot_price"]
                # Add small random basis
                basis_variance = np.random.uniform(-8, 3)  # -8 to +3 bps
                market_data["perp_price"] = spot_px * (1 + basis_variance / 10000)

            # Get funding rate (annualized)
            funding_key = f"funding:{symbol.lower()}:annual"
            funding_rate = self.redis.get(funding_key)
            if funding_rate:
                market_data["funding_annual"] = float(funding_rate)
            else:
                # Mock funding rates
                mock_funding = {"BTC": 0.08, "ETH": 0.12, "SOL": 0.15}  # 8-15% annual
                market_data["funding_annual"] = mock_funding.get(symbol, 0.10)

            # Calculate basis
            market_data["basis_bps"] = self.compute_basis_bps(
                market_data["spot_price"], market_data["perp_price"]
            )

            logger.debug(f"{symbol} market data: {market_data}")
            return market_data

        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {}

    def entry_signal(
        self, basis_bps: float, funding_ann: float, symbol: str = "BTC"
    ) -> Tuple[bool, str]:
        """Check if entry conditions are met."""
        try:
            # Try to get calibrated z-score from basis calibrator
            calib_data = self.redis.hgetall(f"basis:calib:{symbol}")

            if calib_data and "z" in calib_data:
                z_score = float(calib_data["z"])

                # Z-score based entry (more robust)
                if z_score < -1.2:  # Calibrated entry threshold
                    return True, f"z_score_entry_{z_score:.2f}"

            # Fallback to original basis-based logic
            # Entry condition 1: Cheap carry (negative basis)
            if basis_bps < self.config["entry_basis_threshold"]:
                return True, "cheap_carry"

            # Entry condition 2: High positive funding (paid to be long)
            if funding_ann > self.config["entry_funding_threshold"]:
                return True, "high_funding"

            return False, "no_entry"

        except Exception as e:
            logger.error(f"Error checking entry signal: {e}")
            return False, "error"

    def exit_signal(self, position: Dict, current_data: Dict) -> Tuple[bool, str]:
        """Check if position should be closed."""
        try:
            position_id = position["position_id"]
            symbol = position["symbol"]
            entry_time = position["entry_time"]
            entry_basis_bps = position["entry_basis_bps"]
            current_basis_bps = current_data["basis_bps"]

            # Calculate unrealized PnL
            unrealized_pnl = self.calculate_position_pnl(position, current_data)
            unrealized_pnl_pct = unrealized_pnl / position["notional_usd"]

            # Try to get calibrated z-score exit signal
            calib_data = self.redis.hgetall(f"basis:calib:{symbol}")

            if calib_data and "z" in calib_data:
                z_score = float(calib_data["z"])

                # Z-score based exit (mean reversion)
                if abs(z_score) < 0.3:  # Calibrated exit threshold
                    return True, f"z_score_revert_{z_score:.2f}"

            # Fallback to original basis-based logic
            # Exit condition 1: Basis mean reversion
            if abs(current_basis_bps) < self.config["exit_basis_threshold"]:
                return True, "basis_mean_revert"

            # Exit condition 2: Stop loss
            if unrealized_pnl_pct < self.config["stop_loss_pct"]:
                return True, "stop_loss"

            # Exit condition 3: Hedge divergence
            hedge_divergence = abs(current_basis_bps - entry_basis_bps)
            if hedge_divergence > self.config["hedge_divergence_limit"]:
                return True, "hedge_divergence"

            # Exit condition 4: Timeout
            position_age = time.time() - entry_time
            timeout_seconds = self.config["timeout_hours"] * 3600
            if position_age > timeout_seconds:
                return True, "timeout"

            return False, "hold"

        except Exception as e:
            logger.error(f"Error checking exit signal: {e}")
            return True, "error_exit"

    def calculate_position_size(self, symbol: str, market_data: Dict) -> float:
        """Calculate optimal position size in USD notional."""
        try:
            # Get current equity
            equity_usd = float(
                self.redis.get("risk:equity_usd") or 100000
            )  # Default $100k

            # Calculate max notional per strategy
            max_notional = equity_usd * self.config["max_gross_per_strategy"]

            # Check current exposure
            current_notional = 0.0
            for pos in self.active_positions.values():
                current_notional += pos["notional_usd"]

            available_notional = max_notional - current_notional

            # Use portion of available notional
            target_notional = min(
                available_notional * 0.5,  # Use 50% of available
                self.config["max_notional_usd"],
            )

            # Ensure minimum size
            if target_notional < self.config["min_notional_usd"]:
                return 0.0

            logger.debug(f"Position sizing for {symbol}: ${target_notional:,.0f}")
            return target_notional

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def open_trade(
        self, symbol: str, notional_usd: float, market_data: Dict
    ) -> Dict[str, Any]:
        """Open a new basis carry trade."""
        try:
            position_id = f"{symbol}_basis_{int(time.time())}"
            entry_time = time.time()

            spot_price = market_data["spot_price"]
            perp_price = market_data["perp_price"]
            basis_bps = market_data["basis_bps"]
            funding_annual = market_data["funding_annual"]

            # Get calibrated hedge ratio (beta) from basis calibrator
            calib_data = self.redis.hgetall(f"basis:calib:{symbol}")
            hedge_ratio = 1.0  # Default 1:1 hedge

            if calib_data and "beta" in calib_data:
                hedge_ratio = float(calib_data["beta"])
                logger.debug(
                    f"Using calibrated hedge ratio for {symbol}: {hedge_ratio:.4f}"
                )

            # Calculate quantities for both legs using calibrated hedge ratio
            spot_quantity = notional_usd / spot_price
            perp_quantity = spot_quantity * hedge_ratio  # Use calibrated beta

            # Determine trade direction based on basis
            if basis_bps < 0:
                # Negative basis: buy spot, sell perp (earn convergence)
                spot_side = "buy"
                perp_side = "sell"
                strategy_type = "basis_convergence"
            else:
                # Positive funding: sell spot, buy perp (earn funding)
                spot_side = "sell"
                perp_side = "buy"
                strategy_type = "funding_capture"

            # Create position record
            position = {
                "position_id": position_id,
                "symbol": symbol,
                "strategy_type": strategy_type,
                "entry_time": entry_time,
                "notional_usd": notional_usd,
                "spot_leg": {
                    "side": spot_side,
                    "price": spot_price,
                    "quantity": spot_quantity,
                    "notional": spot_quantity * spot_price,
                },
                "perp_leg": {
                    "side": perp_side,
                    "price": perp_price,
                    "quantity": perp_quantity,
                    "notional": perp_quantity * perp_price,
                },
                "entry_basis_bps": basis_bps,
                "entry_funding_annual": funding_annual,
                "realized_pnl": 0.0,
                "status": "open",
                "hedge_ratio": (
                    perp_quantity / spot_quantity if spot_quantity > 0 else 1.0
                ),
            }

            # Store in active positions
            self.active_positions[position_id] = position

            # Store in Redis
            redis_key = f"strategy:basis:{symbol}:pos"
            self.redis.set(redis_key, json.dumps(position, default=str))

            # Update metrics
            self.metrics["basis_open_trades"] = len(self.active_positions)
            self.metrics["basis_notional_usd"] += notional_usd
            self.total_trades += 1

            # Log trade opening
            logger.info(
                f"💰 Opened {strategy_type} trade {position_id}: "
                f"{spot_side.upper()} {spot_quantity:.6f} {symbol} spot @ ${spot_price:.2f}, "
                f"{perp_side.upper()} {perp_quantity:.6f} {symbol} perp @ ${perp_price:.2f} "
                f"(basis: {basis_bps:.1f}bps, funding: {funding_annual:.1%})"
            )

            # Store trade event in Redis stream
            trade_event = {
                "position_id": position_id,
                "symbol": symbol,
                "action": "open",
                "strategy_type": strategy_type,
                "notional_usd": notional_usd,
                "entry_basis_bps": basis_bps,
                "entry_funding_annual": funding_annual,
                "timestamp": entry_time,
            }
            self.redis.xadd("strategy:basis:events", trade_event)

            return {"success": True, "position_id": position_id, "position": position}

        except Exception as e:
            logger.error(f"Error opening trade for {symbol}: {e}")
            return {"success": False, "error": str(e)}

    def calculate_position_pnl(self, position: Dict, current_data: Dict) -> float:
        """Calculate current unrealized PnL for position."""
        try:
            spot_leg = position["spot_leg"]
            perp_leg = position["perp_leg"]

            current_spot_price = current_data["spot_price"]
            current_perp_price = current_data["perp_price"]

            # Calculate P&L for each leg
            spot_pnl = 0.0
            perp_pnl = 0.0

            if spot_leg["side"] == "buy":
                spot_pnl = spot_leg["quantity"] * (
                    current_spot_price - spot_leg["price"]
                )
            else:
                spot_pnl = spot_leg["quantity"] * (
                    spot_leg["price"] - current_spot_price
                )

            if perp_leg["side"] == "buy":
                perp_pnl = perp_leg["quantity"] * (
                    current_perp_price - perp_leg["price"]
                )
            else:
                perp_pnl = perp_leg["quantity"] * (
                    perp_leg["price"] - current_perp_price
                )

            total_pnl = spot_pnl + perp_pnl
            return total_pnl

        except Exception as e:
            logger.error(f"Error calculating position PnL: {e}")
            return 0.0

    def close_trade(
        self, position: Dict, market_data: Dict, reason: str
    ) -> Dict[str, Any]:
        """Close an existing basis carry trade."""
        try:
            position_id = position["position_id"]
            symbol = position["symbol"]

            # Calculate realized PnL
            realized_pnl = self.calculate_position_pnl(position, market_data)

            # Update position record
            position["exit_time"] = time.time()
            position["exit_basis_bps"] = market_data["basis_bps"]
            position["realized_pnl"] = realized_pnl
            position["status"] = "closed"
            position["exit_reason"] = reason

            # Calculate position metrics
            notional_usd = position["notional_usd"]
            pnl_pct = realized_pnl / notional_usd
            holding_time = position["exit_time"] - position["entry_time"]

            # Update totals
            self.total_pnl += realized_pnl
            if realized_pnl > 0:
                self.win_trades += 1

            # Remove from active positions
            if position_id in self.active_positions:
                del self.active_positions[position_id]

            # Update metrics
            self.metrics["basis_open_trades"] = len(self.active_positions)
            self.metrics["basis_notional_usd"] = sum(
                pos["notional_usd"] for pos in self.active_positions.values()
            )
            self.metrics["basis_pnl_usd"] = self.total_pnl
            self.metrics["basis_win_rate"] = (
                self.win_trades / self.total_trades if self.total_trades > 0 else 0
            )
            self.metrics["basis_total_trades"] = self.total_trades

            # Store closed position in history
            self.position_history.append(position.copy())
            if len(self.position_history) > 1000:
                self.position_history = self.position_history[-500:]

            # Clean up Redis
            redis_key = f"strategy:basis:{symbol}:pos"
            self.redis.delete(redis_key)

            # Store P&L in Redis
            pnl_key = f"strategy:basis:{symbol}:pnl"
            self.redis.set(pnl_key, realized_pnl)

            # Log trade closure
            logger.info(
                f"🎯 Closed basis trade {position_id}: "
                f"${realized_pnl:+,.2f} ({pnl_pct:+.2%}) "
                f"in {holding_time/3600:.1f}h, reason: {reason}"
            )

            # Store trade event
            trade_event = {
                "position_id": position_id,
                "symbol": symbol,
                "action": "close",
                "reason": reason,
                "realized_pnl": realized_pnl,
                "pnl_pct": pnl_pct,
                "holding_time_hours": holding_time / 3600,
                "timestamp": time.time(),
            }
            self.redis.xadd("strategy:basis:events", trade_event)

            return {
                "success": True,
                "position_id": position_id,
                "realized_pnl": realized_pnl,
                "pnl_pct": pnl_pct,
            }

        except Exception as e:
            logger.error(f"Error closing trade {position.get('position_id')}: {e}")
            return {"success": False, "error": str(e)}

    def check_risk_limits(self) -> Tuple[bool, str]:
        """Check if we're within risk limits."""
        try:
            # Get current equity
            equity_usd = float(self.redis.get("risk:equity_usd") or 100000)

            # Calculate total notional exposure
            total_notional = sum(
                pos["notional_usd"] for pos in self.active_positions.values()
            )
            exposure_pct = total_notional / equity_usd

            # Check gross exposure limit
            if exposure_pct > self.config["max_gross_per_strategy"]:
                return False, f"gross_exposure_exceeded_{exposure_pct:.1%}"

            # Check if system is in halt mode
            mode = self.redis.get("mode")
            if mode == "halt":
                return False, "system_halt"

            # Check capital cap
            capital_effective = float(self.redis.get("risk:capital_effective") or 0.4)
            if capital_effective < 0.1:  # Less than 10% capital allocation
                return False, f"low_capital_allocation_{capital_effective:.1%}"

            return True, "within_limits"

        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False, "risk_check_error"

    def tick(self) -> Dict[str, Any]:
        """Main strategy tick - check all symbols and manage positions."""
        try:
            tick_start = time.time()
            actions_taken = {"opens": 0, "closes": 0, "holds": 0, "errors": 0}

            # Check risk limits
            within_limits, limit_reason = self.check_risk_limits()
            if not within_limits:
                logger.warning(f"⚠️ Risk limits exceeded: {limit_reason}")
                # Close positions if in halt mode
                if "halt" in limit_reason:
                    for position in list(self.active_positions.values()):
                        market_data = self.get_market_data(position["symbol"])
                        if market_data:
                            self.close_trade(position, market_data, "system_halt")
                            actions_taken["closes"] += 1

            # Process each symbol
            for symbol in self.config["symbols"]:
                try:
                    # Get market data
                    market_data = self.get_market_data(symbol)
                    if not market_data:
                        continue

                    basis_bps = market_data["basis_bps"]
                    funding_annual = market_data["funding_annual"]

                    # Check existing positions for this symbol
                    symbol_positions = [
                        pos
                        for pos in self.active_positions.values()
                        if pos["symbol"] == symbol
                    ]

                    if symbol_positions:
                        # Manage existing positions
                        for position in symbol_positions:
                            should_exit, exit_reason = self.exit_signal(
                                position, market_data
                            )

                            if should_exit:
                                result = self.close_trade(
                                    position, market_data, exit_reason
                                )
                                if result["success"]:
                                    actions_taken["closes"] += 1
                                else:
                                    actions_taken["errors"] += 1
                            else:
                                actions_taken["holds"] += 1

                                # Update unrealized PnL in Redis
                                unrealized_pnl = self.calculate_position_pnl(
                                    position, market_data
                                )
                                pnl_key = f"strategy:basis:{symbol}:unrealized_pnl"
                                self.redis.set(pnl_key, unrealized_pnl)

                    else:
                        # Check for new entry opportunities
                        if within_limits:
                            should_enter, entry_reason = self.entry_signal(
                                basis_bps, funding_annual
                            )

                            if should_enter:
                                notional_usd = self.calculate_position_size(
                                    symbol, market_data
                                )

                                if notional_usd > 0:
                                    result = self.open_trade(
                                        symbol, notional_usd, market_data
                                    )
                                    if result["success"]:
                                        actions_taken["opens"] += 1
                                    else:
                                        actions_taken["errors"] += 1

                except Exception as e:
                    logger.error(f"Error processing {symbol} in tick: {e}")
                    actions_taken["errors"] += 1

            # Update metrics in Redis
            for metric, value in self.metrics.items():
                self.redis.set(f"metric:{metric}", value)

            tick_duration = time.time() - tick_start

            # Log summary periodically
            total_actions = sum(actions_taken.values())
            if total_actions > 0:
                logger.info(
                    f"🔄 Tick summary: {actions_taken['opens']} opens, {actions_taken['closes']} closes, "
                    f"{actions_taken['holds']} holds, active positions: {len(self.active_positions)}"
                )

            return {
                "timestamp": tick_start,
                "status": "completed",
                "actions_taken": actions_taken,
                "active_positions": len(self.active_positions),
                "total_pnl": self.total_pnl,
                "tick_duration": tick_duration,
                "within_limits": within_limits,
                "limit_reason": limit_reason,
            }

        except Exception as e:
            logger.error(f"Error in strategy tick: {e}")
            return {"timestamp": time.time(), "status": "error", "error": str(e)}

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive strategy status."""
        try:
            # Calculate portfolio metrics
            total_notional = sum(
                pos["notional_usd"] for pos in self.active_positions.values()
            )

            # Get recent trades
            recent_trades = self.position_history[-5:] if self.position_history else []

            status = {
                "strategy": "spot_perp_basis_carry",
                "timestamp": time.time(),
                "config": self.config,
                "active_positions": len(self.active_positions),
                "total_notional_usd": total_notional,
                "total_pnl_usd": self.total_pnl,
                "total_trades": self.total_trades,
                "win_rate": (
                    self.win_trades / self.total_trades if self.total_trades > 0 else 0
                ),
                "metrics": self.metrics.copy(),
                "positions": [
                    {
                        "position_id": pos["position_id"],
                        "symbol": pos["symbol"],
                        "strategy_type": pos["strategy_type"],
                        "notional_usd": pos["notional_usd"],
                        "entry_basis_bps": pos["entry_basis_bps"],
                        "age_hours": (time.time() - pos["entry_time"]) / 3600,
                        "unrealized_pnl": self.calculate_position_pnl(
                            pos, self.get_market_data(pos["symbol"])
                        ),
                    }
                    for pos in self.active_positions.values()
                ],
                "recent_trades": recent_trades,
            }

            return status

        except Exception as e:
            return {
                "strategy": "spot_perp_basis_carry",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }

    def loop(self):
        """Main strategy loop."""
        logger.info("🚀 Starting basis carry strategy loop")

        try:
            while True:
                try:
                    # Run strategy tick
                    result = self.tick()

                    if result["status"] == "completed":
                        total_actions = sum(result["actions_taken"].values())
                        if total_actions > 0 or len(self.active_positions) > 0:
                            logger.debug(
                                f"📊 Tick: {total_actions} actions, "
                                f"{len(self.active_positions)} positions, "
                                f"P&L: ${self.total_pnl:+,.2f}"
                            )

                    # Sleep until next tick
                    time.sleep(self.config["tick_interval"])

                except Exception as e:
                    logger.error(f"Error in strategy loop: {e}")
                    time.sleep(30)  # Wait longer on error

        except KeyboardInterrupt:
            logger.info("Basis carry strategy stopped by user")

            # Close all positions on exit
            logger.info("Closing all active positions...")
            for position in list(self.active_positions.values()):
                market_data = self.get_market_data(position["symbol"])
                if market_data:
                    self.close_trade(position, market_data, "strategy_shutdown")

        except Exception as e:
            logger.error(f"Fatal error in strategy loop: {e}")


def main():
    """Main entry point for basis carry strategy."""
    import argparse

    parser = argparse.ArgumentParser(description="Spot-Perp Basis Carry Strategy")
    parser.add_argument("--run", action="store_true", help="Run strategy loop")
    parser.add_argument("--tick", action="store_true", help="Run single strategy tick")
    parser.add_argument("--status", action="store_true", help="Show strategy status")

    args = parser.parse_args()

    # Create strategy
    strategy = SpotPerpBasisCarryStrategy()

    if args.status:
        # Show status report
        status = strategy.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.tick:
        # Run single tick
        result = strategy.tick()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.run:
        # Run strategy loop
        strategy.loop()
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
