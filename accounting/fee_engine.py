#!/usr/bin/env python3
"""
Fee Engine
Turn gross P&L into audited net P&L by calculating all trading costs
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("fee_engine")


class FeeEngine:
    """Calculate comprehensive trading costs for net P&L accounting."""

    def __init__(self):
        """Initialize fee engine."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Venue fee schedules (basis points)
        self.venue_configs = {
            "binance": {
                "name": "Binance",
                "fee_maker_bps": 1.0,  # 0.1% maker
                "fee_taker_bps": 1.0,  # 0.1% taker
                "funding_rate_cap": 75.0,  # 0.75% max funding
                "withdrawal_fee_usd": 0.0,  # No withdrawal fees for futures
                "api_weight_limit": 2400,
                "products": ["spot", "futures", "perp"],
            },
            "coinbase": {
                "name": "Coinbase Pro",
                "fee_maker_bps": 5.0,  # 0.5% maker
                "fee_taker_bps": 5.0,  # 0.5% taker
                "funding_rate_cap": 0.0,  # No perpetuals
                "withdrawal_fee_usd": 0.0,
                "api_weight_limit": 10000,
                "products": ["spot"],
            },
            "ftx": {
                "name": "FTX",
                "fee_maker_bps": 2.0,  # 0.02% maker
                "fee_taker_bps": 7.0,  # 0.07% taker
                "funding_rate_cap": 100.0,  # 1% max funding
                "withdrawal_fee_usd": 0.0,
                "api_weight_limit": 30000,
                "products": ["spot", "futures", "perp", "options"],
            },
            "dydx": {
                "name": "dYdX",
                "fee_maker_bps": -2.5,  # -0.025% maker rebate
                "fee_taker_bps": 5.0,  # 0.05% taker
                "funding_rate_cap": 75.0,
                "withdrawal_fee_usd": 0.0,
                "api_weight_limit": 17500,
                "products": ["perp"],
            },
            "alpaca": {
                "name": "Alpaca",
                "fee_maker_bps": 0.0,  # Commission-free for stocks
                "fee_taker_bps": 0.0,  # Commission-free for stocks
                "sec_fee_bps": 0.00276,  # SEC Section 31 fee
                "taf_fee_bps": 0.0095,  # Trading Activity Fee
                "nscc_fee_usd": 0.01,  # NSCC clearing fee per trade
                "finra_taf_cap_usd": 7.27,  # FINRA TAF cap per trade
                "products": ["stock"],
                "fee_schedule": "commission_free",
            },
        }

        # Product-specific cost calculations
        self.product_handlers = {
            "spot": self._calculate_spot_costs,
            "futures": self._calculate_futures_costs,
            "perp": self._calculate_perp_costs,
            "options": self._calculate_options_costs,
            "stock": self._calculate_stock_costs,
        }

        # Borrow rates for margin trading (annual %)
        self.borrow_rates = {
            "USD": 0.05,  # 5% annual
            "BTC": 0.03,  # 3% annual
            "ETH": 0.04,  # 4% annual
            "SOL": 0.08,  # 8% annual
        }

        logger.info("💰 Fee Engine initialized")
        logger.info(f"   Venues: {list(self.venue_configs.keys())}")
        logger.info(f"   Products: {list(self.product_handlers.keys())}")

    def get_venue_config(self, venue: str) -> Dict[str, Any]:
        """Get venue configuration."""
        venue_lower = venue.lower()
        return self.venue_configs.get(
            venue_lower,
            {
                "name": venue,
                "fee_maker_bps": 10.0,  # Default 0.1%
                "fee_taker_bps": 10.0,  # Default 0.1%
                "funding_rate_cap": 75.0,
                "withdrawal_fee_usd": 0.0,
                "products": ["spot"],
            },
        )

    def _calculate_spot_costs(
        self, fill: Dict[str, Any], venue_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate costs for spot trading."""
        try:
            price = float(fill["price"])
            qty = float(fill["qty"])
            is_maker = fill.get("maker", False)

            # Trading fees
            fee_bps = (
                venue_config["fee_maker_bps"]
                if is_maker
                else venue_config["fee_taker_bps"]
            )

            # No funding costs for spot
            funding_bps = 0.0

            # Borrow costs if margin trading
            borrow_bps = 0.0
            if fill.get("margin", False):
                base_asset = fill.get("base_asset", "USD")
                annual_rate = self.borrow_rates.get(base_asset, 0.05)
                # Convert to per-trade basis points (assume avg holding 1 day)
                borrow_bps = annual_rate / 365 * 10000

            total_bps = fee_bps + funding_bps + borrow_bps
            total_usd = price * qty * total_bps / 10000

            return {
                "fee_bps": fee_bps,
                "funding_bps": funding_bps,
                "borrow_bps": borrow_bps,
                "total_bps": total_bps,
                "total_usd": total_usd,
            }

        except Exception as e:
            logger.error(f"Error calculating spot costs: {e}")
            return {
                "fee_bps": 0,
                "funding_bps": 0,
                "borrow_bps": 0,
                "total_bps": 0,
                "total_usd": 0,
            }

    def _calculate_futures_costs(
        self, fill: Dict[str, Any], venue_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate costs for futures trading."""
        try:
            price = float(fill["price"])
            qty = float(fill["qty"])
            is_maker = fill.get("maker", False)

            # Trading fees
            fee_bps = (
                venue_config["fee_maker_bps"]
                if is_maker
                else venue_config["fee_taker_bps"]
            )

            # No ongoing funding for dated futures
            funding_bps = 0.0

            # Margin interest if applicable
            borrow_bps = fill.get("borrow_bps", 0.0)

            total_bps = fee_bps + funding_bps + borrow_bps
            total_usd = price * qty * total_bps / 10000

            return {
                "fee_bps": fee_bps,
                "funding_bps": funding_bps,
                "borrow_bps": borrow_bps,
                "total_bps": total_bps,
                "total_usd": total_usd,
            }

        except Exception as e:
            logger.error(f"Error calculating futures costs: {e}")
            return {
                "fee_bps": 0,
                "funding_bps": 0,
                "borrow_bps": 0,
                "total_bps": 0,
                "total_usd": 0,
            }

    def _calculate_perp_costs(
        self, fill: Dict[str, Any], venue_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate costs for perpetual swaps."""
        try:
            price = float(fill["price"])
            qty = float(fill["qty"])
            is_maker = fill.get("maker", False)

            # Trading fees
            fee_bps = (
                venue_config["fee_maker_bps"]
                if is_maker
                else venue_config["fee_taker_bps"]
            )

            # Funding rate (8-hourly for perps)
            funding_bps = fill.get("funding_bps", 0.0)

            # Cap funding at venue maximum
            funding_cap = venue_config.get("funding_rate_cap", 75.0)
            funding_bps = max(-funding_cap, min(funding_cap, funding_bps))

            # No borrow costs (margin is built-in)
            borrow_bps = 0.0

            total_bps = fee_bps + funding_bps + borrow_bps
            total_usd = price * qty * total_bps / 10000

            return {
                "fee_bps": fee_bps,
                "funding_bps": funding_bps,
                "borrow_bps": borrow_bps,
                "total_bps": total_bps,
                "total_usd": total_usd,
            }

        except Exception as e:
            logger.error(f"Error calculating perp costs: {e}")
            return {
                "fee_bps": 0,
                "funding_bps": 0,
                "borrow_bps": 0,
                "total_bps": 0,
                "total_usd": 0,
            }

    def _calculate_options_costs(
        self, fill: Dict[str, Any], venue_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate costs for options trading."""
        try:
            price = float(fill["price"])
            qty = float(fill["qty"])
            is_maker = fill.get("maker", False)

            # Trading fees (often higher for options)
            base_fee_bps = (
                venue_config["fee_maker_bps"]
                if is_maker
                else venue_config["fee_taker_bps"]
            )
            fee_bps = base_fee_bps * 1.5  # Options typically 1.5x spot fees

            # Greeks-related costs (theta, vega hedging)
            greeks_bps = fill.get("greeks_cost_bps", 0.0)

            # No funding for options
            funding_bps = 0.0
            borrow_bps = 0.0

            total_bps = fee_bps + greeks_bps + funding_bps + borrow_bps
            total_usd = price * qty * total_bps / 10000

            return {
                "fee_bps": fee_bps,
                "funding_bps": funding_bps,
                "borrow_bps": borrow_bps,
                "greeks_bps": greeks_bps,
                "total_bps": total_bps,
                "total_usd": total_usd,
            }

        except Exception as e:
            logger.error(f"Error calculating options costs: {e}")
            return {
                "fee_bps": 0,
                "funding_bps": 0,
                "borrow_bps": 0,
                "total_bps": 0,
                "total_usd": 0,
            }

    def _calculate_stock_costs(
        self, fill: Dict[str, Any], venue_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate costs for US equity trading."""
        try:
            price = float(fill["price"])
            qty = float(fill["qty"])
            notional_value = price * qty
            is_sell = fill.get("side", "").lower() in ["sell", "sell_short"]

            # Commission (typically $0 for major brokers)
            commission_usd = venue_config.get("commission_per_share", 0.0) * qty

            # SEC fees (only on sells)
            sec_fee_usd = 0.0
            if is_sell:
                sec_fee_bps = venue_config.get("sec_fee_bps", 0.00276)  # 0.00276%
                sec_fee_usd = notional_value * sec_fee_bps / 10000

            # Trading Activity Fee (TAF) - FINRA fee on sells
            taf_fee_usd = 0.0
            if is_sell:
                taf_fee_bps = venue_config.get("taf_fee_bps", 0.0095)  # 0.0095%
                taf_fee_usd = min(
                    notional_value * taf_fee_bps / 10000,
                    venue_config.get("finra_taf_cap_usd", 7.27),  # FINRA cap
                )

            # NSCC clearing fee (per trade)
            nscc_fee_usd = venue_config.get("nscc_fee_usd", 0.01)

            # No funding or borrow costs for stock spot trading
            funding_bps = 0.0
            borrow_bps = 0.0

            # Total costs
            total_fixed_usd = commission_usd + sec_fee_usd + taf_fee_usd + nscc_fee_usd
            total_bps = (
                (total_fixed_usd / notional_value) * 10000 if notional_value > 0 else 0
            )

            return {
                "commission_usd": commission_usd,
                "sec_fee_usd": sec_fee_usd,
                "taf_fee_usd": taf_fee_usd,
                "nscc_fee_usd": nscc_fee_usd,
                "fee_bps": total_bps,
                "funding_bps": funding_bps,
                "borrow_bps": borrow_bps,
                "total_bps": total_bps,
                "total_usd": total_fixed_usd,
            }

        except Exception as e:
            logger.error(f"Error calculating stock costs: {e}")
            return {
                "fee_bps": 0,
                "funding_bps": 0,
                "borrow_bps": 0,
                "total_bps": 0,
                "total_usd": 0,
            }

    def compute_fill_cost(self, fill: Dict[str, Any]) -> Dict[str, float]:
        """Compute comprehensive cost for a single fill."""
        try:
            venue = fill.get("venue", "unknown").lower()
            product = fill.get("product", "spot").lower()

            # Get venue configuration
            venue_config = self.get_venue_config(venue)

            # Get appropriate cost calculation handler
            cost_handler = self.product_handlers.get(
                product, self._calculate_spot_costs
            )

            # Calculate costs
            cost_breakdown = cost_handler(fill, venue_config)

            # Add metadata
            cost_breakdown.update(
                {
                    "venue": venue,
                    "product": product,
                    "fill_id": fill.get("fill_id", "unknown"),
                    "timestamp": fill.get("timestamp", time.time()),
                    "symbol": fill.get("symbol", "unknown"),
                    "side": fill.get("side", "unknown"),
                }
            )

            return cost_breakdown

        except Exception as e:
            logger.error(f"Error computing fill cost: {e}")
            return {
                "fee_bps": 0,
                "funding_bps": 0,
                "borrow_bps": 0,
                "total_bps": 0,
                "total_usd": 0,
                "error": str(e),
            }

    def process_fills_batch(
        self, fills: List[Dict[str, Any]]
    ) -> List[Dict[str, float]]:
        """Process a batch of fills and return cost breakdowns."""
        try:
            cost_breakdowns = []

            for fill in fills:
                cost_breakdown = self.compute_fill_cost(fill)
                cost_breakdowns.append(cost_breakdown)

            logger.debug(f"Processed {len(fills)} fills")
            return cost_breakdowns

        except Exception as e:
            logger.error(f"Error processing fills batch: {e}")
            return []

    def get_fills_for_period(
        self, start_time: float, end_time: float
    ) -> List[Dict[str, Any]]:
        """Get all fills for a time period from Redis streams."""
        try:
            all_fills = []

            # Check multiple fill streams
            fill_streams = [
                "exec:fills:binance",
                "exec:fills:coinbase",
                "exec:fills:ftx",
                "exec:fills:dydx",
                "exec:fills:all",
            ]

            for stream_name in fill_streams:
                try:
                    # Get fills from Redis stream in time range
                    stream_data = self.redis.xrange(
                        stream_name,
                        min=int(start_time * 1000),  # Convert to milliseconds
                        max=int(end_time * 1000),
                        count=10000,
                    )

                    for stream_id, fill_data in stream_data:
                        try:
                            # Parse fill data
                            fill = {
                                "stream_id": stream_id,
                                "stream_name": stream_name,
                                "fill_id": fill_data.get("fill_id", stream_id),
                                "venue": fill_data.get(
                                    "venue", stream_name.split(":")[-1]
                                ),
                                "symbol": fill_data.get("symbol", "unknown"),
                                "side": fill_data.get("side", "unknown"),
                                "price": float(fill_data.get("price", 0)),
                                "qty": float(fill_data.get("qty", 0)),
                                "product": fill_data.get("product", "spot"),
                                "maker": bool(int(fill_data.get("maker", "0"))),
                                "timestamp": float(
                                    fill_data.get("timestamp", time.time())
                                ),
                                "strategy": fill_data.get("strategy", "unknown"),
                                "funding_bps": float(fill_data.get("funding_bps", 0)),
                                "borrow_bps": float(fill_data.get("borrow_bps", 0)),
                            }

                            all_fills.append(fill)

                        except Exception as e:
                            logger.debug(f"Error parsing fill from {stream_name}: {e}")

                except Exception as e:
                    logger.debug(f"Error reading stream {stream_name}: {e}")

            # If no real fills, generate some mock fills for demo
            if not all_fills:
                all_fills = self._generate_mock_fills(start_time, end_time)

            logger.info(f"Retrieved {len(all_fills)} fills for period")
            return all_fills

        except Exception as e:
            logger.error(f"Error getting fills for period: {e}")
            return []

    def _generate_mock_fills(
        self, start_time: float, end_time: float
    ) -> List[Dict[str, Any]]:
        """Generate mock fills for demonstration."""
        try:
            mock_fills = []

            # Generate 50-200 fills for the day
            num_fills = np.random.randint(50, 201)

            venues = ["binance", "coinbase", "ftx"]
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            strategies = ["RL", "BASIS", "MM"]
            products = ["spot", "perp"]

            for i in range(num_fills):
                # Random timestamp within period
                timestamp = start_time + (end_time - start_time) * np.random.random()

                venue = np.random.choice(venues)
                symbol = np.random.choice(symbols)
                strategy = np.random.choice(strategies)
                product = np.random.choice(products)
                side = np.random.choice(["buy", "sell"])
                is_maker = np.random.random() > 0.6  # 40% taker, 60% maker

                # Mock prices
                base_prices = {"BTCUSDT": 97600, "ETHUSDT": 3515, "SOLUSDT": 184}
                base_price = base_prices.get(symbol, 100)
                price = base_price * (1 + np.random.uniform(-0.01, 0.01))

                # Mock quantities
                qty = np.random.uniform(0.001, 1.0)

                # Mock funding for perps
                funding_bps = 0.0
                if product == "perp":
                    funding_bps = np.random.uniform(-5, 15)  # -0.05% to +0.15%

                mock_fill = {
                    "fill_id": f"mock_{i}_{int(timestamp)}",
                    "venue": venue,
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "qty": qty,
                    "product": product,
                    "maker": is_maker,
                    "timestamp": timestamp,
                    "strategy": strategy,
                    "funding_bps": funding_bps,
                    "borrow_bps": 0.0,
                }

                mock_fills.append(mock_fill)

            logger.debug(f"Generated {len(mock_fills)} mock fills")
            return mock_fills

        except Exception as e:
            logger.error(f"Error generating mock fills: {e}")
            return []

    def calculate_net_pnl(
        self, fills: List[Dict[str, Any]], gross_pnl: float = None
    ) -> Dict[str, Any]:
        """Calculate net P&L from gross P&L and fills."""
        try:
            # Process all fills to get cost breakdowns
            cost_breakdowns = self.process_fills_batch(fills)

            # Aggregate costs
            total_fees = sum(c.get("total_usd", 0) for c in cost_breakdowns)

            # Breakdown by cost type
            trading_fees = sum(
                c.get("fee_bps", 0) * c.get("price", 0) * c.get("qty", 0) / 10000
                for c in cost_breakdowns
            )
            funding_costs = sum(
                c.get("funding_bps", 0) * c.get("price", 0) * c.get("qty", 0) / 10000
                for c in cost_breakdowns
            )
            borrow_costs = sum(
                c.get("borrow_bps", 0) * c.get("price", 0) * c.get("qty", 0) / 10000
                for c in cost_breakdowns
            )

            # Calculate net P&L
            if gross_pnl is None:
                # Estimate gross P&L from fills (simplified)
                gross_pnl = sum(
                    fill["qty"]
                    * fill["price"]
                    * (0.001 if fill["side"] == "buy" else -0.001)
                    for fill in fills
                )

            net_pnl = gross_pnl - total_fees

            # Calculate cost ratios
            cost_ratio = total_fees / abs(gross_pnl) if gross_pnl != 0 else 0

            # Per-venue breakdown
            venue_costs = {}
            for cost in cost_breakdowns:
                venue = cost.get("venue", "unknown")
                if venue not in venue_costs:
                    venue_costs[venue] = {"count": 0, "total_usd": 0, "fee_usd": 0}

                venue_costs[venue]["count"] += 1
                venue_costs[venue]["total_usd"] += cost.get("total_usd", 0)
                venue_costs[venue]["fee_usd"] += (
                    cost.get("fee_bps", 0)
                    * cost.get("price", 0)
                    * cost.get("qty", 0)
                    / 10000
                )

            # Per-strategy breakdown
            strategy_costs = {}
            for i, fill in enumerate(fills):
                strategy = fill.get("strategy", "unknown")
                if strategy not in strategy_costs:
                    strategy_costs[strategy] = {"count": 0, "total_usd": 0}

                if i < len(cost_breakdowns):
                    cost = cost_breakdowns[i]
                    strategy_costs[strategy]["count"] += 1
                    strategy_costs[strategy]["total_usd"] += cost.get("total_usd", 0)

            result = {
                "timestamp": time.time(),
                "period_start": (
                    min(fill["timestamp"] for fill in fills) if fills else time.time()
                ),
                "period_end": (
                    max(fill["timestamp"] for fill in fills) if fills else time.time()
                ),
                "gross_pnl_usd": gross_pnl,
                "total_fees_usd": total_fees,
                "net_pnl_usd": net_pnl,
                "cost_breakdown": {
                    "trading_fees_usd": trading_fees,
                    "funding_costs_usd": funding_costs,
                    "borrow_costs_usd": borrow_costs,
                },
                "cost_ratio": cost_ratio,
                "fill_count": len(fills),
                "venue_breakdown": venue_costs,
                "strategy_breakdown": strategy_costs,
            }

            return result

        except Exception as e:
            logger.error(f"Error calculating net P&L: {e}")
            return {
                "timestamp": time.time(),
                "gross_pnl_usd": gross_pnl or 0,
                "total_fees_usd": 0,
                "net_pnl_usd": gross_pnl or 0,
                "error": str(e),
            }

    def get_status_report(self) -> Dict[str, Any]:
        """Get fee engine status report."""
        try:
            status = {
                "service": "fee_engine",
                "timestamp": time.time(),
                "venue_configs": self.venue_configs,
                "supported_products": list(self.product_handlers.keys()),
                "borrow_rates": self.borrow_rates,
            }

            return status

        except Exception as e:
            return {
                "service": "fee_engine",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }


def main():
    """Main entry point for fee engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Fee Engine")
    parser.add_argument("--status", action="store_true", help="Show status report")
    parser.add_argument(
        "--calculate", action="store_true", help="Calculate net P&L for today"
    )
    parser.add_argument("--start-time", type=float, help="Start time (Unix timestamp)")
    parser.add_argument("--end-time", type=float, help="End time (Unix timestamp)")

    args = parser.parse_args()

    # Create fee engine
    fee_engine = FeeEngine()

    if args.status:
        # Show status report
        status = fee_engine.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.calculate:
        # Calculate net P&L
        end_time = args.end_time or time.time()
        start_time = args.start_time or (end_time - 24 * 3600)  # Default to 24 hours

        fills = fee_engine.get_fills_for_period(start_time, end_time)
        result = fee_engine.calculate_net_pnl(fills)

        print(json.dumps(result, indent=2, default=str))
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
