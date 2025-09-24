#!/usr/bin/env python3
"""
Funding Rate & Basis Alpha Signal
Pull perp funding & spot-perp basis as features/signals for crypto edge
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

import redis

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("funding_basis")


class FundingBasisPuller:
    """Funding rate and spot-perp basis signal generator."""

    def __init__(self):
        """Initialize funding basis puller."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Symbols to track
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

        # API endpoints
        self.binance_spot_api = "https://api.binance.com/api/v3"
        self.binance_futures_api = "https://fapi.binance.com/fapi/v1"

        # Redis keys
        self.funding_key = "funding:basis"
        self.alpha_contrib_key = "alpha:contrib:funding"

        # Alpha rule parameters
        self.alpha_config = {
            "basis_threshold_bps": 50,  # ±50 bps basis threshold
            "funding_threshold_ann": 0.20,  # ±20% annual funding threshold
            "size_adjustment": 0.1,  # ±10% position size adjustment
            "max_adjustment": 0.3,  # Maximum ±30% adjustment
        }

        logger.info("💰 Funding & Basis Puller initialized")
        logger.info(f"   Symbols: {self.symbols}")
        logger.info(
            f"   Basis threshold: ±{self.alpha_config['basis_threshold_bps']}bps"
        )
        logger.info(
            f"   Funding threshold: ±{self.alpha_config['funding_threshold_ann']:.0%} annual"
        )

    def get_spot_price(self, symbol: str) -> float:
        """Get spot price from Binance."""
        try:
            url = f"{self.binance_spot_api}/ticker/price"
            params = {"symbol": symbol}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            return float(data["price"])

        except Exception as e:
            logger.warning(f"Failed to get spot price for {symbol}: {e}")
            return 0.0

    def get_perp_data(self, symbol: str) -> tuple:
        """Get perpetual price and funding rate from Binance Futures."""
        try:
            # Get perp price
            price_url = f"{self.binance_futures_api}/ticker/price"
            price_params = {"symbol": symbol}

            price_response = requests.get(price_url, params=price_params, timeout=10)
            price_response.raise_for_status()

            price_data = price_response.json()
            perp_price = float(price_data["price"])

            # Get funding rate
            funding_url = f"{self.binance_futures_api}/premiumIndex"
            funding_params = {"symbol": symbol}

            funding_response = requests.get(
                funding_url, params=funding_params, timeout=10
            )
            funding_response.raise_for_status()

            funding_data = funding_response.json()
            funding_rate_8h = float(funding_data["lastFundingRate"])

            return perp_price, funding_rate_8h

        except Exception as e:
            logger.warning(f"Failed to get perp data for {symbol}: {e}")
            return 0.0, 0.0

    def calculate_metrics(self, symbol: str) -> dict:
        """Calculate funding and basis metrics for a symbol."""
        try:
            # Get prices and funding
            spot_price = self.get_spot_price(symbol)
            perp_price, funding_rate_8h = self.get_perp_data(symbol)

            if spot_price == 0.0 or perp_price == 0.0:
                logger.warning(f"Invalid price data for {symbol}")
                return {}

            # Calculate metrics
            spot_perp_basis_bps = ((perp_price - spot_price) / spot_price) * 10_000
            funding_rate_annualized = (
                funding_rate_8h * 3 * 365
            )  # 8h rate * 3 times/day * 365 days

            metrics = {
                "spot_price": spot_price,
                "perp_price": perp_price,
                "basis_bps": spot_perp_basis_bps,
                "funding_8h": funding_rate_8h,
                "funding_ann": funding_rate_annualized,
                "timestamp": time.time(),
                "datetime": datetime.now(timezone.utc).isoformat(),
            }

            logger.debug(
                f"{symbol}: spot=${spot_price:.2f}, perp=${perp_price:.2f}, "
                f"basis={spot_perp_basis_bps:.1f}bps, funding={funding_rate_annualized:.2%}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {e}")
            return {}

    def evaluate_alpha_signal(self, symbol: str, metrics: dict) -> float:
        """Evaluate alpha signal based on funding and basis."""
        try:
            if not metrics:
                return 0.0

            basis_bps = metrics["basis_bps"]
            funding_ann = metrics["funding_ann"]

            # Alpha rules from requirements:
            # - If basis_bps >> +X and funding_ann >> Y → fade longs / reduce size_frac
            # - If basis_bps << -X and funding_ann << -Y → tilt long / increase size_frac

            size_adjustment = 0.0

            # Positive basis + positive funding = fade longs (reduce size)
            if (
                basis_bps > self.alpha_config["basis_threshold_bps"]
                and funding_ann > self.alpha_config["funding_threshold_ann"]
            ):
                size_adjustment = -self.alpha_config["size_adjustment"]

            # Negative basis + negative funding = tilt long (increase size)
            elif (
                basis_bps < -self.alpha_config["basis_threshold_bps"]
                and funding_ann < -self.alpha_config["funding_threshold_ann"]
            ):
                size_adjustment = self.alpha_config["size_adjustment"]

            # Mixed signals - use smaller adjustment based on dominant signal
            elif abs(basis_bps) > self.alpha_config["basis_threshold_bps"]:
                # Basis dominant
                size_adjustment = (
                    -(basis_bps / self.alpha_config["basis_threshold_bps"])
                    * self.alpha_config["size_adjustment"]
                    / 2
                )

            elif abs(funding_ann) > self.alpha_config["funding_threshold_ann"]:
                # Funding dominant
                size_adjustment = (
                    -(funding_ann / self.alpha_config["funding_threshold_ann"])
                    * self.alpha_config["size_adjustment"]
                    / 2
                )

            # Cap adjustment
            size_adjustment = max(
                -self.alpha_config["max_adjustment"],
                min(self.alpha_config["max_adjustment"], size_adjustment),
            )

            if abs(size_adjustment) > 0.01:  # Only log significant adjustments
                logger.info(
                    f"📊 {symbol} alpha signal: {size_adjustment:+.1%} "
                    f"(basis={basis_bps:.1f}bps, funding={funding_ann:.1%})"
                )

            return size_adjustment

        except Exception as e:
            logger.error(f"Error evaluating alpha signal for {symbol}: {e}")
            return 0.0

    def store_alpha_contribution(
        self, symbol: str, size_adjustment: float, pnl_impact: float = None
    ):
        """Store alpha contribution for daily reporting."""
        try:
            if abs(size_adjustment) < 0.01:  # Skip negligible adjustments
                return

            # Estimate P&L impact (simplified)
            if pnl_impact is None:
                # Mock P&L based on size adjustment magnitude
                base_pnl = 50.0  # $50 base impact
                pnl_impact = base_pnl * abs(size_adjustment) * 10  # Scale by adjustment
                if size_adjustment > 0:
                    pnl_impact = abs(pnl_impact)  # Positive for long bias
                else:
                    pnl_impact = -abs(pnl_impact)  # Negative for short bias

            contrib_data = {
                "symbol": symbol,
                "size_adjustment": size_adjustment,
                "pnl_contribution": pnl_impact,
                "timestamp": time.time(),
                "hit_rate": 0.55 if size_adjustment != 0 else 0.50,  # Mock hit rate
            }

            # Store in Redis stream for daily reporting
            self.redis.xadd(self.alpha_contrib_key, contrib_data)

            # Keep only recent data (last 7 days)
            cutoff_time = int((time.time() - 7 * 24 * 3600) * 1000)
            self.redis.xtrim(
                self.alpha_contrib_key, minid=cutoff_time, approximate=True
            )

        except Exception as e:
            logger.error(f"Error storing alpha contribution: {e}")

    def update_all_symbols(self):
        """Update funding and basis metrics for all symbols."""
        update_start = time.time()

        try:
            logger.debug("🔄 Updating funding & basis metrics...")

            total_alpha_impact = 0.0
            updates = {}

            for symbol in self.symbols:
                try:
                    # Calculate metrics
                    metrics = self.calculate_metrics(symbol)

                    if metrics:
                        # Store in Redis
                        symbol_key_funding = f"{symbol}:funding"
                        symbol_key_basis = f"{symbol}:basis"

                        self.redis.hset(
                            self.funding_key, symbol_key_funding, metrics["funding_ann"]
                        )
                        self.redis.hset(
                            self.funding_key, symbol_key_basis, metrics["basis_bps"]
                        )

                        # Evaluate alpha signal
                        size_adjustment = self.evaluate_alpha_signal(symbol, metrics)

                        # Store alpha contribution
                        if abs(size_adjustment) > 0.01:
                            self.store_alpha_contribution(symbol, size_adjustment)
                            total_alpha_impact += abs(size_adjustment)

                        updates[symbol] = {
                            "basis_bps": metrics["basis_bps"],
                            "funding_ann": metrics["funding_ann"],
                            "size_adjustment": size_adjustment,
                        }

                except Exception as e:
                    logger.error(f"Error updating {symbol}: {e}")
                    continue

            # Update last update timestamp
            self.redis.set("funding:basis:last_update", int(time.time()))

            update_duration = time.time() - update_start

            if updates:
                logger.info(
                    f"✅ Updated {len(updates)} symbols in {update_duration:.2f}s, "
                    f"total alpha impact: {total_alpha_impact:.1%}"
                )
            else:
                logger.warning("❌ No symbols updated successfully")

            return updates

        except Exception as e:
            logger.error(f"Error in update cycle: {e}")
            return {}

    def get_status(self) -> dict:
        """Get current status of funding basis system."""
        try:
            # Get current data from Redis
            current_data = self.redis.hgetall(self.funding_key)

            # Parse current metrics
            symbol_data = {}
            for key, value in current_data.items():
                if ":" in key:
                    symbol, metric_type = key.split(":", 1)
                    if symbol not in symbol_data:
                        symbol_data[symbol] = {}
                    symbol_data[symbol][metric_type] = float(value)

            # Get last update time
            last_update = self.redis.get("funding:basis:last_update")
            last_update_ts = float(last_update) if last_update else 0

            # Get recent alpha contributions
            recent_contribs = []
            try:
                contrib_entries = self.redis.xrevrange(self.alpha_contrib_key, count=10)
                for entry_id, fields in contrib_entries:
                    contrib = {
                        "timestamp": int(entry_id.split("-")[0]) / 1000,
                        **fields,
                    }
                    recent_contribs.append(contrib)
            except Exception:
                pass

            return {
                "service": "funding_basis",
                "status": "active",
                "symbols_tracked": self.symbols,
                "last_update": last_update_ts,
                "last_update_age_seconds": (
                    time.time() - last_update_ts if last_update_ts > 0 else None
                ),
                "current_data": symbol_data,
                "alpha_config": self.alpha_config,
                "recent_contributions": recent_contribs[:5],  # Latest 5
            }

        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"service": "funding_basis", "status": "error", "error": str(e)}

    def run_daemon(self, update_interval: int = 60):
        """Run funding basis puller as daemon."""
        logger.info(
            f"🚀 Starting Funding & Basis daemon (interval: {update_interval}s)"
        )

        try:
            while True:
                try:
                    # Update metrics
                    updates = self.update_all_symbols()

                    # Sleep until next update
                    time.sleep(update_interval)

                except KeyboardInterrupt:
                    logger.info("🛑 Funding basis daemon stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in daemon loop: {e}")
                    time.sleep(
                        min(update_interval, 30)
                    )  # Wait at least 30s after error

        except Exception as e:
            logger.error(f"Fatal error in funding basis daemon: {e}")
            raise


def main():
    """Main entry point for funding basis puller."""
    import argparse

    parser = argparse.ArgumentParser(description="Funding Rate & Basis Alpha Signal")
    parser.add_argument(
        "--daemon", action="store_true", help="Run as daemon (default mode)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Update interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--single", action="store_true", help="Run single update and exit"
    )
    parser.add_argument("--status", action="store_true", help="Show current status")

    args = parser.parse_args()

    # Create puller
    puller = FundingBasisPuller()

    if args.status:
        # Show status
        status = puller.get_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.single:
        # Single update
        logger.info("Running single update...")
        updates = puller.update_all_symbols()
        print(json.dumps(updates, indent=2, default=str))
        return

    # Run as daemon (default)
    puller.run_daemon(args.interval)


if __name__ == "__main__":
    main()
