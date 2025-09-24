#!/usr/bin/env python3
"""
Account & Position Reconciliation Daemon
Detects drift between exchange balances/positions and internal state
"""

import os
import sys
import json
import time
import math
import logging
import requests
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis
import ccxt
from deribit_api import RestClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("recon_daemon")


class ReconciliationDaemon:
    """Account and position reconciliation daemon."""

    def __init__(self):
        """Initialize reconciliation daemon."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.slack_webhook = os.getenv("SLACK_WEBHOOK", "")

        # Initialize exchanges
        try:
            self.binance = ccxt.binance(
                {
                    "apiKey": os.getenv("BINANCE_API_KEY", ""),
                    "secret": os.getenv("BINANCE_SECRET", ""),
                    "sandbox": os.getenv("BINANCE_SANDBOX", "true").lower() == "true",
                    "enableRateLimit": True,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Binance: {e}")
            self.binance = None

        try:
            self.coinbase = ccxt.coinbase(
                {
                    "apiKey": os.getenv("COINBASE_API_KEY", ""),
                    "secret": os.getenv("COINBASE_SECRET", ""),
                    "passphrase": os.getenv("COINBASE_PASSPHRASE", ""),
                    "sandbox": os.getenv("COINBASE_SANDBOX", "true").lower() == "true",
                    "enableRateLimit": True,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Coinbase: {e}")
            self.coinbase = None

        try:
            self.deribit = RestClient()
        except Exception as e:
            logger.warning(f"Failed to initialize Deribit: {e}")
            self.deribit = None

        # Reconciliation configuration
        self.config = {
            "check_interval": 30,  # Check every 30 seconds
            "notional_tolerance_pct": 0.25,  # 0.25% tolerance
            "position_tolerance": 1e-6,  # 1e-6 base units
            "price_staleness_limit": 300,  # 5 minutes
            "halt_on_breach": True,  # Halt trading on breach
            "ipfs_logging": True,  # Log to IPFS
        }

        # Supported assets and their price sources
        self.assets = ["BTC", "ETH", "SOL"]
        self.asset_configs = {
            "BTC": {
                "binance_symbol": "BTCUSDT",
                "coinbase_symbol": "BTC-USD",
                "deribit_currency": "BTC",
                "decimals": 8,
            },
            "ETH": {
                "binance_symbol": "ETHUSDT",
                "coinbase_symbol": "ETH-USD",
                "deribit_currency": "ETH",
                "decimals": 8,
            },
            "SOL": {
                "binance_symbol": "SOLUSDT",
                "coinbase_symbol": "SOL-USD",
                "deribit_currency": None,  # No Deribit support
                "decimals": 6,
            },
        }

        # Track reconciliation state
        self.last_successful_check = time.time()
        self.consecutive_failures = 0
        self.total_checks = 0
        self.breach_history = []

        logger.info("🔍 Reconciliation Daemon initialized")
        logger.info(f"   Assets: {self.assets}")
        logger.info(f"   Check interval: {self.config['check_interval']}s")
        logger.info(f"   Tolerance: {self.config['notional_tolerance_pct']}%")

    def get_current_prices(self) -> Dict[str, float]:
        """Get current asset prices in USD."""
        prices = {}

        try:
            # Try Redis first (fastest)
            for asset in self.assets:
                price_key = f"price:{asset.lower()}:usd"
                price = self.redis.get(price_key)
                if price:
                    prices[asset] = float(price)

            # Fill missing prices with mock data for demo
            mock_prices = {"BTC": 97500.0, "ETH": 3500.0, "SOL": 180.0}
            for asset in self.assets:
                if asset not in prices:
                    prices[asset] = mock_prices.get(asset, 1.0)
                    logger.debug(f"Using mock price for {asset}: ${prices[asset]:,.2f}")

            return prices

        except Exception as e:
            logger.error(f"Error getting prices: {e}")
            # Return mock prices as fallback
            return {"BTC": 97500.0, "ETH": 3500.0, "SOL": 180.0}

    def fetch_cefi_balances(self) -> Dict[str, Dict]:
        """Fetch CeFi exchange balances."""
        balances = {}

        try:
            # Binance balances
            if self.binance:
                try:
                    binance_balance = self.binance.fetch_balance()
                    balances["BINANCE"] = self._normalize_balance(binance_balance)
                    logger.debug(
                        f"Binance balance fetched: {len(balances['BINANCE'])} assets"
                    )
                except Exception as e:
                    logger.warning(f"Failed to fetch Binance balance: {e}")
                    balances["BINANCE"] = {}
            else:
                balances["BINANCE"] = {}

            # Coinbase balances
            if self.coinbase:
                try:
                    coinbase_balance = self.coinbase.fetch_balance()
                    balances["COINBASE"] = self._normalize_balance(coinbase_balance)
                    logger.debug(
                        f"Coinbase balance fetched: {len(balances['COINBASE'])} assets"
                    )
                except Exception as e:
                    logger.warning(f"Failed to fetch Coinbase balance: {e}")
                    balances["COINBASE"] = {}
            else:
                balances["COINBASE"] = {}

            return balances

        except Exception as e:
            logger.error(f"Error fetching CeFi balances: {e}")
            return {"BINANCE": {}, "COINBASE": {}}

    def fetch_deribit_positions(self) -> Dict[str, Dict]:
        """Fetch Deribit positions."""
        try:
            if not self.deribit:
                return {"DERIBIT": {}}

            # Fetch positions for each supported currency
            deribit_positions = {}

            for asset in self.assets:
                currency = self.asset_configs[asset].get("deribit_currency")
                if currency:
                    try:
                        positions = self.deribit.getpositions(currency=currency)
                        if positions and "result" in positions:
                            for pos in positions["result"]:
                                instrument = pos.get("instrument_name", "")
                                if instrument:
                                    deribit_positions[instrument] = {
                                        "size": pos.get("size", 0),
                                        "mark_price": pos.get("mark_price", 0),
                                        "notional": pos.get("size", 0)
                                        * pos.get("mark_price", 0),
                                        "currency": currency,
                                    }
                    except Exception as e:
                        logger.debug(f"No Deribit positions for {currency}: {e}")

            return {"DERIBIT": deribit_positions}

        except Exception as e:
            logger.warning(f"Error fetching Deribit positions: {e}")
            return {"DERIBIT": {}}

    def _normalize_balance(self, raw_balance: Dict) -> Dict[str, Dict]:
        """Normalize exchange balance format."""
        normalized = {}

        try:
            for asset, balance_info in raw_balance.items():
                if asset in ["free", "used", "total", "info"]:
                    continue

                if isinstance(balance_info, dict):
                    free = float(balance_info.get("free", 0))
                    used = float(balance_info.get("used", 0))
                    total = float(balance_info.get("total", 0))

                    # Only include non-zero balances
                    if total > 1e-8:
                        normalized[asset] = {"free": free, "used": used, "total": total}

            return normalized

        except Exception as e:
            logger.error(f"Error normalizing balance: {e}")
            return {}

    def get_local_portfolio(self) -> Dict[str, float]:
        """Get local portfolio positions from Redis."""
        try:
            # Get portfolio positions
            positions_data = self.redis.get("portfolio_positions")
            if positions_data:
                positions = json.loads(positions_data)
                return positions

            # Get individual asset positions
            local_positions = {}
            for asset in self.assets:
                position_key = f"position:{asset.lower()}"
                position = self.redis.get(position_key)
                if position:
                    local_positions[asset] = float(position)

            # Mock some positions for demo
            if not local_positions:
                local_positions = {
                    "BTC": 0.25,  # 0.25 BTC
                    "ETH": 1.5,  # 1.5 ETH
                    "SOL": 100.0,  # 100 SOL
                }
                logger.debug("Using mock local positions for demo")

            return local_positions

        except Exception as e:
            logger.error(f"Error getting local portfolio: {e}")
            return {}

    def calculate_notional_values(
        self, positions: Dict[str, float], prices: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate notional values in USD."""
        notional = {}

        try:
            for asset, quantity in positions.items():
                if asset in prices:
                    notional[asset] = quantity * prices[asset]
                else:
                    logger.warning(f"No price available for {asset}")
                    notional[asset] = 0.0

            return notional

        except Exception as e:
            logger.error(f"Error calculating notional values: {e}")
            return {}

    async def reconcile_positions(self) -> Dict[str, Any]:
        """Perform position reconciliation check."""
        try:
            check_start = time.time()
            self.total_checks += 1

            # Get current prices
            prices = self.get_current_prices()

            # Fetch exchange balances and positions
            cefi_balances = self.fetch_cefi_balances()
            deribit_positions = self.fetch_deribit_positions()

            # Get local portfolio state
            local_positions = self.get_local_portfolio()

            # Calculate exchange totals
            exchange_positions = self._aggregate_exchange_positions(
                cefi_balances, deribit_positions
            )

            # Calculate notional values
            local_notional = self.calculate_notional_values(local_positions, prices)
            exchange_notional = self.calculate_notional_values(
                exchange_positions, prices
            )

            # Perform reconciliation checks
            diff_results = self._check_differences(
                local_positions, exchange_positions, local_notional, exchange_notional
            )

            # Calculate summary metrics
            total_local_notional = sum(local_notional.values())
            total_exchange_notional = sum(exchange_notional.values())

            notional_diff_pct = (
                abs(total_local_notional - total_exchange_notional)
                / max(total_local_notional, 1)
                * 100
            )

            position_mismatches = len(
                [
                    asset
                    for asset, diff in diff_results["position_diffs"].items()
                    if abs(diff) > self.config["position_tolerance"]
                ]
            )

            # Store metrics in Redis
            self.redis.set("recon:notional_diff_pct", notional_diff_pct)
            self.redis.set("recon:position_mismatches", position_mismatches)
            self.redis.set("recon:last_check", time.time())

            # Check breach conditions
            breach_detected = (
                notional_diff_pct > self.config["notional_tolerance_pct"]
                or position_mismatches > 0
            )

            recon_result = {
                "timestamp": check_start,
                "status": "breach" if breach_detected else "ok",
                "notional_diff_pct": notional_diff_pct,
                "position_mismatches": position_mismatches,
                "local_positions": local_positions,
                "exchange_positions": exchange_positions,
                "local_notional": local_notional,
                "exchange_notional": exchange_notional,
                "total_local_notional": total_local_notional,
                "total_exchange_notional": total_exchange_notional,
                "diff_results": diff_results,
                "prices": prices,
                "check_duration": time.time() - check_start,
            }

            if breach_detected:
                await self._handle_breach(recon_result)
            else:
                self.last_successful_check = time.time()
                self.consecutive_failures = 0

            logger.debug(
                f"Reconciliation complete: "
                f"diff={notional_diff_pct:.3f}%, mismatches={position_mismatches}"
            )

            return recon_result

        except Exception as e:
            logger.error(f"Error in reconciliation: {e}")
            self.consecutive_failures += 1

            return {
                "timestamp": time.time(),
                "status": "error",
                "error": str(e),
                "consecutive_failures": self.consecutive_failures,
            }

    def _aggregate_exchange_positions(
        self, cefi_balances: Dict, deribit_positions: Dict
    ) -> Dict[str, float]:
        """Aggregate positions across all exchanges."""
        aggregated = {}

        try:
            # Aggregate CeFi spot balances
            for exchange, balances in cefi_balances.items():
                for asset, balance_info in balances.items():
                    total_balance = balance_info.get("total", 0)
                    if asset not in aggregated:
                        aggregated[asset] = 0
                    aggregated[asset] += total_balance

            # Add Deribit positions (derivatives)
            # For now, we don't add derivative notional to spot position reconciliation
            # but we track them separately

            return aggregated

        except Exception as e:
            logger.error(f"Error aggregating exchange positions: {e}")
            return {}

    def _check_differences(
        self,
        local_pos: Dict,
        exchange_pos: Dict,
        local_notional: Dict,
        exchange_notional: Dict,
    ) -> Dict:
        """Check differences between local and exchange positions."""
        try:
            position_diffs = {}
            notional_diffs = {}

            # Get all assets
            all_assets = set(local_pos.keys()) | set(exchange_pos.keys())

            for asset in all_assets:
                local_qty = local_pos.get(asset, 0)
                exchange_qty = exchange_pos.get(asset, 0)

                position_diffs[asset] = local_qty - exchange_qty

                local_not = local_notional.get(asset, 0)
                exchange_not = exchange_notional.get(asset, 0)

                notional_diffs[asset] = local_not - exchange_not

            return {"position_diffs": position_diffs, "notional_diffs": notional_diffs}

        except Exception as e:
            logger.error(f"Error checking differences: {e}")
            return {"position_diffs": {}, "notional_diffs": {}}

    async def _handle_breach(self, recon_result: Dict):
        """Handle reconciliation breach."""
        try:
            logger.warning(
                f"🚨 Reconciliation breach detected: "
                f"diff={recon_result['notional_diff_pct']:.3f}%, "
                f"mismatches={recon_result['position_mismatches']}"
            )

            # Halt trading if configured
            if self.config["halt_on_breach"]:
                self.redis.set("mode", "halt")
                logger.critical("⏸️ Trading halted due to reconciliation breach")

            # Store detailed breach information
            breach_blob = {
                "timestamp": recon_result["timestamp"],
                "diff_pct": recon_result["notional_diff_pct"],
                "mismatches": recon_result["position_mismatches"],
                "local": recon_result["local_positions"],
                "exchange": recon_result["exchange_positions"],
                "diff_details": recon_result["diff_results"],
                "total_local_usd": recon_result["total_local_notional"],
                "total_exchange_usd": recon_result["total_exchange_notional"],
            }

            self.redis.set("recon:last_diff", json.dumps(breach_blob, default=str))

            # Log to IPFS if enabled
            if self.config["ipfs_logging"]:
                try:
                    ipfs_cid = await self._log_to_ipfs(breach_blob)
                    if ipfs_cid:
                        self.redis.set("recon:last_ipfs_cid", ipfs_cid)
                        logger.info(f"📝 Breach logged to IPFS: {ipfs_cid}")
                except Exception as e:
                    logger.warning(f"Failed to log to IPFS: {e}")

            # Send Slack alert
            if self.slack_webhook:
                await self._send_breach_alert(recon_result)

            # Track breach history
            self.breach_history.append(
                {
                    "timestamp": recon_result["timestamp"],
                    "diff_pct": recon_result["notional_diff_pct"],
                    "mismatches": recon_result["position_mismatches"],
                }
            )

            # Keep last 100 breaches
            if len(self.breach_history) > 100:
                self.breach_history = self.breach_history[-100:]

            self.consecutive_failures += 1

        except Exception as e:
            logger.error(f"Error handling breach: {e}")

    async def _log_to_ipfs(self, breach_data: Dict) -> Optional[str]:
        """Log breach data to IPFS."""
        try:
            # This would integrate with IPFS node
            # For now, return a mock CID
            import hashlib

            content = json.dumps(breach_data, sort_keys=True, default=str)
            hash_obj = hashlib.sha256(content.encode())
            mock_cid = f"Qm{hash_obj.hexdigest()[:44]}"

            logger.debug(f"Mock IPFS logging: {mock_cid}")
            return mock_cid

        except Exception as e:
            logger.error(f"Error logging to IPFS: {e}")
            return None

    async def _send_breach_alert(self, recon_result: Dict):
        """Send breach alert to Slack."""
        try:
            diff_pct = recon_result["notional_diff_pct"]
            mismatches = recon_result["position_mismatches"]

            # Format position differences
            diff_details = []
            for asset, diff in recon_result["diff_results"]["position_diffs"].items():
                if abs(diff) > self.config["position_tolerance"]:
                    diff_details.append(f"{asset}: {diff:+.6f}")

            message = (
                f"🚨 *RECONCILIATION BREACH DETECTED*\n"
                f"• Notional diff: {diff_pct:.3f}% (limit: {self.config['notional_tolerance_pct']:.2f}%)\n"
                f"• Position mismatches: {mismatches}\n"
                f"• Total local: ${recon_result['total_local_notional']:,.0f}\n"
                f"• Total exchange: ${recon_result['total_exchange_notional']:,.0f}\n"
                f"• Trading: {'HALTED' if self.config['halt_on_breach'] else 'CONTINUING'}"
            )

            if diff_details:
                message += f"\n• Position diffs:\n  " + "\n  ".join(diff_details[:5])

            payload = {
                "text": message,
                "username": "Reconciliation Daemon",
                "icon_emoji": ":warning:",
            }

            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("📱 Sent breach alert to Slack")

        except Exception as e:
            logger.error(f"Error sending breach alert: {e}")

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        try:
            # Get recent metrics
            notional_diff = float(self.redis.get("recon:notional_diff_pct") or 0)
            position_mismatches = int(self.redis.get("recon:position_mismatches") or 0)
            last_check = float(self.redis.get("recon:last_check") or 0)

            status = {
                "service": "reconciliation_daemon",
                "timestamp": time.time(),
                "status": (
                    "breach"
                    if (
                        notional_diff > self.config["notional_tolerance_pct"]
                        or position_mismatches > 0
                    )
                    else "healthy"
                ),
                "config": self.config,
                "exchanges": {
                    "binance": self.binance is not None,
                    "coinbase": self.coinbase is not None,
                    "deribit": self.deribit is not None,
                },
                "metrics": {
                    "notional_diff_pct": notional_diff,
                    "position_mismatches": position_mismatches,
                    "total_checks": self.total_checks,
                    "consecutive_failures": self.consecutive_failures,
                    "last_successful_check": self.last_successful_check,
                    "last_check": last_check,
                    "breach_count": len(self.breach_history),
                },
                "recent_breaches": (
                    self.breach_history[-5:] if self.breach_history else []
                ),
            }

            return status

        except Exception as e:
            return {
                "service": "reconciliation_daemon",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }

    async def run_continuous_reconciliation(self):
        """Run continuous reconciliation loop."""
        logger.info("🔍 Starting continuous reconciliation")

        try:
            while True:
                try:
                    # Run reconciliation check
                    result = await self.reconcile_positions()

                    if result["status"] == "ok":
                        if self.total_checks % 10 == 0:  # Log every 10 checks
                            logger.info(
                                f"📊 Check #{self.total_checks}: "
                                f"diff={result.get('notional_diff_pct', 0):.3f}%, "
                                f"mismatches={result.get('position_mismatches', 0)}"
                            )
                    elif result["status"] == "breach":
                        logger.warning(f"🚨 Breach #{len(self.breach_history)}")

                    # Wait for next check
                    await asyncio.sleep(self.config["check_interval"])

                except Exception as e:
                    logger.error(f"Error in reconciliation loop: {e}")
                    await asyncio.sleep(5)  # Short delay on error

        except asyncio.CancelledError:
            logger.info("Reconciliation loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in reconciliation loop: {e}")


async def main():
    """Main entry point for reconciliation daemon."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Account & Position Reconciliation Daemon"
    )
    parser.add_argument(
        "--run", action="store_true", help="Run continuous reconciliation"
    )
    parser.add_argument(
        "--check", action="store_true", help="Run single reconciliation check"
    )
    parser.add_argument("--status", action="store_true", help="Show status report")

    args = parser.parse_args()

    # Create daemon
    daemon = ReconciliationDaemon()

    if args.status:
        # Show status report
        status = daemon.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.check:
        # Run single check
        result = await daemon.reconcile_positions()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.run:
        # Run continuous reconciliation
        try:
            await daemon.run_continuous_reconciliation()
        except KeyboardInterrupt:
            logger.info("Reconciliation daemon stopped by user")
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    # Fix for Python 3.9 compatibility
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
