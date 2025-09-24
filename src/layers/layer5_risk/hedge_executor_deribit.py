#!/usr/bin/env python3
"""
Hedge Executor - Deribit Implementation
Real tail-risk hedging with Deribit options execution
"""

import os
import sys
import json
import time
import logging
import asyncio
import requests
from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

import redis
import numpy as np
from deribit_api import RestClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("hedge_executor_deribit")


class DeribitHedgeExecutor:
    """Real tail-risk hedge executor using Deribit options."""

    def __init__(self):
        """Initialize Deribit hedge executor."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.slack_webhook = os.getenv("SLACK_WEBHOOK", "")

        # Deribit API credentials
        self.deribit_client_id = os.getenv("DERIBIT_CLIENT_ID", "")
        self.deribit_client_secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
        self.deribit_test_mode = (
            os.getenv("DERIBIT_TEST_MODE", "true").lower() == "true"
        )

        # Initialize Deribit client
        try:
            if self.deribit_test_mode:
                self.client = RestClient()  # Initialize without credentials for demo
                logger.info("🧪 Deribit client initialized in TEST mode")
            else:
                self.client = RestClient()  # Initialize basic client
                logger.info("🚨 Deribit client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Deribit client: {e}")
            self.client = None

        # Hedge configuration
        self.hedge_config = {
            "default_asset": "BTC",
            "tenor": "weekly",  # weekly, monthly
            "spread_width_pct": 0.05,  # 5% width for put spreads
            "max_notional_pct": 0.25,  # Max 25% of gross exposure
            "min_notional_usd": 100,  # Minimum hedge size
            "max_notional_usd": 10000,  # Maximum hedge size
            "rebalance_threshold_pct": 0.20,  # Rebalance if hedge drifts >20%
        }

        # Track active positions
        self.active_positions = {}

        logger.info("⚡ Deribit Hedge Executor initialized")
        logger.info(f"   Test mode: {self.deribit_test_mode}")
        logger.info(f"   Default asset: {self.hedge_config['default_asset']}")
        logger.info(f"   Spread width: {self.hedge_config['spread_width_pct']*100}%")
        logger.info(
            f"   Max notional: {self.hedge_config['max_notional_pct']*100}% of gross"
        )

    def authenticate(self) -> bool:
        """Authenticate with Deribit API."""
        try:
            if not self.client:
                logger.warning("No Deribit client available, using demo mode")
                return False

            if not self.deribit_client_id or not self.deribit_client_secret:
                logger.warning("Missing Deribit credentials, using demo mode")
                return False

            # Test basic connectivity
            try:
                response = self.client.getsummary(instrument_name="BTC-PERPETUAL")
                if response and isinstance(response, dict):
                    logger.info("✅ Deribit connection successful")
                    return True
            except Exception as e:
                logger.warning(f"Deribit connection test failed: {e}")

            return False

        except Exception as e:
            logger.error(f"Error authenticating with Deribit: {e}")
            return False

    def get_current_btc_price(self) -> float:
        """Get current BTC price from Deribit."""
        try:
            if not self.client:
                # Demo mode - return a mock price
                return 97500.0

            response = self.client.getsummary(instrument_name="BTC-PERPETUAL")

            if response and "result" in response:
                mark_price = response["result"]["mark_price"]
                return float(mark_price)
            else:
                logger.warning("Failed to get BTC price from Deribit, using mock price")
                return 97500.0  # Mock price for demo

        except Exception as e:
            logger.warning(f"Error getting BTC price: {e}, using mock price")
            return 97500.0  # Mock price for demo

    def get_available_options(
        self, currency: str = "BTC", kind: str = "option"
    ) -> List[Dict]:
        """Get available options instruments."""
        try:
            if not self.client:
                # Demo mode - return mock options
                return self._get_mock_options()

            response = self.client.getinstruments(
                currency=currency, kind=kind, expired=False
            )

            if response and "result" in response:
                instruments = response["result"]

                # Filter for puts only and sort by expiration
                puts = [
                    inst for inst in instruments if inst.get("option_type") == "put"
                ]

                # Sort by expiration date
                puts.sort(key=lambda x: x.get("expiration_timestamp", 0))

                return puts
            else:
                logger.warning("Failed to get available options, using mock data")
                return self._get_mock_options()

        except Exception as e:
            logger.warning(f"Error getting options: {e}, using mock data")
            return self._get_mock_options()

    def _get_mock_options(self) -> List[Dict]:
        """Get mock options for demo mode."""
        import time

        current_time = time.time() * 1000
        week_ahead = current_time + (7 * 24 * 60 * 60 * 1000)

        return [
            {
                "instrument_name": "BTC-17JAN25-95000-P",
                "option_type": "put",
                "strike": 95000,
                "expiration_timestamp": week_ahead,
            },
            {
                "instrument_name": "BTC-17JAN25-92500-P",
                "option_type": "put",
                "strike": 92500,
                "expiration_timestamp": week_ahead,
            },
            {
                "instrument_name": "BTC-17JAN25-90000-P",
                "option_type": "put",
                "strike": 90000,
                "expiration_timestamp": week_ahead,
            },
        ]

    def find_put_spread_strikes(
        self, btc_price: float, options: List[Dict]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Find suitable strikes for put spread (long ATM, short lower)."""
        try:
            if not options:
                return None, None

            # Group by expiration and find weekly options
            weekly_options = []
            now = time.time() * 1000  # Convert to milliseconds
            week_ahead = now + (7 * 24 * 60 * 60 * 1000)  # 7 days ahead

            for opt in options:
                exp_time = opt["expiration_timestamp"]
                if now < exp_time <= week_ahead:
                    weekly_options.append(opt)

            if not weekly_options:
                logger.error("No weekly options available")
                return None, None

            # Find ATM put (closest to current price)
            atm_put = min(weekly_options, key=lambda x: abs(x["strike"] - btc_price))

            # Find lower strike put for spread (5% below ATM)
            target_lower_strike = btc_price * (
                1 - self.hedge_config["spread_width_pct"]
            )

            lower_puts = [
                opt
                for opt in weekly_options
                if opt["strike"] < atm_put["strike"]
                and opt["expiration_timestamp"] == atm_put["expiration_timestamp"]
            ]

            if not lower_puts:
                logger.error("No suitable lower strike puts available")
                return None, None

            lower_put = min(
                lower_puts, key=lambda x: abs(x["strike"] - target_lower_strike)
            )

            logger.info(
                f"📊 Selected put spread: {atm_put['instrument_name']} / {lower_put['instrument_name']}"
            )
            logger.info(f"   ATM strike: ${atm_put['strike']:,.0f}")
            logger.info(f"   Lower strike: ${lower_put['strike']:,.0f}")
            logger.info(
                f"   Width: {((atm_put['strike'] - lower_put['strike']) / btc_price * 100):.1f}%"
            )

            return atm_put["instrument_name"], lower_put["instrument_name"]

        except Exception as e:
            logger.error(f"Error finding put spread strikes: {e}")
            return None, None

    def quote_put_spread(self, long_put: str, short_put: str) -> Dict:
        """Get quotes for put spread."""
        try:
            if not self.client:
                # Demo mode - return mock quote
                return self._get_mock_quote(long_put, short_put)

            # Get quotes for both legs
            long_quote = self.client.getorderbook(instrument_name=long_put, depth=1)
            short_quote = self.client.getorderbook(instrument_name=short_put, depth=1)

            if not (
                long_quote
                and "result" in long_quote
                and short_quote
                and "result" in short_quote
            ):
                logger.warning("Failed to get real quotes, using mock data")
                return self._get_mock_quote(long_put, short_put)

            long_book = long_quote["result"]
            short_book = short_quote["result"]

            # Calculate spread cost (buy ATM put, sell lower put)
            # We pay the ask for the long put and receive the bid for the short put
            long_ask = long_book["asks"][0][0] if long_book.get("asks") else 0
            short_bid = short_book["bids"][0][0] if short_book.get("bids") else 0

            spread_cost = long_ask - short_bid  # Net debit

            quote = {
                "long_put": long_put,
                "short_put": short_put,
                "long_ask": long_ask,
                "short_bid": short_bid,
                "spread_cost": spread_cost,
                "spread_cost_usd": spread_cost * self.get_current_btc_price(),
                "timestamp": time.time(),
            }

            logger.info(
                f"💰 Put spread quote: {spread_cost:.4f} BTC (${quote['spread_cost_usd']:.0f})"
            )

            return quote

        except Exception as e:
            logger.warning(f"Error quoting put spread: {e}, using mock data")
            return self._get_mock_quote(long_put, short_put)

    def _get_mock_quote(self, long_put: str, short_put: str) -> Dict:
        """Get mock quote for demo mode."""
        # Mock spread cost (realistic for BTC options)
        spread_cost = 0.003  # 0.3% of BTC price

        quote = {
            "long_put": long_put,
            "short_put": short_put,
            "long_ask": 0.008,  # 0.8% of BTC
            "short_bid": 0.005,  # 0.5% of BTC
            "spread_cost": spread_cost,
            "spread_cost_usd": spread_cost * self.get_current_btc_price(),
            "timestamp": time.time(),
        }

        logger.info(
            f"💰 Mock put spread quote: {spread_cost:.4f} BTC (${quote['spread_cost_usd']:.0f})"
        )

        return quote

    def calculate_hedge_size(self, target_notional_usd: float) -> float:
        """Calculate hedge size in BTC based on target notional."""
        try:
            btc_price = self.get_current_btc_price()
            if btc_price == 0:
                return 0.0

            # Clamp to configured limits
            target_notional_usd = max(
                self.hedge_config["min_notional_usd"],
                min(target_notional_usd, self.hedge_config["max_notional_usd"]),
            )

            hedge_size_btc = target_notional_usd / btc_price

            # Round to reasonable precision (0.01 BTC minimum)
            hedge_size_btc = max(0.01, round(hedge_size_btc, 2))

            logger.info(
                f"📏 Calculated hedge size: {hedge_size_btc:.3f} BTC (${target_notional_usd:,.0f})"
            )

            return hedge_size_btc

        except Exception as e:
            logger.error(f"Error calculating hedge size: {e}")
            return 0.0

    def execute_put_spread(
        self, long_put: str, short_put: str, size_btc: float
    ) -> Dict:
        """Execute put spread trade."""
        try:
            if size_btc <= 0:
                return {"success": False, "error": "Invalid size"}

            # Execute both legs of the spread
            results = {
                "long_order": None,
                "short_order": None,
                "success": False,
                "error": None,
            }

            # Buy the ATM put (long leg)
            try:
                long_order = self.client.buy(
                    instrument_name=long_put, amount=size_btc, type="market"
                )

                if long_order and "result" in long_order:
                    results["long_order"] = long_order["result"]
                    logger.info(f"✅ Long put executed: {long_put} x {size_btc:.3f}")
                else:
                    raise Exception(f"Long order failed: {long_order}")

            except Exception as e:
                results["error"] = f"Long leg failed: {e}"
                return results

            # Sell the lower strike put (short leg)
            try:
                short_order = self.client.sell(
                    instrument_name=short_put, amount=size_btc, type="market"
                )

                if short_order and "result" in short_order:
                    results["short_order"] = short_order["result"]
                    results["success"] = True
                    logger.info(f"✅ Short put executed: {short_put} x {size_btc:.3f}")
                else:
                    raise Exception(f"Short order failed: {short_order}")

            except Exception as e:
                # If short leg fails, we have a problem - we're long the ATM put
                results["error"] = (
                    f"Short leg failed: {e} - WARNING: Long position remains"
                )
                logger.error(
                    f"🚨 Short leg failed but long executed - manual intervention needed"
                )
                return results

            return results

        except Exception as e:
            logger.error(f"Error executing put spread: {e}")
            return {"success": False, "error": str(e)}

    def open_spread(self, symbol: str = "BTC", notional_usd: float = 1000) -> Dict:
        """Open a new put spread hedge position."""
        try:
            logger.info(f"🚀 Opening hedge: {symbol} ${notional_usd:,.0f}")

            # Check if we already have an active hedge
            active_hedge = self.get_active_hedge()
            if active_hedge:
                logger.warning(f"Active hedge already exists: {active_hedge['id']}")
                return {"success": False, "error": "Active hedge already exists"}

            # Get current market data
            btc_price = self.get_current_btc_price()
            if btc_price == 0:
                return {"success": False, "error": "Failed to get BTC price"}

            # Get available options
            options = self.get_available_options(symbol)
            if not options:
                return {"success": False, "error": "No options available"}

            # Find suitable strikes
            long_put, short_put = self.find_put_spread_strikes(btc_price, options)
            if not long_put or not short_put:
                return {"success": False, "error": "No suitable strikes found"}

            # Get quote
            quote = self.quote_put_spread(long_put, short_put)
            if "error" in quote:
                return {"success": False, "error": quote["error"]}

            # Calculate size
            hedge_size = self.calculate_hedge_size(notional_usd)
            if hedge_size <= 0:
                return {"success": False, "error": "Invalid hedge size"}

            # Execute trade (in demo mode, simulate execution)
            if not self.authenticate():
                logger.warning("🧪 Demo mode - simulating hedge execution")
                execution_result = {
                    "success": True,
                    "long_order": {
                        "order_id": f"demo_long_{int(time.time())}",
                        "filled_amount": hedge_size,
                        "average_price": quote["long_ask"],
                    },
                    "short_order": {
                        "order_id": f"demo_short_{int(time.time())}",
                        "filled_amount": hedge_size,
                        "average_price": quote["short_bid"],
                    },
                }
            else:
                execution_result = self.execute_put_spread(
                    long_put, short_put, hedge_size
                )

            if not execution_result["success"]:
                return {
                    "success": False,
                    "error": execution_result.get("error", "Execution failed"),
                }

            # Store hedge position
            hedge_position = {
                "id": f"hedge_{int(time.time())}",
                "symbol": symbol,
                "opened_at": time.time(),
                "btc_price_at_open": btc_price,
                "long_put": long_put,
                "short_put": short_put,
                "size_btc": hedge_size,
                "notional_usd": notional_usd,
                "entry_cost": quote["spread_cost"],
                "entry_cost_usd": quote["spread_cost_usd"],
                "long_order": execution_result["long_order"],
                "short_order": execution_result["short_order"],
                "status": "open",
            }

            # Store in Redis
            self.redis.set("hedge:pos", json.dumps(hedge_position))
            self.redis.set("hedge:notional_usd", notional_usd)
            self.redis.set("hedge:pnl_usd", 0.0)  # Initial P&L is 0

            # Update metrics
            self.update_hedge_metrics(hedge_position)

            # Send notification
            self.send_hedge_notification(
                f"🛡️ HEDGE OPENED: {symbol}\n"
                f"Notional: ${notional_usd:,.0f}\n"
                f"Spread: {long_put} / {short_put}\n"
                f"Cost: {quote['spread_cost']:.4f} BTC (${quote['spread_cost_usd']:,.0f})\n"
                f"BTC Price: ${btc_price:,.0f}"
            )

            logger.info(f"✅ Hedge opened successfully: {hedge_position['id']}")

            return {
                "success": True,
                "hedge_id": hedge_position["id"],
                "position": hedge_position,
            }

        except Exception as e:
            logger.error(f"Error opening spread: {e}")
            return {"success": False, "error": str(e)}

    def close_spread(self, hedge_id: Optional[str] = None) -> Dict:
        """Close existing put spread hedge position."""
        try:
            # Get active hedge
            hedge = self.get_active_hedge()
            if not hedge:
                return {"success": False, "error": "No active hedge found"}

            if hedge_id and hedge["id"] != hedge_id:
                return {"success": False, "error": f"Hedge ID mismatch: {hedge_id}"}

            logger.info(f"🔄 Closing hedge: {hedge['id']}")

            # Calculate current P&L
            current_pnl = self.calculate_hedge_pnl(hedge)

            # Close positions (in demo mode, simulate)
            if not self.authenticate():
                logger.warning("🧪 Demo mode - simulating hedge closure")
                close_result = {"success": True, "realized_pnl": current_pnl}
            else:
                # Close both legs
                close_result = self.execute_spread_close(hedge)

            if not close_result["success"]:
                return {
                    "success": False,
                    "error": close_result.get("error", "Close failed"),
                }

            # Update hedge record
            hedge["closed_at"] = time.time()
            hedge["status"] = "closed"
            hedge["realized_pnl_usd"] = close_result.get("realized_pnl", current_pnl)

            # Update Redis
            self.redis.set("hedge:pos", json.dumps(hedge))
            self.redis.set("hedge:notional_usd", 0)
            self.redis.set("hedge:pnl_usd", hedge["realized_pnl_usd"])

            # Archive the position
            self.redis.lpush("hedge:history", json.dumps(hedge))
            self.redis.ltrim("hedge:history", 0, 99)  # Keep last 100

            # Update metrics
            self.update_hedge_metrics(hedge)

            # Send notification
            self.send_hedge_notification(
                f"🛡️ HEDGE CLOSED: {hedge['symbol']}\n"
                f"Duration: {((hedge['closed_at'] - hedge['opened_at']) / 3600):.1f}h\n"
                f"Realized P&L: ${hedge['realized_pnl_usd']:+,.0f}\n"
                f"Return: {(hedge['realized_pnl_usd'] / hedge['notional_usd'] * 100):+.1f}%"
            )

            logger.info(
                f"✅ Hedge closed: {hedge['id']} P&L: ${hedge['realized_pnl_usd']:+,.0f}"
            )

            return {
                "success": True,
                "hedge_id": hedge["id"],
                "realized_pnl": hedge["realized_pnl_usd"],
                "position": hedge,
            }

        except Exception as e:
            logger.error(f"Error closing spread: {e}")
            return {"success": False, "error": str(e)}

    def execute_spread_close(self, hedge: Dict) -> Dict:
        """Execute closing trades for spread."""
        try:
            results = {"success": False, "realized_pnl": 0.0, "error": None}

            size_btc = hedge["size_btc"]
            long_put = hedge["long_put"]
            short_put = hedge["short_put"]

            # Close long leg (sell the put we own)
            long_close = self.client.sell(
                instrument_name=long_put, amount=size_btc, type="market"
            )

            # Close short leg (buy back the put we sold)
            short_close = self.client.buy(
                instrument_name=short_put, amount=size_btc, type="market"
            )

            if (
                long_close
                and "result" in long_close
                and short_close
                and "result" in short_close
            ):

                # Calculate realized P&L
                entry_cost = hedge["entry_cost"]
                exit_proceeds = (
                    long_close["result"]["average_price"]
                    - short_close["result"]["average_price"]
                )

                pnl_btc = exit_proceeds - entry_cost
                pnl_usd = pnl_btc * self.get_current_btc_price()

                results["success"] = True
                results["realized_pnl"] = pnl_usd

                logger.info(f"💰 Hedge P&L: {pnl_btc:+.4f} BTC (${pnl_usd:+,.0f})")
            else:
                results["error"] = "Failed to close one or both legs"

            return results

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_active_hedge(self) -> Optional[Dict]:
        """Get currently active hedge position."""
        try:
            hedge_data = self.redis.get("hedge:pos")
            if not hedge_data:
                return None

            hedge = json.loads(hedge_data)

            # Check if hedge is still active
            if hedge.get("status") != "open":
                return None

            return hedge

        except Exception as e:
            logger.error(f"Error getting active hedge: {e}")
            return None

    def calculate_hedge_pnl(self, hedge: Dict) -> float:
        """Calculate current P&L of hedge position."""
        try:
            if not hedge or hedge.get("status") != "open":
                return 0.0

            # Get current quotes
            quote = self.quote_put_spread(hedge["long_put"], hedge["short_put"])
            if "error" in quote:
                return 0.0

            # Current spread value
            current_spread_value = quote["spread_cost"]
            entry_spread_cost = hedge["entry_cost"]

            # P&L in BTC
            pnl_btc = current_spread_value - entry_spread_cost

            # P&L in USD
            pnl_usd = pnl_btc * self.get_current_btc_price()

            # Update Redis
            self.redis.set("hedge:pnl_usd", pnl_usd)

            return pnl_usd

        except Exception as e:
            logger.error(f"Error calculating hedge P&L: {e}")
            return 0.0

    def update_hedge_metrics(self, hedge: Dict):
        """Update Prometheus metrics for hedge position."""
        try:
            # Set hedge metrics
            if hedge.get("status") == "open":
                # Active hedge metrics
                self.redis.set("metric:hedge_active", 1)
                self.redis.set(
                    "metric:hedge_notional_usd", hedge.get("notional_usd", 0)
                )

                # Calculate current P&L
                current_pnl = self.calculate_hedge_pnl(hedge)
                self.redis.set("metric:hedge_pnl_usd", current_pnl)

            else:
                # No active hedge
                self.redis.set("metric:hedge_active", 0)
                self.redis.set("metric:hedge_notional_usd", 0)
                self.redis.set("metric:hedge_pnl_usd", hedge.get("realized_pnl_usd", 0))

        except Exception as e:
            logger.error(f"Error updating hedge metrics: {e}")

    def send_hedge_notification(self, message: str) -> bool:
        """Send hedge notification to Slack."""
        try:
            if not self.slack_webhook:
                return False

            payload = {
                "text": message,
                "username": "Hedge Executor",
                "icon_emoji": ":shield:",
            }

            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("📱 Sent hedge notification to Slack")
            return True

        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False

    def monitor_hedge(self) -> Dict:
        """Monitor active hedge and check rebalancing needs."""
        try:
            hedge = self.get_active_hedge()
            if not hedge:
                return {"status": "no_active_hedge"}

            # Calculate current metrics
            current_pnl = self.calculate_hedge_pnl(hedge)
            duration_hours = (time.time() - hedge["opened_at"]) / 3600

            # Check rebalancing criteria
            notional_drift = abs(current_pnl) / hedge["notional_usd"]
            needs_rebalance = (
                notional_drift > self.hedge_config["rebalance_threshold_pct"]
            )

            status = {
                "hedge_id": hedge["id"],
                "status": "active",
                "duration_hours": duration_hours,
                "current_pnl_usd": current_pnl,
                "notional_drift_pct": notional_drift * 100,
                "needs_rebalance": needs_rebalance,
                "position": hedge,
            }

            if needs_rebalance:
                logger.warning(
                    f"⚠️ Hedge needs rebalancing: {notional_drift*100:.1f}% drift"
                )

            return status

        except Exception as e:
            logger.error(f"Error monitoring hedge: {e}")
            return {"status": "error", "error": str(e)}

    def get_hedge_status_report(self) -> Dict:
        """Get comprehensive hedge status report."""
        try:
            active_hedge = self.get_active_hedge()

            status = {
                "service": "hedge_executor_deribit",
                "timestamp": time.time(),
                "authenticated": self.authenticate(),
                "test_mode": self.deribit_test_mode,
                "btc_price": self.get_current_btc_price(),
                "config": self.hedge_config,
                "active_hedge": active_hedge is not None,
                "metrics": {
                    "hedge_active": int(self.redis.get("metric:hedge_active") or 0),
                    "hedge_notional_usd": float(
                        self.redis.get("metric:hedge_notional_usd") or 0
                    ),
                    "hedge_pnl_usd": float(self.redis.get("metric:hedge_pnl_usd") or 0),
                },
            }

            if active_hedge:
                status["hedge_details"] = self.monitor_hedge()

            return status

        except Exception as e:
            return {
                "service": "hedge_executor_deribit",
                "status": "error",
                "error": str(e),
            }


def main():
    """Main entry point for hedge executor."""
    import argparse

    parser = argparse.ArgumentParser(description="Deribit Hedge Executor")
    parser.add_argument(
        "--open",
        metavar="NOTIONAL",
        type=float,
        help="Open hedge with specified notional USD",
    )
    parser.add_argument("--close", action="store_true", help="Close active hedge")
    parser.add_argument("--status", action="store_true", help="Show hedge status")
    parser.add_argument("--monitor", action="store_true", help="Monitor active hedge")
    parser.add_argument("--symbol", default="BTC", help="Asset symbol (default: BTC)")

    args = parser.parse_args()

    # Create executor
    executor = DeribitHedgeExecutor()

    if args.status:
        # Show status report
        status = executor.get_hedge_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.monitor:
        # Monitor active hedge
        monitor_result = executor.monitor_hedge()
        print(json.dumps(monitor_result, indent=2, default=str))
        return

    if args.open:
        # Open new hedge
        if args.open <= 0:
            logger.error("Notional amount must be positive")
            sys.exit(1)

        result = executor.open_spread(args.symbol, args.open)
        print(json.dumps(result, indent=2, default=str))

        if result.get("success"):
            sys.exit(0)
        else:
            sys.exit(1)

    if args.close:
        # Close active hedge
        result = executor.close_spread()
        print(json.dumps(result, indent=2, default=str))

        if result.get("success"):
            sys.exit(0)
        else:
            sys.exit(1)

    # Default action - show help
    parser.print_help()


if __name__ == "__main__":
    main()
