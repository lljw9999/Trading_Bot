#!/usr/bin/env python3
"""
PDT Guard Daemon

Monitors account equity and day trade count via Alpaca API.
Publishes PDT restrictions to Redis for StrategyGuard to consume.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import redis
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST
from alpaca_trade_api.common import URL

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.layers.layer5_risk.pdt_guard import create_pdt_guard
from src.utils.logger import get_logger


class PDTGuardDaemon:
    """Daemon that monitors PDT status via Alpaca API"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.redis_client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True
        )

        # Load config from Redis
        config = self.redis_client.hgetall("risk:pdt_guard")

        self.guard = create_pdt_guard(
            min_equity_usd=float(config.get("min_equity_usd", 26000)),
            max_daytrades_5d=int(config.get("max_daytrades_5d", 3)),
        )

        self.enabled = config.get("enabled", "1") == "1"
        self.check_interval = 300.0  # 5 minutes (don't hammer the API)

        # Initialize Alpaca API
        self.api = None
        self.dry_run = True

        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        base_url = os.getenv("ALPACA_PAPER_BASE", "https://paper-api.alpaca.markets")

        if api_key and api_secret:
            try:
                self.api = REST(
                    key_id=api_key,
                    secret_key=api_secret,
                    base_url=URL(base_url),
                    api_version="v2",
                )
                self.dry_run = False
                self.logger.info("✅ Alpaca API connected for PDT monitoring")
            except Exception as e:
                self.logger.error(f"Failed to connect to Alpaca API: {e}")

        self.logger.info(
            f"PDT Guard Daemon initialized (enabled: {self.enabled}, dry_run: {self.dry_run})"
        )

    async def run(self):
        """Main daemon loop"""
        self.logger.info("🛡️ Starting PDT monitoring...")

        while True:
            try:
                if self.enabled:
                    await self._check_pdt_status()
                await asyncio.sleep(self.check_interval)
            except KeyboardInterrupt:
                self.logger.info("Received shutdown signal")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_pdt_status(self):
        """Check PDT status and publish to Redis"""
        try:
            # Get account data
            if self.dry_run:
                # Mock data for testing
                account_equity = 15000.0
                daytrades_5d = 2
            else:
                account = self.api.get_account()
                account_equity = float(account.equity)

                # Get day trade count (approximate - would need to count actual trades)
                # For now, use a simple heuristic or rely on broker's PDT flag
                daytrades_5d = self._count_recent_daytrades()

            # Run PDT check
            pdt_result = self.guard.check(account_equity, daytrades_5d)

            # Publish results to Redis
            self.redis_client.hset("risk:pdt_status", mapping=pdt_result)

            # Set block flag if PDT violation
            if pdt_result["should_block_intraday"]:
                self.redis_client.set("risk:pdt_block", "1")
                self.logger.warning("🚨 PDT block active")

                # Alert
                alert = {
                    "type": "PDT_VIOLATION",
                    "message": f"PDT block: equity ${account_equity:,.0f}, daytrades {daytrades_5d}",
                    "pdt_status": pdt_result,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self.redis_client.xadd("alerts:pdt", alert)
            else:
                self.redis_client.delete("risk:pdt_block")

            # Update metrics
            self.redis_client.set("metrics:pdt_guard:account_equity", account_equity)
            self.redis_client.set("metrics:pdt_guard:daytrades_5d", daytrades_5d)
            self.redis_client.set(
                "metrics:pdt_guard:is_blocked",
                "1" if pdt_result["should_block_intraday"] else "0",
            )
            self.redis_client.set("metrics:pdt_guard:last_check", time.time())

        except Exception as e:
            self.logger.error(f"Error checking PDT status: {e}")

    def _count_recent_daytrades(self) -> int:
        """
        Count day trades in the last 5 business days.

        Returns:
            Number of day trades
        """
        if self.dry_run:
            return 2  # Mock data

        try:
            # Get orders from last 5 days
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(
                days=7
            )  # Get extra days to account for weekends

            orders = self.api.list_orders(
                status="all",
                after=start_date.isoformat(),
                until=end_date.isoformat(),
                limit=1000,
            )

            # Count day trades (simplified - would need more sophisticated logic)
            # For now, count sell orders as potential day trades
            day_trade_count = 0
            for order in orders:
                if order.side == "sell" and order.status == "filled":
                    day_trade_count += 1

            # Cap at reasonable number
            return min(day_trade_count, 10)

        except Exception as e:
            self.logger.error(f"Error counting day trades: {e}")
            return 0


def main():
    """Run the PDT guard daemon"""
    daemon = PDTGuardDaemon()

    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        print("\nShutdown complete")


if __name__ == "__main__":
    main()
