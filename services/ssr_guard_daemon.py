#!/usr/bin/env python3
"""
SSR Guard Daemon

Monitors symbols for Short Sale Restriction conditions.
Publishes SSR flags to Redis for execution layer to enforce.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import redis
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.layers.layer5_risk.ssr_guard import create_ssr_guard
from src.utils.logger import get_logger


class SSRGuardDaemon:
    """Daemon that monitors SSR conditions across symbols"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.redis_client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True
        )

        # Load config
        config = self.redis_client.hgetall("risk:ssr_guard")
        self.enabled = config.get("enabled", "1") == "1"

        self.guard = create_ssr_guard()
        self.check_interval = 30.0  # Check every 30 seconds

        # Get symbols to monitor
        self.symbols = list(
            self.redis_client.smembers("symbols:stocks")
            or ["AAPL", "MSFT", "NVDA", "SPY"]
        )

        self.logger.info(f"SSR Guard Daemon initialized (enabled: {self.enabled})")
        self.logger.info(f"Monitoring symbols: {self.symbols}")

    async def run(self):
        """Main daemon loop"""
        self.logger.info("🛡️ Starting SSR monitoring...")

        while True:
            try:
                if self.enabled:
                    await self._check_all_symbols()
                await asyncio.sleep(self.check_interval)
            except KeyboardInterrupt:
                self.logger.info("Received shutdown signal")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_all_symbols(self):
        """Check SSR status for all monitored symbols"""
        for symbol in self.symbols:
            try:
                await self._check_symbol_ssr(symbol)
            except Exception as e:
                self.logger.error(f"Error checking SSR for {symbol}: {e}")

    async def _check_symbol_ssr(self, symbol: str):
        """Check SSR status for a specific symbol"""
        try:
            # Get current and previous prices
            # In practice, this would come from market data feed
            current_price = self._get_current_price(symbol)
            prev_close = self._get_previous_close(symbol)

            if current_price is None or prev_close is None:
                return

            # Evaluate SSR
            ssr_result = self.guard.evaluate(symbol, current_price, prev_close)

            # Publish to Redis
            self.redis_client.hset(f"risk:ssr_status:{symbol}", mapping=ssr_result)

            # Set/clear SSR block flag
            if ssr_result["is_ssr_active"]:
                self.redis_client.set(f"risk:ssr:{symbol}", "1")
                # Set expiration for end of trading day
                self.redis_client.expire(f"risk:ssr:{symbol}", 86400)  # 24 hours

                # Alert on new SSR activation
                existing_ssr = self.redis_client.get(f"ssr:alerted:{symbol}")
                if not existing_ssr:
                    alert = {
                        "type": "SSR_ACTIVATED",
                        "symbol": symbol,
                        "message": f"SSR activated for {symbol}: {ssr_result['price_change_pct']:.1f}% drop",
                        "ssr_status": ssr_result,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    self.redis_client.xadd("alerts:ssr", alert)
                    self.redis_client.setex(f"ssr:alerted:{symbol}", 86400, "1")
            else:
                self.redis_client.delete(f"risk:ssr:{symbol}")
                self.redis_client.delete(f"ssr:alerted:{symbol}")

            # Update metrics
            self.redis_client.set(
                f"metrics:ssr_guard:{symbol}:is_active",
                "1" if ssr_result["is_ssr_active"] else "0",
            )
            self.redis_client.set(
                f"metrics:ssr_guard:{symbol}:price_change_pct",
                ssr_result["price_change_pct"],
            )

        except Exception as e:
            self.logger.error(f"Error checking SSR for {symbol}: {e}")

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        # Mock data for now - would integrate with market data feed
        mock_prices = {"AAPL": 175.50, "MSFT": 310.25, "NVDA": 450.75, "SPY": 425.30}
        return mock_prices.get(symbol)

    def _get_previous_close(self, symbol: str) -> Optional[float]:
        """Get previous close price for symbol"""
        # Mock data - would come from market data or Redis cache
        mock_closes = {
            "AAPL": 180.00,  # -2.5% from close
            "MSFT": 315.00,  # -1.5% from close
            "NVDA": 500.00,  # -9.85% from close (approaching SSR)
            "SPY": 430.00,  # -1.1% from close
        }
        return mock_closes.get(symbol)


def main():
    """Run the SSR guard daemon"""
    daemon = SSRGuardDaemon()

    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        print("\nShutdown complete")


if __name__ == "__main__":
    main()
