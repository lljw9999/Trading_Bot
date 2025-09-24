#!/usr/bin/env python3
"""
Halts and LULD Daemon

Monitors for trading halts and LULD band violations.
Publishes halt flags to Redis for TTL watchdog and execution layer.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import redis

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.layers.layer5_risk.halts_luld_monitor import create_halt_luld_monitor
from src.utils.logger import get_logger


class HaltLULDDaemon:
    """Daemon that monitors halt and LULD conditions"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.redis_client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True
        )

        self.monitor = create_halt_luld_monitor()
        self.check_interval = 5.0  # Check every 5 seconds

        # Get symbols to monitor
        self.symbols = list(
            self.redis_client.smembers("symbols:stocks")
            or ["AAPL", "MSFT", "NVDA", "SPY"]
        )

        self.logger.info(f"Halt/LULD Daemon initialized")
        self.logger.info(f"Monitoring symbols: {self.symbols}")

    async def run(self):
        """Main daemon loop"""
        self.logger.info("🛑 Starting halt/LULD monitoring...")

        while True:
            try:
                await self._monitor_all_symbols()
                await asyncio.sleep(self.check_interval)
            except KeyboardInterrupt:
                self.logger.info("Received shutdown signal")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _monitor_all_symbols(self):
        """Monitor halt/LULD status for all symbols"""
        for symbol in self.symbols:
            try:
                await self._monitor_symbol(symbol)
            except Exception as e:
                self.logger.error(f"Error monitoring {symbol}: {e}")

    async def _monitor_symbol(self, symbol: str):
        """Monitor halt/LULD status for a specific symbol"""
        try:
            # Get current market data (mock for now)
            price = self._get_mock_price(symbol)
            luld_bands = self._get_mock_luld_bands(symbol)

            if price is None:
                return

            # Process tick through monitor
            result = self.monitor.on_tick(symbol, price, luld_bands)

            if not result:
                return

            # Publish status to Redis
            self.redis_client.hset(f"market:halt_luld_status:{symbol}", mapping=result)

            # Set halt flags
            if result["halt_status"]["is_halted"]:
                self.redis_client.set(f"risk:halted:{symbol}", "1")
                self.redis_client.expire(f"risk:halted:{symbol}", 3600)  # 1 hour expiry
            else:
                self.redis_client.delete(f"risk:halted:{symbol}")

            # Set LULD bands if available
            if luld_bands:
                self.redis_client.hset(f"market:luld:{symbol}", mapping=luld_bands)

            # Update metrics
            self.redis_client.set(
                f"metrics:halt_luld:{symbol}:is_halted",
                "1" if result["halt_status"]["is_halted"] else "0",
            )
            self.redis_client.set(
                f"metrics:halt_luld:{symbol}:near_luld",
                "1" if result["luld_status"]["near_bands"] else "0",
            )
            self.redis_client.set(f"metrics:halt_luld:{symbol}:last_check", time.time())

        except Exception as e:
            self.logger.error(f"Error monitoring {symbol}: {e}")

    def _get_mock_price(self, symbol: str) -> Optional[float]:
        """Get mock current price (would integrate with real market data)"""
        mock_prices = {"AAPL": 175.50, "MSFT": 310.25, "NVDA": 450.75, "SPY": 425.30}
        return mock_prices.get(symbol)

    def _get_mock_luld_bands(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get mock LULD bands (would come from market data feed)"""
        mock_bands = {
            "AAPL": {"up": 185.00, "down": 165.00},
            "MSFT": {"up": 325.00, "down": 295.00},
            "NVDA": {"up": 475.00, "down": 425.00},
            "SPY": {"up": 435.00, "down": 415.00},
        }
        return mock_bands.get(symbol)


def main():
    """Run the halt/LULD daemon"""
    daemon = HaltLULDDaemon()

    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        print("\nShutdown complete")


if __name__ == "__main__":
    main()
