#!/usr/bin/env python3
"""
Market Hours Guard Daemon

Publishes market open/closed status to Redis every 5 seconds.
Sets mode=halt when market is closed unless allow_afterhours=1.
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

from src.layers.layer5_risk.market_hours_guard import create_market_hours_guard
from src.utils.logger import get_logger


class MarketHoursGuardDaemon:
    """Daemon that monitors and enforces market hours"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.redis_client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True
        )

        # Load config from Redis
        config = self.redis_client.hgetall("risk:market_hours_guard")

        self.guard = create_market_hours_guard(
            tz=config.get("tz", "America/New_York"),
            pre=config.get("pre", "09:25"),
            post=config.get("post", "16:05"),
        )

        self.enabled = config.get("enabled", "1") == "1"
        self.allow_afterhours = config.get("allow_afterhours", "0") == "1"
        self.check_interval = 5.0  # seconds

        self.logger.info(f"Market Hours Guard Daemon initialized")
        self.logger.info(
            f"Enabled: {self.enabled}, Allow afterhours: {self.allow_afterhours}"
        )

    async def run(self):
        """Main daemon loop"""
        self.logger.info("🕐 Starting market hours monitoring...")

        while True:
            try:
                await self._check_and_publish()
                await asyncio.sleep(self.check_interval)
            except KeyboardInterrupt:
                self.logger.info("Received shutdown signal")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_and_publish(self):
        """Check market status and publish to Redis"""
        if not self.enabled:
            return

        try:
            # Get market status
            status = self.guard.get_market_status()
            is_open = status["is_open"]
            should_block = status["should_block"]

            # Publish market status
            self.redis_client.set("risk:market_open", "1" if is_open else "0")
            self.redis_client.hset("risk:market_status", mapping=status)

            # Enforce trading halt if market closed and after-hours not allowed
            if should_block and not self.allow_afterhours:
                current_mode = self.redis_client.get("mode")
                if current_mode != "halt":
                    self.logger.warning(
                        f"🔴 Market closed, setting mode=halt (was: {current_mode})"
                    )
                    self.redis_client.set("mode", "halt")

                    # Log to alert stream
                    alert = {
                        "type": "MARKET_HOURS_HALT",
                        "message": "Trading halted - market closed",
                        "status": status,
                        "timestamp": datetime.utcnow().isoformat(),
                        "previous_mode": current_mode,
                    }
                    self.redis_client.xadd("alerts:market_hours", alert)

            elif is_open and not should_block:
                # Market is open - restore auto mode if currently halted due to hours
                current_mode = self.redis_client.get("mode")
                if current_mode == "halt":
                    # Check if halt was due to market hours (not other reasons)
                    last_halt_reason = self.redis_client.get("last_halt_reason")
                    if last_halt_reason == "market_hours" or not last_halt_reason:
                        self.logger.info("🟢 Market open, restoring auto mode")
                        self.redis_client.set("mode", "auto")
                        self.redis_client.delete("last_halt_reason")

            # Update metrics
            self.redis_client.set(
                "metrics:market_hours_guard:is_open", "1" if is_open else "0"
            )
            self.redis_client.set(
                "metrics:market_hours_guard:should_block", "1" if should_block else "0"
            )
            self.redis_client.set("metrics:market_hours_guard:last_check", time.time())

        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")


def main():
    """Run the market hours guard daemon"""
    daemon = MarketHoursGuardDaemon()

    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        print("\nShutdown complete")


if __name__ == "__main__":
    main()
