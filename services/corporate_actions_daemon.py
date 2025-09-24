#!/usr/bin/env python3
"""
Corporate Actions Daemon

Polls broker/vendor for splits & dividends and processes them through FIFO ledger.
Integrates with existing WORM archive and audit systems.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import redis

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from accounting.fifo_ledger import FIFOLedger
from src.utils.logger import get_logger


class CorporateActionsDaemon:
    """Daemon that monitors and processes corporate actions"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.redis_client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True
        )

        # Initialize FIFO ledger for corporate actions processing
        self.fifo_ledger = FIFOLedger()

        self.check_interval = (
            3600.0  # Check every hour (corporate actions are not frequent)
        )

        # Get symbols to monitor
        self.symbols = list(
            self.redis_client.smembers("symbols:stocks")
            or ["AAPL", "MSFT", "NVDA", "SPY"]
        )

        self.logger.info(f"Corporate Actions Daemon initialized")
        self.logger.info(f"Monitoring symbols: {self.symbols}")

    async def run(self):
        """Main daemon loop"""
        self.logger.info("📅 Starting corporate actions monitoring...")

        while True:
            try:
                await self._check_corporate_actions()
                await asyncio.sleep(self.check_interval)
            except KeyboardInterrupt:
                self.logger.info("Received shutdown signal")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_corporate_actions(self):
        """Check for new corporate actions across all symbols"""
        self.logger.info("🔍 Checking for corporate actions...")

        for symbol in self.symbols:
            try:
                # Check for stock splits
                await self._check_splits(symbol)

                # Check for dividends
                await self._check_dividends(symbol)

            except Exception as e:
                self.logger.error(f"Error checking corporate actions for {symbol}: {e}")

    async def _check_splits(self, symbol: str):
        """Check for stock splits for a symbol"""
        try:
            # Mock implementation - would integrate with real corporate actions feed
            # Check if we've already processed recent splits
            last_split = self.redis_client.hget(
                f"corporate_actions:{symbol}", "last_split_check"
            )
            current_time = time.time()

            if (
                last_split and current_time - float(last_split) < 86400
            ):  # Check once per day
                return

            # Mock: simulate finding a split (in practice, would poll API)
            mock_split_data = self._get_mock_split_data(symbol)

            if mock_split_data:
                split_result = self.fifo_ledger.apply_split(
                    symbol=symbol,
                    ratio_from=mock_split_data["ratio_from"],
                    ratio_to=mock_split_data["ratio_to"],
                    effective_date=mock_split_data["effective_date"],
                )

                # Publish to corporate actions stream
                self.redis_client.xadd(
                    "corp_actions:events",
                    {
                        "type": "STOCK_SPLIT",
                        "symbol": symbol,
                        "data": json.dumps(split_result),
                    },
                )

                self.logger.info(
                    f"✅ Processed stock split for {symbol}: {split_result}"
                )

            # Update last check time
            self.redis_client.hset(
                f"corporate_actions:{symbol}", "last_split_check", current_time
            )

        except Exception as e:
            self.logger.error(f"Error checking splits for {symbol}: {e}")

    async def _check_dividends(self, symbol: str):
        """Check for dividend payments for a symbol"""
        try:
            # Mock implementation - would integrate with real corporate actions feed
            last_dividend = self.redis_client.hget(
                f"corporate_actions:{symbol}", "last_dividend_check"
            )
            current_time = time.time()

            if (
                last_dividend and current_time - float(last_dividend) < 86400
            ):  # Check once per day
                return

            # Mock: simulate finding a dividend
            mock_dividend_data = self._get_mock_dividend_data(symbol)

            if mock_dividend_data:
                dividend_result = self.fifo_ledger.record_dividend(
                    symbol=symbol,
                    gross_amount=mock_dividend_data["gross_amount"],
                    tax_withheld=mock_dividend_data["tax_withheld"],
                    record_date=mock_dividend_data["record_date"],
                )

                # Publish to corporate actions stream
                self.redis_client.xadd(
                    "corp_actions:events",
                    {
                        "type": "DIVIDEND",
                        "symbol": symbol,
                        "data": json.dumps(dividend_result),
                    },
                )

                self.logger.info(
                    f"✅ Processed dividend for {symbol}: ${dividend_result['gross_amount']:.2f}"
                )

            # Update last check time
            self.redis_client.hset(
                f"corporate_actions:{symbol}", "last_dividend_check", current_time
            )

        except Exception as e:
            self.logger.error(f"Error checking dividends for {symbol}: {e}")

    def _get_mock_split_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get mock split data (would integrate with real data source)"""
        # Return None most of the time (splits are rare)
        # In practice, would poll Alpaca/Polygon/IEX for corporate actions
        return None

    def _get_mock_dividend_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get mock dividend data (would integrate with real data source)"""
        # Return None most of the time
        # In practice, would poll corporate actions API
        return None

    async def process_manual_corporate_action(self, action_data: Dict[str, Any]):
        """Process manually triggered corporate action"""
        try:
            action_type = action_data.get("action_type")
            symbol = action_data.get("symbol")

            if action_type == "STOCK_SPLIT":
                result = self.fifo_ledger.apply_split(
                    symbol=symbol,
                    ratio_from=action_data["ratio_from"],
                    ratio_to=action_data["ratio_to"],
                    effective_date=action_data.get("effective_date"),
                )

            elif action_type == "DIVIDEND":
                result = self.fifo_ledger.record_dividend(
                    symbol=symbol,
                    gross_amount=action_data["gross_amount"],
                    tax_withheld=action_data.get("tax_withheld", 0.0),
                    record_date=action_data.get("record_date"),
                )
            else:
                raise ValueError(f"Unsupported action type: {action_type}")

            # Publish result
            self.redis_client.xadd(
                "corp_actions:events",
                {
                    "type": action_type,
                    "symbol": symbol,
                    "data": json.dumps(result),
                    "source": "MANUAL",
                },
            )

            self.logger.info(
                f"✅ Processed manual corporate action: {action_type} for {symbol}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error processing manual corporate action: {e}")
            raise


def main():
    """Run the corporate actions daemon"""
    daemon = CorporateActionsDaemon()

    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        print("\nShutdown complete")


if __name__ == "__main__":
    main()
