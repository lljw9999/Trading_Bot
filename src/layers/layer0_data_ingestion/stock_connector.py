"""
Stock Data Connector for IEX Cloud REST API

Implements market data polling from IEX Cloud for stock data.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional, Dict, Any, AsyncIterator
import aiohttp

from .base_connector import BaseDataConnector
from .schemas import MarketTick
from ...utils.config_manager import config


class IEXStockConnector(BaseDataConnector):
    """IEX Cloud REST API stock data connector."""

    def __init__(self, symbols: list[str], **kwargs):
        """
        Initialize IEX stock connector.

        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'MSFT'])
        """
        super().__init__(
            symbols=symbols, exchange_name="iex", asset_type="stock", **kwargs
        )

        # IEX configuration
        self.api_key = config.get("IEX_CLOUD_API_KEY", "pk_test_default")
        self.base_url = "https://cloud.iexapis.com/stable"
        self.polling_interval = 5.0  # 5 seconds for stock data

        # Rate limiting
        self.rate_limit = config.get(
            "data_ingestion.sources.stocks.iex.rate_limit", 100
        )
        self.last_request_time = datetime.now(timezone.utc)

        self.logger.info(f"IEX connector initialized for {len(symbols)} symbols")

    async def _connect_impl(self) -> None:
        """Initialize connection to IEX Cloud API."""
        try:
            # Test API connection with a simple request
            test_url = f"{self.base_url}/time"
            async with self.session.get(
                test_url, params={"token": self.api_key}
            ) as response:
                if response.status == 200:
                    server_time = await response.json()
                    self.logger.info(
                        f"Connected to IEX Cloud API, server time: {server_time}"
                    )
                else:
                    raise Exception(f"API test failed with status {response.status}")

        except Exception as e:
            self.logger.error(f"Failed to connect to IEX API: {e}")
            raise

    async def _subscribe_impl(self) -> None:
        """IEX uses polling, so this just validates symbols."""
        try:
            # Validate symbols by fetching quote for each
            valid_symbols = []

            for symbol in self.symbols:
                url = f"{self.base_url}/stock/{symbol}/quote"
                params = {"token": self.api_key}

                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        valid_symbols.append(symbol)
                        self.logger.info(f"Validated symbol: {symbol}")
                    else:
                        self.logger.warning(
                            f"Invalid symbol or API error for {symbol}: {response.status}"
                        )

            self.symbols = valid_symbols
            self.logger.info(f"Subscribed to {len(self.symbols)} valid symbols")

        except Exception as e:
            self.logger.error(f"Failed to validate symbols: {e}")
            raise

    async def _stream_data(self) -> AsyncIterator[Dict[str, Any]]:
        """Poll IEX API for stock data at regular intervals."""
        while True:
            try:
                # Respect rate limits
                await self._rate_limit_delay()

                # Fetch batch quotes for all symbols
                symbols_str = ",".join(self.symbols)
                url = f"{self.base_url}/stock/market/batch"
                params = {
                    "symbols": symbols_str,
                    "types": "quote",
                    "token": self.api_key,
                }

                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Yield each symbol's data
                        for symbol, symbol_data in data.items():
                            if "quote" in symbol_data:
                                yield {
                                    "symbol": symbol,
                                    "quote": symbol_data["quote"],
                                    "timestamp": datetime.now(timezone.utc),
                                }
                    else:
                        self.logger.error(
                            f"API request failed with status {response.status}"
                        )
                        await asyncio.sleep(
                            self.polling_interval * 2
                        )  # Back off on error

                # Wait before next poll
                await asyncio.sleep(self.polling_interval)

            except Exception as e:
                self.logger.error(f"Error in stock data polling: {e}")
                await asyncio.sleep(self.polling_interval)

    async def _parse_tick(self, raw_data: Dict[str, Any]) -> Optional[MarketTick]:
        """Parse IEX quote data into MarketTick format."""
        try:
            symbol = raw_data["symbol"]
            quote = raw_data["quote"]

            # Extract relevant fields from IEX quote
            bid_price = quote.get("iexBidPrice")
            ask_price = quote.get("iexAskPrice")
            last_price = quote.get("latestPrice")
            bid_size = quote.get("iexBidSize")
            ask_size = quote.get("iexAskSize")
            volume = quote.get("latestVolume")

            # Handle market hours - IEX may not have bid/ask outside trading hours
            if not bid_price or not ask_price:
                # Use latest price as both bid and ask if real bid/ask not available
                if last_price:
                    bid_price = last_price
                    ask_price = last_price

            return MarketTick(
                symbol=symbol,
                exchange=self.exchange_name,
                asset_type=self.asset_type,
                timestamp=raw_data["timestamp"],
                exchange_timestamp=(
                    datetime.fromtimestamp(quote.get("latestUpdate", 0) / 1000)
                    if quote.get("latestUpdate")
                    else None
                ),
                bid=Decimal(str(bid_price)) if bid_price else None,
                ask=Decimal(str(ask_price)) if ask_price else None,
                last=Decimal(str(last_price)) if last_price else None,
                bid_size=Decimal(str(bid_size)) if bid_size else None,
                ask_size=Decimal(str(ask_size)) if ask_size else None,
                volume=Decimal(str(volume)) if volume else None,
                metadata={
                    "type": "quote",
                    "market_cap": quote.get("marketCap"),
                    "pe_ratio": quote.get("peRatio"),
                    "week_52_high": quote.get("week52High"),
                    "week_52_low": quote.get("week52Low"),
                    "raw": quote,
                },
            )

        except Exception as e:
            self.logger.error(f"Error parsing IEX data: {e}, Data: {raw_data}")
            return None

    async def _rate_limit_delay(self) -> None:
        """Implement rate limiting for API requests."""
        now = datetime.now(timezone.utc)
        time_since_last = (now - self.last_request_time).total_seconds()
        min_interval = 1.0 / self.rate_limit  # seconds between requests

        if time_since_last < min_interval:
            delay = min_interval - time_since_last
            await asyncio.sleep(delay)

        self.last_request_time = datetime.now(timezone.utc)


async def test_iex_connector():
    """Test function for IEX stock connector."""
    import logging

    logging.basicConfig(level=logging.INFO)

    symbols = ["AAPL", "MSFT", "GOOGL"]

    async with IEXStockConnector(symbols) as connector:
        print(f"Testing IEX connector with symbols: {symbols}")

        tick_count = 0
        async for tick in connector.start_data_stream():
            print(f"Received tick {tick_count + 1}: {tick.symbol} @ {tick.last}")
            tick_count += 1

            if tick_count >= 10:  # Test with 10 ticks
                break

        print(f"Connector stats: {connector.get_stats()}")


if __name__ == "__main__":
    asyncio.run(test_iex_connector())
