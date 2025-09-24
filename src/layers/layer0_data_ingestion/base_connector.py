"""
Base Data Connector for Market Data Sources

Provides abstract base class for all data connectors with async context management.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, AsyncIterator, Dict, Any, Callable
import aiohttp

from .schemas import MarketTick
from ...utils.logger import get_logger


class BaseDataConnector(ABC):
    """
    Abstract base class for all market data connectors.

    Provides common functionality for connecting, subscribing, parsing, and publishing
    market data with async context management.
    """

    def __init__(
        self,
        symbols: list[str],
        exchange_name: str,
        asset_type: str,
        publish_callback: Optional[Callable[[MarketTick], None]] = None,
    ):
        """
        Initialize the base connector.

        Args:
            symbols: List of symbols to subscribe to
            exchange_name: Name of the exchange
            asset_type: Type of asset ('crypto' or 'stock')
            publish_callback: Optional callback for publishing data
        """
        self.symbols = symbols
        self.exchange_name = exchange_name
        self.asset_type = asset_type
        self.publish_callback = publish_callback

        self.logger = get_logger(f"connector.{exchange_name}")
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[aiohttp.ClientWebSocketResponse] = None

        # Connection state
        self.is_connected = False
        self.is_subscribed = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

        # Statistics
        self.ticks_received = 0
        self.ticks_processed = 0
        self.last_tick_time: Optional[datetime] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Establish connection to the data source."""
        try:
            self.logger.info(f"Connecting to {self.exchange_name}...")

            # Create HTTP session
            self.session = aiohttp.ClientSession()

            # Perform exchange-specific connection
            await self._connect_impl()

            self.is_connected = True
            self.logger.info(f"Connected to {self.exchange_name}")

        except Exception as e:
            self.logger.error(f"Failed to connect to {self.exchange_name}: {e}")
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        """Disconnect from the data source."""
        try:
            self.logger.info(f"Disconnecting from {self.exchange_name}...")

            self.is_connected = False
            self.is_subscribed = False

            # Close WebSocket
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()

            # Close HTTP session
            if self.session and not self.session.closed:
                await self.session.close()

            self.logger.info(f"Disconnected from {self.exchange_name}")

        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")

    async def subscribe(self) -> None:
        """Subscribe to market data for configured symbols."""
        if not self.is_connected:
            raise RuntimeError("Must be connected before subscribing")

        try:
            self.logger.info(
                f"Subscribing to {len(self.symbols)} symbols: {self.symbols}"
            )

            await self._subscribe_impl()

            self.is_subscribed = True
            self.logger.info("Successfully subscribed to market data")

        except Exception as e:
            self.logger.error(f"Failed to subscribe: {e}")
            raise

    async def start_data_stream(self) -> AsyncIterator[MarketTick]:
        """
        Start the data stream and yield market ticks.

        Yields:
            MarketTick objects as they are received
        """
        if not self.is_subscribed:
            await self.subscribe()

        self.logger.info("Starting data stream...")

        try:
            async for tick in self._stream_data():
                self.ticks_received += 1
                self.last_tick_time = datetime.utcnow()

                try:
                    # Parse the raw data into MarketTick
                    parsed_tick = await self._parse_tick(tick)

                    if parsed_tick:
                        self.ticks_processed += 1

                        # Call publish callback if provided
                        if self.publish_callback:
                            self.publish_callback(parsed_tick)

                        yield parsed_tick

                except Exception as e:
                    self.logger.error(f"Error parsing tick: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error in data stream: {e}")

            # Attempt reconnection
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                self.logger.info(
                    f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}"
                )

                await self.disconnect()
                await asyncio.sleep(2**self.reconnect_attempts)  # Exponential backoff
                await self.connect()

                # Recursively restart stream
                async for tick in self.start_data_stream():
                    yield tick
            else:
                self.logger.error("Max reconnection attempts reached")
                raise

    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            "exchange": self.exchange_name,
            "symbols": self.symbols,
            "is_connected": self.is_connected,
            "is_subscribed": self.is_subscribed,
            "ticks_received": self.ticks_received,
            "ticks_processed": self.ticks_processed,
            "last_tick_time": (
                self.last_tick_time.isoformat() if self.last_tick_time else None
            ),
            "reconnect_attempts": self.reconnect_attempts,
        }

    # Abstract methods to be implemented by subclasses

    @abstractmethod
    async def _connect_impl(self) -> None:
        """Exchange-specific connection implementation."""
        pass

    @abstractmethod
    async def _subscribe_impl(self) -> None:
        """Exchange-specific subscription implementation."""
        pass

    @abstractmethod
    async def _stream_data(self) -> AsyncIterator[Dict[str, Any]]:
        """Exchange-specific data streaming implementation."""
        pass

    @abstractmethod
    async def _parse_tick(self, raw_data: Dict[str, Any]) -> Optional[MarketTick]:
        """Parse raw exchange data into MarketTick format."""
        pass
