#!/usr/bin/env python3
"""
Nautilus-Redis Bridge

Bidirectional bridge between NautilusTrader event bus and Redis streams.
Allows existing monitoring, risk management, and compliance systems to
work seamlessly with Nautilus execution.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.utils.aredis import (
        get_redis,
        get_batch_writer,
        set_metric,
        publish_trade_event,
        publish_metrics_batch,
    )

    ASYNC_REDIS_AVAILABLE = True
except ImportError:
    ASYNC_REDIS_AVAILABLE = False

# Try to import Nautilus components
try:
    from nautilus_trader.core.nautilus_pyo3 import UUID4
    from nautilus_trader.model.events import OrderEvent, FillEvent, OrderFilled
    from nautilus_trader.model.data import QuoteTick, TradeTick
    from nautilus_trader.model.identifiers import InstrumentId, ClientOrderId
    from nautilus_trader.model.enums import OrderSide, OrderStatus
    from nautilus_trader.common.component import Component
    from nautilus_trader.common.logging import Logger

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False

logger = logging.getLogger("nautilus_bridge")


@dataclass
class BridgeConfig:
    """Configuration for Nautilus-Redis bridge."""

    # Redis stream mappings
    orders_stream: str = "exec:orders"
    fills_stream: str = "exec:fills"
    market_data_stream: str = "market.raw.crypto"
    metrics_stream: str = "metrics:nautilus"

    # Feature control
    publish_orders: bool = True
    publish_fills: bool = True
    publish_market_data: bool = True
    publish_metrics: bool = True

    # Risk control integration
    subscribe_to_halt: bool = True
    subscribe_to_mode: bool = True
    halt_redis_key: str = "mode"

    # Performance settings
    batch_size: int = 100
    flush_interval: float = 0.1
    max_event_age: float = 60.0  # Max age for events to process


class NautilusRedisBridge:
    """
    Bridge between NautilusTrader and Redis.

    Provides bidirectional integration:
    - Publishes Nautilus events (orders, fills, market data) to Redis
    - Subscribes to Redis commands (halt, mode changes) and sends to Nautilus
    - Maintains existing monitoring and compliance integrations
    """

    def __init__(self, config: BridgeConfig = None):
        """
        Initialize Nautilus-Redis bridge.

        Args:
            config: Bridge configuration
        """
        self.config = config or BridgeConfig()
        self._running = False
        self._redis = None
        self._batch_writer = None

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {
            "order": [],
            "fill": [],
            "quote": [],
            "trade": [],
        }

        # Performance tracking
        self.stats = {
            "events_processed": 0,
            "orders_published": 0,
            "fills_published": 0,
            "quotes_published": 0,
            "trades_published": 0,
            "redis_errors": 0,
            "bridge_uptime": 0.0,
        }

        self._start_time = time.time()

        if not NAUTILUS_AVAILABLE:
            logger.warning(
                "NautilusTrader not available - bridge will operate in mock mode"
            )

        if not ASYNC_REDIS_AVAILABLE:
            logger.warning("Async Redis not available - some features disabled")

        logger.info("Initialized Nautilus-Redis bridge")

    async def start(self):
        """Start the bridge."""
        try:
            logger.info("🌊 Starting Nautilus-Redis bridge")

            if ASYNC_REDIS_AVAILABLE:
                self._redis = await get_redis()
                self._batch_writer = await get_batch_writer()

            self._running = True

            # Start background tasks
            tasks = [self._stats_publisher(), self._redis_command_subscriber()]

            await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"Error starting Nautilus bridge: {e}")
            self._running = False
            raise

    async def stop(self):
        """Stop the bridge gracefully."""
        logger.info("Stopping Nautilus-Redis bridge")
        self._running = False

        # Flush any remaining events
        if self._batch_writer:
            await self._batch_writer.flush()

    # Event Publishing (Nautilus -> Redis)

    async def handle_order_event(self, event):
        """
        Handle order event from Nautilus.

        Args:
            event: Nautilus order event
        """
        try:
            if not self.config.publish_orders:
                return

            # Convert Nautilus order event to Redis format
            order_data = await self._convert_order_event(event)

            if order_data and ASYNC_REDIS_AVAILABLE:
                await publish_trade_event(order_data, self.config.orders_stream)
                self.stats["orders_published"] += 1

            # Call registered handlers
            for handler in self._event_handlers["order"]:
                await handler(order_data)

        except Exception as e:
            logger.error(f"Error handling order event: {e}")
            self.stats["redis_errors"] += 1

    async def handle_fill_event(self, event):
        """
        Handle fill event from Nautilus.

        Args:
            event: Nautilus fill event
        """
        try:
            if not self.config.publish_fills:
                return

            # Convert Nautilus fill event to Redis format
            fill_data = await self._convert_fill_event(event)

            if fill_data and ASYNC_REDIS_AVAILABLE:
                await publish_trade_event(fill_data, self.config.fills_stream)
                self.stats["fills_published"] += 1

                # Also publish to FIFO ledger for compliance
                await self._publish_to_fifo_ledger(fill_data)

            # Call registered handlers
            for handler in self._event_handlers["fill"]:
                await handler(fill_data)

        except Exception as e:
            logger.error(f"Error handling fill event: {e}")
            self.stats["redis_errors"] += 1

    async def handle_quote_tick(self, tick):
        """
        Handle quote tick from Nautilus.

        Args:
            tick: Nautilus quote tick
        """
        try:
            if not self.config.publish_market_data:
                return

            # Convert to Redis market data format
            quote_data = await self._convert_quote_tick(tick)

            if quote_data and ASYNC_REDIS_AVAILABLE:
                stream_name = (
                    f"{self.config.market_data_stream}.{quote_data['symbol'].lower()}"
                )
                await publish_trade_event(quote_data, stream_name)
                self.stats["quotes_published"] += 1

            # Call registered handlers
            for handler in self._event_handlers["quote"]:
                await handler(quote_data)

        except Exception as e:
            logger.error(f"Error handling quote tick: {e}")
            self.stats["redis_errors"] += 1

    async def handle_trade_tick(self, tick):
        """
        Handle trade tick from Nautilus.

        Args:
            tick: Nautilus trade tick
        """
        try:
            if not self.config.publish_market_data:
                return

            # Convert to Redis trade format
            trade_data = await self._convert_trade_tick(tick)

            if trade_data and ASYNC_REDIS_AVAILABLE:
                stream_name = f"{self.config.market_data_stream}.trades.{trade_data['symbol'].lower()}"
                await publish_trade_event(trade_data, stream_name)
                self.stats["trades_published"] += 1

            # Call registered handlers
            for handler in self._event_handlers["trade"]:
                await handler(trade_data)

        except Exception as e:
            logger.error(f"Error handling trade tick: {e}")
            self.stats["redis_errors"] += 1

    # Event Conversion (Nautilus -> Redis format)

    async def _convert_order_event(self, event) -> Optional[Dict[str, Any]]:
        """Convert Nautilus order event to Redis format."""
        try:
            if not NAUTILUS_AVAILABLE:
                return None

            # Extract common fields
            order_data = {
                "order_id": str(event.client_order_id),
                "instrument": str(event.instrument_id),
                "symbol": str(event.instrument_id).split(".")[0],  # Extract symbol
                "side": (
                    event.order_side.name.lower()
                    if hasattr(event, "order_side")
                    else "unknown"
                ),
                "status": (
                    event.order_status.name.lower()
                    if hasattr(event, "order_status")
                    else "unknown"
                ),
                "timestamp": event.ts_event,
                "ts_ns": event.ts_event,
                "event_type": type(event).__name__,
            }

            # Add event-specific fields
            if hasattr(event, "quantity"):
                order_data["quantity"] = float(event.quantity)
            if hasattr(event, "price"):
                order_data["price"] = float(event.price)
            if hasattr(event, "venue_order_id"):
                order_data["venue_order_id"] = str(event.venue_order_id)

            return order_data

        except Exception as e:
            logger.error(f"Error converting order event: {e}")
            return None

    async def _convert_fill_event(self, event) -> Optional[Dict[str, Any]]:
        """Convert Nautilus fill event to Redis format."""
        try:
            if not NAUTILUS_AVAILABLE:
                return None

            fill_data = {
                "fill_id": (
                    str(UUID4())
                    if NAUTILUS_AVAILABLE
                    else f"fill_{int(time.time() * 1000)}"
                ),
                "order_id": (
                    str(event.client_order_id)
                    if hasattr(event, "client_order_id")
                    else ""
                ),
                "instrument": (
                    str(event.instrument_id) if hasattr(event, "instrument_id") else ""
                ),
                "symbol": (
                    str(event.instrument_id).split(".")[0]
                    if hasattr(event, "instrument_id")
                    else ""
                ),
                "side": (
                    event.order_side.name.lower()
                    if hasattr(event, "order_side")
                    else "unknown"
                ),
                "quantity": (
                    float(event.last_qty) if hasattr(event, "last_qty") else 0.0
                ),
                "price": float(event.last_px) if hasattr(event, "last_px") else 0.0,
                "timestamp": (
                    event.ts_event if hasattr(event, "ts_event") else time.time()
                ),
                "ts_ns": (
                    event.ts_event
                    if hasattr(event, "ts_event")
                    else int(time.time() * 1e9)
                ),
                "venue": str(event.venue) if hasattr(event, "venue") else "unknown",
            }

            # Add fees if available
            if hasattr(event, "commission"):
                fill_data["fee"] = float(event.commission.as_decimal())
                fill_data["fee_currency"] = str(event.commission.currency)

            return fill_data

        except Exception as e:
            logger.error(f"Error converting fill event: {e}")
            return None

    async def _convert_quote_tick(self, tick) -> Optional[Dict[str, Any]]:
        """Convert Nautilus quote tick to Redis format."""
        try:
            if not NAUTILUS_AVAILABLE:
                return None

            quote_data = {
                "symbol": str(tick.instrument_id).split(".")[0],
                "bid_price": float(tick.bid_price),
                "ask_price": float(tick.ask_price),
                "bid_size": float(tick.bid_size),
                "ask_size": float(tick.ask_size),
                "mid": (float(tick.bid_price) + float(tick.ask_price)) / 2,
                "timestamp": tick.ts_event,
                "ts_ns": tick.ts_event,
                "venue": (
                    str(tick.instrument_id).split(".")[1]
                    if "." in str(tick.instrument_id)
                    else "unknown"
                ),
            }

            return quote_data

        except Exception as e:
            logger.error(f"Error converting quote tick: {e}")
            return None

    async def _convert_trade_tick(self, tick) -> Optional[Dict[str, Any]]:
        """Convert Nautilus trade tick to Redis format."""
        try:
            if not NAUTILUS_AVAILABLE:
                return None

            trade_data = {
                "symbol": str(tick.instrument_id).split(".")[0],
                "price": float(tick.price),
                "size": float(tick.size),
                "side": (
                    tick.aggressor_side.name.lower()
                    if hasattr(tick, "aggressor_side")
                    else "unknown"
                ),
                "timestamp": tick.ts_event,
                "ts_ns": tick.ts_event,
                "venue": (
                    str(tick.instrument_id).split(".")[1]
                    if "." in str(tick.instrument_id)
                    else "unknown"
                ),
                "trade_id": str(tick.trade_id) if hasattr(tick, "trade_id") else "",
            }

            return trade_data

        except Exception as e:
            logger.error(f"Error converting trade tick: {e}")
            return None

    # Redis Command Subscription (Redis -> Nautilus)

    async def _redis_command_subscriber(self):
        """Subscribe to Redis commands and forward to Nautilus."""
        try:
            if not ASYNC_REDIS_AVAILABLE:
                return

            while self._running:
                try:
                    # Check for halt/mode commands
                    if self.config.subscribe_to_halt:
                        await self._check_halt_command()

                    # Check for other risk commands
                    await self._check_risk_commands()

                    await asyncio.sleep(1)  # Check every second

                except Exception as e:
                    logger.error(f"Error in Redis command subscriber: {e}")
                    await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Fatal error in Redis command subscriber: {e}")

    async def _check_halt_command(self):
        """Check for halt/mode commands from Redis."""
        try:
            if not self._redis:
                return

            mode = await self._redis.get(self.config.halt_redis_key)

            if mode == "halt":
                logger.warning("🛑 Halt command received from Redis")
                # In real implementation, would send halt command to Nautilus
                await self._send_halt_to_nautilus()

        except Exception as e:
            logger.error(f"Error checking halt command: {e}")

    async def _check_risk_commands(self):
        """Check for other risk management commands."""
        try:
            if not self._redis:
                return

            # Check for position limits
            limit_breaches = await self._redis.get("risk:limit_breaches")
            if limit_breaches and int(limit_breaches) > 0:
                logger.warning("⚠️ Position limit breach detected")
                # In real implementation, would adjust position limits in Nautilus

        except Exception as e:
            logger.error(f"Error checking risk commands: {e}")

    async def _send_halt_to_nautilus(self):
        """Send halt command to Nautilus (placeholder)."""
        try:
            # In real implementation, would interact with Nautilus engine
            # to cancel all orders and halt trading
            logger.info("Sending halt command to Nautilus engine")

        except Exception as e:
            logger.error(f"Error sending halt to Nautilus: {e}")

    # Compliance Integration

    async def _publish_to_fifo_ledger(self, fill_data: Dict[str, Any]):
        """Publish fill to FIFO ledger for compliance."""
        try:
            if not ASYNC_REDIS_AVAILABLE:
                return

            # Format for FIFO ledger
            fifo_data = {
                "fill_id": fill_data.get("fill_id"),
                "symbol": fill_data.get("symbol"),
                "side": fill_data.get("side"),
                "quantity": fill_data.get("quantity"),
                "price": fill_data.get("price"),
                "timestamp": fill_data.get("timestamp"),
                "venue": fill_data.get("venue", "nautilus"),
                "strategy": "nautilus_bridge",
                "source": "nautilus",
            }

            await publish_trade_event(fifo_data, "accounting:fifo:fills")

        except Exception as e:
            logger.error(f"Error publishing to FIFO ledger: {e}")

    # Performance Monitoring

    async def _stats_publisher(self):
        """Publish bridge statistics to Redis."""
        try:
            while self._running:
                try:
                    # Update uptime
                    self.stats["bridge_uptime"] = time.time() - self._start_time

                    # Publish metrics
                    if ASYNC_REDIS_AVAILABLE:
                        for metric_name, value in self.stats.items():
                            await set_metric(f"nautilus_bridge_{metric_name}", value)

                    await asyncio.sleep(10)  # Update every 10 seconds

                except Exception as e:
                    logger.error(f"Error publishing bridge stats: {e}")
                    await asyncio.sleep(30)

        except Exception as e:
            logger.error(f"Fatal error in stats publisher: {e}")

    # Event Handler Registration

    def register_order_handler(self, handler: Callable):
        """Register a handler for order events."""
        self._event_handlers["order"].append(handler)

    def register_fill_handler(self, handler: Callable):
        """Register a handler for fill events."""
        self._event_handlers["fill"].append(handler)

    def register_quote_handler(self, handler: Callable):
        """Register a handler for quote events."""
        self._event_handlers["quote"].append(handler)

    def register_trade_handler(self, handler: Callable):
        """Register a handler for trade events."""
        self._event_handlers["trade"].append(handler)

    # Status and Diagnostics

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status and statistics."""
        return {
            "running": self._running,
            "config": asdict(self.config),
            "stats": self.stats.copy(),
            "nautilus_available": NAUTILUS_AVAILABLE,
            "redis_available": ASYNC_REDIS_AVAILABLE,
            "event_handlers": {
                event_type: len(handlers)
                for event_type, handlers in self._event_handlers.items()
            },
        }


# Mock implementations for testing without Nautilus


class MockOrderEvent:
    """Mock order event for testing."""

    def __init__(self):
        self.client_order_id = "test_order_123"
        self.instrument_id = "BTCUSDT.BINANCE"
        self.order_side = type("", (), {"name": "BUY"})()
        self.order_status = type("", (), {"name": "FILLED"})()
        self.ts_event = int(time.time() * 1e9)
        self.quantity = 1.0
        self.price = 50000.0


class MockFillEvent:
    """Mock fill event for testing."""

    def __init__(self):
        self.client_order_id = "test_order_123"
        self.instrument_id = "BTCUSDT.BINANCE"
        self.order_side = type("", (), {"name": "BUY"})()
        self.last_qty = 1.0
        self.last_px = 50000.0
        self.ts_event = int(time.time() * 1e9)
        self.venue = "BINANCE"


async def main():
    """Test the Nautilus-Redis bridge."""
    import argparse

    parser = argparse.ArgumentParser(description="Nautilus-Redis Bridge")
    parser.add_argument("--test", action="store_true", help="Run test with mock events")
    parser.add_argument("--run", action="store_true", help="Start bridge")

    args = parser.parse_args()

    if args.test:
        # Test with mock events
        bridge = NautilusRedisBridge()

        # Test order event
        mock_order = MockOrderEvent()
        await bridge.handle_order_event(mock_order)

        # Test fill event
        mock_fill = MockFillEvent()
        await bridge.handle_fill_event(mock_fill)

        # Get status
        status = bridge.get_status()
        print("Bridge Status:")
        print(json.dumps(status, indent=2, default=str))

    elif args.run:
        # Start bridge
        bridge = NautilusRedisBridge()
        try:
            await bridge.start()
        except KeyboardInterrupt:
            await bridge.stop()

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
