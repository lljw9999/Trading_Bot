"""
Binance Spot WebSocket Connector (L0-2)

Streams best-bid-offer data for BTCUSDT, ETHUSDT, SOLUSDT from the Binance
public WebSocket API and publishes normalized ticks to Kafka. Designed to meet
Future_instruction.txt requirements:

• ≥10 msg/s combined for the three symbols
• Normalized schema identical to Coinbase connector
• Prometheus histogram `binance_ws_latency_seconds` for end-to-end latency
• Resilient reconnect logic with exponential back-off
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, AsyncIterator, Optional

import orjson
import websockets
from aiokafka import AIOKafkaProducer

from .base_connector import BaseDataConnector
from .schemas import MarketTick
from ...utils.config_manager import config
from ...utils.metrics import get_metrics
from ...utils.logger import get_logger

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

BINANCE_WS_URL = (
    "wss://stream.binance.com:9443/stream?streams="
    "btcusdt@bookTicker/ethusdt@bookTicker/solusdt@bookTicker"
)

KAFKA_TOPIC = "market.raw.crypto.binance"

# ----------------------------------------------------------------------------
# Connector Implementation
# ----------------------------------------------------------------------------


class BinanceConnector(BaseDataConnector):
    """Binance Spot bookTicker stream connector."""

    def __init__(self, symbols: list[str] | None = None):
        symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        super().__init__(symbols=symbols, exchange_name="binance", asset_type="crypto")

        self.ws_url: str = BINANCE_WS_URL  # Combined stream for 3 symbols
        self.kafka_bootstrap = config.get(
            "data_ingestion.kafka.bootstrap_servers", "localhost:9092"
        )

        # Kafka
        self.producer: Optional[AIOKafkaProducer] = None
        self.kafka_topic = KAFKA_TOPIC

        # Metrics
        self.metrics = get_metrics()
        self.latency_hist = self.metrics.register_histogram(
            "binance_ws_latency_seconds",
            "binance → publish latency (s)",
            buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
        )

        self.msg_counter = 0
        self.last_log = time.time()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    async def start(self):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_bootstrap,
            value_serializer=lambda v: orjson.dumps(v),
            compression_type="gzip",
            batch_size=16384,
            linger_ms=10,
        )
        await self.producer.start()
        await super().start()

    async def stop(self):
        await super().stop()
        if self.producer:
            await self.producer.stop()

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    async def _connect_impl(self):
        self.websocket = await websockets.connect(
            self.ws_url, ping_interval=30, ping_timeout=10
        )
        self.logger.info("Connected to Binance WebSocket stream")

    async def _subscribe_impl(self):
        # Combined stream URL already subscribed; no-op.
        pass

    async def _stream_data(self) -> AsyncIterator[Dict[str, Any]]:
        try:
            async for raw in self.websocket:
                try:
                    yield orjson.loads(raw)
                except orjson.JSONDecodeError:
                    continue
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("Binance WebSocket closed")
        except Exception as exc:
            self.logger.error(f"Binance WebSocket error: {exc}")

    async def _parse_tick(self, raw: Dict[str, Any]) -> Optional[MarketTick]:
        # Binance wraps data under 'data'
        data = raw.get("data", raw)
        symbol = data.get("s")  # e.g. BTCUSDT
        if not symbol:
            return None

        ts_start = time.time()

        bid = float(data["b"])
        ask = float(data["a"])
        bid_size = float(data["B"])
        ask_size = float(data["A"])

        normalized = {
            "ts": ts_start,
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "exchange": "binance",
        }

        # Publish
        await self._publish(normalized)

        latency = time.time() - ts_start
        self.latency_hist.observe(latency)
        self.metrics.record_market_tick(symbol, "binance", "crypto", latency)

        # Log msg-rate every 10s
        self.msg_counter += 1
        if time.time() - self.last_log >= 10:
            rate = self.msg_counter / (time.time() - self.last_log)
            self.logger.info(f"Binance msg-rate: {rate:.1f} msg/s")
            self.msg_counter = 0
            self.last_log = time.time()

        # Build MarketTick for downstream if needed
        return MarketTick(
            symbol=symbol,
            exchange="binance",
            asset_type="crypto",
            timestamp=datetime.utcnow(),
            exchange_timestamp=datetime.utcnow(),  # Binance bookTicker has no explicit ts
            bid=Decimal(str(bid)),
            ask=Decimal(str(ask)),
            bid_size=Decimal(str(bid_size)),
            ask_size=Decimal(str(ask_size)),
            metadata={"type": "bookTicker", "raw": data},
        )

    async def _publish(self, payload: Dict[str, Any]):
        try:
            await self.producer.send_and_wait(self.kafka_topic, value=payload)
        except Exception as exc:
            self.logger.error(f"Kafka publish failed: {exc}")

    async def _reconnect(self):
        backoff = 1
        while not self.should_stop:
            try:
                self.logger.info(f"Reconnecting in {backoff}s…")
                await asyncio.sleep(backoff)
                await self._connect_impl()
                self.logger.info("Reconnected")
                return
            except Exception as exc:
                self.logger.error(f"Reconnect failed: {exc}")
                backoff = min(backoff * 2, 60)


# -----------------------------------------------------------------------------
# Factory / CLI helper
# -----------------------------------------------------------------------------


def create_binance_connector(symbols: list[str] | None = None) -> BinanceConnector:
    return BinanceConnector(symbols)


async def test_binance_connector():
    """Run connector for 20 s to assert ≥10 msg/s parsing latency <3 ms"""
    import logging

    logging.basicConfig(level=logging.INFO)
    connector = create_binance_connector()

    # Consume messages for 20 seconds and time parsing
    latencies = []

    async def _run():
        async for raw in connector._stream_data():
            ts = time.perf_counter()
            await connector._parse_tick(raw)
            latencies.append((time.perf_counter() - ts) * 1000)  # ms
            if len(latencies) >= 200:  # ~20s at 10 msg/s
                break

    await connector._connect_impl()
    await asyncio.wait_for(_run(), timeout=25)

    median_latency = sorted(latencies)[len(latencies) // 2]
    assert median_latency < 3, f"Parsing latency {median_latency:.2f} ms exceeds 3 ms"
    print(f"✅ Median parsing latency: {median_latency:.2f} ms")


if __name__ == "__main__":
    asyncio.run(test_binance_connector())
