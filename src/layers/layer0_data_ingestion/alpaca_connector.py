"""
Alpaca Stocks WebSocket Connector (L0-3)

• Streams real-time trade messages for AAPL, TSLA, MSFT from Alpaca's IEX SIP
  WebSocket endpoint (free tier).
• Normalizes each trade into Tick schema (fills bid/ask as last ±0.005).
• Publishes to Kafka topic `market.raw.stocks.alpaca` with same schema as
  crypto connectors.
• ≥5 msg/s combined target.
• Prometheus histogram `alpaca_ws_latency_seconds` for end-to-end latency.
"""

from __future__ import annotations

import os
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

ALPACA_WS_URL = "wss://stream.data.alpaca.markets/v2/sip"
KAFKA_TOPIC = "market.raw.stocks.alpaca"
SYMBOLS = ["AAPL", "TSLA", "MSFT"]


class AlpacaConnector(BaseDataConnector):
    """Real-time Alpaca trade connector."""

    def __init__(self, symbols: list[str] | None = None):
        symbols = symbols or SYMBOLS
        super().__init__(symbols=symbols, exchange_name="alpaca", asset_type="stock")

        self.ws_url = ALPACA_WS_URL
        self.kafka_bootstrap = config.get(
            "data_ingestion.kafka.bootstrap_servers", "localhost:9092"
        )

        self.producer: Optional[AIOKafkaProducer] = None
        self.kafka_topic = KAFKA_TOPIC

        self.metrics = get_metrics()
        self.latency_hist = self.metrics.register_histogram(
            "alpaca_ws_latency_seconds",
            "alpaca → publish latency (s)",
            buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05),
        )

        self.msg_counter = 0
        self.last_log = time.time()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_bootstrap,
            value_serializer=lambda v: orjson.dumps(v),
            compression_type="gzip",
            batch_size=16384,
            linger_ms=5,
        )
        await self.producer.start()
        await super().start()

    async def stop(self):
        await super().stop()
        if self.producer:
            await self.producer.stop()

    # ------------------------------------------------------------------
    # WebSocket helpers
    # ------------------------------------------------------------------

    async def _connect_impl(self):
        self.websocket = await websockets.connect(
            self.ws_url, ping_interval=30, ping_timeout=10
        )
        self.logger.info("Connected to Alpaca WS")
        await self._authenticate()

    async def _authenticate(self):
        key = os.getenv("ALPACA_KEY_ID", config.get("alpaca.key_id", "demo"))
        secret = os.getenv("ALPACA_SECRET_KEY", config.get("alpaca.secret_key", "demo"))
        auth_msg = {"action": "auth", "key": key, "secret": secret}
        await self.websocket.send(orjson.dumps(auth_msg).decode())
        resp = await self.websocket.recv()
        if b"authorized" in resp.encode() or "authorized" in resp:
            self.logger.info("Alpaca auth successful")
        else:
            self.logger.error(f"Alpaca auth failed: {resp}")
            raise RuntimeError("Alpaca auth failed")
        sub_msg = {
            "action": "subscribe",
            "trades": self.symbols,
        }
        await self.websocket.send(orjson.dumps(sub_msg).decode())
        self.logger.info(f"Subscribed to trades: {self.symbols}")

    async def _subscribe_impl(self):
        # Done during auth
        pass

    async def _stream_data(self) -> AsyncIterator[Dict[str, Any]]:
        try:
            async for raw in self.websocket:
                try:
                    yield orjson.loads(raw)
                except orjson.JSONDecodeError:
                    continue
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("Alpaca WS closed")
        except Exception as exc:
            self.logger.error(f"Alpaca WS error: {exc}")

    # ------------------------------------------------------------------
    async def _parse_tick(self, raw: Dict[str, Any]):
        # Alpaca sends list of messages
        if isinstance(raw, list):
            for msg in raw:
                await self._handle_trade(msg)
        else:
            await self._handle_trade(raw)
        return None  # We publish internally

    async def _handle_trade(self, msg: Dict[str, Any]):
        if msg.get("T") != "t":  # trade message
            return
        symbol = msg["S"]
        price = float(msg["p"])
        size = float(msg["s"])

        ts_start = time.time()

        bid = price - 0.00005  # minus half cent
        ask = price + 0.00005

        payload = {
            "ts": ts_start,
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "bid_size": size,
            "ask_size": size,
            "exchange": "alpaca",
        }
        await self._publish(payload)
        latency = time.time() - ts_start
        self.latency_hist.observe(latency)
        self.metrics.record_market_tick(symbol, "alpaca", "stock", latency)

        # msg-rate log
        self.msg_counter += 1
        if time.time() - self.last_log >= 10:
            rate = self.msg_counter / (time.time() - self.last_log)
            self.logger.info(f"Alpaca msg-rate: {rate:.1f} msg/s")
            self.msg_counter = 0
            self.last_log = time.time()

        # Build MarketTick for downstream if needed
        return MarketTick(
            symbol=symbol,
            exchange="alpaca",
            asset_type="stock",
            timestamp=datetime.utcnow(),
            exchange_timestamp=datetime.fromtimestamp(msg["t"] / 1e9),
            bid=Decimal(str(bid)),
            ask=Decimal(str(ask)),
            bid_size=Decimal(str(size)),
            ask_size=Decimal(str(size)),
            last=Decimal(str(price)),
            metadata=msg,
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
                self.logger.info("Reconnected Alpaca")
                return
            except Exception as exc:
                self.logger.error(f"Reconnect failed: {exc}")
                backoff = min(backoff * 2, 60)


# ------------------------------------------------------------------


def create_alpaca_connector(symbols: list[str] | None = None) -> AlpacaConnector:
    return AlpacaConnector(symbols)


async def test_alpaca_connector():
    """Parse sample messages and assert <3 ms latency."""
    sample = [
        {
            "T": "t",
            "S": "AAPL",
            "i": 123,
            "x": "V",
            "p": 180.25,
            "s": 100,
            "t": int(time.time() * 1e9),
        }
    ]
    connector = create_alpaca_connector(["AAPL"])
    ts = time.perf_counter()
    await connector._handle_trade(sample[0])
    latency_ms = (time.perf_counter() - ts) * 1000
    assert latency_ms < 3, f"Latency {latency_ms:.2f} ms exceeds 3 ms"
    print(f"✅ Parse latency {latency_ms:.2f} ms")


if __name__ == "__main__":
    asyncio.run(test_alpaca_connector())
