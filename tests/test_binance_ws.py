import asyncio
import json
import pytest

pytest.importorskip(
    "pytest_asyncio", reason="pytest-asyncio plugin required for async websocket tests"
)

from src.layers.layer0_data_ingestion.binance_ws import BinanceWS


class DummyRedis:
    def __init__(self):
        self.data = {}

    def rpush(self, key, value):
        self.data.setdefault(key, []).append(value)


@pytest.mark.asyncio
async def test_on_message(monkeypatch):
    symbols = ["BTCUSDT"]
    ws = BinanceWS(symbols)
    dummy_redis = DummyRedis()
    # Patch the REDIS global in the module
    monkeypatch.setattr(
        "src.layers.layer0_data_ingestion.binance_ws.REDIS", dummy_redis
    )
    msg = {"T": 1718357400000, "p": "66500.12", "q": "0.002"}
    await ws._on_message(msg, "BTCUSDT")
    key = "market.raw.crypto.BTCUSDT"
    assert key in dummy_redis.data
    tick = json.loads(dummy_redis.data[key][0])
    assert tick["ts"] == 1718357400000
    assert tick["price"] == 66500.12
    assert tick["qty"] == 0.002
