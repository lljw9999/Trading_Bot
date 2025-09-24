import json
import asyncio
import logging
import redis
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

URI_TMPL = "wss://stream.binance.com/ws/{symbol}@trade"

# Setup logging
log = logging.getLogger("connector.binance")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Initialize Redis connection
try:
    REDIS = redis.Redis(host="localhost", port=6379, decode_responses=True)
    REDIS.ping()
    log.info("✅ Redis connected")
except Exception as e:
    log.warning(f"⚠️  Redis not available: {e} - will log only")
    REDIS = None


class BinanceWS:
    def __init__(self, symbols):
        self._uris = [URI_TMPL.format(symbol=s.lower()) for s in symbols]
        self.symbols = [s.upper() for s in symbols]

    async def _on_message(self, msg: dict, symbol: str):
        tick = {
            "ts": msg["T"],
            "price": float(msg["p"]),
            "qty": float(msg["q"]),
        }
        if REDIS:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: REDIS.rpush(
                    f"market.raw.crypto.{symbol.upper()}", json.dumps(tick)
                ),
            )
        log.info(f"Tick: {symbol} {tick}")

    async def _connect_and_listen(self, uri, symbol):
        reconnect_delay = 5
        while True:
            try:
                async with websockets.connect(
                    uri, ping_interval=30, ping_timeout=10
                ) as ws:
                    log.info(f"Connected to {uri}")
                    async for message in ws:
                        msg = json.loads(message)
                        await self._on_message(msg, symbol)
            except (ConnectionClosed, WebSocketException, Exception) as e:
                log.warning(
                    f"WebSocket error for {symbol}: {e}. Reconnecting in {reconnect_delay}s..."
                )
                await asyncio.sleep(reconnect_delay)

    async def run(self):
        tasks = [
            self._connect_and_listen(uri, symbol)
            for uri, symbol in zip(self._uris, self.symbols)
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols", type=str, default="BTCUSDT,ETHUSDT", help="Comma-separated symbols"
    )
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    ws = BinanceWS(symbols)
    asyncio.run(ws.run())
