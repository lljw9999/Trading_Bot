"""
Options Flow Signal Generator - Unusual Whales API Integration
Captures institutional options sweeps for forward-looking alpha
"""

import asyncio
import websockets
import aiohttp
import json
import redis
import time
import hashlib
from typing import Dict, Any, Optional
import logging


class UnusualWhalesClient:
    """
    Async WebSocket client for Unusual Whales options flow data.
    Captures large institutional sweeps for alpha generation.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "DEMO_KEY"
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.logger = logging.getLogger("options_flow")

        # Demo mode if no real API key
        self.demo_mode = self.api_key == "DEMO_KEY"

        if self.demo_mode:
            self.logger.info(
                "🔸 Running in demo mode - generating synthetic options flow"
            )

    async def connect_and_stream(self):
        """Main connection and streaming loop."""
        if self.demo_mode:
            await self._demo_stream()
        else:
            await self._real_stream()

    async def _real_stream(self):
        """Connect to real Unusual Whales WebSocket API."""
        ws_url = f"wss://api.unusualwhales.com/ws?token={self.api_key}"

        try:
            async with websockets.connect(ws_url) as websocket:
                self.logger.info("🐋 Connected to Unusual Whales options feed")

                # Subscribe to relevant streams
                await websocket.send(
                    json.dumps(
                        {
                            "action": "subscribe",
                            "symbols": ["BTC", "ETH", "BTCUSDT", "ETHUSDT"],
                            "types": ["sweep", "block", "unusual_activity"],
                        }
                    )
                )

                async for message in websocket:
                    data = json.loads(message)
                    await self._process_options_event(data)

        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
            await asyncio.sleep(5)  # Retry delay

    async def _demo_stream(self):
        """Generate demo options flow data for testing."""
        import random

        while True:
            # Generate synthetic options sweep
            event = {
                "trade_id": f"demo_{int(time.time())}_{random.randint(1000, 9999)}",
                "timestamp": int(time.time()),
                "symbol": random.choice(["BTC", "ETH"]),
                "option_type": random.choice(["call", "put"]),
                "premium_paid": random.randint(100000, 10000000),  # $100K - $10M
                "strike": (
                    random.randint(40000, 80000)
                    if random.choice([True, False])
                    else random.randint(2000, 5000)
                ),
                "expiry": "2025-12-25",
                "volume": random.randint(100, 5000),
                "unusual_activity": random.choice([True, False]),
                "sweep_type": random.choice(["aggressive", "passive", "block"]),
            }

            await self._process_options_event(event)
            await asyncio.sleep(random.uniform(5, 30))  # Random intervals

    async def _process_options_event(self, event: Dict[str, Any]):
        """Process and store options flow event."""
        try:
            # Deduplication by trade_id as specified in task brief
            trade_id = event.get("trade_id", "")
            trade_hash = hashlib.sha1(trade_id.encode()).hexdigest()

            if self.redis.sismember("options:seen", trade_hash):
                return  # Already processed

            # Mark as seen
            self.redis.sadd("options:seen", trade_hash)

            # Extract key metrics
            symbol = event.get("symbol", "").upper()
            option_type = event.get("option_type", "").lower()
            premium = float(event.get("premium_paid", 0))
            unusual = event.get("unusual_activity", False)

            # Create event payload
            payload = {
                "ts": int(time.time()),
                "symbol": symbol,
                "option_type": option_type,
                "premium_usd": premium,
                "unusual": str(int(unusual)),
                "sweep_type": event.get("sweep_type", ""),
                "volume": event.get("volume", 0),
            }

            # Push to Redis stream event:options as specified
            self.redis.xadd("event:options", payload, maxlen=10000)

            # Update aggregated metrics for state builder
            await self._update_sweep_aggregates(symbol, option_type, premium)

            # Log significant events
            if premium > 1000000 or unusual:  # $1M+ or unusual activity
                self.logger.info(
                    f"🎯 Large options sweep: {symbol} {option_type.upper()} ${premium:,.0f}"
                )

        except Exception as e:
            self.logger.error(f"Error processing options event: {e}")

    async def _update_sweep_aggregates(
        self, symbol: str, option_type: str, premium: float
    ):
        """Update aggregated sweep metrics for state builder access."""
        try:
            # Get current aggregates
            call_usd = float(self.redis.hget("optsweep", "call_usd") or 0)
            put_usd = float(self.redis.hget("optsweep", "put_usd") or 0)

            # Decay factor (5-minute half-life for real-time signals)
            decay_factor = 0.95
            call_usd *= decay_factor
            put_usd *= decay_factor

            # Add new sweep
            if option_type == "call":
                call_usd += premium
            elif option_type == "put":
                put_usd += premium

            # Update Redis hash for state builder access
            self.redis.hset(
                "optsweep",
                mapping={
                    "call_usd": call_usd,
                    "put_usd": put_usd,
                    "last_update": int(time.time()),
                    "call_put_ratio": call_usd
                    / max(put_usd, 1),  # Avoid division by zero
                    "net_flow": call_usd - put_usd,
                },
            )

        except Exception as e:
            self.logger.error(f"Error updating sweep aggregates: {e}")

    def get_current_flow(self) -> Dict[str, float]:
        """Get current options flow metrics for analysis."""
        try:
            flow_data = self.redis.hgetall("optsweep")

            return {
                "call_sweep_usd": float(flow_data.get("call_usd", 0)),
                "put_sweep_usd": float(flow_data.get("put_usd", 0)),
                "call_put_ratio": float(flow_data.get("call_put_ratio", 1.0)),
                "net_flow": float(flow_data.get("net_flow", 0)),
                "last_update": int(flow_data.get("last_update", 0)),
            }

        except Exception as e:
            self.logger.error(f"Error getting current flow: {e}")
            return {
                "call_sweep_usd": 0,
                "put_sweep_usd": 0,
                "call_put_ratio": 1.0,
                "net_flow": 0,
                "last_update": 0,
            }


async def run_options_flow_daemon():
    """Main daemon function for options flow ingestion."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger("options_daemon")

    logger.info("🚀 Starting Unusual Whales options flow daemon")

    # Initialize client
    import os

    api_key = os.getenv("UNUSUAL_WHALES_API_KEY", "DEMO_KEY")
    client = UnusualWhalesClient(api_key)

    # Run with automatic reconnection
    retry_count = 0
    max_retries = 10

    while retry_count < max_retries:
        try:
            await client.connect_and_stream()

        except Exception as e:
            retry_count += 1
            wait_time = min(60, 5 * retry_count)  # Exponential backoff, max 60s

            logger.error(f"Connection failed ({retry_count}/{max_retries}): {e}")
            logger.info(f"Retrying in {wait_time} seconds...")

            await asyncio.sleep(wait_time)

    logger.critical("Maximum retries exceeded - daemon shutting down")


if __name__ == "__main__":
    # Run the daemon
    asyncio.run(run_options_flow_daemon())
