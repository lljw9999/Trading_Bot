#!/usr/bin/env python3
"""
Test Coinbase WebSocket Connection
"""

import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_coinbase_websocket():
    """Test WebSocket connection to Coinbase."""

    # Try different URLs
    urls = [
        "wss://ws-feed.pro.coinbase.com",  # Current production URL
        "wss://ws-feed.exchange.coinbase.com",  # Alternative URL
        "wss://advanced-trade-ws.coinbase.com",  # New Advanced Trade API
    ]

    for url in urls:
        try:
            logger.info(f"Testing: {url}")

            async with websockets.connect(url, ping_interval=30) as ws:
                logger.info(f"✅ SUCCESS: Connected to {url}")

                # Send subscription message
                subscribe_message = {
                    "type": "subscribe",
                    "product_ids": ["BTC-USD"],
                    "channels": ["ticker"],
                }

                await ws.send(json.dumps(subscribe_message))
                logger.info("Subscription message sent")

                # Try to receive a few messages
                for i in range(3):
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=10)
                        data = json.loads(message)
                        logger.info(
                            f"Message {i+1}: {data.get('type', 'unknown')} - {data.get('product_id', 'N/A')}"
                        )

                        if data.get("type") == "ticker":
                            logger.info(
                                f"✅ Ticker data received: {data.get('price', 'N/A')}"
                            )
                            return True

                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout waiting for message {i+1}")

        except Exception as e:
            logger.error(f"Failed: {url} - {e}")
            continue

    logger.error("All WebSocket URLs failed")
    return False


if __name__ == "__main__":
    success = asyncio.run(test_coinbase_websocket())
    if success:
        print("✅ Coinbase WebSocket test PASSED")
    else:
        print("❌ Coinbase WebSocket test FAILED")
