#!/usr/bin/env python3
"""
Test NOWNodes WebSocket Connection
"""

import asyncio
import websockets
import ssl
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_websocket():
    """Test WebSocket connection to NOWNodes."""
    api_key = "dabc07ca-7694-4d01-af08-c9114313fa0c"

    # Try different URL formats
    urls = [
        f"wss://btc.blockbook.ws.nownodes.io/?api_key={api_key}",
        f"wss://btc.blockbook.ws.nownodes.io?api_key={api_key}",
        f"wss://btc.ws.nownodes.io?api_key={api_key}",
        f"wss://btc.nownodes.io?api_key={api_key}",
    ]

    for url in urls:
        try:
            logger.info(f"Testing: {url}")

            # Test with SSL context
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    logger.info(f"✅ SUCCESS: {url}")

                    # Try to receive a message
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=5)
                        logger.info(f"Received: {message[:100]}...")
                    except asyncio.TimeoutError:
                        logger.info("No immediate message received")

                    return  # Success, exit

            except Exception as e:
                logger.error(f"Failed: {url} - {e}")

        except Exception as e:
            logger.error(f"Failed: {url} - {e}")

    logger.error("All WebSocket URLs failed")


if __name__ == "__main__":
    asyncio.run(test_websocket())
