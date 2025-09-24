#!/usr/bin/env python3
"""
News Stream Daemon

Background service to continuously stream crypto news to Redis following 
Future_instruction.txt specifications for news integration.

Runs as a daemon to:
- Poll CryptoPanic every 60s 
- Stream new articles to Redis news:cryptopanic
- De-duplicate by URL hash
- Maintain rolling window of news data
"""

import os
import sys
import time
import logging
import signal
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from layers.layer1_signal_generation.cryptopanic_client import CryptoPanicClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/news_daemon.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class NewsStreamDaemon:
    """Background daemon for streaming crypto news to Redis."""

    def __init__(self):
        """Initialize the news stream daemon."""
        self.running = False
        self.cryptopanic_client = None
        self.poll_interval = 60  # seconds

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def initialize(self):
        """Initialize the CryptoPanic client."""
        try:
            self.cryptopanic_client = CryptoPanicClient()
            logger.info("✅ CryptoPanic client initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize CryptoPanic client: {e}")
            return False

    def stream_news_batch(self):
        """Stream a batch of news to Redis."""
        try:
            # Stream news for BTC and ETH
            new_items = self.cryptopanic_client.stream_to_redis(
                currencies="BTC,ETH,ADA,DOT,SOL,MATIC,LINK,UNI",
                stream_key="news:cryptopanic",
                max_items=1000,
            )

            if new_items > 0:
                logger.info(f"📰 Streamed {new_items} new news items to Redis")
            else:
                logger.debug("📰 No new news items found")

            return new_items

        except Exception as e:
            logger.error(f"❌ Error streaming news batch: {e}")
            return 0

    def health_check(self):
        """Perform health checks on the daemon."""
        try:
            # Check Redis connectivity
            if self.cryptopanic_client and self.cryptopanic_client.redis_client:
                self.cryptopanic_client.redis_client.ping()
                logger.debug("✅ Redis health check passed")

            # Check API key availability
            if not os.getenv("CryptoPanic_API"):
                raise ValueError("CryptoPanic API key not available")

            logger.debug("✅ Health check passed")
            return True

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False

    def run(self):
        """Main daemon loop."""
        logger.info("🚀 Starting News Stream Daemon")

        if not self.initialize():
            logger.error("❌ Failed to initialize daemon")
            sys.exit(1)

        self.running = True
        logger.info(f"📡 Daemon started, polling every {self.poll_interval} seconds")

        # Initial news fetch to populate Redis
        logger.info("📰 Performing initial news fetch...")
        self.stream_news_batch()

        last_health_check = time.time()
        health_check_interval = 300  # 5 minutes

        while self.running:
            try:
                start_time = time.time()

                # Stream news batch
                self.stream_news_batch()

                # Periodic health checks
                if time.time() - last_health_check > health_check_interval:
                    self.health_check()
                    last_health_check = time.time()

                # Calculate sleep time to maintain consistent interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.poll_interval - elapsed)

                if sleep_time > 0:
                    logger.debug(f"⏱️ Sleeping for {sleep_time:.1f} seconds")
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("🛑 Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"❌ Unexpected error in main loop: {e}")
                # Sleep longer on errors to avoid spam
                time.sleep(min(self.poll_interval * 2, 300))

        logger.info("👋 News Stream Daemon stopped")

    def status(self):
        """Get daemon status information."""
        try:
            if not self.cryptopanic_client:
                return {"status": "not_initialized"}

            # Get news count from Redis
            news_count = self.cryptopanic_client.redis_client.llen(
                "news:cryptopanic:latest"
            )

            # Get latest sentiment summary
            sentiment = self.cryptopanic_client.get_sentiment_summary()

            return {
                "status": "running" if self.running else "stopped",
                "news_in_cache": news_count,
                "last_update": datetime.now().isoformat(),
                "sentiment_summary": sentiment,
                "poll_interval": self.poll_interval,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        daemon = NewsStreamDaemon()

        if command == "start":
            daemon.run()
        elif command == "status":
            if daemon.initialize():
                status = daemon.status()
                print("📊 News Stream Daemon Status:")
                print(f"  Status: {status.get('status', 'unknown')}")
                print(f"  News in cache: {status.get('news_in_cache', 0)}")
                print(f"  Poll interval: {status.get('poll_interval', 'unknown')}s")
                if status.get("sentiment_summary"):
                    s = status["sentiment_summary"]
                    print(
                        f"  Sentiment: {s.get('bullish_pct', 0)}% bullish, {s.get('bearish_pct', 0)}% bearish"
                    )
            else:
                print("❌ Failed to initialize daemon for status check")
        elif command == "test":
            if daemon.initialize():
                print("🧪 Testing news fetch...")
                items = daemon.stream_news_batch()
                print(f"✅ Successfully fetched {items} news items")
            else:
                print("❌ Failed to initialize daemon for testing")
        else:
            print("Usage: python news_stream_daemon.py {start|status|test}")
            sys.exit(1)
    else:
        print("Usage: python news_stream_daemon.py {start|status|test}")
        sys.exit(1)


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    main()
