#!/usr/bin/env python3
"""
Whale Stream Daemon
Continuously monitors whale transactions and pushes to Redis
"""

import sys
import os
import time
import logging

# Add the src directory to Python path
sys.path.append(
    os.path.join(
        os.path.dirname(__file__), "..", "src", "layers", "layer1_signal_generation"
    )
)

try:
    from whale_alert_client import push_events
except ImportError:
    print("❌ Could not import whale_alert_client")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        (
            logging.FileHandler("/tmp/whale_stream.log")
            if os.path.exists("/tmp")
            else logging.StreamHandler()
        ),
    ],
)

logger = logging.getLogger(__name__)


def main():
    """Main daemon loop"""
    logger.info("🐋 Starting Whale Stream Daemon")
    logger.info("   Monitoring whale transactions every 30 seconds")
    logger.info("   Press Ctrl+C to stop")

    error_count = 0
    max_errors = 10

    while True:
        try:
            events_added = push_events()

            if events_added > 0:
                logger.info(f"📊 Added {events_added} new whale events")
            else:
                logger.debug("No new whale transactions")

            logger.info("Whale-Alert poll complete")
            error_count = 0  # Reset error count on success
            time.sleep(30)

        except KeyboardInterrupt:
            logger.info("🛑 Whale stream daemon stopped by user")
            break

        except Exception as e:
            error_count += 1
            logger.error(f"❌ Whale stream error ({error_count}/{max_errors}): {e}")

            if error_count >= max_errors:
                logger.critical(
                    f"💥 Too many errors ({max_errors}), shutting down daemon"
                )
                sys.exit(1)

            # Exponential backoff on errors
            sleep_time = min(300, 30 * (2 ** min(error_count, 4)))
            logger.info(f"⏳ Waiting {sleep_time}s before retry...")
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
