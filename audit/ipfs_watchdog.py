#!/usr/bin/env python3
"""
IPFS Watchdog - Pins last 100 CIDs from Redis audit stream every 5 minutes
Ensures immutable audit trail persistence through IPFS pinning
"""

import time
import redis
import ipfshttpclient
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("ipfs_watchdog")


class IPFSWatchdog:
    """Monitors Redis audit streams and pins CIDs to IPFS for persistence."""

    def __init__(self):
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.ipfs = ipfshttpclient.connect()
        logger.info("🔍 IPFS Watchdog initialized")

    def get_recent_cids(self, limit: int = 100) -> List[str]:
        """Get recent CIDs from audit streams."""
        try:
            # Get CIDs from audit:orders stream
            order_cids = []
            order_entries = self.redis.xrevrange("audit:orders", count=limit)

            for entry_id, fields in order_entries:
                if "cid" in fields:
                    order_cids.append(fields["cid"])

            logger.debug(f"Found {len(order_cids)} CIDs in audit:orders stream")
            return order_cids

        except Exception as e:
            logger.error(f"Error getting recent CIDs: {e}")
            return []

    def pin_cid(self, cid: str) -> bool:
        """Pin a single CID to IPFS."""
        try:
            # Check if already pinned
            try:
                pin_ls_result = self.ipfs.pin.ls(cid)
                if pin_ls_result:
                    logger.debug(f"CID {cid} already pinned")
                    return True
            except Exception:
                # CID not pinned, continue to pin it
                pass

            # Pin the CID
            result = self.ipfs.pin.add(cid)
            logger.info(f"📌 Pinned CID: {cid}")
            return True

        except Exception as e:
            logger.error(f"Failed to pin CID {cid}: {e}")
            return False

    def pin_recent_cids(self) -> Dict[str, int]:
        """Pin recent CIDs and return statistics."""
        cids = self.get_recent_cids(limit=100)

        if not cids:
            logger.info("No CIDs found to pin")
            return {"total": 0, "pinned": 0, "failed": 0}

        pinned_count = 0
        failed_count = 0

        for cid in cids:
            if self.pin_cid(cid):
                pinned_count += 1
            else:
                failed_count += 1

        stats = {"total": len(cids), "pinned": pinned_count, "failed": failed_count}

        logger.info(f"📊 Pinning stats: {stats}")
        return stats

    def health_check(self) -> bool:
        """Check if Redis and IPFS connections are healthy."""
        try:
            # Check Redis connection
            self.redis.ping()

            # Check IPFS connection
            self.ipfs.id()

            logger.debug("✅ Health check passed")
            return True

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False

    def run_once(self) -> bool:
        """Run one iteration of the watchdog."""
        try:
            if not self.health_check():
                return False

            stats = self.pin_recent_cids()

            # Store watchdog metrics in Redis
            timestamp = int(time.time())
            metrics = {
                "timestamp": timestamp,
                "total_cids": stats["total"],
                "pinned_cids": stats["pinned"],
                "failed_cids": stats["failed"],
                "success_rate": stats["pinned"] / max(stats["total"], 1),
            }

            self.redis.hset("ipfs:watchdog:metrics", mapping=metrics)

            return True

        except Exception as e:
            logger.error(f"Error in watchdog run: {e}")
            return False

    def run_forever(self, interval_seconds: int = 300):
        """Run watchdog continuously with specified interval."""
        logger.info(f"🚀 Starting IPFS Watchdog (interval: {interval_seconds}s)")

        while True:
            try:
                success = self.run_once()

                if success:
                    logger.info(f"💤 Sleeping for {interval_seconds} seconds")
                else:
                    logger.warning(
                        f"⚠️ Run failed, retrying in {interval_seconds} seconds"
                    )

                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("🛑 Watchdog stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(interval_seconds)


def main():
    """Main entry point."""
    watchdog = IPFSWatchdog()

    # Run once for testing, then continuously
    logger.info("🧪 Running initial test...")
    success = watchdog.run_once()

    if not success:
        logger.error("❌ Initial test failed - check Redis and IPFS connections")
        return 1

    logger.info("✅ Initial test successful")

    # Run continuously
    watchdog.run_forever(interval_seconds=300)  # 5 minutes as specified

    return 0


if __name__ == "__main__":
    exit(main())
