#!/usr/bin/env python3
"""
Memory Allocation Probe for Trading System

Snapshots memory usage every 10 minutes, keeps last 3 snapshots,
and diff-prints top 20 allocations to /var/log/mem_snap.log.

Usage: python scripts/mem_probe.py [--background]
"""

import sys
import os
import time
import tracemalloc
import datetime
import pickle
import gc
import threading
import signal
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class MemoryProbe:
    """Memory allocation tracker and leak detector."""

    def __init__(self, snapshot_interval: int = 600, max_snapshots: int = 3):
        """Initialize memory probe."""
        self.snapshot_interval = snapshot_interval  # 10 minutes
        self.max_snapshots = max_snapshots
        self.snapshots_dir = "/tmp/mem_snapshots"
        self.log_file = "logs/mem_snap.log"  # Use writable logs directory
        self.running = False

        # Create directories
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # Setup logging to file
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(file_handler)

        # Start tracemalloc
        tracemalloc.start()
        logger.info("🔍 Memory probe initialized - tracking allocations")

    def take_snapshot(self) -> str:
        """Take a memory snapshot and save it."""
        timestamp = datetime.datetime.now()
        snapshot = tracemalloc.take_snapshot()

        # Generate filename
        filename = f"mem_snapshot_{timestamp.strftime('%Y%m%d_%H%M%S')}.pkl"
        filepath = os.path.join(self.snapshots_dir, filename)

        # Save snapshot
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "timestamp": timestamp,
                    "snapshot": snapshot,
                    "total_memory_mb": self._get_total_memory_mb(),
                },
                f,
            )

        logger.info(f"📸 Snapshot saved: {filename}")

        # Clean old snapshots
        self._cleanup_old_snapshots()

        return filepath

    def _get_total_memory_mb(self) -> float:
        """Get total memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback if psutil not available
            return 0.0

    def _cleanup_old_snapshots(self):
        """Keep only the last N snapshots."""
        snapshots = []
        for filename in os.listdir(self.snapshots_dir):
            if filename.startswith("mem_snapshot_") and filename.endswith(".pkl"):
                filepath = os.path.join(self.snapshots_dir, filename)
                snapshots.append((os.path.getctime(filepath), filepath))

        # Sort by creation time and keep only max_snapshots
        snapshots.sort()
        while len(snapshots) > self.max_snapshots:
            _, old_file = snapshots.pop(0)
            try:
                os.remove(old_file)
                logger.info(f"🗑️  Removed old snapshot: {os.path.basename(old_file)}")
            except OSError:
                pass

    def compare_snapshots(self) -> None:
        """Compare the last two snapshots and log differences."""
        snapshots = []
        for filename in os.listdir(self.snapshots_dir):
            if filename.startswith("mem_snapshot_") and filename.endswith(".pkl"):
                filepath = os.path.join(self.snapshots_dir, filename)
                snapshots.append((os.path.getctime(filepath), filepath))

        snapshots.sort()

        if len(snapshots) < 2:
            logger.info("📊 Need at least 2 snapshots to compare")
            return

        # Load the last two snapshots
        try:
            with open(snapshots[-2][1], "rb") as f:
                old_data = pickle.load(f)
            with open(snapshots[-1][1], "rb") as f:
                new_data = pickle.load(f)

            old_snapshot = old_data["snapshot"]
            new_snapshot = new_data["snapshot"]
            old_memory = old_data["total_memory_mb"]
            new_memory = new_data["total_memory_mb"]

            # Calculate memory growth
            memory_growth_mb = new_memory - old_memory
            memory_growth_pct = (
                (memory_growth_mb / old_memory * 100) if old_memory > 0 else 0
            )

            logger.info(f"📈 Memory Growth Analysis:")
            logger.info(f"   Previous: {old_memory:.1f}MB")
            logger.info(f"   Current:  {new_memory:.1f}MB")
            logger.info(
                f"   Growth:   {memory_growth_mb:+.1f}MB ({memory_growth_pct:+.1f}%)"
            )

            # Compare snapshots
            top_stats = new_snapshot.compare_to(old_snapshot, "lineno")

            logger.info("🔍 Top 20 Memory Allocation Changes:")
            logger.info("=" * 80)

            for index, stat in enumerate(top_stats[:20]):
                frame = stat.traceback.format()[-1]
                logger.info(
                    f"{index+1:2d}. {stat.size_diff/1024:.1f} KB | {frame.strip()}"
                )

                # Check for potential culprits
                if "redis" in frame.lower():
                    logger.warning(f"    ⚠️  Redis-related allocation detected")
                elif "pandas" in frame.lower() or "dataframe" in frame.lower():
                    logger.warning(f"    ⚠️  Pandas DataFrame leak potential")
                elif "async" in frame.lower() or "generator" in frame.lower():
                    logger.warning(f"    ⚠️  Async generator leak potential")

            logger.info("=" * 80)

            # Alert if memory growth is concerning
            if memory_growth_pct > 5.0:
                logger.error(
                    f"🚨 HIGH MEMORY GROWTH: {memory_growth_pct:.1f}% - Investigation required!"
                )
            elif memory_growth_pct > 3.0:
                logger.warning(f"⚠️  Elevated memory growth: {memory_growth_pct:.1f}%")

        except Exception as e:
            logger.error(f"❌ Failed to compare snapshots: {e}")

    def run_background(self):
        """Run memory probe in background mode."""
        logger.info(
            f"🚀 Starting background memory probe (interval: {self.snapshot_interval}s)"
        )
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        try:
            while self.running:
                # Force garbage collection before snapshot
                gc.collect()

                # Take snapshot
                self.take_snapshot()

                # Compare with previous if available
                self.compare_snapshots()

                # Wait for next interval
                time.sleep(self.snapshot_interval)

        except KeyboardInterrupt:
            logger.info("🛑 Memory probe stopped by user")
        except Exception as e:
            logger.error(f"❌ Memory probe error: {e}")
        finally:
            self.running = False
            logger.info("📊 Memory probe stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"🛑 Received signal {signum}, stopping memory probe...")
        self.running = False

    def run_single_snapshot(self):
        """Take a single snapshot and compare."""
        logger.info("📸 Taking single memory snapshot...")
        self.take_snapshot()
        self.compare_snapshots()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Memory allocation probe for trading system"
    )
    parser.add_argument(
        "--background", action="store_true", help="Run in background mode"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=600,
        help="Snapshot interval in seconds (default: 600)",
    )

    args = parser.parse_args()

    probe = MemoryProbe(snapshot_interval=args.interval)

    if args.background:
        probe.run_background()
    else:
        probe.run_single_snapshot()


if __name__ == "__main__":
    main()
