#!/usr/bin/env python3
"""
Memory Leak Simulation Script

Simulates memory allocations and releases for testing the memory probe:
1. Allocates 200 MB gradually
2. Holds memory for specified time
3. Releases memory to simulate recovery
4. Memory probe should detect spike then drop

Usage: python scripts/leak_sim.py --secs 30
"""

import argparse
import time
import logging
import gc
import psutil
import os
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryLeakSimulator:
    """Simulates memory allocation/deallocation patterns for testing."""

    def __init__(self):
        self.allocated_chunks: List[bytes] = []
        self.process = psutil.Process(os.getpid())

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def allocate_memory(self, total_mb: int = 200, chunk_size_mb: int = 10):
        """Allocate memory in chunks."""
        logger.info(
            f"📈 Starting memory allocation: {total_mb}MB in {chunk_size_mb}MB chunks"
        )

        start_memory = self.get_memory_usage_mb()
        logger.info(f"   💾 Initial memory: {start_memory:.1f}MB")

        num_chunks = total_mb // chunk_size_mb
        chunk_size_bytes = chunk_size_mb * 1024 * 1024

        for i in range(num_chunks):
            # Allocate chunk filled with random data
            chunk = b"X" * chunk_size_bytes
            self.allocated_chunks.append(chunk)

            current_memory = self.get_memory_usage_mb()
            logger.info(
                f"   📦 Allocated chunk {i+1}/{num_chunks}: {current_memory:.1f}MB"
            )
            time.sleep(0.5)  # Small delay to make allocation visible

        final_memory = self.get_memory_usage_mb()
        allocated_amount = final_memory - start_memory
        logger.info(
            f"🚨 Memory allocation complete: {final_memory:.1f}MB (+{allocated_amount:.1f}MB)"
        )

        return allocated_amount

    def hold_memory(self, duration_secs: int):
        """Hold allocated memory for specified duration."""
        logger.info(f"⏳ Holding memory for {duration_secs} seconds...")

        for i in range(duration_secs):
            current_memory = self.get_memory_usage_mb()
            logger.info(
                f"   🔒 Holding memory: {current_memory:.1f}MB ({i+1}/{duration_secs}s)"
            )
            time.sleep(1)

    def release_memory(self):
        """Release all allocated memory."""
        logger.info("📉 Starting memory release...")

        start_memory = self.get_memory_usage_mb()
        logger.info(f"   💾 Memory before release: {start_memory:.1f}MB")

        # Clear allocated chunks
        num_chunks = len(self.allocated_chunks)
        for i in range(num_chunks):
            self.allocated_chunks.pop()
            if (i + 1) % 5 == 0:  # Log every 5 chunks
                current_memory = self.get_memory_usage_mb()
                logger.info(
                    f"   🗑️ Released {i+1}/{num_chunks} chunks: {current_memory:.1f}MB"
                )

        # Force garbage collection
        gc.collect()

        final_memory = self.get_memory_usage_mb()
        released_amount = start_memory - final_memory
        logger.info(
            f"✅ Memory release complete: {final_memory:.1f}MB (-{released_amount:.1f}MB)"
        )

        return released_amount

    def simulate_spike_and_recovery(
        self, duration_secs: int = 30, allocation_mb: int = 200
    ):
        """Run complete simulation: allocate → hold → release."""
        logger.info("🎭 Starting memory leak simulation")
        logger.info(f"   📊 Target allocation: {allocation_mb}MB")
        logger.info(f"   ⏱️ Hold duration: {duration_secs}s")

        # Record initial state
        initial_memory = self.get_memory_usage_mb()

        try:
            # Phase 1: Allocate memory (simulate leak)
            allocated = self.allocate_memory(allocation_mb)

            # Phase 2: Hold memory (simulate sustained leak)
            self.hold_memory(duration_secs)

            # Phase 3: Release memory (simulate recovery)
            released = self.release_memory()

            # Summary
            final_memory = self.get_memory_usage_mb()
            net_change = final_memory - initial_memory

            logger.info("🎉 Simulation Summary")
            logger.info(f"   📊 Initial: {initial_memory:.1f}MB")
            logger.info(
                f"   📈 Peak: {initial_memory + allocated:.1f}MB (+{allocated:.1f}MB)"
            )
            logger.info(f"   📉 Final: {final_memory:.1f}MB")
            logger.info(f"   🔄 Net change: {net_change:+.1f}MB")

            if abs(net_change) < 10:  # Within 10MB of original
                logger.info("✅ Simulation successful - memory recovered")
                return True
            else:
                logger.warning(f"⚠️ Significant memory difference: {net_change:.1f}MB")
                return False

        except Exception as e:
            logger.error(f"❌ Simulation failed: {e}")
            # Attempt cleanup
            try:
                self.release_memory()
            except:
                pass
            return False


def main():
    """Main entry point for memory leak simulation."""
    parser = argparse.ArgumentParser(
        description="Memory leak simulation for testing memory probe"
    )
    parser.add_argument(
        "--secs",
        type=int,
        default=30,
        help="Duration to hold allocated memory (default: 30)",
    )
    parser.add_argument(
        "--mb",
        type=int,
        default=200,
        help="Amount of memory to allocate in MB (default: 200)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Chunk size for allocation in MB (default: 10)",
    )

    args = parser.parse_args()

    print("🧪 Memory Leak Simulation")
    print("=" * 40)
    print(f"📊 Allocation: {args.mb}MB")
    print(f"⏱️ Duration: {args.secs}s")
    print(f"📦 Chunk size: {args.chunk_size}MB")
    print()

    simulator = MemoryLeakSimulator()
    success = simulator.simulate_spike_and_recovery(
        duration_secs=args.secs, allocation_mb=args.mb
    )

    if success:
        print("\n🎯 Expected Result: Memory probe should detect spike then recovery")
        print("📝 Check logs/mem_snap.log for memory growth alerts")
        return 0
    else:
        print("\n❌ Simulation failed")
        return 1


if __name__ == "__main__":
    exit(main())
