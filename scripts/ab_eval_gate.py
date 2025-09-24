#!/usr/bin/env python3
"""
A/B Evaluation Gate
Check if A/B testing gate is passing (4 consecutive passes)
"""

import sys
import time
import logging
from typing import Dict, Any

import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("ab_eval_gate")


def check_ab_gate(dry_run: bool = False) -> Dict[str, Any]:
    """Check A/B evaluation gate status."""
    try:
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Get recent A/B test results
        consecutive_passes = int(r.get("ab:last4:exec") or 0)
        total_tests = int(r.get("ab:total_tests") or 0)
        last_test_time = float(r.get("ab:last_test_time") or 0)

        # Check if tests are recent enough
        time_since_last = time.time() - last_test_time
        tests_recent = time_since_last < 3600  # Within last hour

        # Gate passes if we have 4+ consecutive passes and tests are recent
        gate_passing = consecutive_passes >= 4 and tests_recent

        result = {
            "status": "PASS" if gate_passing else "FAIL",
            "consecutive_passes": consecutive_passes,
            "total_tests": total_tests,
            "time_since_last_test": time_since_last,
            "tests_recent": tests_recent,
            "dry_run": dry_run,
            "timestamp": time.time(),
        }

        if dry_run:
            # Simulate some test data for smoke test
            result.update(
                {
                    "consecutive_passes": 4,
                    "total_tests": 12,
                    "time_since_last_test": 300,  # 5 minutes ago
                    "tests_recent": True,
                    "status": "PASS",
                }
            )
            logger.info("🧪 A/B Gate check (DRY RUN): SIMULATED PASS")
        else:
            status_msg = "PASS" if gate_passing else "FAIL"
            logger.info(
                f"📊 A/B Gate check: {status_msg} ({consecutive_passes}/4 consecutive passes)"
            )

        return result

    except Exception as e:
        logger.error(f"Error checking A/B gate: {e}")
        return {
            "status": "ERROR",
            "error": str(e),
            "dry_run": dry_run,
            "timestamp": time.time(),
        }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="A/B Evaluation Gate")
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run mode for testing"
    )

    args = parser.parse_args()

    result = check_ab_gate(args.dry_run)

    if result["status"] == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
