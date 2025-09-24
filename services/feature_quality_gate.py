#!/usr/bin/env python3
"""
Feature Quality Gate Service
Halt/soft-degrade trading if inputs are stale or corrupt (saves bad days)
"""

import os
import sys
import json
import time
import math
import logging
import requests
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("feature_quality_gate")


class FeatureQualityGate:
    """Feature quality monitoring and circuit breaker service."""

    def __init__(self):
        """Initialize feature quality gate."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.slack_webhook = os.getenv("SLACK_WEBHOOK", "")

        # Feed definitions and staleness thresholds (seconds)
        self.feeds = {
            "ticks": "market.raw.crypto.BTCUSDT",
            "orderbook": "orderbook:BTC",
            "sentiment": "sentiment:latest",
            "options": "event:options",
            "whale": "event:whale",
        }

        self.staleness_thresholds = {
            "ticks": 5,  # Market ticks should be very fresh
            "orderbook": 5,  # Order book should be very fresh
            "sentiment": 300,  # Sentiment can be up to 5 minutes old
            "options": 600,  # Options flow can be up to 10 minutes old
            "whale": 600,  # Whale movements can be up to 10 minutes old
        }

        # Quality check parameters
        self.outlier_z_threshold = 8.0  # Z-score threshold for outliers
        self.max_nan_ratio = 0.1  # Maximum ratio of NaN values allowed

        # State tracking
        self.last_check_time = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3

        # Feature flag integration
        self.feature_flags_key = "features:flags"
        self.mode_key = "mode"

        logger.info("🚨 Feature Quality Gate initialized")
        logger.info(f"   Monitoring feeds: {list(self.feeds.keys())}")
        logger.info(
            f"   Slack webhook: {'✅ Configured' if self.slack_webhook else '❌ Missing'}"
        )

    def check_feed_staleness(self) -> list:
        """Check if any feeds are stale."""
        now = time.time()
        stale_feeds = []

        try:
            for feed_name, stream_key in self.feeds.items():
                threshold = self.staleness_thresholds[feed_name]

                try:
                    # Get the most recent entry from the stream
                    recent_entries = self.redis.xrevrange(stream_key, "+", "-", count=1)

                    if not recent_entries:
                        stale_feeds.append((feed_name, "empty_stream"))
                        continue

                    # Extract timestamp from Redis stream ID
                    entry_id = recent_entries[0][0]
                    timestamp_ms = float(entry_id.split("-")[0])
                    timestamp_sec = timestamp_ms / 1000

                    age_seconds = now - timestamp_sec

                    if age_seconds > threshold:
                        stale_feeds.append((feed_name, f"stale_{age_seconds:.1f}s"))
                        logger.warning(
                            f"Feed {feed_name} is stale: {age_seconds:.1f}s old "
                            f"(threshold: {threshold}s)"
                        )

                except Exception as e:
                    stale_feeds.append((feed_name, f"error_{str(e)[:20]}"))
                    logger.error(f"Error checking feed {feed_name}: {e}")

            return stale_feeds

        except Exception as e:
            logger.error(f"Error in staleness check: {e}")
            return [("staleness_check", f"error_{str(e)[:20]}")]

    def check_state_quality(self) -> list:
        """Check quality of the live state vector."""
        quality_issues = []

        try:
            # Get most recent state from live stream
            state_entries = self.redis.xrevrange("state:live", "+", "-", count=1)

            if not state_entries:
                quality_issues.append(("state", "no_live_state"))
                return quality_issues

            # Extract state values
            state_data = state_entries[0][1]
            values = []

            for key, value_str in state_data.items():
                try:
                    # Try to convert to float
                    if isinstance(value_str, str):
                        # Handle string that might represent a number
                        if (
                            value_str.replace(".", "", 1)
                            .replace("-", "", 1)
                            .replace("e", "", 1)
                            .replace("E", "", 1)
                            .isdigit()
                        ):
                            values.append(float(value_str))
                    else:
                        values.append(float(value_str))
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    continue

            if not values:
                quality_issues.append(("state", "no_numeric_values"))
                return quality_issues

            # Check for NaN/Inf values
            nan_count = 0
            inf_count = 0

            for value in values:
                if math.isnan(value):
                    nan_count += 1
                elif math.isinf(value):
                    inf_count += 1

            # Check NaN ratio
            nan_ratio = nan_count / len(values)
            if nan_ratio > self.max_nan_ratio:
                quality_issues.append(("state", f"high_nan_ratio_{nan_ratio:.2%}"))

            # Check for any Inf values
            if inf_count > 0:
                quality_issues.append(("state", f"inf_values_{inf_count}"))

            # Check for extreme outliers using z-score
            if len(values) > 3:  # Need at least a few values for meaningful statistics
                try:
                    values_array = np.array(
                        [v for v in values if not (math.isnan(v) or math.isinf(v))]
                    )

                    if len(values_array) > 1:
                        z_scores = np.abs(stats.zscore(values_array))
                        extreme_outliers = np.sum(z_scores > self.outlier_z_threshold)

                        if extreme_outliers > 0:
                            quality_issues.append(
                                ("state", f"extreme_outliers_{extreme_outliers}")
                            )

                except Exception as e:
                    # Stats calculation failed
                    quality_issues.append(("state", f"stats_error_{str(e)[:15]}"))

            logger.debug(
                f"State quality check: {len(values)} values, "
                f"{nan_count} NaN, {inf_count} Inf"
            )

            return quality_issues

        except Exception as e:
            logger.error(f"Error in state quality check: {e}")
            return [("state_check", f"error_{str(e)[:20]}")]

    def check_data_monotonicity(self) -> list:
        """Check for timestamp monotonicity issues."""
        monotonicity_issues = []

        try:
            # Check a few key streams for timestamp consistency
            critical_streams = ["market.raw.crypto.BTCUSDT", "state:live"]

            for stream_key in critical_streams:
                try:
                    # Get recent entries
                    entries = self.redis.xrevrange(stream_key, "+", "-", count=10)

                    if len(entries) < 2:
                        continue

                    # Extract timestamps
                    timestamps = []
                    for entry_id, _ in entries:
                        timestamp_ms = float(entry_id.split("-")[0])
                        timestamps.append(timestamp_ms)

                    # Check if timestamps are decreasing (as expected from xrevrange)
                    for i in range(1, len(timestamps)):
                        if timestamps[i] > timestamps[i - 1]:
                            monotonicity_issues.append(
                                (stream_key, "timestamp_disorder")
                            )
                            break

                    # Check for duplicate timestamps
                    if len(set(timestamps)) < len(timestamps):
                        monotonicity_issues.append((stream_key, "duplicate_timestamps"))

                except Exception as e:
                    monotonicity_issues.append(
                        (stream_key, f"monotonicity_error_{str(e)[:15]}")
                    )

            return monotonicity_issues

        except Exception as e:
            logger.error(f"Error in monotonicity check: {e}")
            return [("monotonicity_check", f"error_{str(e)[:20]}")]

    def perform_quality_check(self) -> dict:
        """Perform comprehensive quality check."""
        check_start = time.time()

        try:
            # Run all quality checks
            stale_feeds = self.check_feed_staleness()
            state_issues = self.check_state_quality()
            monotonicity_issues = self.check_data_monotonicity()

            # Combine all issues
            all_issues = stale_feeds + state_issues + monotonicity_issues

            # Categorize issues by severity
            critical_issues = []
            warning_issues = []

            for issue_source, issue_desc in all_issues:
                # Critical issues that should halt trading
                if any(
                    keyword in issue_desc.lower()
                    for keyword in [
                        "empty",
                        "nan",
                        "inf",
                        "stale_ticks",
                        "stale_orderbook",
                    ]
                ):
                    critical_issues.append((issue_source, issue_desc))
                else:
                    warning_issues.append((issue_source, issue_desc))

            check_duration = time.time() - check_start

            return {
                "timestamp": time.time(),
                "check_duration_ms": check_duration * 1000,
                "critical_issues": critical_issues,
                "warning_issues": warning_issues,
                "total_issues": len(all_issues),
                "feeds_checked": len(self.feeds),
                "status": (
                    "critical"
                    if critical_issues
                    else ("warning" if warning_issues else "healthy")
                ),
            }

        except Exception as e:
            logger.error(f"Error in quality check: {e}")
            return {"timestamp": time.time(), "status": "error", "error": str(e)}

    def take_action(self, quality_result: dict) -> dict:
        """Take appropriate action based on quality check results."""
        actions_taken = []

        try:
            status = quality_result.get("status", "unknown")
            critical_issues = quality_result.get("critical_issues", [])
            warning_issues = quality_result.get("warning_issues", [])

            if status == "critical":
                # Critical issues - halt trading and disable risky features

                # Set system to halt mode
                self.redis.set(self.mode_key, "halt")
                actions_taken.append("set_halt_mode")

                # Disable risky feature flags
                risky_flags = ["LLM_SENTIMENT", "EXEC_RL_LIVE", "BANDIT_WEIGHTS"]
                flag_updates = {flag: 0 for flag in risky_flags}
                self.redis.hset(self.feature_flags_key, mapping=flag_updates)
                actions_taken.append(f"disabled_flags_{len(risky_flags)}")

                # Send critical alert
                alert_message = (
                    f"🚨 CRITICAL: Feature Quality Gate triggered!\n"
                    f"Issues: {len(critical_issues)} critical, {len(warning_issues)} warnings\n"
                    f"Actions: Trading HALTED, risky features DISABLED\n"
                    f"Critical issues: {critical_issues[:3]}"  # Show first 3
                )

                if self.send_alert(alert_message, severity="critical"):
                    actions_taken.append("sent_critical_alert")

                self.consecutive_failures += 1

                logger.error(
                    f"🚨 CRITICAL quality issues detected: {len(critical_issues)} issues, "
                    f"HALTED trading, disabled {len(risky_flags)} flags"
                )

            elif status == "warning":
                # Warning issues - soft degradation

                # Disable only LLM sentiment (least critical)
                self.redis.hset(self.feature_flags_key, "LLM_SENTIMENT", 0)
                actions_taken.append("disabled_llm_sentiment")

                # Send warning alert (less frequent)
                if self.consecutive_failures % 5 == 0:  # Every 5th warning
                    alert_message = (
                        f"⚠️ WARNING: Feature quality degraded\n"
                        f"Issues: {len(warning_issues)} warnings\n"
                        f"Action: LLM sentiment disabled\n"
                        f"Warning issues: {warning_issues[:2]}"
                    )

                    if self.send_alert(alert_message, severity="warning"):
                        actions_taken.append("sent_warning_alert")

                self.consecutive_failures += 1

                logger.warning(
                    f"⚠️ Quality warnings: {len(warning_issues)} issues, disabled LLM sentiment"
                )

            else:
                # Healthy status - reset failure counter and ensure normal operation
                if self.consecutive_failures > 0:
                    logger.info("✅ Quality checks passed, resetting failure count")

                    # Re-enable normal mode if we were in halt
                    current_mode = self.redis.get(self.mode_key)
                    if current_mode == "halt":
                        self.redis.delete(self.mode_key)  # Remove halt mode
                        actions_taken.append("cleared_halt_mode")

                        # Send recovery alert
                        alert_message = (
                            "✅ Quality gate: All clear! Resuming normal operation."
                        )
                        if self.send_alert(alert_message, severity="recovery"):
                            actions_taken.append("sent_recovery_alert")

                self.consecutive_failures = 0

            self.last_check_time = time.time()

            return {
                "actions_taken": actions_taken,
                "consecutive_failures": self.consecutive_failures,
                "quality_status": status,
            }

        except Exception as e:
            logger.error(f"Error taking action: {e}")
            return {"actions_taken": ["error"], "error": str(e)}

    def send_alert(self, message: str, severity: str = "info") -> bool:
        """Send alert to Slack."""
        try:
            if not self.slack_webhook:
                logger.debug("No Slack webhook configured")
                return False

            # Choose emoji based on severity
            emoji_map = {
                "critical": "🚨",
                "warning": "⚠️",
                "recovery": "✅",
                "info": "ℹ️",
            }

            emoji = emoji_map.get(severity, "ℹ️")

            payload = {
                "text": f"{emoji} {message}",
                "username": "Quality Gate",
                "icon_emoji": ":shield:",
            }

            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"📱 Sent {severity} alert to Slack")
            return True

        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False

    def get_status(self) -> dict:
        """Get current status of the quality gate."""
        try:
            current_mode = self.redis.get(self.mode_key) or "normal"

            # Get current feature flag states
            feature_flags = self.redis.hgetall(self.feature_flags_key)

            return {
                "service": "feature_quality_gate",
                "status": "active",
                "current_mode": current_mode,
                "consecutive_failures": self.consecutive_failures,
                "last_check_time": self.last_check_time,
                "feeds_monitored": list(self.feeds.keys()),
                "staleness_thresholds": self.staleness_thresholds,
                "feature_flags": {k: bool(int(v)) for k, v in feature_flags.items()},
                "quality_parameters": {
                    "outlier_z_threshold": self.outlier_z_threshold,
                    "max_nan_ratio": self.max_nan_ratio,
                    "max_consecutive_failures": self.max_consecutive_failures,
                },
            }

        except Exception as e:
            return {
                "service": "feature_quality_gate",
                "status": "error",
                "error": str(e),
            }

    def run_daemon(self, check_interval: int = 5):
        """Run the quality gate daemon."""
        logger.info(
            f"🚨 Starting Feature Quality Gate daemon (interval: {check_interval}s)"
        )

        try:
            while True:
                try:
                    # Perform quality check
                    quality_result = self.perform_quality_check()

                    # Take appropriate action
                    action_result = self.take_action(quality_result)

                    # Log summary
                    if quality_result.get("total_issues", 0) > 0:
                        logger.info(
                            f"Quality check: {quality_result['status']} status, "
                            f"{quality_result['total_issues']} issues, "
                            f"actions: {action_result['actions_taken']}"
                        )
                    else:
                        logger.debug("Quality check: healthy")

                    # Wait for next check
                    time.sleep(check_interval)

                except KeyboardInterrupt:
                    logger.info("🛑 Quality gate daemon stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in daemon loop: {e}")
                    time.sleep(check_interval)  # Continue after error

        except Exception as e:
            logger.error(f"Fatal error in quality gate daemon: {e}")
            raise


# Import scipy.stats with fallback
try:
    from scipy import stats
except ImportError:
    # Fallback z-score implementation
    class FallbackStats:
        @staticmethod
        def zscore(a):
            return (a - np.mean(a)) / np.std(a)

    stats = FallbackStats()


def main():
    """Main entry point for feature quality gate."""
    import argparse

    parser = argparse.ArgumentParser(description="Feature Quality Gate")
    parser.add_argument(
        "--check-interval", type=int, default=5, help="Check interval in seconds"
    )
    parser.add_argument(
        "--single-check", action="store_true", help="Run single check and exit"
    )
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument(
        "--reset", action="store_true", help="Reset halt mode and failure counters"
    )

    args = parser.parse_args()

    # Create quality gate
    gate = FeatureQualityGate()

    if args.status:
        # Show status
        status = gate.get_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.reset:
        # Reset halt mode
        gate.redis.delete(gate.mode_key)
        gate.consecutive_failures = 0
        logger.info("🔄 Reset quality gate state")
        return

    if args.single_check:
        # Run single check
        logger.info("Running single quality check...")
        quality_result = gate.perform_quality_check()
        action_result = gate.take_action(quality_result)

        print(
            json.dumps(
                {"quality_result": quality_result, "action_result": action_result},
                indent=2,
                default=str,
            )
        )

        return

    # Run daemon
    gate.run_daemon(args.check_interval)


if __name__ == "__main__":
    main()
