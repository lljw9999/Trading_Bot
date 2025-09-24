#!/usr/bin/env python3
"""
Capital Auto-Ramp System
Safely scale live exposure from 10% → 100% based on recent performance
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import redis
import requests
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("capital_autoramp")


class CapitalAutoRamp:
    """Capital auto-ramp system for safe exposure scaling."""

    def __init__(self):
        """Initialize capital auto-ramp system."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.slack_webhook = os.getenv("SLACK_WEBHOOK", "")

        # Risk management parameters
        self.thresholds = {
            "high_sharpe": 0.8,  # Minimum 1h Sharpe to increase capital
            "low_sharpe": 0.2,  # Below this, decrease capital
            "max_dd_increase": 0.015,  # Max 24h drawdown to allow increase (1.5%)
            "max_dd_decrease": 0.03,  # Max 24h drawdown before decrease (3%)
            "capital_step_up": 0.10,  # Increase capital by 10%
            "capital_step_down": 0.20,  # Decrease capital by 20%
            "min_capital": 0.10,  # Minimum capital factor (10%)
            "max_capital": 1.00,  # Maximum capital factor (100%)
        }

        # Redis keys for metrics
        self.keys = {
            "capital_factor": "risk:capital_factor",
            "sharpe_1h": "sharpe:1h:live",
            "dd_24h": "risk:dd_24h",
            "drift_flag": "drift:flag",
            "pnl_stream": "pnl:stream",
            "decisions_log": "capital:decisions",
        }

        logger.info("💰 Capital Auto-Ramp initialized")
        logger.info(
            f"   Thresholds: Sharpe {self.thresholds['low_sharpe']}-{self.thresholds['high_sharpe']}"
        )
        logger.info(
            f"   DD limits: {self.thresholds['max_dd_increase']:.1%}-{self.thresholds['max_dd_decrease']:.1%}"
        )
        logger.info(
            f"   Capital range: {self.thresholds['min_capital']:.0%}-{self.thresholds['max_capital']:.0%}"
        )

    def get_metric(self, key: str, default: float = 0.0) -> float:
        """Get a metric from Redis with fallback."""
        try:
            value = self.redis.get(key)
            return float(value) if value else default
        except Exception:
            return default

    def calculate_sharpe_1h(self) -> float:
        """Calculate 1-hour Sharpe ratio from P&L stream."""
        try:
            # Get P&L data from last hour
            one_hour_ago = int((datetime.now() - timedelta(hours=1)).timestamp() * 1000)
            now = int(datetime.now().timestamp() * 1000)

            pnl_entries = self.redis.xrange(
                self.keys["pnl_stream"], min=one_hour_ago, max=now
            )

            if len(pnl_entries) < 2:
                logger.debug("Insufficient P&L data for Sharpe calculation")
                return 0.0

            # Extract P&L values
            pnl_values = []
            for entry_id, fields in pnl_entries:
                try:
                    pnl = float(fields.get("total_pnl", 0))
                    pnl_values.append(pnl)
                except Exception:
                    continue

            if len(pnl_values) < 2:
                return 0.0

            # Calculate returns
            pnl_returns = np.diff(pnl_values)

            if len(pnl_returns) == 0:
                return 0.0

            # Sharpe calculation (annualized)
            mean_return = np.mean(pnl_returns)
            std_return = np.std(pnl_returns)

            if std_return <= 1e-8:
                return 0.0

            # Annualize assuming ~60 updates per hour
            sharpe = (mean_return / std_return) * np.sqrt(60 * 24 * 365)

            return float(sharpe)

        except Exception as e:
            logger.error(f"Error calculating 1h Sharpe: {e}")
            return 0.0

    def calculate_drawdown_24h(self) -> float:
        """Calculate 24-hour maximum drawdown."""
        try:
            # Get P&L data from last 24 hours
            yesterday = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000)
            now = int(datetime.now().timestamp() * 1000)

            pnl_entries = self.redis.xrange(
                self.keys["pnl_stream"], min=yesterday, max=now
            )

            if len(pnl_entries) < 2:
                logger.debug("Insufficient P&L data for drawdown calculation")
                return 0.0

            # Extract P&L values
            pnl_values = []
            for entry_id, fields in pnl_entries:
                try:
                    pnl = float(fields.get("total_pnl", 0))
                    pnl_values.append(pnl)
                except Exception:
                    continue

            if len(pnl_values) < 2:
                return 0.0

            # Calculate maximum drawdown
            pnl_array = np.array(pnl_values)
            peak = np.maximum.accumulate(pnl_array)
            drawdown = (peak - pnl_array) / np.maximum(np.abs(peak), 1)
            max_dd = np.max(drawdown)

            return float(max_dd)

        except Exception as e:
            logger.error(f"Error calculating 24h drawdown: {e}")
            return 0.0

    def check_drift_flag(self) -> bool:
        """Check if drift detection flag is set."""
        try:
            drift_value = self.redis.get(self.keys["drift_flag"])
            return bool(drift_value and int(drift_value))
        except Exception:
            return False

    def get_current_capital_factor(self) -> float:
        """Get current capital factor."""
        try:
            current = self.redis.get(self.keys["capital_factor"])
            return float(current) if current else self.thresholds["min_capital"]
        except Exception:
            return self.thresholds["min_capital"]

    def log_decision(self, decision_data: dict):
        """Log capital adjustment decision to Redis stream."""
        try:
            self.redis.xadd(self.keys["decisions_log"], decision_data)
        except Exception as e:
            logger.error(f"Error logging decision: {e}")

    def send_slack_notification(self, message: str) -> bool:
        """Send notification to Slack."""
        try:
            if not self.slack_webhook:
                logger.debug("No Slack webhook configured")
                return False

            payload = {
                "text": message,
                "username": "Capital AutoRamp",
                "icon_emoji": ":moneybag:",
            }

            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("📱 Sent notification to Slack")
            return True

        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False

    def evaluate_and_adjust_capital(self) -> dict:
        """Evaluate performance metrics and adjust capital factor."""
        evaluation_start = time.time()

        try:
            # Get current state
            current_capital = self.get_current_capital_factor()

            # Calculate or retrieve metrics
            sharpe_1h = self.get_metric(self.keys["sharpe_1h"])
            if sharpe_1h == 0.0:  # Fallback to calculated Sharpe
                sharpe_1h = self.calculate_sharpe_1h()

            dd_24h = self.get_metric(self.keys["dd_24h"])
            if dd_24h == 0.0:  # Fallback to calculated DD
                dd_24h = self.calculate_drawdown_24h()

            drift_detected = self.check_drift_flag()

            # Decision logic
            decision = "no_change"
            new_capital = current_capital
            reason = ""

            # Conditions for increasing capital
            if (
                sharpe_1h >= self.thresholds["high_sharpe"]
                and dd_24h <= self.thresholds["max_dd_increase"]
                and not drift_detected
                and current_capital < self.thresholds["max_capital"]
            ):

                new_capital = min(
                    self.thresholds["max_capital"],
                    current_capital + self.thresholds["capital_step_up"],
                )
                decision = "increase"
                reason = f"Strong performance: Sharpe={sharpe_1h:.2f}, DD={dd_24h:.2%}"

            # Conditions for decreasing capital
            elif (
                sharpe_1h < self.thresholds["low_sharpe"]
                or dd_24h > self.thresholds["max_dd_decrease"]
                or drift_detected
            ):

                new_capital = max(
                    self.thresholds["min_capital"],
                    current_capital - self.thresholds["capital_step_down"],
                )
                decision = "decrease"

                reasons = []
                if sharpe_1h < self.thresholds["low_sharpe"]:
                    reasons.append(f"Low Sharpe={sharpe_1h:.2f}")
                if dd_24h > self.thresholds["max_dd_decrease"]:
                    reasons.append(f"High DD={dd_24h:.2%}")
                if drift_detected:
                    reasons.append("Drift detected")
                reason = "Poor performance: " + ", ".join(reasons)

            else:
                reason = f"Stable performance: Sharpe={sharpe_1h:.2f}, DD={dd_24h:.2%}"

            # Update capital factor if changed
            if new_capital != current_capital:
                self.redis.set(self.keys["capital_factor"], new_capital)

                # Log decision
                decision_data = {
                    "timestamp": time.time(),
                    "old_capital": current_capital,
                    "new_capital": new_capital,
                    "decision": decision,
                    "reason": reason,
                    "sharpe_1h": sharpe_1h,
                    "dd_24h": dd_24h,
                    "drift_detected": drift_detected,
                }

                self.log_decision(decision_data)

                # Send Slack notification
                emoji = "📈" if decision == "increase" else "📉"
                slack_message = (
                    f"{emoji} Capital factor adjusted: {current_capital:.2f} → {new_capital:.2f}\n"
                    f"Decision: {decision.upper()}\n"
                    f"Metrics: Sharpe={sharpe_1h:.2f}, DD={dd_24h:.2%}, Drift={drift_detected}\n"
                    f"Reason: {reason}"
                )

                self.send_slack_notification(slack_message)

                logger.info(
                    f"💰 Capital adjusted: {current_capital:.2f} → {new_capital:.2f} "
                    f"({decision})"
                )
            else:
                logger.info(f"💰 Capital unchanged: {current_capital:.2f} ({reason})")

            evaluation_time = time.time() - evaluation_start

            return {
                "timestamp": time.time(),
                "evaluation_time_ms": evaluation_time * 1000,
                "old_capital_factor": current_capital,
                "new_capital_factor": new_capital,
                "decision": decision,
                "reason": reason,
                "metrics": {
                    "sharpe_1h": sharpe_1h,
                    "dd_24h": dd_24h,
                    "drift_detected": drift_detected,
                },
                "thresholds": self.thresholds,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Error in capital evaluation: {e}")
            return {"timestamp": time.time(), "status": "error", "error": str(e)}

    def get_status(self) -> dict:
        """Get current status of capital auto-ramp system."""
        try:
            current_capital = self.get_current_capital_factor()

            # Get recent decisions
            recent_decisions = []
            try:
                decision_entries = self.redis.xrevrange(
                    self.keys["decisions_log"], count=5
                )
                for entry_id, fields in decision_entries:
                    timestamp_ms = int(entry_id.split("-")[0])
                    decision = {
                        "timestamp": timestamp_ms / 1000,
                        "datetime": datetime.fromtimestamp(
                            timestamp_ms / 1000
                        ).isoformat(),
                        **fields,
                    }
                    recent_decisions.append(decision)
            except Exception:
                pass

            return {
                "service": "capital_autoramp",
                "status": "active",
                "current_capital_factor": current_capital,
                "capital_utilization": f"{current_capital:.1%}",
                "thresholds": self.thresholds,
                "metrics_keys": self.keys,
                "recent_decisions": recent_decisions,
            }

        except Exception as e:
            return {"service": "capital_autoramp", "status": "error", "error": str(e)}

    def run_single_evaluation(self) -> dict:
        """Run single capital evaluation (for cron job)."""
        logger.info("🔍 Running capital auto-ramp evaluation...")

        result = self.evaluate_and_adjust_capital()

        if result["status"] == "success":
            logger.info(
                f"✅ Capital evaluation complete: "
                f"{result['old_capital_factor']:.2f} → {result['new_capital_factor']:.2f} "
                f"({result['decision']})"
            )
        else:
            logger.error(
                f"❌ Capital evaluation failed: {result.get('error', 'Unknown error')}"
            )

        return result


def main():
    """Main entry point for capital auto-ramp."""
    import argparse

    parser = argparse.ArgumentParser(description="Capital Auto-Ramp System")
    parser.add_argument(
        "--evaluate", action="store_true", help="Run single evaluation and exit"
    )
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument(
        "--reset-capital",
        type=float,
        metavar="FACTOR",
        help="Reset capital factor to specific value (0.1-1.0)",
    )

    args = parser.parse_args()

    # Create auto-ramp system
    autoramp = CapitalAutoRamp()

    if args.status:
        # Show status
        status = autoramp.get_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.reset_capital is not None:
        # Reset capital factor
        if not (0.1 <= args.reset_capital <= 1.0):
            logger.error("Capital factor must be between 0.1 and 1.0")
            sys.exit(1)

        autoramp.redis.set(autoramp.keys["capital_factor"], args.reset_capital)
        logger.info(f"🔄 Reset capital factor to {args.reset_capital:.2f}")
        return

    # Run single evaluation (default behavior)
    result = autoramp.run_single_evaluation()

    # Print result for cron job logging
    print(json.dumps(result, indent=2, default=str))

    # Exit with appropriate code
    if result["status"] == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
