#!/usr/bin/env python3
"""
Capital Ramp Guard
Two consecutive morning greens required before allowing next ramp step
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import redis
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("capital_ramp_guard")


class CapitalRampGuard:
    """Guards capital ramping based on consecutive morning greenlight passes."""

    def __init__(self):
        """Initialize capital ramp guard."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.slack_webhook = os.getenv("SLACK_WEBHOOK")

        # Guard configuration
        self.config = {
            "consecutive_days_required": 2,  # Need 2 consecutive green mornings
            "max_single_ramp": 0.20,  # Max 20% increase per ramp
            "ramp_schedule": [0.10, 0.20, 0.40, 0.70, 1.00],  # Standard ramp schedule
            "recon_breach_threshold": 0,  # Zero tolerance for recon breaches
            "min_hours_between_ramps": 24,  # Minimum 24 hours between ramps
        }

        logger.info("🛡️ Capital Ramp Guard initialized")

    def get_recent_greenlight_history(self, days: int = 7) -> Dict[str, Any]:
        """Get recent morning greenlight history."""
        try:
            history = {}

            for i in range(days):
                date = datetime.now(timezone.utc) - timedelta(days=i)
                date_str = date.strftime("%Y-%m-%d")

                # Try to get greenlight result from Redis
                greenlight_key = f"greenlight:history:{date_str}"
                result_data = self.redis.get(greenlight_key)

                if result_data:
                    try:
                        result = json.loads(result_data)
                        history[date_str] = {
                            "status": result.get("overall_status", "UNKNOWN"),
                            "score": result.get("weighted_score", 0.0),
                            "checks_passed": result.get("checks", {}).get("passed", 0),
                            "checks_failed": result.get("checks", {}).get("failed", 0),
                        }
                    except json.JSONDecodeError:
                        history[date_str] = {"status": "ERROR", "error": "Invalid JSON"}
                else:
                    # Check if we have a report file
                    reports_dir = (
                        Path(__file__).parent.parent / "reports" / "morning_greenlight"
                    )
                    report_file = reports_dir / f"greenlight_{date_str}.json"

                    if report_file.exists():
                        try:
                            with open(report_file, "r") as f:
                                result = json.load(f)
                            history[date_str] = {
                                "status": result.get("overall_status", "UNKNOWN"),
                                "score": result.get("weighted_score", 0.0),
                                "checks_passed": result.get("checks", {}).get(
                                    "passed", 0
                                ),
                                "checks_failed": result.get("checks", {}).get(
                                    "failed", 0
                                ),
                            }
                        except (json.JSONDecodeError, FileNotFoundError):
                            history[date_str] = {"status": "NO_DATA"}
                    else:
                        history[date_str] = {"status": "NO_DATA"}

            return history

        except Exception as e:
            logger.error(f"Error getting greenlight history: {e}")
            return {}

    def check_consecutive_greens(self) -> Dict[str, Any]:
        """Check if we have required consecutive green mornings."""
        try:
            history = self.get_recent_greenlight_history()

            # Sort dates in reverse chronological order (most recent first)
            sorted_dates = sorted(history.keys(), reverse=True)

            consecutive_greens = 0
            consecutive_days = []

            for date in sorted_dates:
                day_status = history[date].get("status", "NO_DATA")

                if day_status == "GREEN":
                    consecutive_greens += 1
                    consecutive_days.append(date)
                else:
                    break  # Chain is broken

            meets_requirement = (
                consecutive_greens >= self.config["consecutive_days_required"]
            )

            result = {
                "meets_requirement": meets_requirement,
                "consecutive_greens": consecutive_greens,
                "required_greens": self.config["consecutive_days_required"],
                "consecutive_days": consecutive_days,
                "history": history,
            }

            logger.info(
                f"🟢 Consecutive green check: {consecutive_greens}/{self.config['consecutive_days_required']} "
                f"(requirement {'MET' if meets_requirement else 'NOT MET'})"
            )

            return result

        except Exception as e:
            logger.error(f"Error checking consecutive greens: {e}")
            return {"meets_requirement": False, "error": str(e)}

    def check_recon_breaches(self) -> Dict[str, Any]:
        """Check for reconciliation breaches."""
        try:
            breaches_24h = int(self.redis.get("recon:breaches_24h") or 0)
            position_mismatches = int(self.redis.get("recon:position_mismatches") or 0)
            total_issues = breaches_24h + position_mismatches

            recon_clean = total_issues <= self.config["recon_breach_threshold"]

            result = {
                "recon_clean": recon_clean,
                "breaches_24h": breaches_24h,
                "position_mismatches": position_mismatches,
                "total_issues": total_issues,
                "threshold": self.config["recon_breach_threshold"],
            }

            logger.info(f"🔍 Recon check: {total_issues} issues (clean: {recon_clean})")

            return result

        except Exception as e:
            logger.error(f"Error checking recon breaches: {e}")
            return {"recon_clean": False, "error": str(e)}

    def get_current_capital_state(self) -> Dict[str, Any]:
        """Get current capital allocation state."""
        try:
            capital_effective = float(self.redis.get("risk:capital_effective") or 0.0)
            capital_staged = self.redis.get("risk:capital_stage_request")
            capital_next_week = float(
                self.redis.get("risk:capital_cap_next_week") or 0.0
            )

            # Get last ramp time
            last_ramp_time = float(self.redis.get("capital:last_ramp_time") or 0)
            hours_since_last_ramp = (
                (time.time() - last_ramp_time) / 3600 if last_ramp_time > 0 else 999
            )

            result = {
                "capital_effective": capital_effective,
                "capital_staged": float(capital_staged) if capital_staged else None,
                "capital_next_week": capital_next_week,
                "last_ramp_time": last_ramp_time,
                "hours_since_last_ramp": hours_since_last_ramp,
                "min_hours_required": self.config["min_hours_between_ramps"],
            }

            return result

        except Exception as e:
            logger.error(f"Error getting capital state: {e}")
            return {"error": str(e)}

    def calculate_next_ramp_target(self, current_capital: float) -> Optional[float]:
        """Calculate the next valid ramp target."""
        try:
            # Find current position in ramp schedule
            ramp_schedule = self.config["ramp_schedule"]

            # Find the next level in the schedule
            for target in ramp_schedule:
                if (
                    target > current_capital + 0.01
                ):  # Small epsilon for float comparison
                    return target

            # Already at max
            return None

        except Exception as e:
            logger.error(f"Error calculating next ramp target: {e}")
            return None

    def approve_ramp_request(self, target_pct: float) -> bool:
        """Approve a capital ramp request."""
        try:
            # Set the new capital cap
            self.redis.set("risk:capital_cap_next_week", target_pct)
            self.redis.set("capital:last_ramp_time", int(time.time()))
            self.redis.set("capital:last_approved_by", "ramp_guard")

            # Clear any staging request
            self.redis.delete("risk:capital_stage_request")

            logger.info(f"✅ Approved capital ramp to {target_pct:.0%}")
            return True

        except Exception as e:
            logger.error(f"Error approving ramp: {e}")
            return False

    def deny_ramp_request(self, reason: str, current_level: float) -> bool:
        """Deny ramp request and pin at current level."""
        try:
            # Pin at current level
            self.redis.set("risk:capital_stage_request", current_level)
            self.redis.set("capital:ramp_deny_reason", reason)
            self.redis.set("capital:ramp_deny_time", int(time.time()))

            logger.warning(f"❌ Denied capital ramp: {reason}")
            return True

        except Exception as e:
            logger.error(f"Error denying ramp: {e}")
            return False

    def send_slack_notification(self, message: str, urgent: bool = False):
        """Send notification to Slack."""
        try:
            if not self.slack_webhook:
                logger.warning("No Slack webhook configured")
                return

            payload = {
                "text": message,
                "username": "Capital Ramp Guard",
                "icon_emoji": ":shield:" if not urgent else ":warning:",
            }

            if urgent:
                payload["channel"] = "#trading-alerts"

            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"📱 Sent Slack notification: {message[:100]}...")

        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")

    def run_ramp_guard_check(self) -> Dict[str, Any]:
        """Run the complete capital ramp guard check."""
        try:
            guard_start = time.time()
            logger.info("🛡️ Running capital ramp guard check...")

            # Get current capital state
            capital_state = self.get_current_capital_state()
            current_capital = capital_state.get("capital_effective", 0.0)

            # Check if there's a pending ramp request
            staged_capital = capital_state.get("capital_staged")
            if not staged_capital or staged_capital <= current_capital:
                logger.info(
                    "ℹ️ No ramp request pending or already at/above staged level"
                )
                return {
                    "status": "no_action_needed",
                    "current_capital": current_capital,
                    "staged_capital": staged_capital,
                    "reason": "No ramp request pending",
                }

            # Check minimum time between ramps
            hours_since_last = capital_state.get("hours_since_last_ramp", 999)
            if hours_since_last < self.config["min_hours_between_ramps"]:
                reason = f"Too soon since last ramp ({hours_since_last:.1f}h < {self.config['min_hours_between_ramps']}h)"
                self.deny_ramp_request(reason, current_capital)
                self.send_slack_notification(f"🛡️ Capital ramp DENIED: {reason}")

                return {
                    "status": "denied",
                    "reason": reason,
                    "current_capital": current_capital,
                    "staged_capital": staged_capital,
                }

            # Check consecutive green mornings
            green_check = self.check_consecutive_greens()

            # Check reconciliation breaches
            recon_check = self.check_recon_breaches()

            # Determine if ramp should be approved
            can_ramp = green_check.get("meets_requirement", False) and recon_check.get(
                "recon_clean", False
            )

            guard_result = {
                "status": "approved" if can_ramp else "denied",
                "current_capital": current_capital,
                "staged_capital": staged_capital,
                "green_check": green_check,
                "recon_check": recon_check,
                "duration": time.time() - guard_start,
                "timestamp": time.time(),
            }

            if can_ramp:
                # Approve the ramp
                if self.approve_ramp_request(staged_capital):
                    guard_result["action"] = "approved_ramp"
                    self.send_slack_notification(
                        f"🟢 Capital ramp APPROVED: {current_capital:.0%} → {staged_capital:.0%}\n"
                        f"✅ {green_check['consecutive_greens']} consecutive green mornings\n"
                        f"✅ {recon_check['total_issues']} reconciliation issues"
                    )
                    logger.info(
                        f"✅ Capital ramp approved: {current_capital:.0%} → {staged_capital:.0%}"
                    )
                else:
                    guard_result["status"] = "error"
                    guard_result["action"] = "approval_failed"
            else:
                # Deny the ramp
                reasons = []
                if not green_check.get("meets_requirement", False):
                    reasons.append(
                        f"Only {green_check.get('consecutive_greens', 0)}/2 consecutive green mornings"
                    )
                if not recon_check.get("recon_clean", False):
                    reasons.append(
                        f"{recon_check.get('total_issues', 0)} reconciliation issues"
                    )

                reason = "; ".join(reasons)
                self.deny_ramp_request(reason, current_capital)
                guard_result["action"] = "denied_ramp"
                guard_result["reason"] = reason

                self.send_slack_notification(
                    f"🔴 Capital ramp DENIED: {current_capital:.0%} ↛ {staged_capital:.0%}\n"
                    f"Reason: {reason}"
                )
                logger.warning(f"❌ Capital ramp denied: {reason}")

            return guard_result

        except Exception as e:
            logger.error(f"Error in ramp guard check: {e}")
            return {"status": "error", "error": str(e), "timestamp": time.time()}


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Capital Ramp Guard")
    parser.add_argument(
        "--run", action="store_true", help="Run capital ramp guard check"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show current guard status"
    )
    parser.add_argument(
        "--history", action="store_true", help="Show greenlight history"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    guard = CapitalRampGuard()

    if args.history:
        history = guard.get_recent_greenlight_history()
        if args.json:
            print(json.dumps(history, indent=2, default=str))
        else:
            print("📊 Recent Greenlight History:")
            for date, data in sorted(history.items(), reverse=True):
                status = data.get("status", "UNKNOWN")
                emoji = (
                    "✅"
                    if status == "GREEN"
                    else ("❌" if status in ["RED", "YELLOW"] else "❓")
                )
                print(f"  {emoji} {date}: {status}")
        return

    if args.status:
        capital_state = guard.get_current_capital_state()
        green_check = guard.check_consecutive_greens()
        recon_check = guard.check_recon_breaches()

        status = {
            "capital_state": capital_state,
            "green_check": green_check,
            "recon_check": recon_check,
            "timestamp": time.time(),
        }

        if args.json:
            print(json.dumps(status, indent=2, default=str))
        else:
            print(f"🛡️ Capital Ramp Guard Status:")
            print(f"  Current Capital: {capital_state.get('capital_effective', 0):.0%}")
            print(f"  Staged Capital: {capital_state.get('capital_staged', 'None')}")
            print(f"  Consecutive Greens: {green_check.get('consecutive_greens', 0)}/2")
            print(
                f"  Recon Clean: {'✅' if recon_check.get('recon_clean', False) else '❌'}"
            )
        return

    if args.run or not sys.argv[1:]:  # Default to run
        result = guard.run_ramp_guard_check()

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            status = result["status"]
            emoji = (
                "✅" if status == "approved" else ("❌" if status == "denied" else "❓")
            )
            print(f"{emoji} Capital Ramp Guard: {status.upper()}")

            if "reason" in result:
                print(f"  Reason: {result['reason']}")

        # Exit code based on status
        if result["status"] == "approved":
            sys.exit(0)
        elif result["status"] == "no_action_needed":
            sys.exit(0)
        else:
            sys.exit(1)

    parser.print_help()


if __name__ == "__main__":
    main()
