#!/usr/bin/env python3
"""
Morning Greenlight Check
Daily green light check at market open with single ✅/❌ Slack card
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple

import redis
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("morning_greenlight")


class MorningGreenlight:
    """Daily morning system health check and greenlight."""

    def __init__(self):
        """Initialize morning greenlight checker."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.slack_webhook = os.getenv("SLACK_WEBHOOK")

        # Greenlight criteria thresholds
        self.criteria = {
            "system_mode": {"expected": "auto", "weight": 0.25},
            "recon_clean": {"max_breaches": 0, "weight": 0.20},
            "entropy_healthy": {"min_entropy": 0.7, "weight": 0.15},
            "q_spread_healthy": {"max_q_spread": 0.05, "weight": 0.15},
            "hedge_ready": {"min_hedge_ratio": 0.8, "weight": 0.10},
            "slo_tier": {"min_tier": "B", "weight": 0.15},
        }

        logger.info("🌅 Morning Greenlight initialized")

    def check_system_mode(self) -> Dict[str, Any]:
        """Check system is not in halt mode."""
        try:
            mode = self.redis.get("mode") or "unknown"
            is_healthy = mode == self.criteria["system_mode"]["expected"]

            return {
                "name": "System Mode",
                "status": "PASS" if is_healthy else "FAIL",
                "value": mode,
                "expected": self.criteria["system_mode"]["expected"],
                "weight": self.criteria["system_mode"]["weight"],
                "details": f"Mode: {mode}",
            }

        except Exception as e:
            return {
                "name": "System Mode",
                "status": "ERROR",
                "error": str(e),
                "weight": self.criteria["system_mode"]["weight"],
            }

    def check_reconciliation(self) -> Dict[str, Any]:
        """Check no reconciliation breaches or position mismatches."""
        try:
            breaches_24h = int(self.redis.get("recon:breaches_24h") or 0)
            position_mismatches = int(self.redis.get("recon:position_mismatches") or 0)

            total_issues = breaches_24h + position_mismatches
            is_healthy = total_issues <= self.criteria["recon_clean"]["max_breaches"]

            return {
                "name": "Reconciliation",
                "status": "PASS" if is_healthy else "FAIL",
                "value": total_issues,
                "expected": f"≤ {self.criteria['recon_clean']['max_breaches']}",
                "weight": self.criteria["recon_clean"]["weight"],
                "details": f"24h breaches: {breaches_24h}, pos mismatches: {position_mismatches}",
            }

        except Exception as e:
            return {
                "name": "Reconciliation",
                "status": "ERROR",
                "error": str(e),
                "weight": self.criteria["recon_clean"]["weight"],
            }

    def check_entropy_health(self) -> Dict[str, Any]:
        """Check model entropy is healthy (not collapsed)."""
        try:
            # Get latest entropy from RL model
            entropy = float(self.redis.get("rl:entropy") or 0.8)
            is_healthy = entropy >= self.criteria["entropy_healthy"]["min_entropy"]

            return {
                "name": "Model Entropy",
                "status": "PASS" if is_healthy else "FAIL",
                "value": f"{entropy:.3f}",
                "expected": f"≥ {self.criteria['entropy_healthy']['min_entropy']}",
                "weight": self.criteria["entropy_healthy"]["weight"],
                "details": f"Current entropy: {entropy:.3f}",
            }

        except Exception as e:
            return {
                "name": "Model Entropy",
                "status": "ERROR",
                "error": str(e),
                "weight": self.criteria["entropy_healthy"]["weight"],
            }

    def check_q_spread(self) -> Dict[str, Any]:
        """Check Q-value spread is not diverged."""
        try:
            # Get Q-spread metric from RL model
            q_spread = float(self.redis.get("rl:q_spread") or 0.02)
            is_healthy = q_spread <= self.criteria["q_spread_healthy"]["max_q_spread"]

            return {
                "name": "Q-Value Spread",
                "status": "PASS" if is_healthy else "FAIL",
                "value": f"{q_spread:.4f}",
                "expected": f"≤ {self.criteria['q_spread_healthy']['max_q_spread']}",
                "weight": self.criteria["q_spread_healthy"]["weight"],
                "details": f"Current Q-spread: {q_spread:.4f}",
            }

        except Exception as e:
            return {
                "name": "Q-Value Spread",
                "status": "ERROR",
                "error": str(e),
                "weight": self.criteria["q_spread_healthy"]["weight"],
            }

    def check_hedge_readiness(self) -> Dict[str, Any]:
        """Check hedge ratios are ready and calibrated."""
        try:
            # Check if hedge system is enabled
            hedge_enabled = bool(int(self.redis.get("HEDGE_ENABLED") or 1))

            # Get average hedge ratio across symbols
            symbols = ["BTC", "ETH", "SOL"]
            hedge_ratios = []

            for symbol in symbols:
                calib_data = self.redis.hgetall(f"basis:calib:{symbol}")
                if calib_data and "beta" in calib_data:
                    hedge_ratios.append(float(calib_data["beta"]))

            if hedge_ratios:
                avg_hedge_ratio = sum(hedge_ratios) / len(hedge_ratios)
                is_healthy = (
                    hedge_enabled
                    and avg_hedge_ratio
                    >= self.criteria["hedge_ready"]["min_hedge_ratio"]
                )
            else:
                avg_hedge_ratio = 0.0
                is_healthy = False

            return {
                "name": "Hedge System",
                "status": "PASS" if is_healthy else "FAIL",
                "value": f"{avg_hedge_ratio:.3f}",
                "expected": f"≥ {self.criteria['hedge_ready']['min_hedge_ratio']} & enabled",
                "weight": self.criteria["hedge_ready"]["weight"],
                "details": f"Enabled: {hedge_enabled}, Avg ratio: {avg_hedge_ratio:.3f}",
            }

        except Exception as e:
            return {
                "name": "Hedge System",
                "status": "ERROR",
                "error": str(e),
                "weight": self.criteria["hedge_ready"]["weight"],
            }

    def check_slo_tier(self) -> Dict[str, Any]:
        """Check SLO tier is B+ (not C)."""
        try:
            slo_tier = self.redis.get("slo:tier") or "A"

            # Convert tier to numeric for comparison
            tier_values = {"A": 3, "B": 2, "C": 1}
            current_value = tier_values.get(slo_tier, 0)
            min_value = tier_values.get(self.criteria["slo_tier"]["min_tier"], 2)

            is_healthy = current_value >= min_value

            return {
                "name": "SLO Tier",
                "status": "PASS" if is_healthy else "FAIL",
                "value": slo_tier,
                "expected": f"≥ {self.criteria['slo_tier']['min_tier']}",
                "weight": self.criteria["slo_tier"]["weight"],
                "details": f"Current tier: {slo_tier}",
            }

        except Exception as e:
            return {
                "name": "SLO Tier",
                "status": "ERROR",
                "error": str(e),
                "weight": self.criteria["slo_tier"]["weight"],
            }

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all morning greenlight checks."""
        try:
            check_start = time.time()
            logger.info("🌅 Running morning greenlight checks...")

            # Run all checks
            checks = [
                self.check_system_mode(),
                self.check_reconciliation(),
                self.check_entropy_health(),
                self.check_q_spread(),
                self.check_hedge_readiness(),
                self.check_slo_tier(),
            ]

            # Calculate weighted score
            total_weight = 0.0
            weighted_score = 0.0

            pass_count = 0
            fail_count = 0
            error_count = 0

            for check in checks:
                weight = check.get("weight", 0)
                total_weight += weight

                if check["status"] == "PASS":
                    weighted_score += weight
                    pass_count += 1
                elif check["status"] == "FAIL":
                    fail_count += 1
                else:  # ERROR
                    error_count += 1

            # Calculate final score
            final_score = weighted_score / total_weight if total_weight > 0 else 0.0

            # Determine overall status
            if error_count > 0:
                overall_status = "ERROR"
                status_emoji = "⚠️"
            elif final_score >= 0.8:  # 80% threshold for green light
                overall_status = "GREEN"
                status_emoji = "✅"
            elif final_score >= 0.6:  # 60% threshold for yellow
                overall_status = "YELLOW"
                status_emoji = "⚠️"
            else:
                overall_status = "RED"
                status_emoji = "❌"

            check_duration = time.time() - check_start

            result = {
                "timestamp": time.time(),
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "overall_status": overall_status,
                "status_emoji": status_emoji,
                "weighted_score": final_score,
                "checks": {
                    "total": len(checks),
                    "passed": pass_count,
                    "failed": fail_count,
                    "errors": error_count,
                },
                "check_results": checks,
                "duration": check_duration,
            }

            logger.info(
                f"🎯 Morning greenlight: {overall_status} "
                f"(score: {final_score:.1%}, {pass_count}/{len(checks)} passed)"
            )

            return result

        except Exception as e:
            logger.error(f"Error in morning greenlight checks: {e}")
            return {
                "timestamp": time.time(),
                "overall_status": "ERROR",
                "status_emoji": "💥",
                "error": str(e),
            }

    def create_slack_card(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create Slack card with greenlight status."""
        try:
            status = result["overall_status"]
            emoji = result["status_emoji"]
            score = result.get("weighted_score", 0)

            # Main status text
            if status == "GREEN":
                main_text = f"{emoji} *MORNING GREENLIGHT: ALL SYSTEMS GO*"
                color = "#36a64f"  # Green
            elif status == "YELLOW":
                main_text = f"{emoji} *MORNING GREENLIGHT: CAUTION*"
                color = "#ffcc00"  # Yellow
            else:
                main_text = f"{emoji} *MORNING GREENLIGHT: ISSUES DETECTED*"
                color = "#ff0000"  # Red

            # Summary stats
            checks = result.get("checks", {})
            summary_text = f"""*System Health Score:* {score:.1%}
*Checks:* {checks.get('passed', 0)} passed, {checks.get('failed', 0)} failed, {checks.get('errors', 0)} errors
*Date:* {result.get('date', 'unknown')}"""

            # Detailed check results
            check_fields = []
            for check in result.get("check_results", []):
                status_icon = (
                    "✅"
                    if check["status"] == "PASS"
                    else ("❌" if check["status"] == "FAIL" else "⚠️")
                )
                value = check.get("value", "N/A")

                check_fields.append(
                    {
                        "type": "mrkdwn",
                        "text": f"{status_icon} *{check['name']}*\n{value}",
                    }
                )

            # Create Slack blocks
            blocks = [
                {"type": "section", "text": {"type": "mrkdwn", "text": main_text}},
                {"type": "section", "text": {"type": "mrkdwn", "text": summary_text}},
            ]

            # Add check results in groups of 2 (Slack limit)
            for i in range(0, len(check_fields), 2):
                fields_group = check_fields[i : i + 2]
                blocks.append({"type": "section", "fields": fields_group})

            # Add action buttons based on status
            if status != "GREEN":
                blocks.append(
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "📊 View Details",
                                },
                                "action_id": "greenlight_details",
                                "value": "details",
                            },
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "🚨 Emergency Response",
                                },
                                "style": "danger",
                                "action_id": "emergency_response",
                                "value": "emergency",
                            },
                        ],
                    }
                )

            return {
                "text": main_text,
                "blocks": blocks,
                "attachments": [
                    {"color": color, "fallback": f"Morning Greenlight: {status}"}
                ],
            }

        except Exception as e:
            logger.error(f"Error creating Slack card: {e}")
            return {
                "text": f"💥 Morning Greenlight Error: {e}",
                "attachments": [{"color": "#ff0000"}],
            }

    def send_greenlight_report(self, result: Dict[str, Any]) -> bool:
        """Send greenlight report to Slack."""
        try:
            if not self.slack_webhook:
                logger.warning("No Slack webhook configured")
                return False

            # Create Slack card
            slack_card = self.create_slack_card(result)

            # Send to Slack
            response = requests.post(self.slack_webhook, json=slack_card, timeout=10)
            response.raise_for_status()

            logger.info("📱 Sent morning greenlight report to Slack")
            return True

        except Exception as e:
            logger.error(f"Error sending Slack report: {e}")
            return False

    def save_daily_report(self, result: Dict[str, Any]) -> str:
        """Save daily report to file."""
        try:
            # Create reports directory
            reports_dir = (
                Path(__file__).parent.parent / "reports" / "morning_greenlight"
            )
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Generate report filename
            date_str = result.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            report_file = reports_dir / f"greenlight_{date_str}.json"

            # Save result as JSON
            with open(report_file, "w") as f:
                json.dump(result, f, indent=2, default=str)

            logger.info(f"💾 Saved daily report: {report_file}")
            return str(report_file)

        except Exception as e:
            logger.error(f"Error saving daily report: {e}")
            return ""

    def run_morning_greenlight(self) -> Dict[str, Any]:
        """Run complete morning greenlight check."""
        try:
            logger.info("🌅 Starting morning greenlight check")

            # Run all checks
            result = self.run_all_checks()

            # Save daily report
            report_file = self.save_daily_report(result)
            result["report_file"] = report_file

            # Send Slack notification
            slack_sent = self.send_greenlight_report(result)
            result["slack_sent"] = slack_sent

            # Store result in Redis for other systems
            self.redis.set("greenlight:latest", json.dumps(result, default=str))
            self.redis.set("greenlight:last_run", int(time.time()))

            status = result["overall_status"]
            logger.info(f"✅ Morning greenlight completed: {status}")

            return result

        except Exception as e:
            logger.error(f"Error in morning greenlight: {e}")
            return {
                "timestamp": time.time(),
                "overall_status": "ERROR",
                "status_emoji": "💥",
                "error": str(e),
            }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Morning Greenlight Check")
    parser.add_argument(
        "--run", action="store_true", help="Run morning greenlight check"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    greenlight = MorningGreenlight()

    if args.run or not sys.argv[1:]:  # Default to run
        result = greenlight.run_morning_greenlight()

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            status = result["overall_status"]
            print(f"{result.get('status_emoji', '❓')} Morning Greenlight: {status}")

        # Exit code based on status
        if result["overall_status"] == "GREEN":
            sys.exit(0)
        elif result["overall_status"] == "YELLOW":
            sys.exit(1)  # Warning
        else:
            sys.exit(2)  # Error/Red

    parser.print_help()


if __name__ == "__main__":
    main()
