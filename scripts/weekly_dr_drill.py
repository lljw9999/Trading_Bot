#!/usr/bin/env python3
"""
Weekly DR Drill
Automated weekly disaster recovery drill with RTO/RPO reporting
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("weekly_dr_drill")


class WeeklyDRDrill:
    """Manages weekly disaster recovery drills and reporting."""

    def __init__(self):
        """Initialize weekly DR drill."""
        self.slack_webhook = os.getenv("SLACK_WEBHOOK")

        # DR drill configuration
        self.config = {
            "drill_day": "saturday",  # Run on Saturday nights
            "drill_hour": 2,  # 2 AM local time
            "max_rto_minutes": 30,  # SLA: 30 minutes RTO
            "max_rpo_minutes": 60,  # SLA: 60 minutes RPO
            "max_downtime_minutes": 45,  # SLA: 45 minutes max downtime
            "reports_retention_days": 90,
        }

        # Initialize reports directory
        self.reports_dir = Path(__file__).parent.parent / "reports" / "dr_drills"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("📅 Weekly DR Drill initialized")

    def should_run_drill(self) -> bool:
        """Check if drill should run today."""
        try:
            now = datetime.now()

            # Check if it's the right day
            current_day = now.strftime("%A").lower()
            if current_day != self.config["drill_day"]:
                logger.info(
                    f"Not drill day (today: {current_day}, drill day: {self.config['drill_day']})"
                )
                return False

            # Check if it's the right hour (within 1 hour window)
            current_hour = now.hour
            drill_hour = self.config["drill_hour"]

            if not (drill_hour <= current_hour < drill_hour + 1):
                logger.info(
                    f"Not drill time (current: {current_hour}h, drill: {drill_hour}h)"
                )
                return False

            # Check if drill already ran this week
            last_drill_file = self.reports_dir / "last_drill_date.txt"
            if last_drill_file.exists():
                with open(last_drill_file, "r") as f:
                    last_drill_date = f.read().strip()

                try:
                    last_drill = datetime.strptime(last_drill_date, "%Y-%m-%d")
                    days_since = (now.date() - last_drill.date()).days

                    if days_since < 7:
                        logger.info(
                            f"Drill already ran this week ({days_since} days ago)"
                        )
                        return False
                except ValueError:
                    logger.warning("Invalid last drill date format")

            return True

        except Exception as e:
            logger.error(f"Error checking if drill should run: {e}")
            return False

    def record_drill_execution(self):
        """Record that drill was executed today."""
        try:
            last_drill_file = self.reports_dir / "last_drill_date.txt"
            with open(last_drill_file, "w") as f:
                f.write(datetime.now().strftime("%Y-%m-%d"))
        except Exception as e:
            logger.error(f"Error recording drill execution: {e}")

    def run_restore_drill(self) -> Dict[str, Any]:
        """Run the disaster recovery restore drill."""
        try:
            logger.info("🔄 Running restore drill...")

            # Import and run restore script
            import subprocess

            result = subprocess.run(
                ["python3", "scripts/restore_from_s3.py", "--dry-run"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            if result.returncode == 0:
                # Parse output for metrics (if JSON format)
                try:
                    output_lines = result.stdout.strip().split("\n")
                    for line in reversed(output_lines):
                        if line.startswith("{"):
                            drill_data = json.loads(line)
                            break
                    else:
                        # Fallback if no JSON output
                        drill_data = {"status": "completed", "dry_run": True}
                except json.JSONDecodeError:
                    drill_data = {"status": "completed", "dry_run": True}

                return {
                    "drill_type": "restore",
                    "status": "success",
                    "data": drill_data,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            else:
                return {
                    "drill_type": "restore",
                    "status": "failed",
                    "error": result.stderr,
                    "stdout": result.stdout,
                }

        except Exception as e:
            logger.error(f"Error running restore drill: {e}")
            return {"drill_type": "restore", "status": "error", "error": str(e)}

    def run_game_day_drill(self) -> Dict[str, Any]:
        """Run the game day drill (dry run only for weekly)."""
        try:
            logger.info("🎮 Running game day drill (dry run)...")

            import subprocess

            result = subprocess.run(
                ["python3", "scripts/dr_game_day.py", "--dry-run"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            if result.returncode == 0:
                try:
                    output_lines = result.stdout.strip().split("\n")
                    for line in reversed(output_lines):
                        if line.startswith("{"):
                            drill_data = json.loads(line)
                            break
                    else:
                        drill_data = {"status": "completed", "dry_run": True}
                except json.JSONDecodeError:
                    drill_data = {"status": "completed", "dry_run": True}

                return {
                    "drill_type": "game_day",
                    "status": "success",
                    "data": drill_data,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            else:
                return {
                    "drill_type": "game_day",
                    "status": "failed",
                    "error": result.stderr,
                    "stdout": result.stdout,
                }

        except Exception as e:
            logger.error(f"Error running game day drill: {e}")
            return {"drill_type": "game_day", "status": "error", "error": str(e)}

    def analyze_drill_results(
        self, restore_result: Dict, game_day_result: Dict
    ) -> Dict[str, Any]:
        """Analyze drill results and calculate metrics."""
        try:
            analysis = {
                "timestamp": time.time(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "overall_status": "pass",
                "drills_run": 2,
                "drills_passed": 0,
                "drills_failed": 0,
                "metrics": {},
                "sla_compliance": {},
                "recommendations": [],
            }

            # Analyze restore drill
            if restore_result["status"] == "success":
                analysis["drills_passed"] += 1

                # Extract metrics from restore data
                restore_data = restore_result.get("data", {})
                metrics = restore_data.get("metrics", {})

                rto_minutes = metrics.get("rto_minutes", 0)
                rpo_minutes = metrics.get("rpo_minutes", 0)

                analysis["metrics"]["restore_rto_minutes"] = rto_minutes
                analysis["metrics"]["restore_rpo_minutes"] = rpo_minutes

                # Check SLA compliance
                analysis["sla_compliance"]["rto_compliant"] = (
                    rto_minutes <= self.config["max_rto_minutes"]
                )
                analysis["sla_compliance"]["rpo_compliant"] = (
                    rpo_minutes <= self.config["max_rpo_minutes"]
                )

                if not analysis["sla_compliance"]["rto_compliant"]:
                    analysis["recommendations"].append(
                        f"RTO exceeded SLA: {rto_minutes:.1f}min > {self.config['max_rto_minutes']}min"
                    )
                    analysis["overall_status"] = "warning"

                if not analysis["sla_compliance"]["rpo_compliant"]:
                    analysis["recommendations"].append(
                        f"RPO exceeded SLA: {rpo_minutes:.0f}min > {self.config['max_rpo_minutes']}min"
                    )
                    analysis["overall_status"] = "warning"

            else:
                analysis["drills_failed"] += 1
                analysis["overall_status"] = "fail"
                analysis["recommendations"].append(
                    "Restore drill failed - investigate backup and restore process"
                )

            # Analyze game day drill
            if game_day_result["status"] == "success":
                analysis["drills_passed"] += 1

                game_day_data = game_day_result.get("data", {})
                metrics = game_day_data.get("metrics", {})

                downtime_minutes = metrics.get("total_downtime_minutes", 0)
                analysis["metrics"]["game_day_downtime_minutes"] = downtime_minutes

                analysis["sla_compliance"]["downtime_compliant"] = (
                    downtime_minutes <= self.config["max_downtime_minutes"]
                )

                if not analysis["sla_compliance"]["downtime_compliant"]:
                    analysis["recommendations"].append(
                        f"Downtime exceeded SLA: {downtime_minutes:.1f}min > {self.config['max_downtime_minutes']}min"
                    )
                    analysis["overall_status"] = "warning"

            else:
                analysis["drills_failed"] += 1
                analysis["overall_status"] = "fail"
                analysis["recommendations"].append(
                    "Game day drill failed - investigate failover process"
                )

            # Overall status
            if analysis["drills_failed"] > 0:
                analysis["overall_status"] = "fail"
            elif not analysis["recommendations"]:
                analysis["overall_status"] = "pass"

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing drill results: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }

    def generate_drill_report(
        self, restore_result: Dict, game_day_result: Dict, analysis: Dict
    ) -> str:
        """Generate comprehensive drill report."""
        try:
            report_date = datetime.now().strftime("%Y-%m-%d")

            # Status indicators
            status_emoji = {"pass": "🟢", "warning": "🟡", "fail": "🔴", "error": "💥"}

            overall_emoji = status_emoji.get(analysis["overall_status"], "❓")

            report = f"""# Weekly DR Drill Report

**Date:** {report_date}
**Overall Status:** {overall_emoji} {analysis["overall_status"].upper()}
**Drills Passed:** {analysis["drills_passed"]}/{analysis["drills_run"]}

## Executive Summary

| Metric | Value | SLA | Status |
|--------|-------|-----|---------|"""

            metrics = analysis.get("metrics", {})
            sla_compliance = analysis.get("sla_compliance", {})

            if "restore_rto_minutes" in metrics:
                rto_status = (
                    "✅" if sla_compliance.get("rto_compliant", False) else "❌"
                )
                report += f"\n| **RTO (Restore)** | {metrics['restore_rto_minutes']:.1f} min | ≤ {self.config['max_rto_minutes']} min | {rto_status} |"

            if "restore_rpo_minutes" in metrics:
                rpo_status = (
                    "✅" if sla_compliance.get("rpo_compliant", False) else "❌"
                )
                report += f"\n| **RPO (Restore)** | {metrics['restore_rpo_minutes']:.0f} min | ≤ {self.config['max_rpo_minutes']} min | {rpo_status} |"

            if "game_day_downtime_minutes" in metrics:
                downtime_status = (
                    "✅" if sla_compliance.get("downtime_compliant", False) else "❌"
                )
                report += f"\n| **Downtime (Game Day)** | {metrics['game_day_downtime_minutes']:.1f} min | ≤ {self.config['max_downtime_minutes']} min | {downtime_status} |"

            # Drill results
            report += "\n\n## Drill Results\n\n"

            # Restore drill
            restore_emoji = "✅" if restore_result["status"] == "success" else "❌"
            report += f"### {restore_emoji} Restore Drill\n"
            report += f"**Status:** {restore_result['status'].upper()}\n"

            if restore_result["status"] == "success":
                restore_data = restore_result.get("data", {})
                if "metrics" in restore_data:
                    restore_metrics = restore_data["metrics"]
                    report += f"- Components restored: {', '.join(restore_metrics.get('components_restored', []))}\n"
                    report += f"- Components failed: {', '.join(restore_metrics.get('components_failed', [])) or 'None'}\n"
            else:
                report += f"- Error: {restore_result.get('error', 'Unknown error')}\n"

            report += "\n"

            # Game day drill
            game_day_emoji = "✅" if game_day_result["status"] == "success" else "❌"
            report += f"### {game_day_emoji} Game Day Drill (Dry Run)\n"
            report += f"**Status:** {game_day_result['status'].upper()}\n"

            if game_day_result["status"] == "success":
                game_day_data = game_day_result.get("data", {})
                phases = game_day_data.get("phases", [])
                report += f"- Phases completed: {', '.join(phases)}\n"
                if "health_checks" in game_day_data:
                    health_checks = game_day_data["health_checks"]
                    passed = sum(health_checks.values())
                    total = len(health_checks)
                    report += f"- Health checks: {passed}/{total} passed\n"
            else:
                report += f"- Error: {game_day_result.get('error', 'Unknown error')}\n"

            # Recommendations
            report += "\n## Recommendations\n\n"

            if analysis["recommendations"]:
                for i, rec in enumerate(analysis["recommendations"], 1):
                    report += f"{i}. {rec}\n"
            else:
                report += "🎉 All drills passed within SLA requirements. No immediate action needed.\n"

            # Next actions
            report += "\n## Next Actions\n\n"

            if analysis["overall_status"] == "pass":
                report += "- Continue weekly automated drills\n"
                report += "- Monitor trending metrics for degradation\n"
                report += "- Schedule quarterly full game day drill\n"
            else:
                report += "- **IMMEDIATE:** Address failed drills before next production deployment\n"
                report += "- Review and update DR procedures\n"
                report += "- Consider additional backup strategies\n"

            report += f"""
---

*Report generated by Weekly DR Drill*
*Next drill: {(datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")}*
*SLA Targets: RTO ≤ {self.config['max_rto_minutes']}min, RPO ≤ {self.config['max_rpo_minutes']}min, Downtime ≤ {self.config['max_downtime_minutes']}min*
"""

            return report

        except Exception as e:
            logger.error(f"Error generating drill report: {e}")
            return f"# Weekly DR Drill Report Error\n\nError generating report: {e}\n"

    def save_drill_report(self, report_content: str) -> str:
        """Save drill report to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.reports_dir / f"weekly_dr_drill_{timestamp}.md"

            with open(report_file, "w") as f:
                f.write(report_content)

            logger.info(f"💾 Saved drill report: {report_file}")
            return str(report_file)

        except Exception as e:
            logger.error(f"Error saving drill report: {e}")
            return ""

    def send_drill_summary(self, analysis: Dict[str, Any], report_file: str):
        """Send drill summary to Slack."""
        try:
            if not self.slack_webhook:
                logger.warning("No Slack webhook configured")
                return

            status = analysis["overall_status"]
            emoji = {"pass": "🟢", "warning": "🟡", "fail": "🔴", "error": "💥"}.get(
                status, "❓"
            )

            metrics = analysis.get("metrics", {})
            rto = metrics.get("restore_rto_minutes", 0)
            rpo = metrics.get("restore_rpo_minutes", 0)

            message = f"""{emoji} **Weekly DR Drill Summary**

**Status:** {status.upper()}
**Date:** {analysis.get('date', 'unknown')}

**Metrics:**
• RTO: {rto:.1f} min (SLA: ≤{self.config['max_rto_minutes']} min)
• RPO: {rpo:.0f} min (SLA: ≤{self.config['max_rpo_minutes']} min)

**Drills:** {analysis['drills_passed']}/{analysis['drills_run']} passed"""

            if analysis["recommendations"]:
                message += f"\n\n**Action Required:**\n• " + "\n• ".join(
                    analysis["recommendations"][:3]
                )

            payload = {
                "text": message,
                "username": "Weekly DR Drill",
                "icon_emoji": ":calendar:",
                "attachments": [
                    {
                        "color": (
                            "#36a64f"
                            if status == "pass"
                            else "#ffcc00" if status == "warning" else "#dc2626"
                        )
                    }
                ],
            }

            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("📱 Sent drill summary to Slack")

        except Exception as e:
            logger.error(f"Error sending drill summary: {e}")

    def cleanup_old_reports(self):
        """Clean up old drill reports."""
        try:
            cutoff_date = datetime.now() - timedelta(
                days=self.config["reports_retention_days"]
            )

            for report_file in self.reports_dir.glob("weekly_dr_drill_*.md"):
                if report_file.stat().st_mtime < cutoff_date.timestamp():
                    report_file.unlink()
                    logger.info(f"🗑️ Cleaned up old report: {report_file.name}")

        except Exception as e:
            logger.error(f"Error cleaning up old reports: {e}")

    def run_weekly_drill(self, force: bool = False) -> Dict[str, Any]:
        """Run complete weekly DR drill."""
        try:
            logger.info("📅 Starting weekly DR drill...")

            # Check if drill should run
            if not force and not self.should_run_drill():
                return {
                    "status": "skipped",
                    "reason": "Not scheduled to run",
                    "timestamp": time.time(),
                }

            drill_start = time.time()

            # Run drills
            logger.info("🔄 Running restore drill...")
            restore_result = self.run_restore_drill()

            logger.info("🎮 Running game day drill...")
            game_day_result = self.run_game_day_drill()

            # Analyze results
            logger.info("📊 Analyzing drill results...")
            analysis = self.analyze_drill_results(restore_result, game_day_result)

            # Generate report
            logger.info("📝 Generating drill report...")
            report_content = self.generate_drill_report(
                restore_result, game_day_result, analysis
            )
            report_file = self.save_drill_report(report_content)

            # Send summary
            self.send_drill_summary(analysis, report_file)

            # Record execution
            self.record_drill_execution()

            # Cleanup old reports
            self.cleanup_old_reports()

            drill_duration = time.time() - drill_start

            result = {
                "status": "completed",
                "drill_duration_minutes": drill_duration / 60,
                "overall_status": analysis["overall_status"],
                "drills_passed": analysis["drills_passed"],
                "drills_total": analysis["drills_run"],
                "report_file": report_file,
                "restore_result": restore_result,
                "game_day_result": game_day_result,
                "analysis": analysis,
                "timestamp": time.time(),
            }

            logger.info(
                f"📅 Weekly DR drill completed: {analysis['overall_status']} "
                f"({analysis['drills_passed']}/{analysis['drills_run']} passed in {drill_duration/60:.1f}min)"
            )

            return result

        except Exception as e:
            logger.error(f"Error in weekly DR drill: {e}")
            return {"status": "error", "error": str(e), "timestamp": time.time()}


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Weekly DR Drill")
    parser.add_argument("--run", action="store_true", help="Run weekly DR drill")
    parser.add_argument(
        "--force", action="store_true", help="Force run regardless of schedule"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    drill = WeeklyDRDrill()

    if args.run or args.force or not sys.argv[1:]:  # Default to run
        result = drill.run_weekly_drill(force=args.force)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            status = result["status"]
            emoji = (
                "✅"
                if status == "completed"
                else ("⏭️" if status == "skipped" else "❌")
            )

            print(f"{emoji} Weekly DR Drill: {status.upper()}")

            if status == "completed":
                overall = result["overall_status"]
                passed = result["drills_passed"]
                total = result["drills_total"]
                print(f"Overall: {overall.upper()} ({passed}/{total} drills passed)")

                if result["report_file"]:
                    print(f"Report: {result['report_file']}")
            elif status == "skipped":
                print(f"Reason: {result['reason']}")

        # Exit code based on drill results
        if result["status"] == "completed":
            if result["overall_status"] == "pass":
                sys.exit(0)
            else:
                sys.exit(1)  # Warning or fail
        elif result["status"] == "skipped":
            sys.exit(0)
        else:
            sys.exit(1)  # Error

    parser.print_help()


if __name__ == "__main__":
    main()
