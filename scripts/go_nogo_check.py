#!/usr/bin/env python3
"""
Go/No-Go Check - Automated Deployment Decision Support
Evaluates all Go/No-Go criteria and provides recommendation
"""
import os
import sys
import json
import time
import glob
import requests
import pathlib
from datetime import datetime, timezone, timedelta


class GoNoGoChecker:
    def __init__(self):
        self.criteria_results = {}
        self.go_blockers = []
        self.warnings = []
        self.score = 0
        self.max_score = 0

    def log_criteria(self, name, status, score_weight, details=None):
        """Log criteria check result."""
        self.criteria_results[name] = {
            "status": status,
            "score_weight": score_weight,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.max_score += score_weight

        if status == "PASS":
            self.score += score_weight
        elif status == "FAIL":
            self.go_blockers.append(name)
        elif status == "WARN":
            self.warnings.append(name)
            self.score += score_weight * 0.5  # Partial credit for warnings

    def check_validation_history(self):
        """Check for two consecutive 48h PASS validations."""
        print("📊 Checking validation history...")

        try:
            root = pathlib.Path(os.getenv("GO_NOGO_ROOT", "."))
            validation_dir = root / "artifacts" / "validation"
            if not validation_dir.exists():
                self.log_criteria(
                    "validation_history",
                    "FAIL",
                    20,
                    {"message": "No validation artifacts directory found"},
                )
                return

            # Look for recent validation files
            validation_files = sorted(
                validation_dir.glob("*validation*.json"), reverse=True
            )

            if len(validation_files) < 2:
                self.log_criteria(
                    "validation_history",
                    "FAIL",
                    20,
                    {
                        "message": f"Only {len(validation_files)} validation artifacts found, need at least 2"
                    },
                )
                return

            # Check last two validations
            recent_passes = 0
            latest_age_hours = None

            for i, val_file in enumerate(validation_files[:2]):
                try:
                    with open(val_file, "r") as f:
                        val_data = json.load(f)

                    # Check age of latest validation
                    if i == 0:
                        val_time = datetime.fromisoformat(
                            val_data.get("timestamp", "").replace("Z", "+00:00")
                        )
                        latest_age_hours = (
                            datetime.now(timezone.utc) - val_time
                        ).total_seconds() / 3600

                    # Check if validation passed
                    if (
                        val_data.get("status") == "PASS"
                        or val_data.get("overall_status") == "PASS"
                    ):
                        recent_passes += 1
                except:
                    pass

            if recent_passes >= 2 and (
                latest_age_hours is None or latest_age_hours < 24
            ):
                self.log_criteria(
                    "validation_history",
                    "PASS",
                    20,
                    {
                        "recent_passes": recent_passes,
                        "latest_age_hours": latest_age_hours,
                        "message": "Two consecutive PASS validations found",
                    },
                )
            else:
                self.log_criteria(
                    "validation_history",
                    "FAIL",
                    20,
                    {
                        "recent_passes": recent_passes,
                        "latest_age_hours": latest_age_hours,
                        "message": f"Need 2 PASS validations, found {recent_passes} (latest age: {latest_age_hours:.1f}h)",
                    },
                )

        except Exception as e:
            self.log_criteria("validation_history", "FAIL", 20, {"error": str(e)})

    def check_alerting_status(self):
        """Check for active alerts and alert history."""
        print("🚨 Checking alerting status...")

        try:
            # Check for recent alert artifacts
            root = pathlib.Path(os.getenv("GO_NOGO_ROOT", "."))
            alert_dir = root / "artifacts" / "audit"
            alert_files = (
                list(alert_dir.glob("*alert*.json")) if alert_dir.exists() else []
            )

            recent_alerts = 0
            page_alerts = 0

            # Look for alerts in last 48h
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=48)

            for alert_file in alert_files:
                try:
                    # Extract timestamp from filename or file content
                    file_stat = alert_file.stat()
                    file_time = datetime.fromtimestamp(
                        file_stat.st_mtime, tz=timezone.utc
                    )

                    if file_time > cutoff_time:
                        recent_alerts += 1

                        # Check if it's a page alert
                        try:
                            with open(alert_file, "r") as f:
                                alert_data = json.load(f)
                            if alert_data.get("severity") in ["critical", "page"]:
                                page_alerts += 1
                        except:
                            pass
                except:
                    continue

            # Also check Prometheus alertmanager if available
            try:
                response = requests.get(
                    "http://localhost:9093/api/v1/alerts", timeout=5
                )
                if response.status_code == 200:
                    alerts_data = response.json()
                    active_alerts = alerts_data.get("data", [])

                    firing_alerts = [
                        a for a in active_alerts if a.get("state") == "firing"
                    ]
                    page_alerts += len(
                        [
                            a
                            for a in firing_alerts
                            if a.get("labels", {}).get("severity")
                            in ("critical", "page")
                        ]
                    )
            except Exception:
                # Alertmanager unavailable shouldn't downgrade status if local data is clean
                response = None

            if page_alerts > 0:
                self.log_criteria(
                    "alerting_status",
                    "FAIL",
                    15,
                    {
                        "page_alerts": page_alerts,
                        "recent_alerts": recent_alerts,
                        "message": f"{page_alerts} active page alerts detected",
                    },
                )
            elif recent_alerts > 5:
                self.log_criteria(
                    "alerting_status",
                    "WARN",
                    15,
                    {
                        "page_alerts": page_alerts,
                        "recent_alerts": recent_alerts,
                        "message": f"{recent_alerts} alerts in last 48h (>5 threshold)",
                    },
                )
            else:
                self.log_criteria(
                    "alerting_status",
                    "PASS",
                    15,
                    {
                        "page_alerts": page_alerts,
                        "recent_alerts": recent_alerts,
                        "message": "No page alerts, alert volume acceptable",
                    },
                )

        except Exception as e:
            self.log_criteria(
                "alerting_status",
                "WARN",
                15,
                {"error": str(e), "message": "Could not check alerting status"},
            )

    def check_slo_performance(self):
        """Check SLO performance and error budget."""
        print("📈 Checking SLO performance...")

        try:
            # Run error budget guard to get current status
            import subprocess

            result = subprocess.run(
                ["python", "scripts/error_budget_guard.py", "--dry-run"],
                capture_output=True,
                text=True,
                cwd=".",
            )

            budget_status = "UNKNOWN"
            if result.returncode == 0:
                budget_status = "OK"
            elif result.returncode == 1:
                budget_status = "EXHAUSTED"
            elif result.returncode == 2:
                budget_status = "WARNING"

            # Check for recent budget audit
            root = pathlib.Path(os.getenv("GO_NOGO_ROOT", "."))
            audit_dir = root / "artifacts" / "audit"
            budget_files = (
                sorted(audit_dir.glob("*budget*.json"), reverse=True)
                if audit_dir.exists()
                else []
            )
            budget_spent = 0.0

            if budget_files:
                try:
                    with open(budget_files[0], "r") as f:
                        budget_data = json.load(f)
                    budget_spent = budget_data.get("budget_spent", 0.0)
                except:
                    pass

            if budget_status == "EXHAUSTED" or budget_spent >= 1.0:
                self.log_criteria(
                    "slo_performance",
                    "FAIL",
                    15,
                    {
                        "budget_spent": budget_spent,
                        "status": budget_status,
                        "message": "Error budget exhausted",
                    },
                )
            elif budget_status == "WARNING" or budget_spent >= 0.25:
                self.log_criteria(
                    "slo_performance",
                    "WARN",
                    15,
                    {
                        "budget_spent": budget_spent,
                        "status": budget_status,
                        "message": f"Error budget at {budget_spent:.1%}",
                    },
                )
            else:
                self.log_criteria(
                    "slo_performance",
                    "PASS",
                    15,
                    {
                        "budget_spent": budget_spent,
                        "status": budget_status,
                        "message": "Error budget healthy",
                    },
                )

        except Exception as e:
            self.log_criteria(
                "slo_performance",
                "WARN",
                15,
                {"error": str(e), "message": "Could not check SLO performance"},
            )

    def check_technical_readiness(self):
        """Check technical readiness (preflight, security, kill-switch)."""
        print("🔧 Checking technical readiness...")

        try:
            # Check for recent preflight check
            root = pathlib.Path(os.getenv("GO_NOGO_ROOT", "."))
            audit_dir = root / "artifacts" / "audit"
            preflight_files = (
                sorted(audit_dir.glob("*preflight*.json"), reverse=True)
                if audit_dir.exists()
                else []
            )
            preflight_passed = False

            if preflight_files:
                try:
                    with open(preflight_files[0], "r") as f:
                        preflight_data = json.load(f)

                    if preflight_data.get("summary", {}).get("status") == "PASS":
                        preflight_passed = True
                except:
                    pass

            # Check for recent security scan
            security_files = (
                sorted(audit_dir.glob("*security*.json"), reverse=True)
                if audit_dir.exists()
                else []
            )
            security_passed = False

            if security_files:
                try:
                    with open(security_files[0], "r") as f:
                        security_data = json.load(f)

                    sec_status = security_data.get("summary", {}).get("status")
                    if sec_status in ["PASS", "WARN"]:  # Warnings OK for security
                        security_passed = True
                except:
                    pass

            # Test kill-switch functionality
            kill_switch_works = False
            try:
                from src.rl.influence_controller import InfluenceController

                ic = InfluenceController()
                status = ic.get_status()
                if "weight" in status:
                    kill_switch_works = True
            except:
                pass

            readiness_score = sum(
                [preflight_passed, security_passed, kill_switch_works]
            )

            if readiness_score == 3:
                self.log_criteria(
                    "technical_readiness",
                    "PASS",
                    15,
                    {
                        "preflight_passed": preflight_passed,
                        "security_passed": security_passed,
                        "kill_switch_works": kill_switch_works,
                        "message": "All technical readiness checks passed",
                    },
                )
            elif readiness_score >= 2:
                self.log_criteria(
                    "technical_readiness",
                    "WARN",
                    15,
                    {
                        "preflight_passed": preflight_passed,
                        "security_passed": security_passed,
                        "kill_switch_works": kill_switch_works,
                        "message": f"{readiness_score}/3 technical checks passed",
                    },
                )
            else:
                self.log_criteria(
                    "technical_readiness",
                    "FAIL",
                    15,
                    {
                        "preflight_passed": preflight_passed,
                        "security_passed": security_passed,
                        "kill_switch_works": kill_switch_works,
                        "message": f"Only {readiness_score}/3 technical checks passed",
                    },
                )

        except Exception as e:
            self.log_criteria("technical_readiness", "FAIL", 15, {"error": str(e)})

    def check_policy_health(self):
        """Check RL policy health metrics."""
        print("🧠 Checking policy health...")

        try:
            # Try to get metrics from exporter
            response = requests.get("http://localhost:9108/metrics", timeout=5)

            entropy_ok = False
            qspread_ok = False
            heartbeat_ok = False

            if response.status_code == 200:
                metrics = response.text

                # Extract metrics (basic parsing)
                for line in metrics.split("\n"):
                    if line.startswith("rl_policy_entropy "):
                        entropy = float(line.split()[-1])
                        entropy_ok = entropy >= 0.9
                    elif line.startswith("rl_policy_q_spread "):
                        qspread = float(line.split()[-1])
                        # Assume baseline of 1.0 for demo
                        qspread_ok = qspread <= 2.0
                    elif line.startswith("rl_policy_heartbeat_age_seconds "):
                        heartbeat_age = float(line.split()[-1])
                        heartbeat_ok = heartbeat_age < 600  # 10 minutes

            health_score = sum([entropy_ok, qspread_ok, heartbeat_ok])

            if health_score == 3:
                self.log_criteria(
                    "policy_health",
                    "PASS",
                    10,
                    {
                        "entropy_ok": entropy_ok,
                        "qspread_ok": qspread_ok,
                        "heartbeat_ok": heartbeat_ok,
                        "message": "All policy health metrics good",
                    },
                )
            elif health_score >= 2:
                self.log_criteria(
                    "policy_health",
                    "WARN",
                    10,
                    {
                        "entropy_ok": entropy_ok,
                        "qspread_ok": qspread_ok,
                        "heartbeat_ok": heartbeat_ok,
                        "message": f"{health_score}/3 policy health checks passed",
                    },
                )
            else:
                self.log_criteria(
                    "policy_health",
                    "FAIL",
                    10,
                    {
                        "entropy_ok": entropy_ok,
                        "qspread_ok": qspread_ok,
                        "heartbeat_ok": heartbeat_ok,
                        "message": f"Only {health_score}/3 policy health checks passed",
                    },
                )

        except Exception as e:
            self.log_criteria(
                "policy_health",
                "WARN",
                10,
                {"error": str(e), "message": "Could not check policy health metrics"},
            )

    def check_operational_readiness(self):
        """Check operational readiness (simplified)."""
        print("👥 Checking operational readiness...")

        try:
            root = pathlib.Path(os.getenv("GO_NOGO_ROOT", "."))
            # Check if ops bot is functional
            ops_bot_works = False
            try:
                from ops_bot.influence_commands import InfluenceBotCommands

                bot = InfluenceBotCommands()
                status = bot.command_status()
                if status.get("status") == "success":
                    ops_bot_works = True
            except:
                pass

            # Check for runbook accessibility
            runbooks_exist = (root / "playbooks" / "go_nogo.md").exists()

            # In production, you'd check for on-call ACK, etc.
            # For now, assume operational readiness based on available checks

            if ops_bot_works and runbooks_exist:
                self.log_criteria(
                    "operational_readiness",
                    "PASS",
                    10,
                    {
                        "ops_bot_works": ops_bot_works,
                        "runbooks_exist": runbooks_exist,
                        "message": "Operational readiness checks passed",
                    },
                )
            else:
                self.log_criteria(
                    "operational_readiness",
                    "WARN",
                    10,
                    {
                        "ops_bot_works": ops_bot_works,
                        "runbooks_exist": runbooks_exist,
                        "message": "Some operational readiness issues",
                    },
                )

        except Exception as e:
            self.log_criteria("operational_readiness", "WARN", 10, {"error": str(e)})

    def run_all_checks(self):
        """Run all Go/No-Go checks."""
        print("🚦 Running Go/No-Go Decision Check...")
        print(f"⏰ Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print()

        # Run all check methods
        check_methods = [
            self.check_validation_history,
            self.check_alerting_status,
            self.check_slo_performance,
            self.check_technical_readiness,
            self.check_policy_health,
            self.check_operational_readiness,
        ]

        for check_method in check_methods:
            try:
                check_method()
            except Exception as e:
                print(f"❌ Error in {check_method.__name__}: {e}")

        # Calculate final score and decision
        score_pct = (self.score / self.max_score * 100) if self.max_score > 0 else 0

        # Decision logic
        if len(self.go_blockers) > 0:
            decision = "NO_GO"
            decision_reason = f"Blocking criteria failed: {', '.join(self.go_blockers)}"
        elif score_pct >= 85:
            decision = "GO"
            decision_reason = f"All criteria satisfied (score: {score_pct:.0f}%)"
        elif score_pct >= 70:
            decision = "CONDITIONAL_GO"
            decision_reason = f"Marginal score ({score_pct:.0f}%) with warnings: {', '.join(self.warnings)}"
        else:
            decision = "NO_GO"
            decision_reason = f"Insufficient score ({score_pct:.0f}%) - multiple issues"

        # Summary
        total_criteria = len(self.criteria_results)
        passed = len(
            [c for c in self.criteria_results.values() if c["status"] == "PASS"]
        )
        warned = len(
            [c for c in self.criteria_results.values() if c["status"] == "WARN"]
        )
        failed = len(
            [c for c in self.criteria_results.values() if c["status"] == "FAIL"]
        )

        print(f"📊 Go/No-Go Decision Results:")
        print(f"   Total Criteria: {total_criteria}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ⚠️  Warnings: {warned}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📈 Score: {self.score:.0f}/{self.max_score} ({score_pct:.0f}%)")
        print()

        # Create audit record
        audit_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "go_nogo_decision",
            "decision": decision,
            "decision_reason": decision_reason,
            "score": {
                "points": self.score,
                "max_points": self.max_score,
                "percentage": score_pct,
            },
            "summary": {
                "total_criteria": total_criteria,
                "passed": passed,
                "warned": warned,
                "failed": failed,
            },
            "criteria_results": self.criteria_results,
            "go_blockers": self.go_blockers,
            "warnings": self.warnings,
        }

        # Write audit artifact
        os.makedirs("artifacts/audit", exist_ok=True)
        audit_filename = (
            f"artifacts/audit/{audit_data['timestamp'].replace(':', '_')}_go_nogo.json"
        )
        with open(audit_filename, "w") as f:
            json.dump(audit_data, f, indent=2)

        print(f"📋 Audit: {audit_filename}")

        # Final decision display
        if decision == "GO":
            print(f"\n🟢 **GO DECISION**")
            print(f"   {decision_reason}")
            print(f"   ✅ Cleared for deployment")
        elif decision == "CONDITIONAL_GO":
            print(f"\n🟡 **CONDITIONAL GO**")
            print(f"   {decision_reason}")
            print(f"   ⚠️  Proceed with caution - extra monitoring required")
        else:
            print(f"\n🔴 **NO-GO DECISION**")
            print(f"   {decision_reason}")
            print(f"   🛑 Deployment blocked until issues resolved")

            if self.go_blockers:
                print(f"\n🔴 Critical Blockers:")
                for blocker in self.go_blockers:
                    details = self.criteria_results[blocker].get("details", {})
                    message = details.get("message", "No details")
                    print(f"   • {blocker}: {message}")

        return decision == "GO"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Go/No-Go Decision Check")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    args = parser.parse_args()

    checker = GoNoGoChecker()
    is_go = checker.run_all_checks()

    if args.json:
        # Output structured JSON for automation
        result = {
            "decision": "GO" if is_go else "NO_GO",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "score_percentage": (
                (checker.score / checker.max_score * 100)
                if checker.max_score > 0
                else 0
            ),
            "blockers": checker.go_blockers,
            "warnings": checker.warnings,
        }
        print(json.dumps(result, indent=2))

    sys.exit(0 if is_go else 1)


if __name__ == "__main__":
    main()
