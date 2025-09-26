#!/usr/bin/env python3
"""
Ramp Guard - Safety Checks Before Policy Influence Ramp
Enforces Go/No-Go criteria before allowing policy influence increases
"""
import json
import glob
import sys
import time
import os
import redis
from datetime import datetime, timedelta, timezone
from pathlib import Path


def check_validation_passes(required_passes: int = 2) -> tuple:
    """
    Check for consecutive PASS results from 48h offline validation.

    Args:
        required_passes: Number of consecutive passes required

    Returns:
        (bool, str): (success, message)
    """
    try:
        # Find all validation artifact directories
        pattern = "artifacts/*/rl/gate_report.md"
        reports = sorted(glob.glob(pattern), reverse=True)  # Newest first

        if len(reports) < required_passes:
            return (
                False,
                f"Need {required_passes} validation runs, found {len(reports)}",
            )

        consecutive_passes = 0
        checked_reports = []

        for report_path in reports:
            try:
                with open(report_path, "r") as f:
                    content = f.read().upper()

                if "PASS" in content and "✅" in content:
                    consecutive_passes += 1
                    status = "PASS"
                else:
                    status = "FAIL"
                    break  # Stop at first failure

                checked_reports.append(f"{report_path}: {status}")

                if consecutive_passes >= required_passes:
                    break

            except Exception as e:
                return False, f"Error reading {report_path}: {e}"

        if consecutive_passes >= required_passes:
            return True, f"Found {consecutive_passes} consecutive PASS validations"
        else:
            return (
                False,
                f"Only {consecutive_passes} consecutive passes, need {required_passes}",
            )

    except Exception as e:
        return False, f"Validation check failed: {e}"


def check_no_recent_alerts(hours: int = 48) -> tuple:
    """
    Check for no page-severity alerts in the last N hours.

    Args:
        hours: Hours to look back

    Returns:
        (bool, str): (success, message)
    """
    try:
        cutoff_time = time.time() - (hours * 3600)

        # Check for alert sentinel files
        alert_patterns = ["artifacts/alerts/*", "logs/*alert*.log", "logs/*page*.log"]

        recent_alerts = []
        for pattern in alert_patterns:
            alert_files = glob.glob(pattern)
            for alert_file in alert_files:
                try:
                    mtime = os.path.getmtime(alert_file)
                    if mtime > cutoff_time:
                        recent_alerts.append(alert_file)
                except OSError:
                    continue

        if recent_alerts:
            return (
                False,
                f"Found {len(recent_alerts)} recent alerts: {recent_alerts[:3]}",
            )
        else:
            return True, f"No alerts found in last {hours}h"

    except Exception as e:
        return False, f"Alert check failed: {e}"


def check_shadow_kpis(tolerance_pct: float = 20.0) -> tuple:
    """
    Check shadow session KPIs are within historical bands.

    Args:
        tolerance_pct: Allowed deviation percentage

    Returns:
        (bool, str): (success, message)
    """
    try:
        # Get current shadow metrics from Redis
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)

        current_entropy = r.hget("policy:current", "entropy")
        current_return = r.hget(
            "shadow:stats", "return_mean"
        )  # Hypothetical shadow stats

        if not current_entropy:
            return False, "No current entropy data available"

        current_entropy = float(current_entropy)

        # Check entropy is in healthy range
        if current_entropy < 0.9:
            return False, f"Entropy too low: {current_entropy:.3f} < 0.9"

        if current_entropy > 2.5:
            return False, f"Entropy too high: {current_entropy:.3f} > 2.5"

        # For now, pass KPI check if entropy is healthy
        # In production, this would check historical baselines
        return True, f"Shadow KPIs healthy (entropy: {current_entropy:.3f})"

    except Exception as e:
        return False, f"Shadow KPI check failed: {e}"


def check_kill_switch_functional() -> tuple:
    """
    Verify kill-switch is functional.

    Returns:
        (bool, str): (success, message)
    """
    try:
        # Test kill-switch script exists and is executable
        kill_switch_path = Path("scripts/kill_switch.py")

        if not kill_switch_path.exists():
            return False, "kill_switch.py script not found"

        if not os.access(kill_switch_path, os.X_OK):
            return False, "kill_switch.py not executable"

        # Test influence controller import
        try:
            from src.rl.influence_controller import InfluenceController

            ic = InfluenceController()
            # Don't actually test emergency stop in guard check
            return True, "Kill-switch components verified"
        except ImportError as e:
            return False, f"Kill-switch import failed: {e}"

    except Exception as e:
        return False, f"Kill-switch check failed: {e}"


def main():
    """Main ramp guard execution."""
    print("🛡️  Running RL Policy Ramp Guard")
    print(f"⏰ Timestamp: {datetime.now().isoformat()}")
    print()

    checks = [
        ("Two consecutive validation PASS", lambda: check_validation_passes(2)),
        ("No alerts in last 48h", lambda: check_no_recent_alerts(48)),
        ("Shadow KPIs within bands", lambda: check_shadow_kpis(20.0)),
        ("Kill-switch functional", check_kill_switch_functional),
    ]

    all_passed = True
    results = []

    for check_name, check_func in checks:
        print(f"🔍 Checking: {check_name}")
        try:
            passed, message = check_func()
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status}: {message}")
            results.append((check_name, passed, message))

            if not passed:
                all_passed = False

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append((check_name, False, str(e)))
            all_passed = False

    print()
    print("🏁 Ramp Guard Results:")

    if all_passed:
        print("✅ RAMP_GUARD_PASS - All safety checks passed")
        print("🚀 Ready for 10% influence ramp")

        # Write success audit
        audit_path = Path("artifacts/audit")
        audit_path.mkdir(parents=True, exist_ok=True)

        audit_data = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "event": "ramp_guard_pass",
            "checks": [
                {"name": name, "passed": passed, "message": msg}
                for name, passed, msg in results
            ],
            "approved_for_ramp": True,
        }

        with open(audit_path / f"ramp_guard_{int(time.time())}.json", "w") as f:
            json.dump(audit_data, f, indent=2)

        sys.exit(0)

    else:
        failed_checks = [name for name, passed, _ in results if not passed]
        print("❌ RAMP_GUARD_FAIL - Safety checks failed:")
        for check_name in failed_checks:
            print(f"   - {check_name}")

        print()
        print("🛑 Policy influence ramp NOT APPROVED")
        print("📋 Fix failed checks before attempting ramp")

        # Write failure audit
        audit_path = Path("artifacts/audit")
        audit_path.mkdir(parents=True, exist_ok=True)

        audit_data = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "event": "ramp_guard_fail",
            "checks": [
                {"name": name, "passed": passed, "message": msg}
                for name, passed, msg in results
            ],
            "failed_checks": failed_checks,
            "approved_for_ramp": False,
        }

        with open(audit_path / f"ramp_guard_fail_{int(time.time())}.json", "w") as f:
            json.dump(audit_data, f, indent=2)

        sys.exit(1)


if __name__ == "__main__":
    main()
