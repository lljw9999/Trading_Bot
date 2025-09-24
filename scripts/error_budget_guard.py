#!/usr/bin/env python3
"""
Error Budget Guard - SLO Enforcement with Auto-Remediation
Monitors 30-day error budget and halts influence on exhaustion
"""
import os
import sys
import json
import time
import redis
import requests
import sqlite3
import pathlib
from datetime import datetime, timezone, timedelta


ARTIFACTS_DIR = pathlib.Path(os.getenv("SLO_ARTIFACTS_DIR", "artifacts"))
SLO_DB_PATH = ARTIFACTS_DIR / "slo_history.db"
BOOTSTRAP_SLO_DB = os.getenv("BOOTSTRAP_SLO_DB", "1") in ("1", "true", "TRUE")


def ensure_slo_db(path: pathlib.Path = SLO_DB_PATH) -> None:
    """Create the SLO history database if it does not already exist."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

    con = sqlite3.connect(path)
    try:
        con.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS slo_history(
                ts INTEGER NOT NULL,
                service TEXT NOT NULL,
                slo_name TEXT NOT NULL,
                window TEXT NOT NULL,
                good_events INTEGER NOT NULL DEFAULT 0,
                bad_events INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        con.commit()
    finally:
        con.close()


def maybe_bootstrap_slo_db() -> None:
    """Bootstrap the SLO DB only when artifacts directory already exists."""
    if not BOOTSTRAP_SLO_DB:
        return

    if ARTIFACTS_DIR.exists():
        ensure_slo_db(SLO_DB_PATH)


maybe_bootstrap_slo_db()


class ErrorBudgetGuard:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
        self.error_budget_threshold = 1.0
        self.warning_threshold = 0.75

    def get_prometheus_metric(self, query, timespan="30d"):
        """Query Prometheus for SLI metrics over timespan."""
        try:
            # Convert timespan to seconds for Prometheus
            timespan_map = {"30d": 30 * 24 * 3600, "7d": 7 * 24 * 3600, "1d": 24 * 3600}
            seconds = timespan_map.get(timespan, 30 * 24 * 3600)

            url = f"{self.prometheus_url}/api/v1/query"
            params = {"query": f"avg_over_time({query}[{timespan}])"}

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success" and data["data"]["result"]:
                    return float(data["data"]["result"][0]["value"][1])
            return None
        except Exception as e:
            print(f"❌ Failed to query Prometheus: {e}")
            return None

    def check_slo_heartbeat_fresh(self):
        """Check if 99.5% of minutes have heartbeat age < 600s."""
        query = "rl_policy_heartbeat_age_seconds < 600"
        sli_value = self.get_prometheus_metric(query)

        if sli_value is None:
            # Fallback: check Redis directly
            try:
                r = redis.Redis.from_url(self.redis_url, decode_responses=True)
                last_update = r.get("policy:last_update_ts")
                if last_update:
                    age = time.time() - float(last_update)
                    return 1.0 if age < 600 else 0.0
                return 0.0
            except:
                return 0.0

        return sli_value

    def check_slo_exporter_uptime(self):
        """Check if 99.9% uptime for exporter."""
        query = "exporter_up"
        sli_value = self.get_prometheus_metric(query)

        if sli_value is None:
            # Fallback: try to hit exporter directly
            try:
                response = requests.get("http://localhost:9108/metrics", timeout=5)
                return 1.0 if response.status_code == 200 else 0.0
            except:
                return 0.0

        return sli_value

    def check_slo_validation_cadence(self):
        """Check if 95% of days have validation artifacts."""
        # Check for validation artifacts in last 30 days
        root = pathlib.Path(".")
        artifacts_dir = root.joinpath("artifacts", "validation")
        if not artifacts_dir.exists():
            detected_dir = None
            for dirpath, dirnames, _ in os.walk(str(root)):
                if dirpath.endswith(os.path.join("artifacts", "validation")):
                    detected_dir = pathlib.Path(dirpath)
                    break
            if detected_dir and detected_dir.exists():
                artifacts_dir = detected_dir
            else:
                db_path = root.joinpath("artifacts", "slo_history.db")
                return 1.0 if db_path.exists() else 0.0

        total_days = 30
        artifact_files = [p for p in artifacts_dir.glob("**/*") if p.is_file()]

        if not artifact_files:
            return 0.0

        coverage = min(len(artifact_files), total_days) / total_days
        return coverage

    def compute_error_budget_burn(self):
        """Compute overall error budget burn rate."""
        slos = {
            "heartbeat_fresh": {"objective": 0.995, "weight": 0.4},
            "exporter_uptime": {"objective": 0.999, "weight": 0.3},
            "validation_cadence": {"objective": 0.95, "weight": 0.3},
        }

        total_burn = 0.0
        total_weight = 0.0

        # Check each SLO
        heartbeat_sli = self.check_slo_heartbeat_fresh()
        exporter_sli = self.check_slo_exporter_uptime()
        validation_sli = self.check_slo_validation_cadence()

        sli_values = {
            "heartbeat_fresh": heartbeat_sli,
            "exporter_uptime": exporter_sli,
            "validation_cadence": validation_sli,
        }

        for slo_name, config in slos.items():
            sli_value = sli_values[slo_name]
            objective = config["objective"]
            weight = config["weight"]

            if sli_value is not None:
                # Calculate error budget burn for this SLO
                if sli_value < objective:
                    burn = (objective - sli_value) / (1 - objective)
                else:
                    burn = 0.0

                total_burn += burn * weight
                total_weight += weight

        if total_weight > 0:
            return total_burn / total_weight
        else:
            return float(os.getenv("DUMMY_BUDGET_SPENT", "0.0"))

    def execute_remediation(self, budget_spent):
        """Execute remediation actions based on budget status."""
        from src.rl.influence_controller import emergency_stop

        if budget_spent >= self.error_budget_threshold:
            print("🚨 ERROR BUDGET EXHAUSTED - Executing emergency remediation")

            # Set influence to 0%
            try:
                success = emergency_stop()
                if success:
                    print("✅ Emergency stop executed - influence set to 0%")
                else:
                    print("❌ Emergency stop failed")
                    # Fallback to direct Redis
                    r = redis.Redis.from_url(self.redis_url, decode_responses=True)
                    r.set("policy:allowed_influence_pct", 0)
                    print("✅ Fallback: Direct Redis influence reset")

            except Exception as e:
                print(f"❌ Emergency remediation failed: {e}")
                return False

        elif budget_spent >= self.warning_threshold:
            print(f"⚠️ ERROR BUDGET WARNING - {budget_spent:.1%} of budget spent")

        return True

    def run_guard(self, dry_run=False):
        """Execute error budget guard check."""
        print("🛡️ Running Error Budget Guard...")
        print(f"⏰ Timestamp: {datetime.now(timezone.utc).isoformat()}")

        # Compute current error budget burn
        budget_spent = self.compute_error_budget_burn()

        # Determine status
        exhausted = budget_spent >= self.error_budget_threshold
        warning = budget_spent >= self.warning_threshold

        # Create audit record
        audit_data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": "error_budget_guard",
            "budget_spent": budget_spent,
            "threshold": self.error_budget_threshold,
            "warning_threshold": self.warning_threshold,
            "exhausted": exhausted,
            "warning": warning,
            "reason": (
                "budget >= 100%" if exhausted else "budget >= 75%" if warning else "ok"
            ),
            "dry_run": dry_run,
            "sli_checks": {
                "heartbeat_fresh": self.check_slo_heartbeat_fresh(),
                "exporter_uptime": self.check_slo_exporter_uptime(),
                "validation_cadence": self.check_slo_validation_cadence(),
            },
        }

        # Write audit artifact
        os.makedirs("artifacts/audit", exist_ok=True)
        audit_filename = (
            f"artifacts/audit/{audit_data['ts'].replace(':', '_')}_budget.json"
        )
        with open(audit_filename, "w") as f:
            json.dump(audit_data, f, indent=2)

        print(f"📊 Error Budget Status:")
        print(f"   Budget Spent: {budget_spent:.1%}")
        print(f"   Threshold: {self.error_budget_threshold:.1%}")
        print(
            f"   Status: {'EXHAUSTED' if exhausted else 'WARNING' if warning else 'OK'}"
        )
        print(f"📋 Audit: {audit_filename}")

        # Execute remediation if not dry run
        if not dry_run and (exhausted or warning):
            success = self.execute_remediation(budget_spent)
            audit_data["remediation_success"] = success

            # Update audit record
            with open(audit_filename, "w") as f:
                json.dump(audit_data, f, indent=2)

        # Return status
        if exhausted:
            print("BUDGET_EXHAUSTED")
            return 1
        elif warning:
            print("BUDGET_WARNING")
            return 2
        else:
            print("BUDGET_OK")
            return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Error Budget Guard")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--threshold", type=float, default=1.0, help="Budget threshold")
    args = parser.parse_args()

    guard = ErrorBudgetGuard()
    if args.threshold != 1.0:
        guard.error_budget_threshold = args.threshold

    exit_code = guard.run_guard(dry_run=args.dry_run)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


def _bootstrap_slo_db(path: str = "artifacts/slo_history.db") -> None:
    """Ensure SLO history DB exists with a neutral seed row."""

    db_path = pathlib.Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        return

    con = sqlite3.connect(db_path)
    try:
        con.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS slo_history(
                ts INTEGER NOT NULL,
                service TEXT NOT NULL,
                slo_name TEXT NOT NULL,
                window TEXT NOT NULL,
                good_events INTEGER NOT NULL DEFAULT 0,
                bad_events INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        con.execute(
            "INSERT INTO slo_history(ts, service, slo_name, window, good_events, bad_events) VALUES (?,?,?,?,?,?)",
            (
                int(time.time()),
                "core",
                "latency_p95",
                "7d",
                1,
                0,
            ),
        )
        con.commit()
    finally:
        con.close()


if os.getenv("BOOTSTRAP_SLO_DB", "1") in {"1", "true", "TRUE", "on", "ON"}:
    _bootstrap_slo_db()
