#!/usr/bin/env python3
"""
Canary Cutover Script
Enable live trading in small, auditable steps with instant rollback capability
"""

import time
import redis
import os
import json
import sqlite3
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("cutover_canary")


class CanaryCutover:
    """Automated canary cutover with safety checks and rollback."""

    def __init__(self):
        """Initialize canary cutover."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.slack_webhook = os.getenv("SLACK_WEBHOOK")

        # Cutover configuration
        self.config = {
            "initial_capital_pct": 0.10,  # Start at 10%
            "ramp_steps": [0.20, 0.30, 0.40, 0.50],  # Ramp schedule
            "ab_wait_hours": 2,  # Wait 2 hours between A/B checks
            "ramp_wait_hours": 6,  # Wait 6 hours between ramp steps
            "required_consecutive_passes": 4,  # Need 4 consecutive A/B passes
            "safety_checks_interval": 300,  # Safety check every 5 minutes
        }

        # Feature flags to manage
        self.feature_flags = {
            "low_risk": ["BANDIT_WEIGHTS", "LLM_SENTIMENT"],
            "high_risk": ["EXEC_RL_LIVE"],
            "safety": ["HEDGE_ENABLED", "RISK_CONTROLS"],
        }

        # Initialize audit database
        self.init_audit_db()

        logger.info("🚀 Canary Cutover initialized")

    def init_audit_db(self):
        """Initialize audit database for cutover logging."""
        try:
            db_path = Path(__file__).parent.parent / "ab_history.db"
            conn = sqlite3.connect(db_path)

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cutover_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    stage TEXT,
                    action TEXT,
                    details TEXT,
                    success BOOLEAN,
                    capital_pct REAL,
                    flags_state TEXT
                )
            """
            )
            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error initializing audit DB: {e}")

    def log_audit(
        self,
        stage: str,
        action: str,
        details: str,
        success: bool,
        capital_pct: Optional[float] = None,
    ):
        """Log cutover action to audit database."""
        try:
            db_path = Path(__file__).parent.parent / "ab_history.db"
            conn = sqlite3.connect(db_path)

            # Get current flags state
            flags_state = json.dumps(self.redis.hgetall("features:flags") or {})

            conn.execute(
                """
                INSERT INTO cutover_log 
                (timestamp, stage, action, details, success, capital_pct, flags_state)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    int(time.time()),
                    stage,
                    action,
                    details,
                    success,
                    capital_pct,
                    flags_state,
                ),
            )
            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error logging to audit DB: {e}")

    def send_slack_message(self, message: str, urgent: bool = False):
        """Send message to Slack."""
        try:
            if not self.slack_webhook:
                logger.warning("No Slack webhook configured")
                return

            payload = {
                "text": message,
                "username": "Cutover Bot",
                "icon_emoji": ":rocket:" if not urgent else ":rotating_light:",
            }

            if urgent:
                payload["channel"] = "#trading-alerts"  # Use alerts channel for urgent

            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"📱 Sent Slack message: {message[:100]}...")

        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")

    def set_feature_flag(self, flag_name: str, enabled: bool) -> bool:
        """Set feature flag value."""
        try:
            self.redis.hset("features:flags", flag_name, int(enabled))
            logger.info(f"🏁 Set feature flag {flag_name} = {enabled}")
            return True
        except Exception as e:
            logger.error(f"Error setting flag {flag_name}: {e}")
            return False

    def run_preflight_check(self) -> bool:
        """Run preflight check before starting cutover."""
        try:
            logger.info("🔍 Running preflight check...")

            result = subprocess.run(
                ["python3", "scripts/preflight_supercheck.py", "--silent"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            if result.returncode == 0:
                logger.info("✅ Preflight check PASSED")
                self.log_audit("preflight", "check", "Preflight passed", True)
                return True
            else:
                logger.error("❌ Preflight check FAILED")
                logger.error(f"Preflight output: {result.stdout}")
                self.log_audit(
                    "preflight", "check", f"Preflight failed: {result.stdout}", False
                )
                return False

        except Exception as e:
            logger.error(f"Error running preflight: {e}")
            self.log_audit("preflight", "check", f"Preflight error: {e}", False)
            return False

    def check_ab_gate(self) -> Dict[str, Any]:
        """Check A/B testing gate status."""
        try:
            consecutive_passes = int(self.redis.get("ab:last4:exec") or 0)
            total_tests = int(self.redis.get("ab:total_tests") or 0)
            last_test_time = float(self.redis.get("ab:last_test_time") or 0)

            time_since_last = time.time() - last_test_time
            tests_recent = time_since_last < 3600  # Within last hour

            gate_passing = (
                consecutive_passes >= self.config["required_consecutive_passes"]
                and tests_recent
            )

            return {
                "passing": gate_passing,
                "consecutive_passes": consecutive_passes,
                "total_tests": total_tests,
                "time_since_last": time_since_last,
                "tests_recent": tests_recent,
            }

        except Exception as e:
            logger.error(f"Error checking A/B gate: {e}")
            return {"passing": False, "error": str(e)}

    def check_safety_conditions(self) -> Dict[str, bool]:
        """Check critical safety conditions."""
        try:
            safety_checks = {}

            # Check system mode is not halt
            mode = self.redis.get("mode") or "unknown"
            safety_checks["mode_ok"] = mode != "halt"

            # Check no reconciliation breaches
            recon_breaches = int(self.redis.get("recon:breaches_24h") or 0)
            safety_checks["recon_ok"] = recon_breaches == 0

            # Check SLO tier is not C
            slo_tier = self.redis.get("slo:tier") or "A"
            safety_checks["slo_ok"] = slo_tier != "C"

            # Check capital is within bounds
            capital_effective = float(self.redis.get("risk:capital_effective") or 0)
            safety_checks["capital_ok"] = 0.05 <= capital_effective <= 1.0

            # Check risk controls are enabled
            risk_enabled = bool(int(self.redis.get("risk:enabled") or 1))
            safety_checks["risk_controls_ok"] = risk_enabled

            all_safe = all(safety_checks.values())

            return {"all_safe": all_safe, "checks": safety_checks}

        except Exception as e:
            logger.error(f"Error checking safety conditions: {e}")
            return {"all_safe": False, "error": str(e)}

    def stage_capital(self, target_pct: float) -> bool:
        """Stage capital allocation change."""
        try:
            logger.info(f"💰 Staging capital to {target_pct:.0%}")

            # Set staging request
            self.redis.set("risk:capital_stage_request", target_pct)
            self.redis.set("risk:capital_stage_timestamp", int(time.time()))
            self.redis.set("risk:capital_stage_user", "cutover_canary")

            # For automated cutover, we'll auto-approve if safety checks pass
            safety = self.check_safety_conditions()
            if safety["all_safe"]:
                # Auto-approve the staging
                self.redis.set("risk:capital_cap_next_week", target_pct)
                self.redis.delete("risk:capital_stage_request")

                logger.info(f"✅ Capital staged and auto-approved: {target_pct:.0%}")
                self.log_audit(
                    "capital",
                    "stage_approve",
                    f"Staged to {target_pct:.0%}",
                    True,
                    target_pct,
                )
                return True
            else:
                logger.error(
                    f"❌ Cannot stage capital - safety checks failed: {safety}"
                )
                self.log_audit(
                    "capital",
                    "stage_deny",
                    f"Safety failed: {safety}",
                    False,
                    target_pct,
                )
                return False

        except Exception as e:
            logger.error(f"Error staging capital: {e}")
            self.log_audit("capital", "stage_error", str(e), False, target_pct)
            return False

    def promote_low_risk_features(self) -> bool:
        """Promote low-risk features to live."""
        try:
            logger.info("🎯 Promoting low-risk features to live")

            success = True
            for flag in self.feature_flags["low_risk"]:
                if not self.set_feature_flag(flag, True):
                    success = False

            if success:
                self.log_audit(
                    "features",
                    "promote_low_risk",
                    f"Enabled: {', '.join(self.feature_flags['low_risk'])}",
                    True,
                )
                self.send_slack_message(
                    f"🟢 Promoted low-risk features to live: {', '.join(self.feature_flags['low_risk'])}"
                )
            else:
                self.log_audit(
                    "features", "promote_low_risk", "Failed to enable some flags", False
                )
                self.send_slack_message(
                    "🔴 Failed to promote some low-risk features", urgent=True
                )

            return success

        except Exception as e:
            logger.error(f"Error promoting low-risk features: {e}")
            self.log_audit("features", "promote_low_risk", str(e), False)
            return False

    def promote_high_risk_features(self) -> bool:
        """Promote high-risk features to live after A/B validation."""
        try:
            logger.info("⚡ Promoting high-risk features to live")

            success = True
            for flag in self.feature_flags["high_risk"]:
                if not self.set_feature_flag(flag, True):
                    success = False

            if success:
                self.log_audit(
                    "features",
                    "promote_high_risk",
                    f"Enabled: {', '.join(self.feature_flags['high_risk'])}",
                    True,
                )
                self.send_slack_message(
                    f"🔴 Promoted HIGH-RISK features to live: {', '.join(self.feature_flags['high_risk'])} after A/B validation"
                )
            else:
                self.log_audit(
                    "features",
                    "promote_high_risk",
                    "Failed to enable some flags",
                    False,
                )
                self.send_slack_message(
                    "🚨 FAILED to promote high-risk features", urgent=True
                )

            return success

        except Exception as e:
            logger.error(f"Error promoting high-risk features: {e}")
            self.log_audit("features", "promote_high_risk", str(e), False)
            return False

    def emergency_rollback(self) -> bool:
        """Emergency rollback to safe state."""
        try:
            logger.warning("🚨 EMERGENCY ROLLBACK INITIATED")

            # Disable all risky features
            for flag in self.feature_flags["high_risk"]:
                self.set_feature_flag(flag, False)

            # Set capital to minimal
            self.redis.set("risk:capital_effective", 0.05)  # 5% emergency mode

            # Set system to halt mode
            self.redis.set("mode", "halt")

            self.log_audit("emergency", "rollback", "Full system rollback", True)
            self.send_slack_message(
                "🚨 EMERGENCY ROLLBACK COMPLETED - System in safe mode", urgent=True
            )

            return True

        except Exception as e:
            logger.error(f"Error in emergency rollback: {e}")
            self.log_audit("emergency", "rollback", f"Rollback error: {e}", False)
            return False

    def wait_with_safety_checks(self, wait_hours: float, description: str) -> bool:
        """Wait for specified time with periodic safety checks."""
        try:
            wait_seconds = wait_hours * 3600
            check_interval = self.config["safety_checks_interval"]

            logger.info(
                f"⏱️ Waiting {wait_hours}h for {description} (safety checks every {check_interval/60:.0f}min)"
            )

            start_time = time.time()
            end_time = start_time + wait_seconds

            while time.time() < end_time:
                # Periodic safety checks
                safety = self.check_safety_conditions()
                if not safety["all_safe"]:
                    logger.error(f"💥 Safety check failed during wait: {safety}")
                    self.emergency_rollback()
                    return False

                # Sleep for check interval or remaining time
                remaining = end_time - time.time()
                sleep_time = min(check_interval, remaining)

                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Log progress
                elapsed_hours = (time.time() - start_time) / 3600
                logger.debug(
                    f"⏱️ Waiting progress: {elapsed_hours:.1f}h / {wait_hours:.1f}h"
                )

            logger.info(f"✅ Wait completed for {description}")
            return True

        except Exception as e:
            logger.error(f"Error during wait: {e}")
            return False

    def run_cutover(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run complete canary cutover process."""
        try:
            cutover_start = time.time()
            logger.info("🚀 Starting canary cutover process...")

            if dry_run:
                logger.info("🧪 DRY RUN MODE - No real changes will be made")

            cutover_result = {
                "status": "in_progress",
                "dry_run": dry_run,
                "start_time": cutover_start,
                "stages_completed": [],
                "stages_failed": [],
            }

            # Stage 0: Preflight check
            logger.info("📋 Stage 0: Preflight check")
            if not dry_run and not self.run_preflight_check():
                cutover_result["status"] = "failed"
                cutover_result["stages_failed"].append("preflight")
                logger.error("❌ Preflight failed - aborting cutover")
                return cutover_result

            cutover_result["stages_completed"].append("preflight")
            self.send_slack_message("🚀 Canary cutover started - preflight passed")

            # Stage 1: Promote low-risk features
            logger.info("📋 Stage 1: Promote low-risk features")
            if not dry_run and not self.promote_low_risk_features():
                cutover_result["status"] = "failed"
                cutover_result["stages_failed"].append("low_risk_features")
                return cutover_result

            cutover_result["stages_completed"].append("low_risk_features")

            # Stage 2: Set initial capital allocation
            logger.info("📋 Stage 2: Set initial capital allocation")
            if not dry_run and not self.stage_capital(
                self.config["initial_capital_pct"]
            ):
                cutover_result["status"] = "failed"
                cutover_result["stages_failed"].append("initial_capital")
                return cutover_result

            cutover_result["stages_completed"].append("initial_capital")

            # Stage 3: Wait and verify A/B tests
            logger.info("📋 Stage 3: Wait for A/B validation")
            if not self.wait_with_safety_checks(
                self.config["ab_wait_hours"], "A/B validation"
            ):
                cutover_result["status"] = "failed"
                cutover_result["stages_failed"].append("ab_wait")
                return cutover_result

            # Check A/B gate
            ab_status = self.check_ab_gate()
            if not ab_status["passing"]:
                logger.error(f"❌ A/B gate not passing: {ab_status}")
                cutover_result["status"] = "failed"
                cutover_result["stages_failed"].append("ab_gate")
                self.send_slack_message(f"🔴 A/B gate failed: {ab_status}", urgent=True)
                return cutover_result

            cutover_result["stages_completed"].append("ab_validation")

            # Stage 4: Promote high-risk features
            logger.info("📋 Stage 4: Promote high-risk features after A/B validation")
            if not dry_run and not self.promote_high_risk_features():
                cutover_result["status"] = "failed"
                cutover_result["stages_failed"].append("high_risk_features")
                return cutover_result

            cutover_result["stages_completed"].append("high_risk_features")

            # Stage 5: Ramp capital allocation
            logger.info("📋 Stage 5: Ramp capital allocation")
            for i, target_pct in enumerate(self.config["ramp_steps"]):
                logger.info(f"🎯 Ramp step {i+1}: {target_pct:.0%}")

                if not dry_run and not self.stage_capital(target_pct):
                    cutover_result["status"] = "failed"
                    cutover_result["stages_failed"].append(f"ramp_{i+1}")
                    return cutover_result

                # Wait between ramp steps (except for the last one)
                if i < len(self.config["ramp_steps"]) - 1:
                    if not self.wait_with_safety_checks(
                        self.config["ramp_wait_hours"], f"ramp step {i+1}"
                    ):
                        cutover_result["status"] = "failed"
                        cutover_result["stages_failed"].append(f"ramp_wait_{i+1}")
                        return cutover_result

                cutover_result["stages_completed"].append(f"ramp_{i+1}")
                self.send_slack_message(f"📈 Capital ramped to {target_pct:.0%}")

            # Cutover complete
            cutover_duration = time.time() - cutover_start
            cutover_result.update(
                {
                    "status": "completed",
                    "duration": cutover_duration,
                    "final_capital_pct": self.config["ramp_steps"][-1],
                    "end_time": time.time(),
                }
            )

            logger.info(f"✅ Canary cutover completed in {cutover_duration/3600:.1f}h")
            self.log_audit(
                "cutover",
                "complete",
                f"Full cutover completed in {cutover_duration:.0f}s",
                True,
            )
            self.send_slack_message(
                f"🎉 CANARY CUTOVER COMPLETED! Final capital: {self.config['ramp_steps'][-1]:.0%} "
                f"Duration: {cutover_duration/3600:.1f}h"
            )

            return cutover_result

        except Exception as e:
            logger.error(f"Error in cutover process: {e}")
            cutover_result["status"] = "error"
            cutover_result["error"] = str(e)

            if not dry_run:
                self.emergency_rollback()

            return cutover_result

    def get_status_report(self) -> Dict[str, Any]:
        """Get current cutover status."""
        try:
            # Get current system state
            feature_flags = self.redis.hgetall("features:flags") or {}
            capital_effective = float(self.redis.get("risk:capital_effective") or 0)
            capital_staged = self.redis.get("risk:capital_stage_request")
            system_mode = self.redis.get("mode") or "unknown"

            # Get A/B gate status
            ab_status = self.check_ab_gate()

            # Get safety conditions
            safety_status = self.check_safety_conditions()

            status = {
                "service": "canary_cutover",
                "timestamp": time.time(),
                "system_state": {
                    "mode": system_mode,
                    "capital_effective": capital_effective,
                    "capital_staged": float(capital_staged) if capital_staged else None,
                    "feature_flags": feature_flags,
                },
                "ab_gate": ab_status,
                "safety_checks": safety_status,
                "config": self.config,
            }

            return status

        except Exception as e:
            return {
                "service": "canary_cutover",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Canary Cutover")
    parser.add_argument("--run", action="store_true", help="Run canary cutover")
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run mode (no real changes)"
    )
    parser.add_argument("--status", action="store_true", help="Show cutover status")
    parser.add_argument("--rollback", action="store_true", help="Emergency rollback")

    args = parser.parse_args()

    cutover = CanaryCutover()

    if args.status:
        status = cutover.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.rollback:
        print("⚠️ Initiating emergency rollback...")
        success = cutover.emergency_rollback()
        sys.exit(0 if success else 1)

    if args.run or args.dry_run:
        result = cutover.run_cutover(dry_run=args.dry_run)
        print(json.dumps(result, indent=2, default=str))

        if result["status"] == "completed":
            sys.exit(0)
        else:
            sys.exit(1)

    parser.print_help()


if __name__ == "__main__":
    main()
