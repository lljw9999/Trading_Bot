#!/usr/bin/env python3
"""
Disaster Recovery: Restore from S3
Rebuild Redis + model deltas + configs on a fresh node; verify RTO/RPO
"""

import os
import sys
import time
import json
import boto3
import redis
import logging
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("restore_from_s3")


class DisasterRecoveryRestore:
    """Handles complete system restore from S3 backups."""

    def __init__(self):
        """Initialize DR restore."""
        self.s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
        self.redis = None  # Will connect after restore
        self.slack_webhook = os.getenv("SLACK_WEBHOOK")

        # DR Configuration
        self.config = {
            "backup_bucket": os.getenv("BACKUP_BUCKET", "trading-system-backups"),
            "backup_prefix": "dr-snapshots",
            "redis_dump_file": "/var/lib/redis/dump.rdb",
            "redis_service": "redis-server",
            "model_cache_dir": "/opt/trader/models",
            "config_dir": "/opt/trader/config",
            "logs_dir": "/opt/trader/logs",
            "restore_timeout_minutes": 30,
            "max_restore_attempts": 3,
        }

        # RTO/RPO tracking
        self.restore_metrics = {
            "start_time": None,
            "end_time": None,
            "rto_minutes": None,  # Recovery Time Objective
            "rpo_minutes": None,  # Recovery Point Objective
            "components_restored": [],
            "components_failed": [],
            "data_loss_detected": False,
        }

        logger.info("🆘 Disaster Recovery Restore initialized")

    def find_latest_backup(self) -> Optional[Dict[str, Any]]:
        """Find the most recent complete backup."""
        try:
            logger.info("🔍 Searching for latest backup...")

            response = self.s3.list_objects_v2(
                Bucket=self.config["backup_bucket"],
                Prefix=f"{self.config['backup_prefix']}/",
                Delimiter="/",
            )

            if "CommonPrefixes" not in response:
                logger.error("No backup snapshots found")
                return None

            # Sort backup directories by timestamp (newest first)
            backup_dirs = sorted(
                [prefix["Prefix"] for prefix in response["CommonPrefixes"]],
                reverse=True,
            )

            # Check each backup for completeness
            for backup_dir in backup_dirs:
                backup_timestamp = backup_dir.split("/")[-2]

                # List objects in this backup
                objects_response = self.s3.list_objects_v2(
                    Bucket=self.config["backup_bucket"], Prefix=backup_dir
                )

                if "Contents" not in objects_response:
                    continue

                # Check for required backup components
                required_files = [
                    "redis_dump.rdb",
                    "model_weights.pkl",
                    "system_config.json",
                    "backup_manifest.json",
                ]

                found_files = [
                    obj["Key"].split("/")[-1] for obj in objects_response["Contents"]
                ]

                if all(req_file in found_files for req_file in required_files):
                    # Parse backup timestamp
                    try:
                        backup_time = datetime.strptime(
                            backup_timestamp, "%Y%m%d_%H%M%S"
                        )

                        return {
                            "backup_dir": backup_dir,
                            "backup_timestamp": backup_timestamp,
                            "backup_time": backup_time,
                            "age_minutes": (
                                datetime.utcnow() - backup_time
                            ).total_seconds()
                            / 60,
                            "files": found_files,
                        }
                    except ValueError:
                        continue

            logger.error("No complete backups found")
            return None

        except Exception as e:
            logger.error(f"Error finding latest backup: {e}")
            return None

    def download_backup_file(
        self, backup_dir: str, filename: str, local_path: str
    ) -> bool:
        """Download a specific backup file from S3."""
        try:
            s3_key = f"{backup_dir}{filename}"

            # Create local directory if needed
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            # Download file
            self.s3.download_file(self.config["backup_bucket"], s3_key, local_path)

            logger.info(f"📥 Downloaded {filename} to {local_path}")
            return True

        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return False

    def restore_redis_data(self, backup_dir: str) -> bool:
        """Restore Redis data from backup."""
        try:
            logger.info("💾 Restoring Redis data...")

            # Stop Redis service
            subprocess.run(
                ["sudo", "systemctl", "stop", self.config["redis_service"]], check=True
            )

            # Backup current dump if exists
            current_dump = Path(self.config["redis_dump_file"])
            if current_dump.exists():
                backup_current = f"{current_dump}.pre_restore_{int(time.time())}"
                shutil.copy2(current_dump, backup_current)
                logger.info(f"💾 Backed up current Redis dump to {backup_current}")

            # Download Redis dump from S3
            if not self.download_backup_file(
                backup_dir, "redis_dump.rdb", self.config["redis_dump_file"]
            ):
                return False

            # Set correct permissions
            subprocess.run(
                ["sudo", "chown", "redis:redis", self.config["redis_dump_file"]],
                check=True,
            )
            subprocess.run(
                ["sudo", "chmod", "640", self.config["redis_dump_file"]], check=True
            )

            # Start Redis service
            subprocess.run(
                ["sudo", "systemctl", "start", self.config["redis_service"]], check=True
            )

            # Wait for Redis to be ready
            time.sleep(5)

            # Test Redis connection
            test_redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
            test_redis.ping()

            logger.info("✅ Redis data restored successfully")
            return True

        except Exception as e:
            logger.error(f"Error restoring Redis data: {e}")
            return False

    def restore_model_weights(self, backup_dir: str) -> bool:
        """Restore ML model weights and checkpoints."""
        try:
            logger.info("🧠 Restoring model weights...")

            model_cache = Path(self.config["model_cache_dir"])
            model_cache.mkdir(parents=True, exist_ok=True)

            # Download model weights
            model_weights_path = model_cache / "restored_weights.pkl"
            if not self.download_backup_file(
                backup_dir, "model_weights.pkl", str(model_weights_path)
            ):
                return False

            # Download model metadata if available
            metadata_files = [
                "model_config.json",
                "training_state.json",
                "feature_stats.pkl",
            ]
            for metadata_file in metadata_files:
                local_path = model_cache / metadata_file
                self.download_backup_file(backup_dir, metadata_file, str(local_path))

            logger.info("✅ Model weights restored successfully")
            return True

        except Exception as e:
            logger.error(f"Error restoring model weights: {e}")
            return False

    def restore_system_configs(self, backup_dir: str) -> bool:
        """Restore system configuration files."""
        try:
            logger.info("⚙️ Restoring system configs...")

            config_dir = Path(self.config["config_dir"])
            config_dir.mkdir(parents=True, exist_ok=True)

            # Download system config
            config_path = config_dir / "system_config.json"
            if not self.download_backup_file(
                backup_dir, "system_config.json", str(config_path)
            ):
                return False

            # Download additional config files if available
            config_files = [
                "exchange_configs.json",
                "feature_configs.json",
                "risk_params.json",
                "strategy_configs.json",
            ]

            for config_file in config_files:
                local_path = config_dir / config_file
                self.download_backup_file(backup_dir, config_file, str(local_path))

            logger.info("✅ System configs restored successfully")
            return True

        except Exception as e:
            logger.error(f"Error restoring system configs: {e}")
            return False

    def verify_data_integrity(self, backup_info: Dict[str, Any]) -> bool:
        """Verify restored data integrity."""
        try:
            logger.info("🔍 Verifying data integrity...")

            # Connect to restored Redis
            self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

            # Check critical Redis keys exist
            critical_keys = [
                "risk:capital_effective",
                "mode",
                "pnl:total",
                "positions:*",
                "orders:*",
            ]

            keys_found = 0
            for key_pattern in critical_keys:
                if "*" in key_pattern:
                    matches = self.redis.keys(key_pattern)
                    if matches:
                        keys_found += 1
                else:
                    if self.redis.exists(key_pattern):
                        keys_found += 1

            integrity_score = keys_found / len(critical_keys)

            if integrity_score < 0.8:  # Require 80% of critical keys
                logger.warning(f"⚠️ Data integrity low: {integrity_score:.1%}")
                return False

            # Calculate RPO (data loss)
            backup_age_minutes = backup_info["age_minutes"]
            self.restore_metrics["rpo_minutes"] = backup_age_minutes

            if backup_age_minutes > 60:  # More than 1 hour old
                logger.warning(
                    f"⚠️ Potential data loss: backup is {backup_age_minutes:.0f} minutes old"
                )
                self.restore_metrics["data_loss_detected"] = True

            logger.info(
                f"✅ Data integrity verified (score: {integrity_score:.1%}, RPO: {backup_age_minutes:.0f}min)"
            )
            return True

        except Exception as e:
            logger.error(f"Error verifying data integrity: {e}")
            return False

    def run_post_restore_checks(self) -> Dict[str, bool]:
        """Run post-restore system health checks."""
        try:
            logger.info("🔎 Running post-restore checks...")

            checks = {}

            # Check preflight
            try:
                result = subprocess.run(
                    ["python3", "scripts/preflight_supercheck.py", "--silent"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                checks["preflight"] = result.returncode == 0
            except Exception as e:
                logger.error(f"Preflight check failed: {e}")
                checks["preflight"] = False

            # Check reconciliation status
            try:
                if self.redis:
                    recon_breaches = int(self.redis.get("recon:breaches_24h") or 0)
                    checks["reconciliation"] = recon_breaches == 0
                else:
                    checks["reconciliation"] = False
            except Exception:
                checks["reconciliation"] = False

            # Check system mode
            try:
                if self.redis:
                    mode = self.redis.get("mode") or "unknown"
                    checks["system_mode"] = mode in ["auto", "manual"]
                else:
                    checks["system_mode"] = False
            except Exception:
                checks["system_mode"] = False

            # Check model availability
            try:
                model_path = (
                    Path(self.config["model_cache_dir"]) / "restored_weights.pkl"
                )
                checks["model_weights"] = model_path.exists()
            except Exception:
                checks["model_weights"] = False

            # Check critical services
            try:
                services = ["redis-server", "prometheus", "grafana-server"]
                service_status = []
                for service in services:
                    result = subprocess.run(
                        ["sudo", "systemctl", "is-active", service],
                        capture_output=True,
                        text=True,
                    )
                    service_status.append(result.returncode == 0)
                checks["services"] = all(service_status)
            except Exception:
                checks["services"] = False

            passed_checks = sum(checks.values())
            total_checks = len(checks)

            logger.info(
                f"📋 Post-restore checks: {passed_checks}/{total_checks} passed"
            )
            return checks

        except Exception as e:
            logger.error(f"Error in post-restore checks: {e}")
            return {}

    def send_restore_summary(self, success: bool, restore_info: Dict[str, Any]):
        """Send disaster recovery restore summary to Slack."""
        try:
            if not self.slack_webhook:
                logger.warning("No Slack webhook configured")
                return

            rto = self.restore_metrics["rto_minutes"] or 0
            rpo = self.restore_metrics["rpo_minutes"] or 0

            if success:
                emoji = "🟢"
                title = "✅ DISASTER RECOVERY: RESTORE COMPLETED"
                color = "#36a64f"
            else:
                emoji = "🔴"
                title = "❌ DISASTER RECOVERY: RESTORE FAILED"
                color = "#dc2626"

            message = f"""{emoji} **{title}**

**RTO:** {rto:.0f} minutes
**RPO:** {rpo:.0f} minutes
**Backup Age:** {restore_info.get('backup_age_minutes', 0):.0f} minutes
**Data Loss:** {'Yes' if self.restore_metrics['data_loss_detected'] else 'No'}

**Components:**
✅ **Restored:** {', '.join(self.restore_metrics['components_restored'])}
❌ **Failed:** {', '.join(self.restore_metrics['components_failed']) if self.restore_metrics['components_failed'] else 'None'}

**Next Steps:** {'Run production verification' if success else 'Investigate failure and retry'}"""

            payload = {
                "text": message,
                "username": "DR Restore Bot",
                "icon_emoji": ":sos:",
                "attachments": [{"color": color}],
            }

            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("📱 Sent restore summary to Slack")

        except Exception as e:
            logger.error(f"Error sending restore summary: {e}")

    def execute_restore(self, dry_run: bool = False) -> Dict[str, Any]:
        """Execute complete disaster recovery restore."""
        try:
            self.restore_metrics["start_time"] = time.time()
            logger.info("🆘 Starting disaster recovery restore...")

            if dry_run:
                logger.info("🧪 DRY RUN MODE - No real changes will be made")

            restore_result = {
                "status": "in_progress",
                "dry_run": dry_run,
                "start_time": self.restore_metrics["start_time"],
                "phases": [],
            }

            # Phase 1: Find latest backup
            logger.info("📋 Phase 1: Finding latest backup")
            backup_info = self.find_latest_backup()
            if not backup_info:
                raise Exception("No valid backup found")

            restore_result["backup_info"] = backup_info
            restore_result["phases"].append("find_backup")

            logger.info(
                f"📦 Found backup: {backup_info['backup_timestamp']} (age: {backup_info['age_minutes']:.0f}min)"
            )

            # Phase 2: Restore Redis data
            if not dry_run:
                logger.info("📋 Phase 2: Restoring Redis data")
                if self.restore_redis_data(backup_info["backup_dir"]):
                    self.restore_metrics["components_restored"].append("redis")
                else:
                    self.restore_metrics["components_failed"].append("redis")
                    raise Exception("Redis restore failed")

            restore_result["phases"].append("restore_redis")

            # Phase 3: Restore model weights
            if not dry_run:
                logger.info("📋 Phase 3: Restoring model weights")
                if self.restore_model_weights(backup_info["backup_dir"]):
                    self.restore_metrics["components_restored"].append("models")
                else:
                    self.restore_metrics["components_failed"].append("models")
                    logger.warning("⚠️ Model restore failed - continuing")

            restore_result["phases"].append("restore_models")

            # Phase 4: Restore configs
            if not dry_run:
                logger.info("📋 Phase 4: Restoring system configs")
                if self.restore_system_configs(backup_info["backup_dir"]):
                    self.restore_metrics["components_restored"].append("configs")
                else:
                    self.restore_metrics["components_failed"].append("configs")
                    logger.warning("⚠️ Config restore failed - continuing")

            restore_result["phases"].append("restore_configs")

            # Phase 5: Verify integrity
            if not dry_run:
                logger.info("📋 Phase 5: Verifying data integrity")
                if not self.verify_data_integrity(backup_info):
                    logger.warning("⚠️ Data integrity verification failed")

            restore_result["phases"].append("verify_integrity")

            # Phase 6: Post-restore checks
            if not dry_run:
                logger.info("📋 Phase 6: Running post-restore checks")
                checks = self.run_post_restore_checks()
                restore_result["post_restore_checks"] = checks

                # Determine if restore is successful
                critical_checks = ["preflight", "reconciliation", "system_mode"]
                critical_passed = sum(
                    1 for check in critical_checks if checks.get(check, False)
                )
                restore_success = critical_passed == len(critical_checks)
            else:
                restore_success = True  # Dry run always "succeeds"

            restore_result["phases"].append("post_restore_checks")

            # Calculate final metrics
            self.restore_metrics["end_time"] = time.time()
            self.restore_metrics["rto_minutes"] = (
                self.restore_metrics["end_time"] - self.restore_metrics["start_time"]
            ) / 60

            restore_result.update(
                {
                    "status": "completed" if restore_success else "failed",
                    "restore_success": restore_success,
                    "metrics": self.restore_metrics,
                    "duration": self.restore_metrics["rto_minutes"],
                }
            )

            # Send Slack notification
            if not dry_run:
                self.send_restore_summary(restore_success, backup_info)

            logger.info(
                f"🆘 Disaster recovery restore {'completed' if restore_success else 'failed'}: "
                f"RTO={self.restore_metrics['rto_minutes']:.1f}min, "
                f"RPO={self.restore_metrics['rpo_minutes']:.0f}min"
            )

            return restore_result

        except Exception as e:
            logger.error(f"Error in disaster recovery restore: {e}")

            self.restore_metrics["end_time"] = time.time()
            self.restore_metrics["rto_minutes"] = (
                (self.restore_metrics["end_time"] - self.restore_metrics["start_time"])
                / 60
                if self.restore_metrics["start_time"]
                else 0
            )

            restore_result = {
                "status": "error",
                "error": str(e),
                "metrics": self.restore_metrics,
                "timestamp": time.time(),
            }

            if not dry_run:
                self.send_restore_summary(False, {"backup_age_minutes": 0})

            return restore_result


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Disaster Recovery Restore")
    parser.add_argument(
        "--execute", action="store_true", help="Execute disaster recovery restore"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (test without real restore)",
    )
    parser.add_argument(
        "--find-backup", action="store_true", help="Find and show latest backup info"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    dr = DisasterRecoveryRestore()

    if args.find_backup:
        backup_info = dr.find_latest_backup()
        if args.json:
            print(json.dumps(backup_info, indent=2, default=str))
        else:
            if backup_info:
                print(f"📦 Latest Backup Found:")
                print(f"  Timestamp: {backup_info['backup_timestamp']}")
                print(f"  Age: {backup_info['age_minutes']:.0f} minutes")
                print(f"  Files: {', '.join(backup_info['files'])}")
            else:
                print("❌ No valid backup found")
        return

    if args.execute or args.dry_run or not sys.argv[1:]:
        if not args.dry_run and not args.execute:
            print("⚠️  This will perform a FULL SYSTEM RESTORE")
            print("   All current data will be replaced with backup data")
            print("   Use --dry-run to test or --execute to confirm")
            sys.exit(1)

        result = dr.execute_restore(dry_run=args.dry_run or not args.execute)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            status = result["status"]
            emoji = (
                "✅"
                if status == "completed"
                else ("❌" if status == "failed" else "❓")
            )
            duration = result.get("duration", 0)

            print(
                f"{emoji} Disaster Recovery Restore: {status.upper()} ({duration:.1f}min)"
            )

            if "metrics" in result:
                metrics = result["metrics"]
                print(f"RTO: {metrics['rto_minutes']:.1f} minutes")
                print(f"RPO: {metrics['rpo_minutes']:.0f} minutes")
                print(
                    f"Components restored: {', '.join(metrics['components_restored'])}"
                )
                if metrics["components_failed"]:
                    print(
                        f"Components failed: {', '.join(metrics['components_failed'])}"
                    )

        # Exit code based on restore success
        if result.get("restore_success", False):
            sys.exit(0)
        else:
            sys.exit(1)

    parser.print_help()


if __name__ == "__main__":
    main()
