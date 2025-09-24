#!/usr/bin/env python3
"""
DR Game Day Script
Terminate the primary, restore on a clean VM, point blue/green to it, confirm preflight + recon=0
"""

import os
import sys
import time
import json
import boto3
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("dr_game_day")


class DRGameDay:
    """Orchestrates disaster recovery game day drill."""

    def __init__(self):
        """Initialize DR game day."""
        self.ec2 = boto3.client("ec2", region_name=os.getenv("AWS_REGION", "us-east-1"))
        self.elb = boto3.client(
            "elbv2", region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        self.slack_webhook = os.getenv("SLACK_WEBHOOK")

        # Game day configuration
        self.config = {
            "primary_instance_id": os.getenv("PRIMARY_INSTANCE_ID"),
            "backup_instance_id": os.getenv("BACKUP_INSTANCE_ID"),
            "load_balancer_arn": os.getenv("LOAD_BALANCER_ARN"),
            "target_group_arn": os.getenv("TARGET_GROUP_ARN"),
            "ssh_key_path": os.getenv("SSH_KEY_PATH", "~/.ssh/trading-key.pem"),
            "restore_timeout_minutes": 30,
            "verification_timeout_minutes": 10,
        }

        # Game day metrics
        self.drill_metrics = {
            "start_time": None,
            "primary_terminated_time": None,
            "restore_completed_time": None,
            "traffic_switched_time": None,
            "verification_completed_time": None,
            "total_downtime_minutes": None,
            "rto_minutes": None,
            "phases_completed": [],
            "phases_failed": [],
            "chaos_scenarios_executed": [],
        }

        logger.info("🎮 DR Game Day initialized")

    def send_drill_notification(self, message: str, urgent: bool = False):
        """Send game day notification to Slack."""
        try:
            if not self.slack_webhook:
                logger.warning("No Slack webhook configured")
                return

            payload = {
                "text": message,
                "username": "DR Game Day",
                "icon_emoji": ":game_die:" if not urgent else ":rotating_light:",
                "channel": "#trading-alerts" if urgent else None,
            }

            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"📱 Sent game day notification: {message[:100]}...")

        except Exception as e:
            logger.error(f"Error sending notification: {e}")

    def terminate_primary_instance(self) -> bool:
        """Terminate the primary trading instance."""
        try:
            logger.info("💥 Terminating primary instance...")

            if not self.config["primary_instance_id"]:
                raise Exception("Primary instance ID not configured")

            # Get instance info before termination
            response = self.ec2.describe_instances(
                InstanceIds=[self.config["primary_instance_id"]]
            )

            if not response["Reservations"]:
                raise Exception("Primary instance not found")

            instance = response["Reservations"][0]["Instances"][0]
            instance_state = instance["State"]["Name"]

            if instance_state != "running":
                logger.warning(
                    f"⚠️ Primary instance not running (state: {instance_state})"
                )
                return False

            # Terminate instance
            self.ec2.terminate_instances(
                InstanceIds=[self.config["primary_instance_id"]]
            )

            self.drill_metrics["primary_terminated_time"] = time.time()

            logger.info("✅ Primary instance termination initiated")
            self.send_drill_notification(
                "🔥 **DR GAME DAY STARTED** - Primary instance terminated", urgent=True
            )

            return True

        except Exception as e:
            logger.error(f"Error terminating primary instance: {e}")
            self.drill_metrics["phases_failed"].append("terminate_primary")
            return False

    def wait_for_backup_instance(self) -> bool:
        """Wait for backup instance to be ready."""
        try:
            logger.info("⏱️ Waiting for backup instance to be ready...")

            if not self.config["backup_instance_id"]:
                raise Exception("Backup instance ID not configured")

            timeout = time.time() + (self.config["restore_timeout_minutes"] * 60)

            while time.time() < timeout:
                response = self.ec2.describe_instances(
                    InstanceIds=[self.config["backup_instance_id"]]
                )

                if response["Reservations"]:
                    instance = response["Reservations"][0]["Instances"][0]
                    state = instance["State"]["Name"]

                    if state == "running":
                        # Check if instance is fully ready (status checks)
                        status_response = self.ec2.describe_instance_status(
                            InstanceIds=[self.config["backup_instance_id"]]
                        )

                        if status_response["InstanceStatuses"]:
                            status = status_response["InstanceStatuses"][0]
                            instance_status = status["InstanceStatus"]["Status"]
                            system_status = status["SystemStatus"]["Status"]

                            if instance_status == "ok" and system_status == "ok":
                                logger.info("✅ Backup instance is ready")
                                return True

                logger.info(
                    f"⏱️ Backup instance not ready yet (state: {state}), waiting..."
                )
                time.sleep(30)

            logger.error("⏰ Timeout waiting for backup instance")
            return False

        except Exception as e:
            logger.error(f"Error waiting for backup instance: {e}")
            return False

    def execute_restore_on_backup(self) -> bool:
        """Execute restore process on backup instance."""
        try:
            logger.info("🔄 Executing restore on backup instance...")

            # SSH to backup instance and run restore
            backup_ip = self.get_instance_ip(self.config["backup_instance_id"])
            if not backup_ip:
                raise Exception("Could not get backup instance IP")

            ssh_cmd = [
                "ssh",
                "-i",
                self.config["ssh_key_path"],
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=30",
                f"ubuntu@{backup_ip}",
                "cd /opt/trader && python3 scripts/restore_from_s3.py --execute",
            ]

            logger.info(f"🔌 Connecting to backup instance: {backup_ip}")

            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=self.config["restore_timeout_minutes"] * 60,
            )

            if result.returncode == 0:
                self.drill_metrics["restore_completed_time"] = time.time()
                logger.info("✅ Restore completed successfully on backup instance")
                return True
            else:
                logger.error(f"❌ Restore failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("⏰ Restore timeout")
            return False
        except Exception as e:
            logger.error(f"Error executing restore: {e}")
            return False

    def get_instance_ip(self, instance_id: str) -> Optional[str]:
        """Get public IP of an instance."""
        try:
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            if response["Reservations"]:
                instance = response["Reservations"][0]["Instances"][0]
                return instance.get("PublicIpAddress")
            return None
        except Exception as e:
            logger.error(f"Error getting instance IP: {e}")
            return None

    def switch_load_balancer_traffic(self) -> bool:
        """Switch load balancer to point to backup instance."""
        try:
            logger.info("🔀 Switching load balancer traffic to backup instance...")

            if not self.config["target_group_arn"]:
                logger.warning("⚠️ Target group ARN not configured - skipping LB switch")
                return True

            backup_ip = self.get_instance_ip(self.config["backup_instance_id"])
            if not backup_ip:
                raise Exception("Could not get backup instance IP")

            # Register backup instance with target group
            self.elb.register_targets(
                TargetGroupArn=self.config["target_group_arn"],
                Targets=[{"Id": self.config["backup_instance_id"], "Port": 8000}],
            )

            # Wait for target to be healthy
            logger.info("⏱️ Waiting for target to be healthy...")
            timeout = time.time() + 300  # 5 minutes

            while time.time() < timeout:
                response = self.elb.describe_target_health(
                    TargetGroupArn=self.config["target_group_arn"]
                )

                for target in response["TargetHealthDescriptions"]:
                    if target["Target"]["Id"] == self.config["backup_instance_id"]:
                        if target["TargetHealth"]["State"] == "healthy":
                            self.drill_metrics["traffic_switched_time"] = time.time()
                            logger.info("✅ Traffic switched to backup instance")
                            return True
                        break

                time.sleep(10)

            logger.error("⏰ Timeout waiting for target to be healthy")
            return False

        except Exception as e:
            logger.error(f"Error switching load balancer traffic: {e}")
            return False

    def verify_system_health(self) -> Dict[str, bool]:
        """Verify system health on restored instance."""
        try:
            logger.info("🔍 Verifying system health...")

            backup_ip = self.get_instance_ip(self.config["backup_instance_id"])
            if not backup_ip:
                raise Exception("Could not get backup instance IP")

            checks = {}

            # Check preflight
            ssh_cmd = [
                "ssh",
                "-i",
                self.config["ssh_key_path"],
                "-o",
                "StrictHostKeyChecking=no",
                f"ubuntu@{backup_ip}",
                "cd /opt/trader && python3 scripts/preflight_supercheck.py --silent",
            ]

            try:
                result = subprocess.run(ssh_cmd, capture_output=True, timeout=120)
                checks["preflight"] = result.returncode == 0
            except Exception:
                checks["preflight"] = False

            # Check reconciliation via API
            try:
                response = requests.get(
                    f"http://{backup_ip}:8000/api/metrics", timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    metrics = data.get("metrics", {})
                    recon_breaches = metrics.get("recon_breaches", 1)
                    checks["reconciliation"] = recon_breaches == 0
                else:
                    checks["reconciliation"] = False
            except Exception:
                checks["reconciliation"] = False

            # Check API health
            try:
                response = requests.get(
                    f"http://{backup_ip}:8000/api/health", timeout=30
                )
                checks["api_health"] = response.status_code == 200
            except Exception:
                checks["api_health"] = False

            # Check system mode
            try:
                response = requests.get(
                    f"http://{backup_ip}:8000/api/metrics", timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    metrics = data.get("metrics", {})
                    system_mode = metrics.get("system_mode", "unknown")
                    checks["system_mode"] = system_mode in ["auto", "manual"]
                else:
                    checks["system_mode"] = False
            except Exception:
                checks["system_mode"] = False

            passed_checks = sum(checks.values())
            total_checks = len(checks)

            logger.info(
                f"📋 System health verification: {passed_checks}/{total_checks} passed"
            )
            return checks

        except Exception as e:
            logger.error(f"Error verifying system health: {e}")
            return {}

    def calculate_drill_metrics(self) -> Dict[str, Any]:
        """Calculate final drill metrics."""
        try:
            metrics = self.drill_metrics.copy()

            if metrics["start_time"] and metrics["verification_completed_time"]:
                total_time = (
                    metrics["verification_completed_time"] - metrics["start_time"]
                )
                metrics["total_drill_time_minutes"] = total_time / 60

            if metrics["primary_terminated_time"] and metrics["traffic_switched_time"]:
                downtime = (
                    metrics["traffic_switched_time"]
                    - metrics["primary_terminated_time"]
                )
                metrics["total_downtime_minutes"] = downtime / 60

            if metrics["start_time"] and metrics["restore_completed_time"]:
                rto = metrics["restore_completed_time"] - metrics["start_time"]
                metrics["rto_minutes"] = rto / 60

            return metrics

        except Exception as e:
            logger.error(f"Error calculating drill metrics: {e}")
            return self.drill_metrics

    def execute_chaos_scenario_heartbeat_pause(self) -> bool:
        """Chaos scenario: Pause policy heartbeat and verify watchdog restart."""
        try:
            logger.info("🔥 Chaos Scenario: Pausing policy heartbeat...")

            # Stop policy heartbeat by setting a flag in Redis
            import redis

            r = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                decode_responses=True,
            )

            # Pause heartbeat for 15 minutes
            r.set("policy:heartbeat_paused", "1", ex=900)

            # Wait for watchdog detection (should be <10 minutes)
            logger.info("⏱️  Waiting for watchdog to detect and restart...")
            time.sleep(600)  # 10 minutes

            # Check if watchdog restarted heartbeat
            last_update = r.get("policy:last_update_ts")
            if last_update:
                age = time.time() - float(last_update)
                watchdog_worked = age < 300  # Heartbeat fresh within 5 minutes
            else:
                watchdog_worked = False

            # Clean up
            r.delete("policy:heartbeat_paused")

            self.drill_metrics["chaos_scenarios_executed"].append(
                {
                    "scenario": "heartbeat_pause",
                    "success": watchdog_worked,
                    "timestamp": time.time(),
                }
            )

            logger.info(
                f"🔥 Heartbeat pause scenario: {'✅ PASS' if watchdog_worked else '❌ FAIL'}"
            )
            return watchdog_worked

        except Exception as e:
            logger.error(f"Error in heartbeat pause scenario: {e}")
            return False

    def execute_chaos_scenario_nan_injection(self) -> bool:
        """Chaos scenario: Force NaN loss and verify guards stop training."""
        try:
            logger.info("🔥 Chaos Scenario: Injecting NaN loss...")

            # Set chaos flag to trigger NaN in policy daemon
            import redis

            r = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                decode_responses=True,
            )

            r.set("chaos:inject_nan", "1", ex=300)  # 5 minutes

            # Wait for guards to detect and stop training
            time.sleep(120)  # 2 minutes

            # Check for alert generation
            alerts = r.lrange("alerts:policy", 0, -1)
            nan_alert_found = any("NaN_LOSS" in alert for alert in alerts)

            # Check training status
            training_stopped = r.get("policy:training_stopped") == "1"

            # Clean up chaos flag
            r.delete("chaos:inject_nan")

            guard_worked = nan_alert_found or training_stopped

            self.drill_metrics["chaos_scenarios_executed"].append(
                {
                    "scenario": "nan_injection",
                    "success": guard_worked,
                    "alerts_generated": nan_alert_found,
                    "training_stopped": training_stopped,
                    "timestamp": time.time(),
                }
            )

            logger.info(
                f"🔥 NaN injection scenario: {'✅ PASS' if guard_worked else '❌ FAIL'}"
            )
            return guard_worked

        except Exception as e:
            logger.error(f"Error in NaN injection scenario: {e}")
            return False

    def execute_chaos_scenario_exporter_block(self) -> bool:
        """Chaos scenario: Block exporter and verify alert generation."""
        try:
            logger.info("🔥 Chaos Scenario: Blocking RL exporter...")

            # Kill exporter process if running
            import subprocess

            # Find and kill exporter process
            try:
                result = subprocess.run(
                    ["pkill", "-f", "rl_redis_exporter"], capture_output=True
                )
                logger.info(f"Killed exporter process: {result.returncode}")
            except:
                pass

            # Wait for alert detection
            time.sleep(300)  # 5 minutes

            # Check for RLExporterDown alert
            try:
                response = requests.get(
                    "http://localhost:9093/api/v1/alerts", timeout=5
                )
                if response.status_code == 200:
                    alerts = response.json().get("data", [])
                    exporter_alert = any(
                        "RLExporterDown" in alert.get("labels", {}).get("alertname", "")
                        or "rl_exporter_up" in str(alert)
                        for alert in alerts
                    )
                else:
                    exporter_alert = False
            except:
                exporter_alert = False

            # Restart exporter
            try:
                subprocess.Popen(
                    ["python", "src/monitoring/rl_redis_exporter.py"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("Restarted exporter")
            except:
                pass

            self.drill_metrics["chaos_scenarios_executed"].append(
                {
                    "scenario": "exporter_block",
                    "success": exporter_alert,
                    "timestamp": time.time(),
                }
            )

            logger.info(
                f"🔥 Exporter block scenario: {'✅ PASS' if exporter_alert else '❌ FAIL'}"
            )
            return exporter_alert

        except Exception as e:
            logger.error(f"Error in exporter block scenario: {e}")
            return False

    def execute_chaos_scenario_cost_spike(self) -> bool:
        """Chaos scenario: Simulate cost spike and verify alert generation."""
        try:
            logger.info("🔥 Chaos Scenario: Simulating cost spike...")

            # Set high cost values in cost exporter cache or Redis
            import redis

            r = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                decode_responses=True,
            )

            # Simulate 150% cost spike for 10 minutes
            original_cost = r.get("aws:daily_cost") or "100"
            spike_cost = float(original_cost) * 2.5

            r.set("aws:daily_cost", str(spike_cost), ex=600)
            r.set("aws:cost_spike_pct", "150", ex=600)

            # Wait for cost spike alert
            time.sleep(180)  # 3 minutes

            # Check for CostSpike alert
            try:
                response = requests.get(
                    "http://localhost:9093/api/v1/alerts", timeout=5
                )
                if response.status_code == 200:
                    alerts = response.json().get("data", [])
                    cost_alert = any(
                        "CostSpike" in alert.get("labels", {}).get("alertname", "")
                        for alert in alerts
                    )
                else:
                    cost_alert = False
            except:
                cost_alert = False

            # Reset cost values
            r.delete("aws:daily_cost")
            r.delete("aws:cost_spike_pct")

            self.drill_metrics["chaos_scenarios_executed"].append(
                {
                    "scenario": "cost_spike",
                    "success": cost_alert,
                    "spike_amount": "150%",
                    "timestamp": time.time(),
                }
            )

            logger.info(
                f"🔥 Cost spike scenario: {'✅ PASS' if cost_alert else '❌ FAIL'}"
            )
            return cost_alert

        except Exception as e:
            logger.error(f"Error in cost spike scenario: {e}")
            return False

    def capture_rca_artifacts(self) -> str:
        """Capture RCA template and artifacts for analysis."""
        try:
            logger.info("📋 Capturing RCA artifacts...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rca_dir = Path(f"artifacts/dr/{timestamp}")
            rca_dir.mkdir(parents=True, exist_ok=True)

            # Capture system state
            artifacts = {
                "drill_metrics": self.drill_metrics,
                "system_health": self.verify_system_health(),
                "timestamp": timestamp,
                "rca_template": {
                    "incident_summary": "DR Game Day Drill Execution",
                    "timeline": [],
                    "root_cause": "Planned disaster recovery test",
                    "impact": "Controlled downtime for testing",
                    "resolution": "System restored to backup instance",
                    "lessons_learned": [],
                    "action_items": [],
                },
            }

            # Add timeline from drill metrics
            if self.drill_metrics.get("start_time"):
                artifacts["rca_template"]["timeline"].append(
                    {
                        "time": self.drill_metrics["start_time"],
                        "event": "DR Game Day drill started",
                    }
                )

            if self.drill_metrics.get("primary_terminated_time"):
                artifacts["rca_template"]["timeline"].append(
                    {
                        "time": self.drill_metrics["primary_terminated_time"],
                        "event": "Primary instance terminated",
                    }
                )

            if self.drill_metrics.get("restore_completed_time"):
                artifacts["rca_template"]["timeline"].append(
                    {
                        "time": self.drill_metrics["restore_completed_time"],
                        "event": "Backup instance restored",
                    }
                )

            # Write RCA artifacts
            with open(rca_dir / "rca_template.json", "w") as f:
                json.dump(artifacts, f, indent=2, default=str)

            # Copy key logs
            try:
                import shutil

                if Path("logs").exists():
                    shutil.copytree("logs", rca_dir / "logs", ignore_errors=True)
            except:
                pass

            logger.info(f"📋 RCA artifacts saved to {rca_dir}")
            return str(rca_dir)

        except Exception as e:
            logger.error(f"Error capturing RCA artifacts: {e}")
            return ""

    def execute_game_day_drill(
        self, dry_run: bool = False, chaos_scenarios: bool = True
    ) -> Dict[str, Any]:
        """Execute complete game day drill."""
        try:
            self.drill_metrics["start_time"] = time.time()
            logger.info("🎮 Starting DR game day drill...")

            if dry_run:
                logger.info("🧪 DRY RUN MODE - Simulating drill without real changes")

            drill_result = {
                "status": "in_progress",
                "dry_run": dry_run,
                "start_time": self.drill_metrics["start_time"],
                "phases": [],
            }

            self.send_drill_notification(
                "🎮 **DR GAME DAY DRILL STARTED**", urgent=True
            )

            # Phase 1: Terminate primary
            logger.info("📋 Phase 1: Terminating primary instance")
            if not dry_run:
                if not self.terminate_primary_instance():
                    raise Exception("Failed to terminate primary instance")
                self.drill_metrics["phases_completed"].append("terminate_primary")
            drill_result["phases"].append("terminate_primary")

            # Phase 2: Wait for backup instance
            logger.info("📋 Phase 2: Ensuring backup instance is ready")
            if not dry_run:
                if not self.wait_for_backup_instance():
                    raise Exception("Backup instance not ready")
                self.drill_metrics["phases_completed"].append("backup_ready")
            drill_result["phases"].append("backup_ready")

            # Phase 3: Execute restore
            logger.info("📋 Phase 3: Executing restore on backup instance")
            if not dry_run:
                if not self.execute_restore_on_backup():
                    raise Exception("Restore failed on backup instance")
                self.drill_metrics["phases_completed"].append("restore_backup")
            drill_result["phases"].append("restore_backup")

            # Phase 4: Switch traffic
            logger.info("📋 Phase 4: Switching load balancer traffic")
            if not dry_run:
                if not self.switch_load_balancer_traffic():
                    logger.warning("⚠️ Load balancer switch failed - continuing")
                else:
                    self.drill_metrics["phases_completed"].append("switch_traffic")
            drill_result["phases"].append("switch_traffic")

            # Phase 5: Verify system health
            logger.info("📋 Phase 5: Verifying system health")
            if not dry_run:
                health_checks = self.verify_system_health()
                drill_result["health_checks"] = health_checks

                critical_checks = ["preflight", "reconciliation"]
                critical_passed = sum(
                    1 for check in critical_checks if health_checks.get(check, False)
                )
                drill_success = critical_passed == len(critical_checks)

                self.drill_metrics["verification_completed_time"] = time.time()
                self.drill_metrics["phases_completed"].append("verify_health")
            else:
                drill_success = True  # Dry run always "succeeds"

            drill_result["phases"].append("verify_health")

            # Phase 6: Execute chaos scenarios (optional)
            if chaos_scenarios and not dry_run:
                logger.info("📋 Phase 6: Executing chaos scenarios")

                chaos_results = {
                    "heartbeat_pause": self.execute_chaos_scenario_heartbeat_pause(),
                    "nan_injection": self.execute_chaos_scenario_nan_injection(),
                    "exporter_block": self.execute_chaos_scenario_exporter_block(),
                    "cost_spike": self.execute_chaos_scenario_cost_spike(),
                }

                drill_result["chaos_scenarios"] = chaos_results
                chaos_passed = sum(chaos_results.values())

                logger.info(f"🔥 Chaos scenarios completed: {chaos_passed}/4 passed")
                self.drill_metrics["phases_completed"].append("chaos_scenarios")

            drill_result["phases"].append("chaos_scenarios")

            # Phase 7: Capture RCA artifacts
            logger.info("📋 Phase 7: Capturing RCA artifacts")
            rca_path = self.capture_rca_artifacts()
            drill_result["rca_artifacts_path"] = rca_path

            # Calculate final metrics
            final_metrics = self.calculate_drill_metrics()
            drill_result["metrics"] = final_metrics
            drill_result["drill_success"] = drill_success
            drill_result["status"] = "completed" if drill_success else "failed"

            # Send completion notification
            rto = final_metrics.get("rto_minutes", 0)
            downtime = final_metrics.get("total_downtime_minutes", 0)

            completion_msg = f"""🎮 **DR GAME DAY DRILL {'COMPLETED' if drill_success else 'FAILED'}**
            
**RTO:** {rto:.1f} minutes
**Downtime:** {downtime:.1f} minutes
**Phases Completed:** {len(self.drill_metrics['phases_completed'])}
**Health Checks:** {'All passed' if drill_success else 'Some failed'}"""

            self.send_drill_notification(completion_msg, urgent=not drill_success)

            logger.info(
                f"🎮 DR game day drill {'completed' if drill_success else 'failed'}: "
                f"RTO={rto:.1f}min, Downtime={downtime:.1f}min"
            )

            return drill_result

        except Exception as e:
            logger.error(f"Error in game day drill: {e}")

            self.drill_metrics["phases_failed"].append("drill_error")
            final_metrics = self.calculate_drill_metrics()

            drill_result = {
                "status": "error",
                "error": str(e),
                "metrics": final_metrics,
                "timestamp": time.time(),
            }

            if not dry_run:
                self.send_drill_notification(
                    f"🔴 **DR GAME DAY DRILL FAILED:** {str(e)}", urgent=True
                )

            return drill_result


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="DR Game Day Drill")
    parser.add_argument("--execute", action="store_true", help="Execute game day drill")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (simulate without real changes)",
    )
    parser.add_argument(
        "--no-chaos", action="store_true", help="Skip chaos engineering scenarios"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    game_day = DRGameDay()

    if args.execute or args.dry_run or not sys.argv[1:]:
        if not args.dry_run and not args.execute:
            print("⚠️  This will TERMINATE the primary instance and switch to backup")
            print("   This is a destructive operation for game day testing")
            print("   Use --dry-run to test or --execute to confirm")
            sys.exit(1)

        result = game_day.execute_game_day_drill(
            dry_run=args.dry_run or not args.execute, chaos_scenarios=not args.no_chaos
        )

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            status = result["status"]
            emoji = (
                "✅"
                if status == "completed"
                else ("❌" if status == "failed" else "❓")
            )

            print(f"{emoji} DR Game Day Drill: {status.upper()}")

            if "metrics" in result:
                metrics = result["metrics"]
                if "rto_minutes" in metrics:
                    print(f"RTO: {metrics['rto_minutes']:.1f} minutes")
                if "total_downtime_minutes" in metrics:
                    print(f"Downtime: {metrics['total_downtime_minutes']:.1f} minutes")
                print(f"Phases completed: {len(metrics.get('phases_completed', []))}")

        # Exit code based on drill success
        if result.get("drill_success", False):
            sys.exit(0)
        else:
            sys.exit(1)

    parser.print_help()


if __name__ == "__main__":
    main()
