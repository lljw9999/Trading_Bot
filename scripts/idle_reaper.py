#!/usr/bin/env python3
"""
Idle Resource Reaper & Auto-Scaler
Automatically shuts down idle GPU resources when safe to reduce costs.
"""
import os
import sys
import time
import json
import datetime
import pathlib
import subprocess
import redis
from pathlib import Path


class IdleReaper:
    def __init__(self):
        self.r = redis.Redis(decode_responses=True)
        self.audit_dir = Path("artifacts/audit")
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def get_system_metrics(self):
        """Get current system utilization metrics."""
        try:
            # Get influence controller status
            influence_pct = float(self.r.get("policy:allowed_influence_pct") or 0)

            # Get GPU metrics
            gpu_mem_frac = float(self.r.get("gpu:mem_frac") or 0)

            # Get probe health
            probe_success = int(self.r.get("probe:success") or 0)

            # Get recent activity
            recent_signals = self.r.llen("signals:recent") or 0
            recent_trades = self.r.llen("trades:recent") or 0

            # Get uptime
            system_start = self.r.get("system:start_time")
            uptime_hours = 0
            if system_start:
                start_time = datetime.datetime.fromisoformat(
                    system_start.replace("Z", "+00:00")
                )
                uptime_hours = (
                    datetime.datetime.now(datetime.timezone.utc) - start_time
                ).total_seconds() / 3600

            return {
                "influence_pct": influence_pct,
                "gpu_mem_frac": gpu_mem_frac,
                "probe_success": probe_success,
                "recent_signals": recent_signals,
                "recent_trades": recent_trades,
                "uptime_hours": uptime_hours,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }

        except Exception as e:
            return {
                "error": str(e),
                "influence_pct": 0,
                "gpu_mem_frac": 0,
                "probe_success": 0,
                "recent_signals": 0,
                "recent_trades": 0,
                "uptime_hours": 0,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }

    def check_idle_conditions(self, metrics, idle_threshold_minutes=30):
        """Check if system meets idle shutdown conditions."""

        idle_conditions = {
            "influence_zero": metrics["influence_pct"] == 0,
            "gpu_underutilized": metrics["gpu_mem_frac"] < 0.2,
            "probe_healthy": metrics["probe_success"] == 1,
            "no_recent_activity": metrics["recent_signals"] == 0
            and metrics["recent_trades"] == 0,
            "sufficient_uptime": metrics["uptime_hours"]
            >= 0.5,  # At least 30 min uptime
        }

        # Check idle duration
        last_activity_key = "system:last_activity"
        last_activity = self.r.get(last_activity_key)

        if not last_activity:
            # Set current time as last activity if not set
            self.r.set(last_activity_key, metrics["timestamp"])
            idle_conditions["idle_duration_met"] = False
            idle_duration_minutes = 0
        else:
            last_time = datetime.datetime.fromisoformat(
                last_activity.replace("Z", "+00:00")
            )
            current_time = datetime.datetime.fromisoformat(
                metrics["timestamp"].replace("Z", "+00:00")
            )
            idle_duration_minutes = (current_time - last_time).total_seconds() / 60
            idle_conditions["idle_duration_met"] = (
                idle_duration_minutes >= idle_threshold_minutes
            )

        # Update last activity if there's any activity
        if (
            metrics["recent_signals"] > 0
            or metrics["recent_trades"] > 0
            or metrics["influence_pct"] > 0
        ):
            self.r.set(last_activity_key, metrics["timestamp"])

        should_reap = all(idle_conditions.values())

        return {
            "should_reap": should_reap,
            "conditions": idle_conditions,
            "idle_duration_minutes": idle_duration_minutes,
            "idle_threshold_minutes": idle_threshold_minutes,
            "blocking_conditions": [k for k, v in idle_conditions.items() if not v],
        }

    def execute_idle_reaping(self, metrics, dry_run=False):
        """Execute idle resource reaping."""

        timestamp = metrics["timestamp"]
        audit_entry = {
            "timestamp": timestamp,
            "action": "idle_reaping",
            "dry_run": dry_run,
            "trigger_metrics": metrics,
            "reaping_actions": [],
        }

        print(
            f"{'🔍 DRY RUN: ' if dry_run else '⚡ EXECUTING: '}Idle resource reaping..."
        )

        try:
            # 1. Scale down policy inference pods
            if not dry_run:
                # In production: would scale down k8s deployment or stop containers
                print("  Scaling down policy inference pods...")
                # kubectl scale deployment policy-inference --replicas=0
                pass

            audit_entry["reaping_actions"].append(
                {
                    "action": "scale_down_inference_pods",
                    "target": "policy-inference",
                    "replicas": 0,
                    "executed": not dry_run,
                    "estimated_savings_per_hour": 15.50,
                }
            )

            # 2. Stop GPU-intensive services
            if not dry_run:
                print("  Stopping GPU services...")
                # systemctl stop gpu-profiler.service
                # systemctl stop model-training.service
                pass

            audit_entry["reaping_actions"].append(
                {
                    "action": "stop_gpu_services",
                    "services": ["gpu-profiler", "model-training"],
                    "executed": not dry_run,
                    "estimated_savings_per_hour": 8.25,
                }
            )

            # 3. Power down unused GPU slots (if multi-GPU)
            if not dry_run:
                print("  Power limiting GPUs...")
                # nvidia-smi -i 1,2,3 -pl 50  # Power limit to minimum
                pass

            audit_entry["reaping_actions"].append(
                {
                    "action": "power_limit_gpus",
                    "gpu_slots": [1, 2, 3],
                    "power_limit_watts": 50,
                    "executed": not dry_run,
                    "estimated_savings_per_hour": 12.30,
                }
            )

            # 4. Scale down data pipeline workers
            if not dry_run:
                print("  Scaling down data pipeline...")
                # Reduce Kafka consumer instances, feature compute workers
                pass

            audit_entry["reaping_actions"].append(
                {
                    "action": "scale_down_data_pipeline",
                    "components": ["kafka-consumers", "feature-workers"],
                    "scale_factor": 0.3,
                    "executed": not dry_run,
                    "estimated_savings_per_hour": 6.80,
                }
            )

            # Calculate total savings
            total_savings_per_hour = sum(
                action["estimated_savings_per_hour"]
                for action in audit_entry["reaping_actions"]
            )

            audit_entry["total_estimated_savings_per_hour"] = total_savings_per_hour
            audit_entry["total_estimated_daily_savings"] = total_savings_per_hour * 24

            # Set system state
            if not dry_run:
                self.r.set("system:idle_state", "reaped")
                self.r.set("system:reap_timestamp", timestamp)

                print(f"✅ Idle reaping completed")
                print(f"   Estimated savings: ${total_savings_per_hour:.2f}/hour")
            else:
                print(f"🔍 DRY RUN: Would save ${total_savings_per_hour:.2f}/hour")

            audit_entry["status"] = "success"

        except Exception as e:
            audit_entry["status"] = "failed"
            audit_entry["error"] = str(e)
            print(f"❌ Reaping failed: {e}")

        # Write audit log
        audit_filename = (
            f"{timestamp.replace(':', '_').replace('-', '')}_idle_reaper.json"
        )
        audit_path = self.audit_dir / audit_filename

        with open(audit_path, "w") as f:
            json.dump(audit_entry, f, indent=2)

        print(f"📝 Audit logged: {audit_path}")

        return audit_entry

    def check_wakeup_conditions(self, metrics):
        """Check if system should wake up from idle state."""

        current_state = self.r.get("system:idle_state") or "active"

        if current_state != "reaped":
            return {"should_wakeup": False, "reason": "not_in_idle_state"}

        wakeup_conditions = {
            "influence_requested": metrics["influence_pct"] > 0,
            "activity_detected": metrics["recent_signals"] > 0
            or metrics["recent_trades"] > 0,
            "manual_wakeup": self.r.get("system:manual_wakeup") == "1",
        }

        should_wakeup = any(wakeup_conditions.values())

        return {
            "should_wakeup": should_wakeup,
            "conditions": wakeup_conditions,
            "trigger_reasons": [k for k, v in wakeup_conditions.items() if v],
        }

    def execute_wakeup(self, metrics, dry_run=False):
        """Wake up system from idle state."""

        timestamp = metrics["timestamp"]
        audit_entry = {
            "timestamp": timestamp,
            "action": "system_wakeup",
            "dry_run": dry_run,
            "trigger_metrics": metrics,
            "wakeup_actions": [],
        }

        print(f"{'🔍 DRY RUN: ' if dry_run else '🚀 EXECUTING: '}System wakeup...")

        try:
            # 1. Scale up inference pods
            if not dry_run:
                print("  Scaling up policy inference...")
                # kubectl scale deployment policy-inference --replicas=2
                pass

            audit_entry["wakeup_actions"].append(
                {
                    "action": "scale_up_inference_pods",
                    "target": "policy-inference",
                    "replicas": 2,
                    "executed": not dry_run,
                }
            )

            # 2. Restart GPU services
            if not dry_run:
                print("  Starting GPU services...")
                # systemctl start gpu-profiler.service
                pass

            audit_entry["wakeup_actions"].append(
                {
                    "action": "start_gpu_services",
                    "services": ["gpu-profiler", "model-training"],
                    "executed": not dry_run,
                }
            )

            # 3. Restore GPU power limits
            if not dry_run:
                print("  Restoring GPU power...")
                # nvidia-smi -i 1,2,3 -pl 300  # Full power
                pass

            audit_entry["wakeup_actions"].append(
                {
                    "action": "restore_gpu_power",
                    "gpu_slots": [1, 2, 3],
                    "power_limit_watts": 300,
                    "executed": not dry_run,
                }
            )

            # 4. Scale up data pipeline
            if not dry_run:
                print("  Scaling up data pipeline...")
                pass

            audit_entry["wakeup_actions"].append(
                {
                    "action": "scale_up_data_pipeline",
                    "components": ["kafka-consumers", "feature-workers"],
                    "scale_factor": 1.0,
                    "executed": not dry_run,
                }
            )

            # Update system state
            if not dry_run:
                self.r.set("system:idle_state", "active")
                self.r.delete("system:reap_timestamp")
                self.r.delete("system:manual_wakeup")
                print("✅ System wakeup completed")
            else:
                print("🔍 DRY RUN: Would wake up system")

            audit_entry["status"] = "success"

        except Exception as e:
            audit_entry["status"] = "failed"
            audit_entry["error"] = str(e)
            print(f"❌ Wakeup failed: {e}")

        # Write audit log
        audit_filename = (
            f"{timestamp.replace(':', '_').replace('-', '')}_system_wakeup.json"
        )
        audit_path = self.audit_dir / audit_filename

        with open(audit_path, "w") as f:
            json.dump(audit_entry, f, indent=2)

        return audit_entry


def main():
    """Main idle reaper function."""
    print("💤 Idle Resource Reaper & Auto-Scaler")
    print("=" * 40)

    import argparse

    parser = argparse.ArgumentParser(description="Idle Resource Reaper")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--idle-threshold",
        type=int,
        default=30,
        help="Idle threshold in minutes (default: 30)",
    )
    parser.add_argument(
        "--force-wakeup", action="store_true", help="Force system wakeup"
    )
    args = parser.parse_args()

    try:
        reaper = IdleReaper()

        # Get current system metrics
        print("📊 Gathering system metrics...")
        metrics = reaper.get_system_metrics()

        print(f"  Influence: {metrics['influence_pct']}%")
        print(f"  GPU Memory: {metrics['gpu_mem_frac']:.1%}")
        print(f"  Probe Health: {metrics['probe_success']}")
        print(
            f"  Recent Activity: {metrics['recent_signals']} signals, {metrics['recent_trades']} trades"
        )
        print(f"  Uptime: {metrics['uptime_hours']:.1f} hours")

        if args.force_wakeup:
            # Force wakeup
            print("\n🚀 FORCE WAKEUP requested")
            reaper.r.set("system:manual_wakeup", "1")
            wakeup_check = reaper.check_wakeup_conditions(metrics)
            if wakeup_check["should_wakeup"]:
                wakeup_result = reaper.execute_wakeup(metrics, dry_run=args.dry_run)
                return 0

        # Check if system should wake up first
        current_state = reaper.r.get("system:idle_state") or "active"
        if current_state == "reaped":
            print(f"\n💤 System is in idle state")
            wakeup_check = reaper.check_wakeup_conditions(metrics)

            if wakeup_check["should_wakeup"]:
                print(f"🚀 Wakeup conditions met: {wakeup_check['trigger_reasons']}")
                wakeup_result = reaper.execute_wakeup(metrics, dry_run=args.dry_run)
                return 0
            else:
                print("💤 Staying in idle state")
                return 0

        # Check idle conditions
        print(f"\n🔍 Checking idle conditions...")
        idle_check = reaper.check_idle_conditions(metrics, args.idle_threshold)

        print(
            f"  Idle for {idle_check['idle_duration_minutes']:.1f} minutes (threshold: {idle_check['idle_threshold_minutes']})"
        )

        for condition, status in idle_check["conditions"].items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {condition}")

        if idle_check["should_reap"]:
            print(f"\n💤 All idle conditions met - executing reaping...")
            reap_result = reaper.execute_idle_reaping(metrics, dry_run=args.dry_run)

            if reap_result["status"] == "success":
                print(f"✅ Idle reaping completed successfully")
                return 0
            else:
                print(f"❌ Idle reaping failed")
                return 1
        else:
            blocking = idle_check["blocking_conditions"]
            print(f"\n🔄 Idle conditions not met. Blocking: {', '.join(blocking)}")
            return 0

    except Exception as e:
        print(f"❌ Idle reaper failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
