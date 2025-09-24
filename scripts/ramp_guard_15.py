#!/usr/bin/env python3
"""
M17: 15% Ramp Hard Guards
Real-time monitoring with instant rollback triggers (2-second SLA).
"""
import os
import sys
import json
import time
import datetime
import argparse
import redis
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


class RampGuard15:
    """Hard guards for 15% ramp with instant rollback."""

    def __init__(self):
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

        # Redis connection for metrics
        try:
            self.redis_client = redis.Redis(decode_responses=True)
            self.redis_client.ping()
        except Exception as e:
            print(f"⚠️ Redis not available: {e}")
            self.redis_client = None

        # Guard configuration
        self.guard_config = {
            # Slippage guards
            "slip_p95_30min_threshold": 15.0,  # 15 bps rolling 30min
            "slip_p95_10min_threshold": 18.0,  # 18 bps rolling 10min (early warning)
            # Risk guards
            "drawdown_2h_threshold": 1.0,  # 1.0% rolling 2h
            "drawdown_30min_threshold": 0.6,  # 0.6% rolling 30min
            # Execution quality guards
            "maker_ratio_30min_threshold": 0.65,  # 65% minimum maker ratio
            "cancel_ratio_30min_threshold": 0.50,  # 50% maximum cancel ratio
            # Latency guards
            "latency_p95_threshold_ms": 150,  # 150ms P95 latency
            "latency_budget_fail_count": 3,  # 3 consecutive budget failures
            # Alert guards
            "alert_page_immediate": True,  # Any page = immediate rollback
            # Rollback SLA
            "rollback_sla_seconds": 2,  # 2-second rollback SLA
        }

        # State tracking
        self.monitoring_active = False
        self.rollback_in_progress = False
        self.metrics_buffer = {}
        self.alert_history = []
        self.last_rollback_time = None

        # WORM audit
        self.worm_dir = self.base_dir / "worm"
        self.worm_dir.mkdir(exist_ok=True)

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics for guard evaluation."""

        metrics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "collection_success": False,
        }

        try:
            # Slippage metrics (simulated from optimized model)
            slip_metrics = self.get_slippage_metrics()
            metrics.update(slip_metrics)

            # Risk metrics
            risk_metrics = self.get_risk_metrics()
            metrics.update(risk_metrics)

            # Execution metrics
            exec_metrics = self.get_execution_metrics()
            metrics.update(exec_metrics)

            # Alert status
            alert_metrics = self.get_alert_status()
            metrics.update(alert_metrics)

            metrics["collection_success"] = True

        except Exception as e:
            metrics["collection_error"] = str(e)
            print(f"⚠️ Metrics collection error: {e}")

        return metrics

    def get_slippage_metrics(self) -> Dict[str, Any]:
        """Get current slippage metrics."""

        # Simulate real-time slippage with M16.1 optimizations
        # In production, this would query actual execution data

        base_slip = 9.4  # M16.1 baseline

        # Add some realistic variation
        current_slip_p95_10min = base_slip + np.random.normal(0, 2.0)
        current_slip_p95_30min = base_slip + np.random.normal(0, 1.5)

        # Occasionally simulate spikes for testing
        if np.random.random() < 0.05:  # 5% chance of spike
            spike_magnitude = np.random.uniform(10, 25)
            current_slip_p95_10min += spike_magnitude
            current_slip_p95_30min += spike_magnitude * 0.6

        return {
            "slip_p95_10min": max(0, current_slip_p95_10min),
            "slip_p95_30min": max(0, current_slip_p95_30min),
            "slip_avg_10min": current_slip_p95_10min * 0.6,
            "slip_samples_10min": np.random.randint(50, 150),
        }

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""

        # Simulate P&L and drawdown
        # In production, would query actual portfolio metrics

        base_pnl_30min = 25  # $25 per 30min
        current_pnl_30min = base_pnl_30min + np.random.normal(0, 15)

        # Calculate rolling drawdown
        current_drawdown_30min = (
            max(0, -current_pnl_30min / 10000) * 100
        )  # As percentage
        current_drawdown_2h = current_drawdown_30min * np.random.uniform(0.8, 1.4)

        # Occasionally simulate drawdown for testing
        if np.random.random() < 0.02:  # 2% chance of significant drawdown
            drawdown_spike = np.random.uniform(0.8, 1.5)
            current_drawdown_30min += drawdown_spike
            current_drawdown_2h += drawdown_spike * 0.8

        return {
            "pnl_30min_usd": current_pnl_30min,
            "pnl_2h_usd": current_pnl_30min * 4,
            "drawdown_30min_pct": current_drawdown_30min,
            "drawdown_2h_pct": current_drawdown_2h,
            "positions_count": np.random.randint(3, 8),
        }

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get current execution quality metrics."""

        # Simulate execution metrics with M16.1 improvements
        base_maker_ratio = 0.87  # M16.1 baseline

        current_maker_ratio = base_maker_ratio + np.random.normal(0, 0.05)
        current_cancel_ratio = 0.25 + np.random.normal(0, 0.08)
        current_latency_p95 = 85 + np.random.normal(0, 20)  # Good latency

        # Occasionally simulate execution degradation
        if np.random.random() < 0.03:  # 3% chance
            current_maker_ratio -= np.random.uniform(0.1, 0.25)
            current_cancel_ratio += np.random.uniform(0.1, 0.3)
            current_latency_p95 += np.random.uniform(50, 150)

        return {
            "maker_ratio_30min": max(0.3, min(0.98, current_maker_ratio)),
            "cancel_ratio_30min": max(0.05, min(0.8, current_cancel_ratio)),
            "latency_p95_ms": max(20, current_latency_p95),
            "fills_30min": np.random.randint(80, 200),
        }

    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert/page status."""

        # Simulate alert system status
        # In production, would query actual alerting system

        active_pages = []

        # Rarely simulate pages for testing
        if np.random.random() < 0.01:  # 1% chance of page
            active_pages.append(
                {
                    "alert": "execution_latency_high",
                    "severity": "warning",
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

        return {
            "active_pages": active_pages,
            "page_count_1h": len(active_pages),
            "last_page_time": active_pages[0]["timestamp"] if active_pages else None,
        }

    def evaluate_guards(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all guard conditions and determine if rollback needed."""

        guard_results = {
            "rollback_required": False,
            "triggered_guards": [],
            "guard_checks": {},
        }

        # 1. Slippage guards
        slip_p95_30min = metrics.get("slip_p95_30min", 0)
        if slip_p95_30min > self.guard_config["slip_p95_30min_threshold"]:
            guard_results["triggered_guards"].append(
                {
                    "guard": "slip_p95_30min",
                    "value": slip_p95_30min,
                    "threshold": self.guard_config["slip_p95_30min_threshold"],
                    "severity": "critical",
                }
            )
            guard_results["rollback_required"] = True

        guard_results["guard_checks"]["slip_p95_30min"] = {
            "value": slip_p95_30min,
            "threshold": self.guard_config["slip_p95_30min_threshold"],
            "status": (
                "FAIL"
                if slip_p95_30min > self.guard_config["slip_p95_30min_threshold"]
                else "PASS"
            ),
        }

        # 2. Drawdown guards
        drawdown_2h = metrics.get("drawdown_2h_pct", 0)
        if drawdown_2h > self.guard_config["drawdown_2h_threshold"]:
            guard_results["triggered_guards"].append(
                {
                    "guard": "drawdown_2h",
                    "value": drawdown_2h,
                    "threshold": self.guard_config["drawdown_2h_threshold"],
                    "severity": "critical",
                }
            )
            guard_results["rollback_required"] = True

        guard_results["guard_checks"]["drawdown_2h"] = {
            "value": drawdown_2h,
            "threshold": self.guard_config["drawdown_2h_threshold"],
            "status": (
                "FAIL"
                if drawdown_2h > self.guard_config["drawdown_2h_threshold"]
                else "PASS"
            ),
        }

        # 3. Maker ratio guard
        maker_ratio = metrics.get("maker_ratio_30min", 1.0)
        if maker_ratio < self.guard_config["maker_ratio_30min_threshold"]:
            guard_results["triggered_guards"].append(
                {
                    "guard": "maker_ratio_30min",
                    "value": maker_ratio,
                    "threshold": self.guard_config["maker_ratio_30min_threshold"],
                    "severity": "high",
                }
            )
            guard_results["rollback_required"] = True

        guard_results["guard_checks"]["maker_ratio_30min"] = {
            "value": maker_ratio,
            "threshold": self.guard_config["maker_ratio_30min_threshold"],
            "status": (
                "FAIL"
                if maker_ratio < self.guard_config["maker_ratio_30min_threshold"]
                else "PASS"
            ),
        }

        # 4. Latency guard
        latency_p95 = metrics.get("latency_p95_ms", 0)
        if latency_p95 > self.guard_config["latency_p95_threshold_ms"]:
            guard_results["triggered_guards"].append(
                {
                    "guard": "latency_p95",
                    "value": latency_p95,
                    "threshold": self.guard_config["latency_p95_threshold_ms"],
                    "severity": "medium",
                }
            )
            # Only rollback on sustained latency issues
            if latency_p95 > self.guard_config["latency_p95_threshold_ms"] * 1.5:
                guard_results["rollback_required"] = True

        guard_results["guard_checks"]["latency_p95"] = {
            "value": latency_p95,
            "threshold": self.guard_config["latency_p95_threshold_ms"],
            "status": (
                "FAIL"
                if latency_p95 > self.guard_config["latency_p95_threshold_ms"]
                else "PASS"
            ),
        }

        # 5. Alert/page guard
        active_pages = metrics.get("active_pages", [])
        if active_pages and self.guard_config["alert_page_immediate"]:
            guard_results["triggered_guards"].append(
                {
                    "guard": "alert_page",
                    "value": len(active_pages),
                    "threshold": 0,
                    "severity": "critical",
                    "pages": active_pages,
                }
            )
            guard_results["rollback_required"] = True

        guard_results["guard_checks"]["alert_pages"] = {
            "value": len(active_pages),
            "threshold": 0,
            "status": "FAIL" if active_pages else "PASS",
        }

        return guard_results

    def execute_rollback(
        self, triggered_guards: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute emergency rollback to 0% influence."""

        if self.rollback_in_progress:
            return {"success": False, "reason": "rollback_already_in_progress"}

        self.rollback_in_progress = True
        rollback_start = time.time()

        print(f"🚨 EMERGENCY ROLLBACK TRIGGERED")
        for guard in triggered_guards:
            print(f"   • {guard['guard']}: {guard['value']} > {guard['threshold']}")

        rollback_log = {
            "rollback_id": f"rollback15_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "trigger_time": datetime.datetime.now().isoformat(),
            "triggered_guards": triggered_guards,
            "rollback_steps": [],
        }

        try:
            # 1. Set all assets to 0% influence immediately
            assets = ["BTC-USD", "ETH-USD", "SOL-USD", "NVDA"]

            for asset in assets:
                step_start = time.time()
                success = self.set_influence_to_zero(asset)
                step_duration = time.time() - step_start

                rollback_log["rollback_steps"].append(
                    {
                        "asset": asset,
                        "success": success,
                        "duration_seconds": step_duration,
                    }
                )

                if not success:
                    print(f"❌ Failed to rollback {asset}")

            # 2. Set global kill switch
            kill_switch_success = self.activate_kill_switch()
            rollback_log["kill_switch_activated"] = kill_switch_success

            total_rollback_time = time.time() - rollback_start
            rollback_log["total_duration_seconds"] = total_rollback_time

            # Check SLA compliance
            sla_met = total_rollback_time <= self.guard_config["rollback_sla_seconds"]
            rollback_log["sla_met"] = sla_met

            if sla_met:
                print(
                    f"✅ Rollback completed in {total_rollback_time:.2f}s (SLA: {self.guard_config['rollback_sla_seconds']}s)"
                )
            else:
                print(
                    f"⚠️ Rollback took {total_rollback_time:.2f}s (exceeded {self.guard_config['rollback_sla_seconds']}s SLA)"
                )

            rollback_log["status"] = "completed"
            self.last_rollback_time = datetime.datetime.now()

        except Exception as e:
            rollback_log["status"] = "failed"
            rollback_log["error"] = str(e)
            print(f"❌ Rollback error: {e}")

        finally:
            self.rollback_in_progress = False

        # Write WORM audit
        self.write_worm_audit("ramp15_rollback", rollback_log)

        return rollback_log

    def set_influence_to_zero(self, asset: str) -> bool:
        """Set specific asset influence to 0%."""

        try:
            # Redis method
            if self.redis_client:
                key = f"influence:{asset}"
                self.redis_client.set(key, 0, ex=86400)  # 24h expiry
                return True

            # Fallback: file method
            influence_dir = self.base_dir / "artifacts" / "influence"
            influence_dir.mkdir(parents=True, exist_ok=True)

            influence_file = influence_dir / f"{asset.replace('-', '_')}_influence.json"

            with open(influence_file, "w") as f:
                json.dump(
                    {
                        "asset": asset,
                        "influence_pct": 0,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "reason": "emergency_rollback",
                    },
                    f,
                    indent=2,
                )

            return True

        except Exception as e:
            print(f"⚠️ Failed to zero {asset}: {e}")
            return False

    def activate_kill_switch(self) -> bool:
        """Activate global kill switch."""

        try:
            kill_switch_file = self.base_dir / "artifacts" / "kill_switch_active"

            with open(kill_switch_file, "w") as f:
                json.dump(
                    {
                        "activated": True,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "reason": "ramp15_guard_triggered",
                        "rollback_id": f"rollback15_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    },
                    f,
                    indent=2,
                )

            print(f"🛑 Kill switch activated: {kill_switch_file}")
            return True

        except Exception as e:
            print(f"⚠️ Kill switch activation failed: {e}")
            return False

    def write_worm_audit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write WORM audit entry."""

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        audit_file = self.worm_dir / f"{event_type}_{timestamp_str}.json"

        audit_entry = {
            "event_type": event_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "data": data,
            "user": os.getenv("USER", "system"),
            "guard_config": self.guard_config,
        }

        try:
            with open(audit_file, "w") as f:
                json.dump(audit_entry, f, indent=2)
        except Exception as e:
            print(f"⚠️ WORM audit error: {e}")

    def monitoring_loop(self, watch_interval: int) -> None:
        """Main monitoring loop."""

        print(f"🛡️ M17 Hard Guards Active")
        print(f"   Watch interval: {watch_interval}s")
        print(f"   Rollback SLA: {self.guard_config['rollback_sla_seconds']}s")
        print(f"   Guards: slippage, drawdown, maker ratio, latency, alerts")
        print("=" * 50)

        self.monitoring_active = True
        iteration = 0

        try:
            while self.monitoring_active:
                iteration += 1
                loop_start = time.time()

                # Collect metrics
                metrics = self.collect_metrics()

                if metrics.get("collection_success", False):
                    # Evaluate guards
                    guard_results = self.evaluate_guards(metrics)

                    # Log status every 10 iterations or if issues detected
                    if iteration % 10 == 0 or guard_results["triggered_guards"]:
                        print(
                            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                            f"Slip: {metrics.get('slip_p95_30min', 0):.1f}bps, "
                            f"DD: {metrics.get('drawdown_2h_pct', 0):.2f}%, "
                            f"Maker: {metrics.get('maker_ratio_30min', 0):.1%}, "
                            f"Latency: {metrics.get('latency_p95_ms', 0):.0f}ms"
                        )

                    # Execute rollback if needed
                    if guard_results["rollback_required"]:
                        rollback_result = self.execute_rollback(
                            guard_results["triggered_guards"]
                        )

                        if rollback_result.get("status") == "completed":
                            print(f"🛑 M17 ROLLBACK COMPLETED - Monitoring continues")
                        else:
                            print(
                                f"❌ M17 ROLLBACK FAILED - Manual intervention required"
                            )

                # Sleep until next check
                loop_duration = time.time() - loop_start
                sleep_time = max(0, watch_interval - loop_duration)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
        finally:
            self.monitoring_active = False
            print("🛡️ M17 Hard Guards deactivated")

    def run_guard(self, watch_interval: int) -> int:
        """Run guard monitoring."""

        try:
            self.monitoring_loop(watch_interval)
            return 0
        except Exception as e:
            print(f"❌ Guard error: {e}")
            return 1


def main():
    """Main M17 guard CLI."""
    parser = argparse.ArgumentParser(description="M17: 15% Ramp Hard Guards")
    parser.add_argument(
        "--watch", type=int, default=30, help="Watch interval in seconds"
    )
    args = parser.parse_args()

    try:
        guard = RampGuard15()
        return guard.run_guard(args.watch)

    except Exception as e:
        print(f"❌ M17 guard startup error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
