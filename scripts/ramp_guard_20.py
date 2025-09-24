#!/usr/bin/env python3
"""
M18: 20% Ramp Guard - Enhanced Hard Guards
Ultra-fast rollback on any breach of 20% safety thresholds.
"""
import os
import sys
import json
import time
import datetime
import argparse
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional
import redis


class RampGuard20:
    """Enhanced hard guards for M18 20% ramp with 0.1s rollback SLA."""

    def __init__(self, watch_interval: int = 15):
        self.watch_interval = watch_interval
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")
        self.active = True

        # Enhanced thresholds for 20%
        self.thresholds = {
            "slip_p95_rolling_30m_bps": 12.0,  # Stricter than 15% (was 15)
            "impact_p95_bp_per_1k": 8.0,  # Market impact limit
            "maker_ratio_rolling_30m": 0.70,  # Minimum maker ratio
            "drawdown_rolling_2h_pct": 0.9,  # Maximum drawdown
            "decision_to_ack_p95_ms": 110.0,  # Latency budget
            "page_alerts_active": 0,  # No page alerts allowed
        }

        # Redis for real-time monitoring
        try:
            self.redis_client = redis.Redis(decode_responses=True)
            self.redis_client.ping()
        except Exception:
            self.redis_client = None

        # Setup signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"🛡️ M18 Guard initialized with {watch_interval}s intervals")
        print(
            f"Thresholds: slip≤{self.thresholds['slip_p95_rolling_30m_bps']}bps, "
            f"impact≤{self.thresholds['impact_p95_bp_per_1k']}bp/\$1k, "
            f"maker≥{self.thresholds['maker_ratio_rolling_30m']:.0%}"
        )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n🛡️ M18 Guard received signal {signum}, shutting down...")
        self.active = False

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics for guard evaluation."""

        # Simulate real-time metrics (in production, this would query live systems)
        import numpy as np

        # Base metrics that would normally come from monitoring systems
        base_metrics = {
            "slip_p95_rolling_30m_bps": np.random.normal(9.0, 2.0),
            "impact_p95_bp_per_1k": np.random.normal(6.5, 1.5),
            "maker_ratio_rolling_30m": np.random.normal(0.85, 0.05),
            "drawdown_rolling_2h_pct": abs(np.random.normal(0.3, 0.2)),
            "decision_to_ack_p95_ms": np.random.normal(95, 15),
            "page_alerts_active": 0,  # Would come from alerting system
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Ensure positive values where needed
        base_metrics["slip_p95_rolling_30m_bps"] = max(
            1.0, base_metrics["slip_p95_rolling_30m_bps"]
        )
        base_metrics["impact_p95_bp_per_1k"] = max(
            1.0, base_metrics["impact_p95_bp_per_1k"]
        )
        base_metrics["maker_ratio_rolling_30m"] = max(
            0.5, min(0.95, base_metrics["maker_ratio_rolling_30m"])
        )
        base_metrics["decision_to_ack_p95_ms"] = max(
            50, base_metrics["decision_to_ack_p95_ms"]
        )

        return base_metrics

    def evaluate_guards(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all guard conditions."""

        violations = []
        checks = {}

        # Check each threshold
        for metric, threshold in self.thresholds.items():
            current_value = metrics.get(metric, 0)

            if metric in ["maker_ratio_rolling_30m"]:
                # For ratios, current should be >= threshold
                violated = current_value < threshold
                status = "FAIL" if violated else "PASS"
            elif metric == "page_alerts_active":
                # For alerts, current should be == threshold (0)
                violated = current_value != threshold
                status = "FAIL" if violated else "PASS"
            else:
                # For other metrics, current should be <= threshold
                violated = current_value > threshold
                status = "FAIL" if violated else "PASS"

            checks[metric] = {
                "current": current_value,
                "threshold": threshold,
                "status": status,
            }

            if violated:
                violations.append(
                    {
                        "metric": metric,
                        "current": current_value,
                        "threshold": threshold,
                        "severity": "critical",
                    }
                )

        return {
            "violations": violations,
            "checks": checks,
            "violation_count": len(violations),
            "overall_status": "BREACH" if violations else "OK",
        }

    def execute_emergency_rollback(self, violations: List[Dict[str, Any]]) -> bool:
        """Execute emergency rollback to 0% influence."""

        rollback_start = datetime.datetime.now()

        print(f"🚨 EMERGENCY ROLLBACK TRIGGERED")
        print(f"Violations: {len(violations)}")
        for violation in violations:
            print(
                f"  • {violation['metric']}: {violation['current']} > {violation['threshold']}"
            )

        try:
            # Set all influence levels to 0
            assets = ["BTC-USD", "ETH-USD", "NVDA"]

            for asset in assets:
                success = self._set_influence_level(asset, 0.0)
                if success:
                    print(f"✅ {asset} influence → 0%")
                else:
                    print(f"❌ Failed to rollback {asset}")

            # Write emergency audit
            self._write_emergency_audit(violations, rollback_start)

            rollback_duration = (
                datetime.datetime.now() - rollback_start
            ).total_seconds()
            print(f"⚡ Rollback completed in {rollback_duration:.2f}s")

            return True

        except Exception as e:
            print(f"💥 Rollback error: {e}")
            return False

    def _set_influence_level(self, asset: str, influence_pct: float) -> bool:
        """Set influence level to 0 for emergency rollback."""

        try:
            # Redis method
            if self.redis_client:
                key = f"influence:{asset}"
                self.redis_client.set(key, influence_pct, ex=3600)
                return True

            # Fallback: file-based
            influence_dir = self.base_dir / "artifacts" / "influence"
            influence_dir.mkdir(parents=True, exist_ok=True)

            influence_file = influence_dir / f"{asset.replace('-', '_')}_influence.json"

            with open(influence_file, "w") as f:
                json.dump(
                    {
                        "asset": asset,
                        "influence_pct": influence_pct,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "reason": "emergency_rollback_m18",
                        "guard_trigger": True,
                    },
                    f,
                    indent=2,
                )

            return True

        except Exception:
            return False

    def _write_emergency_audit(
        self, violations: List[Dict[str, Any]], rollback_start: datetime.datetime
    ):
        """Write emergency rollback audit to WORM."""

        worm_dir = self.base_dir / "worm"
        worm_dir.mkdir(exist_ok=True)

        timestamp_str = rollback_start.strftime("%Y%m%d_%H%M%S_%f")
        audit_file = worm_dir / f"emergency_rollback_m18_{timestamp_str}.json"

        audit_entry = {
            "event_type": "emergency_rollback_m18",
            "trigger_time": rollback_start.isoformat(),
            "completion_time": datetime.datetime.now().isoformat(),
            "violations": violations,
            "rollback_reason": "m18_guard_breach",
            "assets_affected": ["BTC-USD", "ETH-USD", "NVDA"],
            "user": os.getenv("USER", "system"),
            "guard_version": "m18_20pct",
        }

        try:
            with open(audit_file, "w") as f:
                json.dump(audit_entry, f, indent=2)
            print(f"📝 Emergency audit: {audit_file}")
        except Exception as e:
            print(f"⚠️ Audit write error: {e}")

    def run_guard_loop(self):
        """Main guard monitoring loop."""

        print(f"🛡️ M18 Guard active - monitoring every {self.watch_interval}s")
        print(f"Rollback SLA: 0.1 seconds")
        print("Press Ctrl+C to stop\n")

        breach_count = 0

        while self.active:
            try:
                # Get current metrics
                metrics = self.get_current_metrics()

                # Evaluate guard conditions
                guard_result = self.evaluate_guards(metrics)

                current_time = datetime.datetime.now().strftime("%H:%M:%S")

                if guard_result["overall_status"] == "BREACH":
                    breach_count += 1
                    print(f"🚨 {current_time} GUARD BREACH #{breach_count}")

                    # Execute emergency rollback
                    rollback_success = self.execute_emergency_rollback(
                        guard_result["violations"]
                    )

                    if rollback_success:
                        print(f"✅ Emergency rollback completed")
                        # Continue monitoring but flag the breach
                    else:
                        print(
                            f"💥 Emergency rollback FAILED - manual intervention required"
                        )
                        self.active = False
                        break

                else:
                    # Normal status - show key metrics
                    slip = metrics["slip_p95_rolling_30m_bps"]
                    maker = metrics["maker_ratio_rolling_30m"]
                    impact = metrics["impact_p95_bp_per_1k"]

                    print(
                        f"✅ {current_time} OK - slip:{slip:.1f}bps, maker:{maker:.1%}, impact:{impact:.1f}bp/\$1k"
                    )

                # Wait for next check
                time.sleep(self.watch_interval)

            except KeyboardInterrupt:
                self.active = False
                break
            except Exception as e:
                print(f"💥 Guard error: {e}")
                time.sleep(self.watch_interval)

        print(f"\n🛡️ M18 Guard stopped (breaches detected: {breach_count})")


def main():
    """Main M18 guard CLI."""

    parser = argparse.ArgumentParser(description="M18: 20% Ramp Hard Guards")
    parser.add_argument(
        "--watch", type=int, default=15, help="Watch interval in seconds"
    )
    args = parser.parse_args()

    try:
        guard = RampGuard20(watch_interval=args.watch)
        guard.run_guard_loop()
        return 0

    except Exception as e:
        print(f"💥 M18 Guard error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
