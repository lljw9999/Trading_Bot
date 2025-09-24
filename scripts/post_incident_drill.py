#!/usr/bin/env python3
"""
Post-Incident Drill: Stress Test M18 Guard Controls
Pause → inject slip spike → verify guard rollback → resume
"""
import os
import sys
import json
import time
import datetime
import argparse
import signal
from pathlib import Path
from typing import Dict, Any, List
import redis


class PostIncidentDrill:
    """Simulate controlled incidents to test M18 guard responses."""

    def __init__(self):
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")
        self.drill_session_id = (
            f"drill_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Redis for simulation
        try:
            self.redis_client = redis.Redis(decode_responses=True)
            self.redis_client.ping()
        except Exception:
            self.redis_client = None

        # WORM audit trail
        self.worm_dir = self.base_dir / "worm"
        self.worm_dir.mkdir(exist_ok=True)

        print(f"🎪 Post-Incident Drill initialized: {self.drill_session_id}")

    def simulate_slip_spike(
        self, target_bps: float = 25.0, duration_seconds: int = 10
    ) -> Dict[str, Any]:
        """Simulate a slippage spike to test guard response."""

        start_time = datetime.datetime.now()
        print(f"💥 Simulating slip spike: {target_bps} bps for {duration_seconds}s")

        # Create simulated metrics showing high slippage
        spike_metrics = {
            "slip_p95_rolling_30m_bps": target_bps,
            "impact_p95_bp_per_1k": 6.0,
            "maker_ratio_rolling_30m": 0.85,
            "drawdown_rolling_2h_pct": 0.2,
            "decision_to_ack_p95_ms": 95.0,
            "page_alerts_active": 0,
            "simulation": True,
            "spike_start": start_time.isoformat(),
            "spike_target_bps": target_bps,
        }

        # Write simulation signal for guard to pick up
        if self.redis_client:
            self.redis_client.setex(
                "drill:slip_spike", duration_seconds, json.dumps(spike_metrics)
            )

        # Create file-based simulation signal as backup
        sim_file = (
            self.base_dir
            / "artifacts"
            / "drill"
            / f"slip_spike_{self.drill_session_id}.json"
        )
        sim_file.parent.mkdir(parents=True, exist_ok=True)

        with open(sim_file, "w") as f:
            json.dump(spike_metrics, f, indent=2)

        print(f"📡 Spike signal active: redis=drill:slip_spike, file={sim_file}")

        return {
            "spike_start": start_time.isoformat(),
            "target_bps": target_bps,
            "duration_seconds": duration_seconds,
            "simulation_file": str(sim_file),
        }

    def simulate_drawdown_breach(
        self, target_pct: float = 1.2, duration_seconds: int = 8
    ) -> Dict[str, Any]:
        """Simulate a drawdown breach to test guard response."""

        start_time = datetime.datetime.now()
        print(
            f"📉 Simulating drawdown breach: {target_pct:.1%} for {duration_seconds}s"
        )

        breach_metrics = {
            "slip_p95_rolling_30m_bps": 8.0,
            "impact_p95_bp_per_1k": 5.5,
            "maker_ratio_rolling_30m": 0.88,
            "drawdown_rolling_2h_pct": target_pct / 100.0,  # Convert to decimal
            "decision_to_ack_p95_ms": 85.0,
            "page_alerts_active": 0,
            "simulation": True,
            "breach_start": start_time.isoformat(),
            "breach_target_pct": target_pct,
        }

        # Write simulation signal
        if self.redis_client:
            self.redis_client.setex(
                "drill:drawdown_breach", duration_seconds, json.dumps(breach_metrics)
            )

        # File-based backup
        sim_file = (
            self.base_dir
            / "artifacts"
            / "drill"
            / f"drawdown_breach_{self.drill_session_id}.json"
        )

        with open(sim_file, "w") as f:
            json.dump(breach_metrics, f, indent=2)

        print(f"📡 Breach signal active: redis=drill:drawdown_breach, file={sim_file}")

        return {
            "breach_start": start_time.isoformat(),
            "target_pct": target_pct,
            "duration_seconds": duration_seconds,
            "simulation_file": str(sim_file),
        }

    def verify_guard_response(
        self, expected_rollback: bool = True, timeout_seconds: int = 30
    ) -> Dict[str, Any]:
        """Verify that guards responded correctly to the simulated incident."""

        start_check = datetime.datetime.now()
        print(f"🔍 Verifying guard response (timeout: {timeout_seconds}s)...")

        # Look for emergency rollback audit files
        rollback_files = []
        response_detected = False

        while (datetime.datetime.now() - start_check).total_seconds() < timeout_seconds:
            # Check for new emergency rollback WORM files
            emergency_files = list(self.worm_dir.glob(f"emergency_rollback_m18_*.json"))

            for file in emergency_files:
                if file.stat().st_mtime > start_check.timestamp():
                    rollback_files.append(str(file))
                    response_detected = True
                    print(f"✅ Guard response detected: {file.name}")

            # Check influence files for rollback to 0
            influence_dir = self.base_dir / "artifacts" / "influence"
            if influence_dir.exists():
                for influence_file in influence_dir.glob("*_influence.json"):
                    try:
                        with open(influence_file, "r") as f:
                            data = json.load(f)

                        if (
                            data.get("influence_pct", 100) == 0
                            and data.get("reason") == "emergency_rollback_m18"
                        ):
                            response_detected = True
                            print(f"✅ Rollback confirmed: {influence_file.name}")
                    except Exception:
                        continue

            if response_detected:
                break

            time.sleep(1)

        response_time = (datetime.datetime.now() - start_check).total_seconds()

        result = {
            "response_detected": response_detected,
            "response_time_seconds": response_time,
            "rollback_files": rollback_files,
            "expected_rollback": expected_rollback,
            "test_result": "PASS" if response_detected == expected_rollback else "FAIL",
        }

        if response_detected:
            print(f"✅ Guard response verified in {response_time:.1f}s")
        else:
            print(f"❌ No guard response detected within {timeout_seconds}s")

        return result

    def cleanup_simulation(self) -> None:
        """Clean up simulation artifacts."""

        print("🧹 Cleaning up simulation artifacts...")

        # Remove Redis simulation keys
        if self.redis_client:
            for key in ["drill:slip_spike", "drill:drawdown_breach"]:
                self.redis_client.delete(key)

        # Archive drill files
        drill_dir = self.base_dir / "artifacts" / "drill"
        if drill_dir.exists():
            for drill_file in drill_dir.glob(f"*_{self.drill_session_id}.json"):
                archive_name = f"archived_{drill_file.name}"
                drill_file.rename(drill_file.parent / archive_name)
                print(f"📦 Archived: {archive_name}")

    def run_slip_spike_drill(
        self, spike_bps: float = 25.0, duration: int = 10
    ) -> Dict[str, Any]:
        """Run complete slip spike drill scenario."""

        print("🎯 Slip Spike Drill Scenario")
        print("=" * 40)
        print(f"Target: {spike_bps} bps spike for {duration}s")
        print(f"Expected: Guard should trigger rollback")
        print("=" * 40)

        drill_start = datetime.datetime.now()

        try:
            # Step 1: Inject slip spike
            spike_result = self.simulate_slip_spike(spike_bps, duration)

            # Step 2: Wait for guard response
            print(f"⏳ Waiting for guard to detect and respond...")
            time.sleep(2)  # Give guard time to detect

            # Step 3: Verify response
            response_result = self.verify_guard_response(
                expected_rollback=True, timeout_seconds=15
            )

            # Step 4: Clean up
            self.cleanup_simulation()

            drill_result = {
                "drill_type": "slip_spike",
                "drill_session_id": self.drill_session_id,
                "start_time": drill_start.isoformat(),
                "end_time": datetime.datetime.now().isoformat(),
                "spike_config": spike_result,
                "response_verification": response_result,
                "overall_result": response_result["test_result"],
            }

            # Write drill report
            self.write_drill_report(drill_result)

            return drill_result

        except Exception as e:
            print(f"💥 Drill error: {e}")
            self.cleanup_simulation()
            return {"drill_type": "slip_spike", "status": "ERROR", "error": str(e)}

    def run_drawdown_drill(
        self, drawdown_pct: float = 1.2, duration: int = 8
    ) -> Dict[str, Any]:
        """Run complete drawdown breach drill scenario."""

        print("🎯 Drawdown Breach Drill Scenario")
        print("=" * 40)
        print(f"Target: {drawdown_pct:.1%} drawdown for {duration}s")
        print(f"Expected: Guard should trigger rollback")
        print("=" * 40)

        drill_start = datetime.datetime.now()

        try:
            # Step 1: Inject drawdown breach
            breach_result = self.simulate_drawdown_breach(drawdown_pct, duration)

            # Step 2: Wait for guard response
            print(f"⏳ Waiting for guard to detect and respond...")
            time.sleep(2)

            # Step 3: Verify response
            response_result = self.verify_guard_response(
                expected_rollback=True, timeout_seconds=15
            )

            # Step 4: Clean up
            self.cleanup_simulation()

            drill_result = {
                "drill_type": "drawdown_breach",
                "drill_session_id": self.drill_session_id,
                "start_time": drill_start.isoformat(),
                "end_time": datetime.datetime.now().isoformat(),
                "breach_config": breach_result,
                "response_verification": response_result,
                "overall_result": response_result["test_result"],
            }

            # Write drill report
            self.write_drill_report(drill_result)

            return drill_result

        except Exception as e:
            print(f"💥 Drill error: {e}")
            self.cleanup_simulation()
            return {"drill_type": "drawdown_breach", "status": "ERROR", "error": str(e)}

    def write_drill_report(self, drill_result: Dict[str, Any]) -> None:
        """Write drill report to WORM audit."""

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        audit_file = self.worm_dir / f"post_incident_drill_{timestamp_str}.json"

        audit_entry = {
            "event_type": "post_incident_drill",
            "timestamp": datetime.datetime.now().isoformat(),
            "drill_result": drill_result,
            "user": os.getenv("USER", "system"),
            "compliance_test": True,
        }

        try:
            with open(audit_file, "w") as f:
                json.dump(audit_entry, f, indent=2)
            print(f"📝 Drill report: {audit_file}")
        except Exception as e:
            print(f"⚠️ Drill report error: {e}")


def main():
    """Main post-incident drill CLI."""

    parser = argparse.ArgumentParser(
        description="Post-Incident Drill: Test M18 Guard Controls"
    )
    parser.add_argument(
        "--scenario",
        choices=["slip", "drawdown", "both"],
        default="both",
        help="Drill scenario to run",
    )
    parser.add_argument(
        "--slip-bps", type=float, default=25.0, help="Slippage spike target in bps"
    )
    parser.add_argument(
        "--drawdown-pct",
        type=float,
        default=1.2,
        help="Drawdown breach target in percent",
    )
    parser.add_argument(
        "--duration", type=int, default=10, help="Incident duration in seconds"
    )
    args = parser.parse_args()

    try:
        drill = PostIncidentDrill()

        if args.scenario in ["slip", "both"]:
            print("\n" + "=" * 50)
            slip_result = drill.run_slip_spike_drill(args.slip_bps, args.duration)
            print(f"📊 Slip Drill Result: {slip_result.get('overall_result', 'ERROR')}")

        if args.scenario in ["drawdown", "both"]:
            print("\n" + "=" * 50)
            drawdown_result = drill.run_drawdown_drill(args.drawdown_pct, args.duration)
            print(
                f"📊 Drawdown Drill Result: {drawdown_result.get('overall_result', 'ERROR')}"
            )

        print(f"\n✅ Post-incident drill completed!")
        print(f"💡 Review WORM audit files in: {drill.worm_dir}")

        return 0

    except Exception as e:
        print(f"💥 Drill error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
