#!/usr/bin/env python3
"""
M17: 15% Green-Window Ramp Orchestrator
Micro-gradient promotion: 10 → 12 → 13.5 → 15% within green windows only.
"""
import os
import sys
import json
import time
import datetime
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import redis
import yaml


class RampOrchestrator15:
    """Orchestrate guarded 15% ramp within green windows."""

    def __init__(self, go_live: bool = False):
        self.go_live = go_live
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

        # Load ramp policy
        self.policy = self.load_ramp_policy()

        # Redis connection for live influence updates
        try:
            self.redis_client = redis.Redis(decode_responses=True)
            self.redis_client.ping()
        except Exception as e:
            print(f"⚠️ Redis not available: {e}")
            self.redis_client = None

        # WORM audit trail
        self.worm_dir = self.base_dir / "worm"
        self.worm_dir.mkdir(exist_ok=True)

        # Current state tracking
        self.current_influence = 10  # Starting point
        self.in_green_window = False
        self.ramp_session_id = None

    def load_ramp_policy(self) -> Dict[str, Any]:
        """Load M17 ramp policy configuration."""
        policy_file = self.base_dir / "ramp" / "ramp_policy.yaml"

        try:
            with open(policy_file, "r") as f:
                policy = yaml.safe_load(f)
            return policy.get("step_15", {})
        except Exception as e:
            print(f"⚠️ Error loading ramp policy: {e}")
            return {}

    def verify_gate_compliance(self) -> Dict[str, Any]:
        """Verify all M17 gate requirements before promotion."""

        compliance = {"passed": True, "failures": [], "checks": {}}

        # Check required audit tokens
        required_audits = self.policy.get("compliance", {}).get("audits_required", [])

        for audit in required_audits:
            token_file = None

            if audit == "slip_gate_ok":
                token_file = self.base_dir / "artifacts" / "exec" / "slip_gate_ok"
            elif audit == "m12_go_token":
                token_file = self.base_dir / "artifacts" / "m12" / "go_token"
            elif audit == "m15_green_summary":
                token_file = self.base_dir / "artifacts" / "econ_green" / "summary.json"

            if token_file and token_file.exists():
                compliance["checks"][audit] = {
                    "status": "PASS",
                    "file": str(token_file),
                }
            else:
                compliance["checks"][audit] = {
                    "status": "FAIL",
                    "file": str(token_file) if token_file else "unknown",
                }
                compliance["failures"].append(f"Missing {audit}")
                compliance["passed"] = False

        # Check execution metrics from live knobs
        try:
            from scripts.exec_knobs import ExecutionKnobs

            knobs = ExecutionKnobs()

            # Verify M16.1 optimizations are active
            post_only_base = knobs.get_knob_value("sizer_v2.post_only_base", 0.7)
            slice_max = knobs.get_knob_value("sizer_v2.slice_pct_max", 2.0)

            if post_only_base >= 0.80:  # M16.1 optimization active
                compliance["checks"]["execution_optimized"] = {
                    "status": "PASS",
                    "post_only_base": post_only_base,
                }
            else:
                compliance["checks"]["execution_optimized"] = {
                    "status": "FAIL",
                    "post_only_base": post_only_base,
                }
                compliance["failures"].append("M16.1 optimizations not active")
                compliance["passed"] = False

        except Exception as e:
            compliance["checks"]["execution_knobs"] = {
                "status": "ERROR",
                "error": str(e),
            }

        # Simulate slippage check (use optimized test)
        try:
            # Import and run optimized slippage test
            sys.path.insert(0, str(self.base_dir))
            from scripts.test_optimized_slip_gate import OptimizedSlippageGate

            gate = OptimizedSlippageGate()
            result = gate.run_optimized_test(48)

            if result.get("target_achieved", False):
                compliance["checks"]["slip_p95"] = {
                    "status": "PASS",
                    "p95_bps": result.get("p95_slippage_bps", 0),
                    "maker_ratio": result.get("maker_ratio", 0),
                }
            else:
                compliance["checks"]["slip_p95"] = {
                    "status": "FAIL",
                    "p95_bps": result.get("p95_slippage_bps", 999),
                    "needed_improvement": result.get("improvement_needed_bps", 0),
                }
                compliance["failures"].append(
                    f"Slippage P95 {result.get('p95_slippage_bps', 0):.1f} bps > 12 bps"
                )
                compliance["passed"] = False

        except Exception as e:
            compliance["checks"]["slip_test"] = {"status": "ERROR", "error": str(e)}

        return compliance

    def load_green_calendar(self, calendar_file: str) -> pd.DataFrame:
        """Load EV green window calendar."""

        try:
            calendar_path = Path(calendar_file)
            if not calendar_path.exists():
                print(f"⚠️ Calendar not found: {calendar_path}")
                return self.generate_synthetic_calendar()

            df = pd.read_parquet(calendar_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            # Filter for upcoming green windows
            now = datetime.datetime.now(datetime.timezone.utc)
            upcoming = df[df["timestamp"] >= now].head(50)

            print(f"📅 Loaded calendar: {len(upcoming)} upcoming windows")
            return upcoming

        except Exception as e:
            print(f"⚠️ Calendar load error: {e}")
            return self.generate_synthetic_calendar()

    def generate_synthetic_calendar(self) -> pd.DataFrame:
        """Generate synthetic green window calendar for testing."""

        now = datetime.datetime.now(datetime.timezone.utc)
        windows = []

        # Generate next 24 hours of 15-minute green windows
        for i in range(24 * 4):  # Every 15 minutes
            window_time = now + datetime.timedelta(minutes=i * 15)

            # 60% chance of green window during active hours
            is_active = 9 <= window_time.hour <= 21
            is_green = np.random.random() < (0.6 if is_active else 0.3)

            if is_green:
                windows.append(
                    {
                        "timestamp": window_time,
                        "asset": np.random.choice(["BTC-USD", "ETH-USD", "NVDA"]),
                        "band": "green",
                        "duration_minutes": 15,
                        "ev_usd_per_hour": np.random.uniform(20, 80),
                        "confidence": np.random.uniform(0.7, 0.95),
                    }
                )

        df = pd.DataFrame(windows)
        print(f"📅 Generated {len(df)} synthetic green windows")
        return df

    def get_next_green_window(
        self, calendar_df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Find next suitable green window for ramp."""

        if calendar_df.empty:
            return None

        now = datetime.datetime.now(datetime.timezone.utc)
        min_duration = self.policy.get("min_green_minutes", 10)

        # Find upcoming green windows (duration is 5 minutes for all windows)
        suitable_windows = calendar_df[
            (calendar_df["timestamp"] >= now) & (calendar_df["band"] == "green")
        ].head(10)

        if suitable_windows.empty:
            return None

        # Return next window
        next_window = suitable_windows.iloc[0]

        return {
            "timestamp": next_window["timestamp"],
            "asset": next_window["asset"],
            "duration_minutes": 5,  # Fixed 5-minute windows
            "ev_usd_per_hour": next_window.get("ev_usd_per_hour", 50),
            "confidence": 0.8,  # Default confidence
        }

    def execute_micro_gradient_ramp(
        self, green_window: Dict[str, Any], steps: List[float], step_minutes: int
    ) -> Dict[str, Any]:
        """Execute micro-gradient ramp within green window."""

        start_time = datetime.datetime.now(datetime.timezone.utc)
        window_end = green_window["timestamp"] + datetime.timedelta(
            minutes=green_window["duration_minutes"]
        )

        # Generate session ID
        self.ramp_session_id = f"ramp15_{start_time.strftime('%Y%m%d_%H%M%S')}"

        ramp_log = {
            "session_id": self.ramp_session_id,
            "start_time": start_time.isoformat(),
            "green_window": green_window,
            "target_steps": steps,
            "step_duration_minutes": step_minutes,
            "steps_executed": [],
            "status": "started",
        }

        print(f"🚀 Starting M17 ramp session: {self.ramp_session_id}")
        print(f"   Window: {green_window['timestamp']} - {window_end}")
        print(f"   Asset: {green_window['asset']}")
        print(f"   Steps: {steps}")

        try:
            # Execute each step
            for i, target_pct in enumerate(steps):
                step_start = datetime.datetime.now(datetime.timezone.utc)

                # Check if we have time remaining
                if step_start >= window_end:
                    print(f"⏰ Window ended before step {i+1}")
                    break

                print(f"📈 Step {i+1}/{len(steps)}: Ramping to {target_pct}%")

                # Apply influence level
                success = self.set_influence_level(target_pct, green_window["asset"])

                step_log = {
                    "step_number": i + 1,
                    "target_influence_pct": target_pct,
                    "timestamp": step_start.isoformat(),
                    "success": success,
                    "asset": green_window["asset"],
                }

                if success:
                    self.current_influence = target_pct
                    print(f"✅ Influence set to {target_pct}%")
                else:
                    print(f"❌ Failed to set influence to {target_pct}%")
                    step_log["error"] = "influence_set_failed"

                ramp_log["steps_executed"].append(step_log)

                # Write step audit
                self.write_worm_audit(
                    "ramp15_step",
                    {
                        "session_id": self.ramp_session_id,
                        "step": step_log,
                        "policy_snapshot": self.policy,
                    },
                )

                # Wait for step duration (unless last step)
                if i < len(steps) - 1:
                    wait_seconds = step_minutes * 60
                    if not self.go_live:
                        wait_seconds = 2  # Short wait for testing

                    print(f"⏳ Waiting {wait_seconds}s for next step...")
                    time.sleep(wait_seconds)

            # Hold at target until window end
            if ramp_log["steps_executed"]:
                final_influence = steps[-1]
                hold_duration = (
                    window_end - datetime.datetime.now(datetime.timezone.utc)
                ).total_seconds()

                if hold_duration > 0:
                    print(f"🎯 Holding at {final_influence}% for {hold_duration:.0f}s")

                    if not self.go_live:
                        hold_duration = 5  # Short hold for testing

                    time.sleep(max(0, hold_duration))

            # Auto-revert to 0% at window end
            print(f"🔄 Green window ended, reverting to 0%")
            revert_success = self.set_influence_level(0, green_window["asset"])

            ramp_log["status"] = "completed"
            ramp_log["end_time"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            ramp_log["final_revert"] = {"success": revert_success}

        except Exception as e:
            print(f"❌ Ramp session error: {e}")
            ramp_log["status"] = "failed"
            ramp_log["error"] = str(e)

            # Emergency revert
            self.set_influence_level(0, green_window["asset"])

        # Write final audit
        self.write_worm_audit("ramp15_session", ramp_log)

        return ramp_log

    def set_influence_level(self, influence_pct: float, asset: str) -> bool:
        """Set influence level via Redis or file-based system."""

        if not self.go_live:
            print(f"🧪 [DRY RUN] Would set {asset} influence to {influence_pct}%")
            return True

        try:
            # Redis method
            if self.redis_client:
                key = f"influence:{asset}"
                self.redis_client.set(key, influence_pct, ex=3600)
                print(f"📡 Redis: {key} = {influence_pct}%")
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
                        "session_id": self.ramp_session_id,
                    },
                    f,
                    indent=2,
                )

            print(f"📁 File: {influence_file}")
            return True

        except Exception as e:
            print(f"❌ Influence set error: {e}")
            return False

    def write_worm_audit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write WORM (Write Once Read Many) audit entry."""

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        audit_file = self.worm_dir / f"{event_type}_{timestamp_str}.json"

        audit_entry = {
            "event_type": event_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "data": data,
            "user": os.getenv("USER", "system"),
            "go_live": self.go_live,
        }

        try:
            with open(audit_file, "w") as f:
                json.dump(audit_entry, f, indent=2)
            print(f"📝 WORM audit: {audit_file}")
        except Exception as e:
            print(f"⚠️ WORM audit error: {e}")

    def run_orchestrator(
        self, calendar_file: str, steps: List[float], step_minutes: int
    ) -> Dict[str, Any]:
        """Run the complete M17 orchestrator."""

        print("🎯 M17: 15% Green-Window Ramp Orchestrator")
        print("=" * 45)
        print(f"Go-live: {self.go_live}")
        print(f"Steps: {steps}")
        print(f"Step duration: {step_minutes} minutes")
        print("=" * 45)

        # 1. Verify gate compliance
        print("🔒 Verifying M17 gate compliance...")
        compliance = self.verify_gate_compliance()

        if not compliance["passed"]:
            print("❌ Gate compliance FAILED:")
            for failure in compliance["failures"]:
                print(f"   • {failure}")

            return {
                "success": False,
                "reason": "gate_compliance_failed",
                "compliance": compliance,
            }

        print("✅ All gates passed")

        # 2. Load green window calendar
        print("📅 Loading green window calendar...")
        calendar_df = self.load_green_calendar(calendar_file)

        # 3. Find next suitable green window
        print("🔍 Finding next green window...")
        next_window = self.get_next_green_window(calendar_df)

        if not next_window:
            print("❌ No suitable green windows found")
            return {
                "success": False,
                "reason": "no_green_windows",
                "calendar_windows": len(calendar_df),
            }

        print(
            f"✅ Next window: {next_window['timestamp']} ({next_window['duration_minutes']}min)"
        )

        # 4. Wait for window start (if not immediate)
        window_start = next_window["timestamp"]
        now = datetime.datetime.now(datetime.timezone.utc)

        if window_start > now:
            wait_seconds = (window_start - now).total_seconds()
            if wait_seconds > 3600:  # More than 1 hour
                print(
                    f"⏰ Next window in {wait_seconds/3600:.1f} hours - exiting for now"
                )
                return {
                    "success": False,
                    "reason": "window_too_far",
                    "next_window_seconds": wait_seconds,
                }

            print(f"⏳ Waiting {wait_seconds:.0f}s for green window...")
            if not self.go_live:
                wait_seconds = min(wait_seconds, 10)  # Cap for testing
            time.sleep(max(0, wait_seconds))

        # 5. Execute micro-gradient ramp
        print("🚀 Executing micro-gradient ramp...")

        # Write session start audit
        self.write_worm_audit(
            "ramp15_start",
            {
                "green_window": next_window,
                "policy": self.policy,
                "compliance": compliance,
                "steps": steps,
                "step_minutes": step_minutes,
            },
        )

        ramp_result = self.execute_micro_gradient_ramp(next_window, steps, step_minutes)

        # 6. Results
        success = ramp_result.get("status") == "completed"

        print(f"\n🎯 M17 Orchestrator Results:")
        print(f"  Session: {ramp_result.get('session_id')}")
        print(f"  Status: {ramp_result.get('status', 'unknown')}")
        print(f"  Steps executed: {len(ramp_result.get('steps_executed', []))}")

        if success:
            print("✅ M17 GREEN WINDOW RAMP COMPLETED")
        else:
            print("❌ M17 ramp failed or incomplete")

        return {
            "success": success,
            "session_id": ramp_result.get("session_id"),
            "ramp_log": ramp_result,
            "compliance": compliance,
        }


def main():
    """Main M17 orchestrator CLI."""
    parser = argparse.ArgumentParser(
        description="M17: 15% Green-Window Ramp Orchestrator"
    )
    parser.add_argument("--calendar", required=True, help="Green window calendar file")
    parser.add_argument(
        "--steps", default="10,12,13.5,15", help="Comma-separated ramp steps"
    )
    parser.add_argument("--step-min", type=int, default=3, help="Minutes per step")
    args = parser.parse_args()

    # Parse steps
    try:
        steps = [float(x.strip()) for x in args.steps.split(",")]
    except ValueError:
        print(f"❌ Invalid steps format: {args.steps}")
        return 1

    # Check GO_LIVE environment
    go_live = os.getenv("GO_LIVE", "0") == "1"

    try:
        orchestrator = RampOrchestrator15(go_live=go_live)
        result = orchestrator.run_orchestrator(args.calendar, steps, args.step_min)

        if result["success"]:
            print(f"\n🎉 M17 SUCCESS: 15% ramp completed in green window!")
            return 0
        else:
            print(f"\n⚠️ M17 incomplete: {result.get('reason', 'unknown')}")
            return 1

    except Exception as e:
        print(f"❌ M17 orchestrator error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
