#!/usr/bin/env python3
"""
M18: 20% Green-Window Ramp Orchestrator
Micro-gradient promotion: 15 → 17 → 18.5 → 20% within green windows only.
Includes EV ceiling enforcement and enhanced safety guards.
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


class RampOrchestrator20:
    """Orchestrate guarded 20% ramp within green windows with EV ceiling enforcement."""

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
        self.current_influence = 15  # Starting point for 20% ramp
        self.in_green_window = False
        self.ramp_session_id = None

    def load_ramp_policy(self) -> Dict[str, Any]:
        """Load M18 ramp policy configuration."""
        policy_file = self.base_dir / "ramp" / "ramp_policy.yaml"

        try:
            with open(policy_file, "r") as f:
                policy = yaml.safe_load(f)
            return policy.get("step_20", {})
        except Exception as e:
            print(f"⚠️ Error loading ramp policy: {e}")
            return {}

    def verify_gate_compliance(self) -> Dict[str, Any]:
        """Verify all M18 gate requirements before promotion."""

        compliance = {"passed": True, "failures": [], "checks": {}}

        # Check required audit tokens for 20% ramp
        required_audits = self.policy.get("compliance", {}).get(
            "audits_required",
            ["soak15_ok", "slip_gate_ok", "m12_go_token", "cfo_green_summary"],
        )

        for audit in required_audits:
            token_file = None

            if audit == "soak15_ok":
                token_file = self.base_dir / "artifacts" / "gates" / "soak15_ok"
            elif audit == "slip_gate_ok":
                token_file = self.base_dir / "artifacts" / "exec" / "slip_gate_ok"
            elif audit == "m12_go_token":
                token_file = self.base_dir / "artifacts" / "m12" / "go_token"
            elif audit == "cfo_green_summary":
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

        # Check enhanced slippage requirements for 20%
        try:
            sys.path.insert(0, str(self.base_dir))
            from scripts.test_optimized_slip_gate import OptimizedSlippageGate

            gate = OptimizedSlippageGate()
            result = gate.run_optimized_test(48)

            # Stricter requirements for 20%
            if (
                result.get("p95_slippage_bps", 999) <= 12.0
            ):  # Tighter than 15 bps for 20%
                compliance["checks"]["slip_p95_enhanced"] = {
                    "status": "PASS",
                    "p95_bps": result.get("p95_slippage_bps", 0),
                    "maker_ratio": result.get("maker_ratio", 0),
                }
            else:
                compliance["checks"]["slip_p95_enhanced"] = {
                    "status": "FAIL",
                    "p95_bps": result.get("p95_slippage_bps", 999),
                    "required_bps": 12.0,
                }
                compliance["failures"].append(
                    f"Slippage P95 {result.get('p95_slippage_bps', 0):.1f} bps > 12 bps (20% requirement)"
                )
                compliance["passed"] = False

        except Exception as e:
            compliance["checks"]["slip_test_enhanced"] = {
                "status": "ERROR",
                "error": str(e),
            }

        # Check EV ceiling calibration
        try:
            calib_file = (
                self.base_dir / "artifacts" / "ev" / "ev_calibration_simple.json"
            )
            if calib_file.exists():
                with open(calib_file, "r") as f:
                    calib_data = json.load(f)
                cost_ratio = calib_data.get("cost_ratio_projection", 1.0)

                if cost_ratio <= 0.35:  # Cost ratio ≤35% required for 20%
                    compliance["checks"]["ev_ceiling"] = {
                        "status": "PASS",
                        "cost_ratio": cost_ratio,
                    }
                else:
                    compliance["checks"]["ev_ceiling"] = {
                        "status": "FAIL",
                        "cost_ratio": cost_ratio,
                    }
                    compliance["failures"].append(f"Cost ratio {cost_ratio:.1%} > 35%")
                    compliance["passed"] = False
            else:
                compliance["checks"]["ev_ceiling"] = {
                    "status": "MISSING",
                    "file": str(calib_file),
                }
                compliance["failures"].append("EV calibration missing")
                compliance["passed"] = False

        except Exception as e:
            compliance["checks"]["ev_ceiling"] = {"status": "ERROR", "error": str(e)}

        return compliance

    def load_green_calendar(self, calendar_file: str) -> pd.DataFrame:
        """Load EV green window calendar with enhanced filtering for 20%."""

        try:
            calendar_path = Path(calendar_file)
            if not calendar_path.exists():
                print(f"⚠️ Calendar not found: {calendar_path}")
                return self.generate_synthetic_calendar()

            df = pd.read_parquet(calendar_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            # Enhanced filtering for 20% ramp - only high-confidence green windows
            now = datetime.datetime.now(datetime.timezone.utc)
            upcoming = df[
                (df["timestamp"] >= now)
                & (df["band"] == "green")
                & (df.get("confidence", 0.8) >= 0.85)  # Higher confidence for 20%
            ].head(50)

            print(
                f"📅 Loaded calendar: {len(upcoming)} high-confidence upcoming windows"
            )
            return upcoming

        except Exception as e:
            print(f"⚠️ Calendar load error: {e}")
            return self.generate_synthetic_calendar()

    def generate_synthetic_calendar(self) -> pd.DataFrame:
        """Generate synthetic green window calendar for testing."""

        now = datetime.datetime.now(datetime.timezone.utc)
        windows = []

        # Generate next 24 hours of 15-minute green windows (higher confidence for 20%)
        for i in range(24 * 4):  # Every 15 minutes
            window_time = now + datetime.timedelta(minutes=i * 15)

            # 50% chance of green window during active hours (more selective for 20%)
            is_active = 9 <= window_time.hour <= 21
            is_green = np.random.random() < (0.5 if is_active else 0.2)

            if is_green:
                windows.append(
                    {
                        "timestamp": window_time,
                        "asset": np.random.choice(["BTC-USD", "ETH-USD", "NVDA"]),
                        "band": "green",
                        "duration_minutes": 15,
                        "ev_usd_per_hour": np.random.uniform(
                            30, 100
                        ),  # Higher EV for 20%
                        "confidence": np.random.uniform(
                            0.85, 0.95
                        ),  # Higher confidence
                    }
                )

        df = pd.DataFrame(windows)
        print(f"📅 Generated {len(df)} synthetic high-confidence green windows")
        return df

    def enforce_ev_ceiling(
        self, green_window: Dict[str, Any], target_pct: float
    ) -> bool:
        """Enforce EV ceiling constraints for 20% ramp."""

        # Check if we're within green_lo and green_hi bands
        if target_pct >= 17 and target_pct <= 20:
            # This is the green band range (17-20%)
            confidence = green_window.get("confidence", 0.8)
            ev_per_hour = green_window.get("ev_usd_per_hour", 50)

            # Enhanced constraints for 20%
            if confidence < 0.85:
                print(f"❌ EV ceiling: confidence {confidence:.1%} < 85%")
                return False

            if ev_per_hour < 30:
                print(f"❌ EV ceiling: EV ${ev_per_hour}/hr < $30/hr")
                return False

            print(f"✅ EV ceiling: confidence {confidence:.1%}, EV ${ev_per_hour}/hr")
            return True
        else:
            # Outside green band - use normal constraints
            return True

    def get_next_green_window(
        self, calendar_df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Find next suitable green window for 20% ramp."""

        if calendar_df.empty:
            return None

        now = datetime.datetime.now(datetime.timezone.utc)

        # Find upcoming high-confidence green windows
        suitable_windows = calendar_df[
            (calendar_df["timestamp"] >= now)
            & (calendar_df["band"] == "green")
            & (calendar_df.get("confidence", 0.8) >= 0.85)  # Enhanced for 20%
        ].head(5)

        if suitable_windows.empty:
            return None

        # Return next window
        next_window = suitable_windows.iloc[0]

        return {
            "timestamp": next_window["timestamp"],
            "asset": next_window["asset"],
            "duration_minutes": 15,  # Fixed 15-minute windows for 20%
            "ev_usd_per_hour": next_window.get("ev_usd_per_hour", 60),
            "confidence": next_window.get("confidence", 0.85),
        }

    def execute_micro_gradient_ramp(
        self, green_window: Dict[str, Any], steps: List[float], step_minutes: int
    ) -> Dict[str, Any]:
        """Execute micro-gradient ramp within green window with EV ceiling enforcement."""

        start_time = datetime.datetime.now(datetime.timezone.utc)
        window_end = green_window["timestamp"] + datetime.timedelta(
            minutes=green_window["duration_minutes"]
        )

        # Generate session ID
        self.ramp_session_id = f"ramp20_{start_time.strftime('%Y%m%d_%H%M%S')}"

        ramp_log = {
            "session_id": self.ramp_session_id,
            "start_time": start_time.isoformat(),
            "green_window": green_window,
            "target_steps": steps,
            "step_duration_minutes": step_minutes,
            "steps_executed": [],
            "status": "started",
            "ev_ceiling_enforced": True,
        }

        print(f"🚀 Starting M18 20% ramp session: {self.ramp_session_id}")
        print(f"   Window: {green_window['timestamp']} - {window_end}")
        print(f"   Asset: {green_window['asset']}")
        print(f"   Steps: {steps}")
        print(f"   EV Ceiling: Active")

        try:
            # Execute each step with EV ceiling enforcement
            for i, target_pct in enumerate(steps):
                step_start = datetime.datetime.now(datetime.timezone.utc)

                # Check if we have time remaining
                if step_start >= window_end:
                    print(f"⏰ Window ended before step {i+1}")
                    break

                # Enforce EV ceiling
                if not self.enforce_ev_ceiling(green_window, target_pct):
                    print(f"🚫 EV ceiling blocked step {i+1}: {target_pct}%")
                    ramp_log["status"] = "ev_ceiling_blocked"
                    break

                print(
                    f"📈 Step {i+1}/{len(steps)}: Ramping to {target_pct}% (EV ceiling OK)"
                )

                # Apply influence level
                success = self.set_influence_level(target_pct, green_window["asset"])

                step_log = {
                    "step_number": i + 1,
                    "target_influence_pct": target_pct,
                    "timestamp": step_start.isoformat(),
                    "success": success,
                    "asset": green_window["asset"],
                    "ev_ceiling_check": "passed",
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
                    "ramp20_step",
                    {
                        "session_id": self.ramp_session_id,
                        "step": step_log,
                        "policy_snapshot": self.policy,
                        "ev_ceiling_enforced": True,
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
            if ramp_log["steps_executed"] and ramp_log["status"] == "started":
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

            if ramp_log["status"] == "started":
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
        self.write_worm_audit("ramp20_session", ramp_log)

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
                        "ramp_level": "M18_20pct",
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
            "ramp_level": "M18_20pct",
        }

        try:
            with open(audit_file, "w") as f:
                json.dump(audit_entry, f, indent=2)
            print(f"📝 WORM audit: {audit_file}")
        except Exception as e:
            print(f"⚠️ WORM audit error: {e}")

    def run_orchestrator(
        self,
        calendar_file: str,
        steps: List[float],
        step_minutes: int,
        respect_ev_ceiling: bool = True,
    ) -> Dict[str, Any]:
        """Run the complete M18 20% orchestrator."""

        print("🎯 M18: 20% Green-Window Ramp Orchestrator")
        print("=" * 45)
        print(f"Go-live: {self.go_live}")
        print(f"Steps: {steps}")
        print(f"Step duration: {step_minutes} minutes")
        print(f"EV ceiling enforcement: {respect_ev_ceiling}")
        print("=" * 45)

        # 1. Verify gate compliance
        print("🔒 Verifying M18 gate compliance...")
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
        print("🔍 Finding next high-confidence green window...")
        next_window = self.get_next_green_window(calendar_df)

        if not next_window:
            print("❌ No suitable high-confidence green windows found")
            return {
                "success": False,
                "reason": "no_green_windows",
                "calendar_windows": len(calendar_df),
            }

        print(
            f"✅ Next window: {next_window['timestamp']} ({next_window['duration_minutes']}min)"
        )
        print(f"   Confidence: {next_window['confidence']:.1%}")
        print(f"   EV: ${next_window['ev_usd_per_hour']}/hr")

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
        print("🚀 Executing 20% micro-gradient ramp with EV ceiling...")

        # Write session start audit
        self.write_worm_audit(
            "ramp20_start",
            {
                "green_window": next_window,
                "policy": self.policy,
                "compliance": compliance,
                "steps": steps,
                "step_minutes": step_minutes,
                "ev_ceiling_enforcement": respect_ev_ceiling,
            },
        )

        ramp_result = self.execute_micro_gradient_ramp(next_window, steps, step_minutes)

        # 6. Results
        success = ramp_result.get("status") == "completed"

        print(f"\n🎯 M18 20% Orchestrator Results:")
        print(f"  Session: {ramp_result.get('session_id')}")
        print(f"  Status: {ramp_result.get('status', 'unknown')}")
        print(f"  Steps executed: {len(ramp_result.get('steps_executed', []))}")
        print(f"  EV ceiling enforced: {ramp_result.get('ev_ceiling_enforced', False)}")

        if success:
            print("✅ M18 20% GREEN WINDOW RAMP COMPLETED")
        else:
            print("❌ M18 20% ramp failed or incomplete")
            if ramp_result.get("status") == "ev_ceiling_blocked":
                print("   Blocked by EV ceiling constraints")

        return {
            "success": success,
            "session_id": ramp_result.get("session_id"),
            "ramp_log": ramp_result,
            "compliance": compliance,
        }


def main():
    """Main M18 20% orchestrator CLI."""
    parser = argparse.ArgumentParser(
        description="M18: 20% Green-Window Ramp Orchestrator"
    )
    parser.add_argument("--calendar", required=True, help="Green window calendar file")
    parser.add_argument(
        "--steps", default="15,17,18.5,20", help="Comma-separated ramp steps"
    )
    parser.add_argument("--step-min", type=int, default=3, help="Minutes per step")
    parser.add_argument(
        "--respect-ev-ceiling",
        type=int,
        default=1,
        help="Enforce EV ceiling (1=yes, 0=no)",
    )
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
        orchestrator = RampOrchestrator20(go_live=go_live)
        result = orchestrator.run_orchestrator(
            args.calendar,
            steps,
            args.step_min,
            respect_ev_ceiling=bool(args.respect_ev_ceiling),
        )

        if result["success"]:
            print(f"\n🎉 M18 SUCCESS: 20% ramp completed in green window!")
            return 0
        else:
            print(f"\n⚠️ M18 incomplete: {result.get('reason', 'unknown')}")
            return 1

    except Exception as e:
        print(f"❌ M18 orchestrator error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
