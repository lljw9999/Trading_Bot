#!/usr/bin/env python3
"""
Deep Sleep Orchestrator: Infrastructure Hibernation System
Hibernate GPU + pipelines during red spans; wake on green/event windows.
"""
import os
import sys
import json
import datetime
import argparse
import pandas as pd
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class DeepSleepOrchestrator:
    def __init__(self):
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")
        self.current_time = datetime.datetime.now(datetime.timezone.utc)

        # Sleep criteria
        self.red_coverage_threshold = 0.90  # 90% red windows in next hour
        self.min_sleep_duration_minutes = 30  # Minimum sleep duration
        self.wake_lookahead_minutes = 15  # Wake 15 min before green window

        # Services to hibernate
        self.hibernation_services = {
            "policy_server": {
                "description": "RL policy inference server",
                "cost_savings_pct": 0.40,  # 40% cost reduction
                "stop_command": "pkill -f 'policy_server'",
                "start_command": "python src/rl/policy_server.py &",
            },
            "feature_bus": {
                "description": "Real-time feature pipeline",
                "cost_savings_pct": 0.25,  # 25% cost reduction
                "stop_command": "pkill -f 'feature_bus'",
                "start_command": "python src/layers/layer0_data_ingestion/feature_bus.py &",
            },
            "heavy_exporters": {
                "description": "GPU monitoring exporters",
                "cost_savings_pct": 0.15,  # 15% cost reduction
                "stop_command": "pkill -f 'gpu_exporter|model_exporter'",
                "start_command": "python src/monitoring/gpu_exporter.py & python src/monitoring/model_exporter.py &",
            },
            "triton_server": {
                "description": "ONNX model inference server",
                "cost_savings_pct": 0.20,  # 20% cost reduction
                "stop_command": "docker stop triton-server",
                "start_command": "docker start triton-server",
            },
        }

        # Critical services to keep running
        self.critical_services = [
            "redis",
            "prometheus",
            "heartbeat_monitor",
            "lightweight_prober",
        ]

    def load_5min_calendar(self) -> Optional[pd.DataFrame]:
        """Load latest 5-minute EV calendar."""
        try:
            # First try direct path
            calendar_file = self.base_dir / "artifacts" / "ev" / "calendar_5m.parquet"
            if calendar_file.exists():
                df = pd.read_parquet(calendar_file)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                return df

            # Try to find latest timestamped calendar
            ev_dir = self.base_dir / "artifacts" / "ev"
            if ev_dir.exists():
                # Find all timestamped directories
                timestamp_dirs = [
                    d for d in ev_dir.iterdir() if d.is_dir() and d.name.endswith("Z")
                ]
                if timestamp_dirs:
                    # Get the latest one
                    latest_dir = max(timestamp_dirs, key=lambda x: x.name)
                    calendar_file = latest_dir / "calendar_5m.parquet"
                    if calendar_file.exists():
                        print(f"📊 Using calendar: {calendar_file}")
                        df = pd.read_parquet(calendar_file)
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                        return df

        except Exception as e:
            print(f"⚠️ Could not load 5-min calendar: {e}")
        return None

    def get_current_influence_status(self) -> Dict[str, float]:
        """Get current influence levels for all assets."""
        try:
            sys.path.insert(0, str(self.base_dir))
            from src.rl.influence_controller import InfluenceController

            ic = InfluenceController()
            weights = ic.get_all_asset_weights()

            # Convert to percentages
            influence_pct = {asset: weight * 100 for asset, weight in weights.items()}
            total_influence = sum(influence_pct.values())

            return {"per_asset": influence_pct, "total_influence_pct": total_influence}

        except Exception as e:
            print(f"⚠️ Could not get influence status: {e}")
            return {"per_asset": {}, "total_influence_pct": 0.0}

    def analyze_upcoming_windows(
        self, calendar_df: pd.DataFrame, lookahead_minutes: int = 60
    ) -> Dict[str, Any]:
        """Analyze EV windows for next N minutes."""
        if calendar_df is None:
            # No calendar = assume red period, eligible for sleep
            return {
                "red_coverage": 1.0,
                "next_green_window": None,
                "sleep_eligible": True,
                "current_window": self.current_time.replace(
                    minute=(self.current_time.minute // 5) * 5, second=0, microsecond=0
                ).isoformat(),
                "lookahead_minutes": lookahead_minutes,
                "total_windows": 0,
                "red_windows": 0,
                "sleep_duration_minutes": lookahead_minutes,
            }

        # Get current 5-minute window
        current_window = self.current_time.replace(
            minute=(self.current_time.minute // 5) * 5, second=0, microsecond=0
        )

        # Define analysis window
        end_window = current_window + datetime.timedelta(minutes=lookahead_minutes)

        # Filter to upcoming windows
        upcoming_windows = calendar_df[
            (calendar_df["timestamp"] >= current_window)
            & (calendar_df["timestamp"] < end_window)
        ]

        if upcoming_windows.empty:
            # No upcoming windows in analysis period = red period, eligible for sleep
            return {
                "current_window": current_window.isoformat(),
                "lookahead_minutes": lookahead_minutes,
                "total_windows": 0,
                "red_windows": 0,
                "red_coverage": 1.0,
                "next_green_window": None,
                "sleep_eligible": True,
                "sleep_duration_minutes": lookahead_minutes,
            }

        # Calculate red coverage
        total_windows = len(upcoming_windows)
        red_windows = len(upcoming_windows[upcoming_windows["band"] == "red"])
        red_coverage = red_windows / total_windows

        # Find next green window
        green_windows = upcoming_windows[
            upcoming_windows["band"] == "green"
        ].sort_values("timestamp")
        next_green_window = (
            green_windows.iloc[0]["timestamp"] if not green_windows.empty else None
        )

        # Check sleep eligibility
        sleep_eligible = red_coverage >= self.red_coverage_threshold and (
            next_green_window is None
            or (next_green_window - current_window).total_seconds() / 60
            >= self.min_sleep_duration_minutes
        )

        analysis = {
            "current_window": current_window.isoformat(),
            "lookahead_minutes": lookahead_minutes,
            "total_windows": total_windows,
            "red_windows": red_windows,
            "red_coverage": red_coverage,
            "next_green_window": (
                next_green_window.isoformat() if next_green_window else None
            ),
            "sleep_eligible": sleep_eligible,
            "sleep_duration_minutes": (
                (next_green_window - current_window).total_seconds() / 60
                - self.wake_lookahead_minutes
                if next_green_window
                else lookahead_minutes
            ),
        }

        return analysis

    def check_event_gate_activity(self) -> Dict[str, Any]:
        """Check for active event gate overrides."""
        event_activity = {
            "active_events": 0,
            "event_assets": [],
            "next_event_expiry": None,
        }

        try:
            event_dir = self.base_dir / "artifacts" / "ev" / "event_gate_on"
            if not event_dir.exists():
                return event_activity

            active_events = []

            for asset_dir in event_dir.iterdir():
                if asset_dir.is_dir():
                    for token_file in asset_dir.glob("event_green_*.json"):
                        try:
                            with open(token_file, "r") as f:
                                token_data = json.load(f)

                            # Check if token is still valid
                            valid_until_str = token_data.get("valid_until", "")
                            if valid_until_str:
                                valid_until = datetime.datetime.fromisoformat(
                                    valid_until_str.replace("Z", "+00:00")
                                )

                                if valid_until > self.current_time:
                                    active_events.append(
                                        {
                                            "asset": token_data["asset"],
                                            "valid_until": valid_until,
                                            "event_score": token_data["event_score"],
                                        }
                                    )

                        except Exception:
                            continue

            if active_events:
                event_activity["active_events"] = len(active_events)
                event_activity["event_assets"] = [e["asset"] for e in active_events]
                event_activity["next_event_expiry"] = min(
                    e["valid_until"] for e in active_events
                ).isoformat()

        except Exception as e:
            print(f"⚠️ Error checking event gate activity: {e}")

        return event_activity

    def calculate_sleep_savings(self, sleep_duration_minutes: float) -> Dict[str, Any]:
        """Calculate cost savings from deep sleep."""

        # Load cost structure
        try:
            calib_file = (
                self.base_dir / "artifacts" / "ev" / "ev_calibration_simple.json"
            )
            if calib_file.exists():
                with open(calib_file, "r") as f:
                    calib_data = json.load(f)
                cost_per_hour = calib_data["cost_per_active_hour_usd"]
            else:
                cost_per_hour = 4.40  # Fallback
        except:
            cost_per_hour = 4.40

        # Calculate savings by service
        service_savings = {}
        total_savings_pct = 0.0

        for service_name, service_config in self.hibernation_services.items():
            savings_pct = service_config["cost_savings_pct"]
            service_hourly_cost = cost_per_hour * savings_pct
            service_sleep_savings = service_hourly_cost * (sleep_duration_minutes / 60)

            service_savings[service_name] = {
                "hourly_cost": service_hourly_cost,
                "sleep_savings": service_sleep_savings,
                "savings_pct": savings_pct,
            }

            total_savings_pct += savings_pct

        total_hourly_savings = cost_per_hour * total_savings_pct
        total_sleep_savings = total_hourly_savings * (sleep_duration_minutes / 60)

        savings_analysis = {
            "cost_per_hour_baseline": cost_per_hour,
            "total_savings_pct": total_savings_pct,
            "total_hourly_savings": total_hourly_savings,
            "sleep_duration_minutes": sleep_duration_minutes,
            "total_sleep_savings": total_sleep_savings,
            "service_breakdown": service_savings,
        }

        return savings_analysis

    def execute_sleep_sequence(self, dry_run: bool = False) -> Dict[str, Any]:
        """Execute the sleep sequence for hibernation services."""
        sleep_results = {
            "services_stopped": [],
            "services_failed": [],
            "total_services": len(self.hibernation_services),
        }

        for service_name, service_config in self.hibernation_services.items():
            try:
                stop_command = service_config["stop_command"]

                if dry_run:
                    print(f"🧪 DRY RUN: Would stop {service_name} with: {stop_command}")
                    sleep_results["services_stopped"].append(service_name)
                else:
                    print(
                        f"💤 Stopping {service_name}: {service_config['description']}"
                    )
                    result = subprocess.run(
                        stop_command, shell=True, capture_output=True, text=True
                    )

                    if result.returncode == 0:
                        sleep_results["services_stopped"].append(service_name)
                        print(f"✅ {service_name} stopped successfully")
                    else:
                        sleep_results["services_failed"].append(service_name)
                        print(f"⚠️ {service_name} stop failed: {result.stderr}")

            except Exception as e:
                sleep_results["services_failed"].append(service_name)
                print(f"❌ Error stopping {service_name}: {e}")

        return sleep_results

    def execute_wake_sequence(self, dry_run: bool = False) -> Dict[str, Any]:
        """Execute the wake sequence to restore hibernation services."""
        wake_results = {
            "services_started": [],
            "services_failed": [],
            "total_services": len(self.hibernation_services),
        }

        for service_name, service_config in self.hibernation_services.items():
            try:
                start_command = service_config["start_command"]

                if dry_run:
                    print(
                        f"🧪 DRY RUN: Would start {service_name} with: {start_command}"
                    )
                    wake_results["services_started"].append(service_name)
                else:
                    print(
                        f"🌅 Starting {service_name}: {service_config['description']}"
                    )
                    result = subprocess.run(
                        start_command, shell=True, capture_output=True, text=True
                    )

                    if result.returncode == 0:
                        wake_results["services_started"].append(service_name)
                        print(f"✅ {service_name} started successfully")
                    else:
                        wake_results["services_failed"].append(service_name)
                        print(f"⚠️ {service_name} start failed: {result.stderr}")

            except Exception as e:
                wake_results["services_failed"].append(service_name)
                print(f"❌ Error starting {service_name}: {e}")

        return wake_results

    def create_sleep_audit(
        self,
        action: str,
        analysis: Dict[str, Any],
        savings: Dict[str, Any],
        sleep_results: Dict[str, Any],
        dry_run: bool = False,
    ) -> str:
        """Create WORM audit record for deep sleep action."""

        audit_dir = self.base_dir / "artifacts" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self.current_time.strftime("%Y%m%d_%H%M%SZ")
        audit_file = audit_dir / f"deep_sleep_{timestamp}.json"

        audit_record = {
            "timestamp": self.current_time.isoformat(),
            "event_type": "deep_sleep",
            "action": action,  # "enter_sleep", "wake_up", "sleep_check"
            "dry_run": dry_run,
            "analysis": analysis,
            "savings_projection": savings,
            "execution_results": sleep_results,
            "services_config": self.hibernation_services,
            "criteria": {
                "red_coverage_threshold": self.red_coverage_threshold,
                "min_sleep_duration_minutes": self.min_sleep_duration_minutes,
                "wake_lookahead_minutes": self.wake_lookahead_minutes,
            },
        }

        with open(audit_file, "w") as f:
            json.dump(audit_record, f, indent=2)

        print(f"📋 Deep sleep audit saved: {audit_file}")
        return str(audit_file)

    def create_sleep_token(self, action: str, duration_minutes: float = None) -> str:
        """Create sleep status token."""
        token_file = self.base_dir / "artifacts" / "ev" / f"deep_sleep_{action}"

        token_data = {
            "timestamp": self.current_time.isoformat(),
            "action": action,
            "sleep_duration_minutes": duration_minutes,
            "next_wake_time": (
                (
                    self.current_time + datetime.timedelta(minutes=duration_minutes)
                ).isoformat()
                if duration_minutes
                else None
            ),
        }

        with open(token_file, "w") as f:
            json.dump(token_data, f, indent=2)

        print(f"🎫 Sleep token created: {token_file}")
        return str(token_file)

    def run_deep_sleep_orchestrator(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run complete deep sleep orchestration."""

        print("💤 Deep Sleep Orchestrator: Infrastructure Hibernation")
        print("=" * 60)
        print(f"Current time: {self.current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        print("=" * 60)

        # Load 5-minute calendar
        print("📊 Loading 5-minute EV calendar...")
        calendar_df = self.load_5min_calendar()

        # Get current influence status
        print("🎯 Checking current influence status...")
        influence_status = self.get_current_influence_status()

        # Analyze upcoming windows
        print("🔍 Analyzing upcoming 60-minute window...")
        window_analysis = self.analyze_upcoming_windows(
            calendar_df, lookahead_minutes=60
        )

        # Check event gate activity
        print("🚨 Checking event gate activity...")
        event_activity = self.check_event_gate_activity()

        # Determine sleep eligibility
        total_influence = influence_status["total_influence_pct"]
        red_coverage = window_analysis["red_coverage"]
        has_active_events = event_activity["active_events"] > 0

        sleep_eligible = (
            window_analysis["sleep_eligible"]
            and total_influence == 0.0  # No active trading
            and not has_active_events  # No event overrides
        )

        # Calculate potential savings
        sleep_duration = window_analysis.get("sleep_duration_minutes", 60)
        savings_analysis = self.calculate_sleep_savings(sleep_duration)

        # Determine action
        if sleep_eligible:
            action = "enter_sleep"
            print(
                f"✅ Sleep criteria met: {red_coverage:.1%} red coverage, 0% influence, no events"
            )

            # Execute sleep sequence
            sleep_results = self.execute_sleep_sequence(dry_run)

            # Create sleep token
            if not dry_run:
                self.create_sleep_token(action, sleep_duration)

        else:
            action = "sleep_blocked"
            sleep_results = {"reason": "criteria_not_met"}

            blocking_reasons = []
            if not window_analysis["sleep_eligible"]:
                blocking_reasons.append(f"insufficient_red_coverage_{red_coverage:.1%}")
            if total_influence > 0:
                blocking_reasons.append(f"active_influence_{total_influence:.1f}%")
            if has_active_events:
                blocking_reasons.append(
                    f"active_events_{event_activity['active_events']}"
                )

            print(f"❌ Sleep blocked: {'; '.join(blocking_reasons)}")

        # Create audit record
        audit_file = self.create_sleep_audit(
            action, window_analysis, savings_analysis, sleep_results, dry_run
        )

        # Summary
        summary = {
            "timestamp": self.current_time.isoformat(),
            "action": action,
            "sleep_eligible": sleep_eligible,
            "influence_status": influence_status,
            "window_analysis": window_analysis,
            "event_activity": event_activity,
            "savings_analysis": savings_analysis,
            "execution_results": sleep_results,
            "audit_file": audit_file,
            "dry_run": dry_run,
        }

        print(f"\n💤 Deep Sleep Orchestrator Summary:")
        print(f"  Action: {action}")
        print(f"  Red Coverage: {red_coverage:.1%}")
        print(f"  Total Influence: {total_influence:.1f}%")
        print(f"  Active Events: {event_activity['active_events']}")

        if sleep_eligible:
            print(f"  Sleep Duration: {sleep_duration:.0f} minutes")
            print(
                f"  Projected Savings: ${savings_analysis['total_sleep_savings']:.2f}"
            )
            print(
                f"  Services Hibernated: {len(sleep_results.get('services_stopped', []))}"
            )

        if action == "enter_sleep":
            print("💤 Infrastructure hibernation active - cost savings in effect!")

        return summary


def main():
    """Main deep sleep orchestrator function."""
    parser = argparse.ArgumentParser(
        description="Deep Sleep Orchestrator: Infrastructure Hibernation"
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (default)")
    parser.add_argument(
        "--wake", action="store_true", help="Execute wake sequence instead"
    )
    args = parser.parse_args()

    try:
        orchestrator = DeepSleepOrchestrator()

        if args.wake:
            # Execute wake sequence
            print("🌅 Executing wake sequence...")
            wake_results = orchestrator.execute_wake_sequence(dry_run=args.dry_run)

            # Create audit for wake action
            audit_file = orchestrator.create_sleep_audit(
                "wake_up", {}, {}, wake_results, args.dry_run
            )

            print(
                f"🌅 Wake sequence complete: {len(wake_results['services_started'])} services started"
            )

        else:
            # Run sleep orchestration
            summary = orchestrator.run_deep_sleep_orchestrator(
                dry_run=True
            )  # Always dry run for safety

            print(f"💡 Next: Run 'make int8-cal' for optional INT8 optimization")

        return 0

    except Exception as e:
        print(f"❌ Deep sleep orchestrator failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
