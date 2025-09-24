#!/usr/bin/env python3
"""
Duty-Cycler v2: 5-Minute TTL Scheduler with Event Override
Advanced duty cycling with 5-minute granularity, event gate integration, and minute-level TTLs.
"""
import os
import sys
import json
import datetime
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class DutyCyclerV2:
    def __init__(self, calendar_path: str):
        self.calendar_path = calendar_path
        self.current_time = datetime.datetime.now(datetime.timezone.utc)
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

        # Load 5-minute EV calendar
        self.ev_calendar_5m = self.load_5min_calendar()

        # Duty cycle parameters (5-minute precision)
        self.green_pct = 10.0  # 10% influence in green windows (when GO token present)
        self.amber_pct = 5.0  # 5% influence in amber windows
        self.red_pct = 0.0  # 0% influence in red windows
        self.ttl_minutes = 15  # TTL for influence settings (15 min for 5-min precision)

        # Event override parameters
        self.event_green_pct = 15.0  # Higher influence during event windows

    def load_5min_calendar(self) -> pd.DataFrame:
        """Load 5-minute EV calendar from parquet file."""
        try:
            if self.calendar_path.endswith(".parquet"):
                df = pd.read_parquet(self.calendar_path)
            else:
                raise ValueError("Calendar must be parquet file")

            # Ensure timestamp is datetime with timezone
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            print(f"📊 Loaded 5-minute EV calendar: {len(df)} windows")
            return df

        except Exception as e:
            print(
                f"❌ Failed to load 5-minute EV calendar from {self.calendar_path}: {e}"
            )
            raise

    def get_current_5min_window(self) -> datetime.datetime:
        """Get current 5-minute window boundary."""
        # Round current time to 5-minute boundary
        minutes = self.current_time.minute
        rounded_minutes = (minutes // 5) * 5

        window_time = self.current_time.replace(
            minute=rounded_minutes, second=0, microsecond=0
        )

        return window_time

    def get_current_ev_band(self, asset: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Get current EV band for an asset from 5-minute calendar."""
        current_window = self.get_current_5min_window()

        # Find EV data for current 5-minute window and asset
        current_windows = self.ev_calendar_5m[
            (self.ev_calendar_5m["asset"] == asset)
            & (self.ev_calendar_5m["timestamp"] == current_window)
        ]

        if current_windows.empty:
            print(
                f"⚠️ No 5-min EV data for {asset} at {current_window}, defaulting to red"
            )
            return "red", None

        # Take the best venue for this asset/window
        best_window = current_windows.loc[current_windows["ev_usd_per_hour"].idxmax()]
        window_data = {
            "venue": best_window["venue"],
            "ev_usd_per_hour": best_window["ev_usd_per_hour"],
            "window_time": current_window.isoformat(),
        }

        return best_window["band"], window_data

    def check_event_gate_override(
        self, asset: str
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if event gate has created a temporary green window for this asset."""
        try:
            # Check for event gate tokens
            event_dir = self.base_dir / "artifacts" / "ev" / "event_gate_on" / asset

            if not event_dir.exists():
                return False, None

            # Find recent event tokens (within last 30 minutes)
            cutoff_time = self.current_time - datetime.timedelta(minutes=30)
            active_tokens = []

            for token_file in event_dir.glob("event_green_*.json"):
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
                            active_tokens.append(token_data)

                except Exception:
                    continue

            if active_tokens:
                # Use most recent active token
                latest_token = max(active_tokens, key=lambda t: t["timestamp"])
                print(
                    f"🎫 Event gate override active for {asset}: score={latest_token['event_score']:.2f}"
                )
                return True, latest_token

            return False, None

        except Exception as e:
            print(f"⚠️ Error checking event gate for {asset}: {e}")
            return False, None

    def get_go_token_status(self) -> bool:
        """Check if M12 experiment GO token is present."""
        try:
            # Check for GO token from M12 experiment
            token_paths = ["experiments/m11/token_GO", "experiments/m11/decision.json"]

            for token_path in token_paths:
                token_file = self.base_dir / token_path
                if token_file.exists():
                    with open(token_file, "r") as f:
                        token_data = json.load(f)

                    # Check if decision is GO and token is valid
                    if token_data.get("decision") == "GO":
                        # Check token validity if present
                        valid_until = token_data.get("valid_until")
                        if valid_until:
                            valid_until_dt = datetime.datetime.fromisoformat(
                                valid_until.replace("Z", "+00:00")
                            )
                            if self.current_time > valid_until_dt:
                                print("⚠️ GO token has expired")
                                return False
                        print("✅ Valid GO token found")
                        return True

            print("⚠️ No valid GO token found")
            return False

        except Exception as e:
            print(f"⚠️ Error checking GO token: {e}")
            return False

    def calculate_target_influence(
        self, asset: str, ev_band: str, has_go_token: bool, event_override: bool = False
    ) -> float:
        """Calculate target influence percentage for asset based on EV band and event status."""

        # Event override takes priority
        if event_override:
            return self.event_green_pct if has_go_token else 0.0

        # Normal EV band logic
        if ev_band == "red":
            return self.red_pct  # Always 0% in red
        elif ev_band == "amber":
            return self.amber_pct if has_go_token else 0.0
        elif ev_band == "green":
            return self.green_pct if has_go_token else 0.0
        else:
            print(f"⚠️ Unknown EV band: {ev_band}, defaulting to 0%")
            return 0.0

    def get_current_influence(self, asset: str) -> float:
        """Get current influence for an asset."""
        try:
            sys.path.insert(0, str(self.base_dir))
            from src.rl.influence_controller import InfluenceController

            ic = InfluenceController()
            weights = ic.get_all_asset_weights()
            current_pct = weights.get(asset, 0.0) * 100
            return current_pct

        except Exception as e:
            print(f"⚠️ Error getting current influence for {asset}: {e}")
            return 0.0

    def set_asset_influence(
        self,
        asset: str,
        target_pct: float,
        reason: str,
        ttl_minutes: int,
        dry_run: bool = False,
    ) -> bool:
        """Set influence for a specific asset with TTL."""
        try:
            if dry_run:
                print(
                    f"🧪 DRY RUN: Would set {asset} influence to {target_pct}% for {ttl_minutes}min (reason: {reason})"
                )
                return True

            sys.path.insert(0, str(self.base_dir))
            from src.rl.influence_controller import InfluenceController

            ic = InfluenceController()

            # Set weight with TTL
            success = ic.set_weight_asset_with_ttl(
                asset, target_pct, reason, ttl_minutes
            )

            if success:
                print(
                    f"✅ Set {asset} influence to {target_pct}% for {ttl_minutes}min (reason: {reason})"
                )
            else:
                print(f"❌ Failed to set {asset} influence to {target_pct}%")

            return success

        except Exception as e:
            print(f"❌ Error setting influence for {asset}: {e}")
            # Fallback: try without TTL
            try:
                sys.path.insert(0, str(self.base_dir))
                from src.rl.influence_controller import InfluenceController

                ic = InfluenceController()
                success = ic.set_weight_asset(asset, target_pct, reason)
                if success:
                    print(
                        f"✅ Set {asset} influence to {target_pct}% (fallback without TTL)"
                    )
                return success
            except:
                return False

    def create_duty_cycle_audit(
        self, actions: List[Dict[str, Any]], dry_run: bool = False
    ):
        """Create WORM audit record for duty cycle actions."""

        audit_dir = self.base_dir / "artifacts" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self.current_time.strftime("%Y%m%d_%H%M%SZ")
        audit_file = audit_dir / f"duty_cycle_v2_{timestamp}.json"

        audit_record = {
            "timestamp": self.current_time.isoformat(),
            "event_type": "duty_cycle_v2",
            "calendar_source": self.calendar_path,
            "duty_cycle_time": self.current_time.isoformat(),
            "current_5min_window": self.get_current_5min_window().isoformat(),
            "dry_run": dry_run,
            "go_token_present": self.get_go_token_status(),
            "actions": actions,
            "parameters": {
                "green_pct": self.green_pct,
                "amber_pct": self.amber_pct,
                "red_pct": self.red_pct,
                "event_green_pct": self.event_green_pct,
                "ttl_minutes": self.ttl_minutes,
            },
        }

        with open(audit_file, "w") as f:
            json.dump(audit_record, f, indent=2)

        print(f"📋 Duty cycle v2 audit saved: {audit_file}")
        return audit_file

    def create_duty_cycle_token(self, output_dir: str = "artifacts/ev"):
        """Create duty cycle v2 active token file."""
        token_file = Path(output_dir) / "duty_cycle_v2_on"

        token_data = {
            "activated_at": self.current_time.isoformat(),
            "calendar_source": self.calendar_path,
            "current_5min_window": self.get_current_5min_window().isoformat(),
            "parameters": {
                "green_pct": self.green_pct,
                "amber_pct": self.amber_pct,
                "red_pct": self.red_pct,
                "event_green_pct": self.event_green_pct,
                "ttl_minutes": self.ttl_minutes,
            },
        }

        with open(token_file, "w") as f:
            json.dump(token_data, f, indent=2)

        print(f"🎫 Duty cycle v2 token created: {token_file}")
        return token_file

    def run_duty_cycle_v2(self, dry_run: bool = False) -> Dict[str, Any]:
        """Execute 5-minute duty cycle with event override support."""

        print("⚡ Duty-Cycler v2: 5-Minute TTL Scheduler")
        print("=" * 55)
        print(f"Current time: {self.current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(
            f"5-min window: {self.get_current_5min_window().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        print(f"TTL: {self.ttl_minutes} minutes")
        print("=" * 55)

        # Check GO token status
        has_go_token = self.get_go_token_status()
        print(f"GO token present: {'✅' if has_go_token else '❌'}")

        # Get unique assets from calendar
        assets = self.ev_calendar_5m["asset"].unique()

        actions = []
        changes_made = 0
        event_overrides = 0

        for asset in assets:
            # Get current EV band for 5-minute window
            ev_band, window_data = self.get_current_ev_band(asset)

            # Check for event gate override
            event_override, event_data = self.check_event_gate_override(asset)

            # Calculate target influence
            target_pct = self.calculate_target_influence(
                asset, ev_band, has_go_token, event_override
            )

            # Get current influence
            current_pct = self.get_current_influence(asset)

            # Determine if change is needed (smaller tolerance for 5-min precision)
            change_needed = abs(current_pct - target_pct) > 0.05  # 0.05% tolerance

            # Build reason string
            reason_parts = [f"duty_v2:{ev_band}"]
            if event_override:
                reason_parts.append("event_override")
                event_overrides += 1
            reason_parts.append(f"ttl_{self.ttl_minutes}m")
            reason = ":".join(reason_parts)

            action = {
                "asset": asset,
                "ev_band": ev_band,
                "event_override": event_override,
                "current_influence_pct": current_pct,
                "target_influence_pct": target_pct,
                "change_needed": change_needed,
                "success": False,
                "window_data": window_data,
                "event_data": event_data,
            }

            if change_needed:
                success = self.set_asset_influence(
                    asset, target_pct, reason, self.ttl_minutes, dry_run
                )
                action["success"] = success
                action["reason"] = reason

                if success:
                    changes_made += 1
            else:
                print(
                    f"⏭️ {asset}: No change needed ({current_pct:.2f}% -> {target_pct:.2f}%)"
                )
                action["success"] = True
                action["reason"] = "no_change_needed"

            actions.append(action)

        # Create audit record
        audit_file = self.create_duty_cycle_audit(actions, dry_run)

        # Create duty cycle v2 token if not dry run
        token_file = None
        if not dry_run:
            token_file = self.create_duty_cycle_token()

        summary = {
            "timestamp": self.current_time.isoformat(),
            "current_5min_window": self.get_current_5min_window().isoformat(),
            "total_assets": len(assets),
            "changes_made": changes_made,
            "event_overrides": event_overrides,
            "dry_run": dry_run,
            "go_token_present": has_go_token,
            "actions": actions,
            "audit_file": str(audit_file),
            "token_file": str(token_file) if token_file else None,
        }

        print(f"\n⚡ Duty cycle v2 complete:")
        print(f"  Assets processed: {len(assets)}")
        print(f"  Changes made: {changes_made}")
        print(f"  Event overrides: {event_overrides}")
        print(f"  GO token present: {'✅' if has_go_token else '❌'}")
        print(
            f"  Next run: {(self.current_time + datetime.timedelta(minutes=5)).strftime('%H:%M UTC')}"
        )

        return summary


def main():
    """Main duty cycler v2 function."""
    parser = argparse.ArgumentParser(
        description="Duty-Cycler v2: 5-Minute TTL Scheduler"
    )
    parser.add_argument(
        "--calendar", required=True, help="5-minute EV calendar parquet file"
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument(
        "--green-pct",
        type=float,
        default=10.0,
        help="Green window influence percentage",
    )
    parser.add_argument(
        "--amber-pct", type=float, default=5.0, help="Amber window influence percentage"
    )
    parser.add_argument(
        "--event-pct",
        type=float,
        default=15.0,
        help="Event override influence percentage",
    )
    parser.add_argument(
        "--ttl", type=int, default=15, help="TTL in minutes for influence settings"
    )
    args = parser.parse_args()

    # Check GO_LIVE flag for non-dry-run mode
    go_live = os.getenv("GO_LIVE", "0") == "1"

    if not args.dry_run and not go_live:
        print("❌ GO_LIVE flag not set. Use --dry-run or set GO_LIVE=1")
        return 1

    print("⚡ Duty-Cycler v2: 5-Minute TTL Scheduler")
    print("=" * 55)
    print(f"Calendar: {args.calendar}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Green %: {args.green_pct}%")
    print(f"Amber %: {args.amber_pct}%")
    print(f"Event %: {args.event_pct}%")
    print(f"TTL: {args.ttl} minutes")
    print("=" * 55)

    try:
        cycler = DutyCyclerV2(args.calendar)

        # Override default percentages if specified
        cycler.green_pct = args.green_pct
        cycler.amber_pct = args.amber_pct
        cycler.event_green_pct = args.event_pct
        cycler.ttl_minutes = args.ttl

        # Run duty cycle v2
        summary = cycler.run_duty_cycle_v2(dry_run=args.dry_run)

        print(
            f"\n✅ Duty cycle v2 {'simulation' if args.dry_run else 'execution'} complete!"
        )

        if not args.dry_run and summary["changes_made"] > 0:
            print(f"💡 Monitor influence changes with: make asset-influence")
            print(f"📊 View audit trail: cat {summary['audit_file']}")

        if summary["event_overrides"] > 0:
            print(f"🎫 Event overrides active: {summary['event_overrides']} assets")

        print(f"💡 Next: Run 'make sleep-now' to test deep sleep orchestrator")
        return 0

    except Exception as e:
        print(f"❌ Duty cycler v2 failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
