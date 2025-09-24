#!/usr/bin/env python3
"""
Duty-Cycler: Influence Scheduler Based on EV Calendar
Set per-asset influence based on green/amber/red EV bands with TTL and WORM auditing.
"""
import os
import sys
import json
import datetime
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional


class DutyCycler:
    def __init__(self, calendar_path: str):
        self.calendar_path = calendar_path
        self.current_time = datetime.datetime.now()
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

        # Load EV calendar
        self.ev_calendar = self.load_ev_calendar()

        # Duty cycle parameters
        self.green_pct = 10.0  # 10% influence in green windows (when GO token present)
        self.amber_pct = 5.0  # 5% influence in amber windows (half of green)
        self.red_pct = 0.0  # 0% influence in red windows
        self.ttl_minutes = 90  # TTL for influence settings (90 min = 1.5 hours)

    def load_ev_calendar(self) -> pd.DataFrame:
        """Load EV calendar from parquet file."""
        try:
            if self.calendar_path.endswith(".parquet"):
                df = pd.read_parquet(self.calendar_path)
            else:
                raise ValueError("Calendar must be parquet file")

            # Ensure timestamp is datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            print(f"📊 Loaded EV calendar: {len(df)} windows")
            return df

        except Exception as e:
            print(f"❌ Failed to load EV calendar from {self.calendar_path}: {e}")
            raise

    def get_current_ev_band(self, asset: str) -> str:
        """Get current EV band for an asset."""
        current_hour = self.current_time.replace(minute=0, second=0, microsecond=0)

        # Find EV data for current hour and asset
        current_windows = self.ev_calendar[
            (self.ev_calendar["asset"] == asset)
            & (self.ev_calendar["timestamp"] == current_hour)
        ]

        if current_windows.empty:
            print(f"⚠️ No EV data for {asset} at {current_hour}, defaulting to red")
            return "red"

        # Take the best venue for this asset/hour
        best_window = current_windows.loc[current_windows["ev_usd_per_hour"].idxmax()]
        return best_window["band"]

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
                            if (
                                datetime.datetime.now(datetime.timezone.utc)
                                > valid_until_dt
                            ):
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
        self, asset: str, ev_band: str, has_go_token: bool
    ) -> float:
        """Calculate target influence percentage for asset based on EV band."""

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
        self, asset: str, target_pct: float, reason: str, dry_run: bool = False
    ) -> bool:
        """Set influence for a specific asset."""
        try:
            if dry_run:
                print(
                    f"🧪 DRY RUN: Would set {asset} influence to {target_pct}% (reason: {reason})"
                )
                return True

            sys.path.insert(0, str(self.base_dir))
            from src.rl.influence_controller import InfluenceController

            ic = InfluenceController()

            # Set weight with TTL
            success = ic.set_weight_asset(asset, target_pct, reason)

            if success:
                print(f"✅ Set {asset} influence to {target_pct}% (reason: {reason})")
            else:
                print(f"❌ Failed to set {asset} influence to {target_pct}%")

            return success

        except Exception as e:
            print(f"❌ Error setting influence for {asset}: {e}")
            return False

    def create_duty_cycle_audit(
        self, actions: List[Dict[str, Any]], dry_run: bool = False
    ):
        """Create WORM audit record for duty cycle actions."""

        audit_dir = self.base_dir / "artifacts" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%SZ")
        audit_file = audit_dir / f"duty_cycle_{timestamp}.json"

        audit_record = {
            "timestamp": datetime.datetime.now().isoformat() + "Z",
            "event_type": "duty_cycle",
            "calendar_source": self.calendar_path,
            "duty_cycle_time": self.current_time.isoformat() + "Z",
            "dry_run": dry_run,
            "go_token_present": self.get_go_token_status(),
            "actions": actions,
            "parameters": {
                "green_pct": self.green_pct,
                "amber_pct": self.amber_pct,
                "red_pct": self.red_pct,
                "ttl_minutes": self.ttl_minutes,
            },
        }

        with open(audit_file, "w") as f:
            json.dump(audit_record, f, indent=2)

        print(f"📋 Duty cycle audit saved: {audit_file}")
        return audit_file

    def create_duty_cycle_token(self, output_dir: str = "artifacts/ev"):
        """Create duty cycle active token file."""
        token_file = Path(output_dir) / "duty_cycle_on"

        token_data = {
            "activated_at": datetime.datetime.now().isoformat() + "Z",
            "calendar_source": self.calendar_path,
            "parameters": {
                "green_pct": self.green_pct,
                "amber_pct": self.amber_pct,
                "red_pct": self.red_pct,
                "ttl_minutes": self.ttl_minutes,
            },
        }

        with open(token_file, "w") as f:
            json.dump(token_data, f, indent=2)

        print(f"🎫 Duty cycle token created: {token_file}")
        return token_file

    def run_duty_cycle(self, dry_run: bool = False) -> Dict[str, Any]:
        """Execute duty cycle based on current EV calendar."""

        print("⚡ Running duty cycle...")
        print(f"Current time: {self.current_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Check GO token status
        has_go_token = self.get_go_token_status()
        print(f"GO token present: {'✅' if has_go_token else '❌'}")

        # Get unique assets from calendar
        assets = self.ev_calendar["asset"].unique()

        actions = []
        changes_made = 0

        for asset in assets:
            # Get current EV band
            ev_band = self.get_current_ev_band(asset)

            # Calculate target influence
            target_pct = self.calculate_target_influence(asset, ev_band, has_go_token)

            # Get current influence
            current_pct = self.get_current_influence(asset)

            # Determine if change is needed
            change_needed = abs(current_pct - target_pct) > 0.1  # 0.1% tolerance

            action = {
                "asset": asset,
                "ev_band": ev_band,
                "current_influence_pct": current_pct,
                "target_influence_pct": target_pct,
                "change_needed": change_needed,
                "success": False,
            }

            if change_needed:
                reason = f"duty_cycle:{ev_band}:ttl_{self.ttl_minutes}m"
                success = self.set_asset_influence(asset, target_pct, reason, dry_run)
                action["success"] = success
                action["reason"] = reason

                if success:
                    changes_made += 1
            else:
                print(
                    f"⏭️ {asset}: No change needed ({current_pct:.1f}% -> {target_pct:.1f}%)"
                )
                action["success"] = True
                action["reason"] = "no_change_needed"

            actions.append(action)

        # Create audit record
        audit_file = self.create_duty_cycle_audit(actions, dry_run)

        # Create duty cycle token if not dry run
        token_file = None
        if not dry_run:
            token_file = self.create_duty_cycle_token()

        summary = {
            "timestamp": self.current_time.isoformat() + "Z",
            "total_assets": len(assets),
            "changes_made": changes_made,
            "dry_run": dry_run,
            "go_token_present": has_go_token,
            "actions": actions,
            "audit_file": str(audit_file),
            "token_file": str(token_file) if token_file else None,
        }

        print(f"\n⚡ Duty cycle complete:")
        print(f"  Assets processed: {len(assets)}")
        print(f"  Changes made: {changes_made}")
        print(f"  GO token present: {'✅' if has_go_token else '❌'}")

        return summary


def main():
    """Main duty cycler function."""
    parser = argparse.ArgumentParser(description="Duty-Cycler: Influence Scheduler")
    parser.add_argument("--calendar", required=True, help="EV calendar parquet file")
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
    args = parser.parse_args()

    # Check GO_LIVE flag for non-dry-run mode
    go_live = os.getenv("GO_LIVE", "0") == "1"

    if not args.dry_run and not go_live:
        print("❌ GO_LIVE flag not set. Use --dry-run or set GO_LIVE=1")
        return 1

    print("⚡ Duty-Cycler: Influence Scheduler")
    print("=" * 50)
    print(f"Calendar: {args.calendar}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Green %: {args.green_pct}%")
    print(f"Amber %: {args.amber_pct}%")
    print("=" * 50)

    try:
        cycler = DutyCycler(args.calendar)

        # Override default percentages if specified
        cycler.green_pct = args.green_pct
        cycler.amber_pct = args.amber_pct

        # Run duty cycle
        summary = cycler.run_duty_cycle(dry_run=args.dry_run)

        print(
            f"\n✅ Duty cycle {'simulation' if args.dry_run else 'execution'} complete!"
        )

        if not args.dry_run and summary["changes_made"] > 0:
            print(f"💡 Monitor influence changes with: make influence")
            print(f"📊 View audit trail: cat {summary['audit_file']}")

        return 0

    except Exception as e:
        print(f"❌ Duty cycler failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
