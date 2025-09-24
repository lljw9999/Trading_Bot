#!/usr/bin/env python3
"""
Experiment Scheduler
Generate switchback assignments (treatment/control blocks) per asset, stratified by day/time.
"""
import os
import sys
import json
import yaml
import random
import datetime
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any


class SwitchbackScheduler:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.exp_config = self.config["experiment"]

        # Set random seed for reproducibility
        self.rng = random.Random(42)

    def load_config(self) -> Dict[str, Any]:
        """Load experiment configuration."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load experiment config: {e}")

    def should_exclude_window(self, timestamp: datetime.datetime) -> bool:
        """Check if timestamp falls in excluded window."""
        exclude_windows = self.exp_config.get("exclude_windows", [])

        for window in exclude_windows:
            tz = window.get("tz", "UTC")  # Currently only supporting UTC
            dow_exclude = window.get("dow", [])  # Day of week (0=Monday, 6=Sunday)
            hours_exclude = window.get("hours", [])  # Hours to exclude

            # Check day of week
            if timestamp.weekday() in dow_exclude:
                return True

            # Check hour
            if timestamp.hour in hours_exclude:
                return True

        return False

    def generate_time_blocks(
        self, start_date: datetime.date, num_days: int
    ) -> List[datetime.datetime]:
        """Generate time blocks for the experiment period."""
        block_minutes = self.exp_config["block_minutes"]
        blocks = []

        current_dt = datetime.datetime.combine(start_date, datetime.time(0, 0))
        end_dt = current_dt + datetime.timedelta(days=num_days)

        while current_dt < end_dt:
            # Skip excluded windows
            if not self.should_exclude_window(current_dt):
                blocks.append(current_dt)

            current_dt += datetime.timedelta(minutes=block_minutes)

        return blocks

    def assign_treatments_balanced(
        self, blocks: List[datetime.datetime], asset: str
    ) -> List[Tuple[str, str]]:
        """Assign treatment/control labels with balanced randomization."""

        # Create asset-specific random seed to ensure different assets have different patterns
        asset_seed = hash(asset) % (2**31)
        asset_rng = random.Random(42 + asset_seed)

        assignments = []

        # Generate balanced assignments (50/50 treatment/control)
        # Use block randomization within day boundaries

        # Group blocks by day
        daily_blocks = {}
        for block in blocks:
            day_key = block.date()
            if day_key not in daily_blocks:
                daily_blocks[day_key] = []
            daily_blocks[day_key].append(block)

        # Assign within each day for balance
        for day, day_blocks in daily_blocks.items():
            day_assignments = []

            # Ensure even number of blocks per day for perfect balance
            num_blocks = len(day_blocks)
            num_treatment = num_blocks // 2
            num_control = num_blocks - num_treatment

            # Create balanced assignment list
            day_labels = ["treatment"] * num_treatment + ["control"] * num_control

            # Shuffle for randomization
            asset_rng.shuffle(day_labels)

            # Pair blocks with assignments
            for block, label in zip(day_blocks, day_labels):
                time_str = block.strftime("%H:%M")
                assignments.append((time_str, label))

        return assignments

    def generate_assignments_for_date(
        self, target_date: datetime.date
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Generate assignments for a specific date."""
        print(f"📅 Generating assignments for {target_date}")

        # Generate time blocks for this single day
        blocks = self.generate_time_blocks(target_date, 1)

        assignments = {}

        for asset in self.exp_config["assets"]:
            asset_assignments = self.assign_treatments_balanced(blocks, asset)
            assignments[asset] = asset_assignments

            treatment_count = sum(
                1 for _, label in asset_assignments if label == "treatment"
            )
            control_count = len(asset_assignments) - treatment_count

            print(
                f"   {asset}: {len(asset_assignments)} blocks ({treatment_count} treatment, {control_count} control)"
            )

        return assignments

    def validate_assignments(
        self, assignments: Dict[str, List[Tuple[str, str]]]
    ) -> bool:
        """Validate assignment balance and coverage."""

        for asset, asset_assignments in assignments.items():
            if not asset_assignments:
                print(f"❌ No assignments for {asset}")
                return False

            treatment_count = sum(
                1 for _, label in asset_assignments if label == "treatment"
            )
            control_count = len(asset_assignments) - treatment_count
            total_count = len(asset_assignments)

            # Check balance (should be roughly 50/50)
            balance_ratio = treatment_count / total_count if total_count > 0 else 0

            if not (0.4 <= balance_ratio <= 0.6):  # Allow 40-60% range
                print(
                    f"❌ Unbalanced assignments for {asset}: {balance_ratio:.1%} treatment"
                )
                return False

            print(
                f"   ✅ {asset}: {balance_ratio:.1%} treatment ({treatment_count}/{total_count})"
            )

        return True

    def save_assignments(
        self, assignments: Dict[str, List[Tuple[str, str]]], target_date: datetime.date
    ):
        """Save assignments to artifacts directory."""

        # Create output directory
        artifacts_dir = Path(self.config.get("artifacts_dir", "experiments/m11"))
        daily_dir = artifacts_dir / target_date.strftime("%Y-%m-%d")
        daily_dir.mkdir(parents=True, exist_ok=True)

        # Save assignments
        assignment_file = (
            daily_dir / f"assignments_{target_date.strftime('%Y-%m-%d')}.json"
        )

        assignment_data = {
            "date": target_date.isoformat(),
            "experiment": self.exp_config["name"],
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "block_minutes": self.exp_config["block_minutes"],
            "assignments": assignments,
            "summary": self.generate_assignment_summary(assignments),
        }

        with open(assignment_file, "w") as f:
            json.dump(assignment_data, f, indent=2)

        # Create latest symlink
        latest_assignment = artifacts_dir / "assignments_latest.json"
        if latest_assignment.exists():
            latest_assignment.unlink()
        latest_assignment.symlink_to(assignment_file)

        print(f"📄 Assignments saved: {assignment_file}")

        return assignment_file

    def generate_assignment_summary(
        self, assignments: Dict[str, List[Tuple[str, str]]]
    ) -> Dict[str, Any]:
        """Generate summary statistics for assignments."""
        summary = {"total_assets": len(assignments), "per_asset_stats": {}}

        total_blocks = 0
        total_treatment = 0

        for asset, asset_assignments in assignments.items():
            treatment_count = sum(
                1 for _, label in asset_assignments if label == "treatment"
            )
            control_count = len(asset_assignments) - treatment_count

            summary["per_asset_stats"][asset] = {
                "total_blocks": len(asset_assignments),
                "treatment_blocks": treatment_count,
                "control_blocks": control_count,
                "treatment_ratio": (
                    treatment_count / len(asset_assignments) if asset_assignments else 0
                ),
            }

            total_blocks += len(asset_assignments)
            total_treatment += treatment_count

        summary["overall_stats"] = {
            "total_blocks": total_blocks,
            "treatment_blocks": total_treatment,
            "control_blocks": total_blocks - total_treatment,
            "treatment_ratio": (
                total_treatment / total_blocks if total_blocks > 0 else 0
            ),
        }

        return summary

    def generate_schedule_preview(self, num_days: int = 3) -> Dict[str, Any]:
        """Generate preview of assignments for next N days."""
        start_date = datetime.date.today()
        preview = {}

        for i in range(num_days):
            target_date = start_date + datetime.timedelta(days=i)
            assignments = self.generate_assignments_for_date(target_date)

            preview[target_date.isoformat()] = {
                "assignments": assignments,
                "summary": self.generate_assignment_summary(assignments),
            }

        return preview


def main():
    """Main scheduler function."""
    parser = argparse.ArgumentParser(description="Experiment Scheduler")
    parser.add_argument("-c", "--config", required=True, help="Experiment config file")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--preview", type=int, help="Preview assignments for N days")
    args = parser.parse_args()

    try:
        scheduler = SwitchbackScheduler(args.config)

        if args.preview:
            print(f"🔍 Generating {args.preview}-day preview...")
            preview = scheduler.generate_schedule_preview(args.preview)

            for date_str, date_data in preview.items():
                print(f"\n📅 {date_str}:")
                summary = date_data["summary"]["overall_stats"]
                print(f"   Total Blocks: {summary['total_blocks']}")
                print(
                    f"   Treatment: {summary['treatment_blocks']} ({summary['treatment_ratio']:.1%})"
                )
                print(f"   Control: {summary['control_blocks']}")

                for asset, stats in date_data["summary"]["per_asset_stats"].items():
                    print(
                        f"     {asset}: {stats['treatment_blocks']}/{stats['total_blocks']} treatment"
                    )

            return 0

        # Generate assignments for target date
        if args.date:
            target_date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
        else:
            target_date = datetime.date.today()

        assignments = scheduler.generate_assignments_for_date(target_date)

        # Validate assignments
        if not scheduler.validate_assignments(assignments):
            print("❌ Assignment validation failed")
            return 1

        # Save assignments
        assignment_file = scheduler.save_assignments(assignments, target_date)

        # Display summary
        summary = scheduler.generate_assignment_summary(assignments)
        overall = summary["overall_stats"]

        print(f"\n📊 Assignment Summary:")
        print(f"  Date: {target_date}")
        print(f"  Total Blocks: {overall['total_blocks']}")
        print(f"  Treatment Ratio: {overall['treatment_ratio']:.1%}")
        print(
            f"  Balance: ✅ GOOD"
            if 0.4 <= overall["treatment_ratio"] <= 0.6
            else "❌ UNBALANCED"
        )

        return 0

    except Exception as e:
        print(f"❌ Scheduler failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
