#!/usr/bin/env python3
"""
Experiment Metrics Collector
Hourly collection of experiment metrics: net P&L, TCA metrics, tagged by treatment assignment.
"""
import os
import sys
import json
import yaml
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any


class ExperimentMetricsCollector:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.exp_config = self.config["experiment"]
        self.artifacts_dir = Path(self.config.get("artifacts_dir", "experiments/m11"))

    def load_config(self) -> Dict[str, Any]:
        """Load experiment configuration."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load experiment config: {e}")

    def get_current_assignment(self, asset: str, timestamp: datetime.datetime) -> str:
        """Get treatment assignment for asset at given timestamp."""

        # Load assignments for current date
        date_str = timestamp.date().strftime("%Y-%m-%d")
        assignment_file = self.artifacts_dir / date_str / f"assignments_{date_str}.json"

        if not assignment_file.exists():
            # For simulation/testing: use deterministic assignment based on hour + asset
            # This ensures balanced treatment/control when no real assignments exist
            hour = timestamp.hour
            asset_hash = hash(asset) % 2
            assignment_hash = (hour + asset_hash) % 2
            return "treatment" if assignment_hash == 0 else "control"

        try:
            with open(assignment_file, "r") as f:
                assignment_data = json.load(f)

            assignments = assignment_data.get("assignments", {}).get(asset, [])

            # Find matching time block
            block_minutes = self.exp_config["block_minutes"]

            for time_str, treatment in assignments:
                block_time = datetime.datetime.strptime(time_str, "%H:%M").time()
                block_start = datetime.datetime.combine(timestamp.date(), block_time)
                block_end = block_start + datetime.timedelta(minutes=block_minutes)

                if block_start <= timestamp < block_end:
                    return treatment

            # Default to control if no match found
            return "control"

        except Exception as e:
            print(f"⚠️ Error reading assignments: {e}")
            # Fall back to deterministic assignment for simulation
            hour = timestamp.hour
            asset_hash = hash(asset) % 2
            assignment_hash = (hour + asset_hash) % 2
            return "treatment" if assignment_hash == 0 else "control"

    def collect_trading_metrics(self, hour_start: datetime.datetime) -> Dict[str, Any]:
        """Collect trading metrics for the hour (simulated)."""

        # In production, this would:
        # 1. Query fills database for the hour
        # 2. Join with fee engine data
        # 3. Calculate TCA metrics
        # 4. Join with infrastructure cost allocation

        # Simulated metrics for demo
        hour_metrics = {}

        for asset in self.exp_config["assets"]:
            # Get treatment assignment for this hour
            assignment = self.get_current_assignment(asset, hour_start)

            # Generate synthetic metrics based on treatment
            if assignment == "treatment":
                # M11 improvements: better P&L, lower slippage
                base_pnl = np.random.normal(15, 8)  # Better P&L
                slippage_p95 = np.random.gamma(2, 6)  # Lower slippage
                is_bps = np.random.gamma(1.5, 4)  # Better IS
                fill_ratio = np.random.beta(20, 2)  # Higher fill ratio
            else:
                # Control: baseline performance
                base_pnl = np.random.normal(8, 12)  # Baseline P&L
                slippage_p95 = np.random.gamma(3, 8)  # Higher slippage
                is_bps = np.random.gamma(2, 6)  # Baseline IS
                fill_ratio = np.random.beta(15, 3)  # Lower fill ratio

            # Apply market regime effects
            vol_factor = np.random.uniform(0.8, 1.2)
            regime_adjustment = (
                vol_factor if assignment == "treatment" else vol_factor * 0.9
            )

            # Infrastructure costs (fixed per hour)
            infra_cost_per_hour = 95.5 / 24  # From CFO report, daily cost
            asset_infra_allocation = infra_cost_per_hour / len(
                self.exp_config["assets"]
            )

            # Trading fees (proportional to activity)
            trading_volume = abs(base_pnl) * 50  # Synthetic volume
            fee_rate = 0.001  # 10 bps
            trading_fees = trading_volume * fee_rate

            # Net P&L after all costs
            net_pnl = base_pnl - trading_fees - asset_infra_allocation

            # Cost ratio
            gross_pnl = max(base_pnl, 0.01)  # Avoid division by zero
            cost_ratio = (trading_fees + asset_infra_allocation) / gross_pnl

            hour_metrics[asset] = {
                "timestamp": hour_start.isoformat() + "Z",
                "assignment": assignment,
                "gross_pnl_usd": round(base_pnl, 2),
                "trading_fees_usd": round(trading_fees, 2),
                "infra_cost_usd": round(asset_infra_allocation, 2),
                "net_pnl_usd": round(net_pnl, 2),
                "slip_bps_p95": round(slippage_p95, 2),
                "is_bps": round(is_bps, 2),
                "fill_ratio": round(fill_ratio, 4),
                "cost_ratio": round(cost_ratio, 4),
                "trading_volume": round(trading_volume, 2),
                "regime_factor": round(regime_adjustment, 3),
            }

        return hour_metrics

    def collect_covariates(self, hour_start: datetime.datetime) -> Dict[str, Any]:
        """Collect CUPED covariates (pre-period data)."""

        # Look back 24 hours for pre-period data
        pre_period_start = hour_start - datetime.timedelta(hours=24)

        covariates = {}

        for asset in self.exp_config["assets"]:
            # Simulate historical data lookup
            pre_pnl = np.random.normal(10, 15)
            vol_5m = np.random.gamma(2, 0.005)  # 5-minute volatility
            spread_bps = np.random.gamma(3, 3)
            volume = np.random.lognormal(8, 1)

            covariates[asset] = {
                "pre_pnl": round(pre_pnl, 2),
                "vol_5m": round(vol_5m, 6),
                "spread_bps": round(spread_bps, 2),
                "volume": round(volume, 2),
            }

        return covariates

    def save_hourly_metrics(
        self,
        hour_start: datetime.datetime,
        metrics: Dict[str, Any],
        covariates: Dict[str, Any],
    ):
        """Save hourly metrics to artifacts."""

        # Create daily directory
        date_str = hour_start.date().strftime("%Y-%m-%d")
        daily_dir = self.artifacts_dir / date_str
        daily_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed metrics
        hour_str = hour_start.strftime("%H")
        metrics_file = daily_dir / f"metrics_{date_str}_{hour_str}.json"

        metrics_data = {
            "collection_time": datetime.datetime.utcnow().isoformat() + "Z",
            "hour_start": hour_start.isoformat() + "Z",
            "experiment": self.exp_config["name"],
            "metrics": metrics,
            "covariates": covariates,
            "primary_metric": self.exp_config["metrics"]["primary"],
            "secondary_metrics": self.exp_config["metrics"]["secondaries"],
        }

        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)

        # Update daily summary
        self.update_daily_summary(date_str, metrics, covariates)

        return metrics_file

    def update_daily_summary(
        self, date_str: str, hour_metrics: Dict[str, Any], covariates: Dict[str, Any]
    ):
        """Update daily summary with new hourly data."""

        daily_dir = self.artifacts_dir / date_str
        summary_file = daily_dir / "daily_summary.json"

        # Load existing summary or create new
        if summary_file.exists():
            with open(summary_file, "r") as f:
                daily_summary = json.load(f)
        else:
            daily_summary = {
                "date": date_str,
                "experiment": self.exp_config["name"],
                "hours_collected": 0,
                "asset_summaries": {},
                "overall_summary": {},
            }

        # Update hour count
        daily_summary["hours_collected"] += 1

        # Update asset summaries
        for asset, asset_metrics in hour_metrics.items():
            if asset not in daily_summary["asset_summaries"]:
                daily_summary["asset_summaries"][asset] = {
                    "treatment_hours": 0,
                    "control_hours": 0,
                    "total_net_pnl": 0,
                    "total_gross_pnl": 0,
                    "total_fees": 0,
                    "avg_slippage_p95": [],
                    "avg_is_bps": [],
                    "avg_fill_ratio": [],
                    "avg_cost_ratio": [],
                }

            asset_summary = daily_summary["asset_summaries"][asset]

            # Update treatment/control hours
            if asset_metrics["assignment"] == "treatment":
                asset_summary["treatment_hours"] += 1
            else:
                asset_summary["control_hours"] += 1

            # Accumulate metrics
            asset_summary["total_net_pnl"] += asset_metrics["net_pnl_usd"]
            asset_summary["total_gross_pnl"] += asset_metrics["gross_pnl_usd"]
            asset_summary["total_fees"] += asset_metrics["trading_fees_usd"]

            # Append for averaging
            asset_summary["avg_slippage_p95"].append(asset_metrics["slip_bps_p95"])
            asset_summary["avg_is_bps"].append(asset_metrics["is_bps"])
            asset_summary["avg_fill_ratio"].append(asset_metrics["fill_ratio"])
            asset_summary["avg_cost_ratio"].append(asset_metrics["cost_ratio"])

        # Compute overall summary
        overall = daily_summary["overall_summary"]
        total_net_pnl = sum(
            s["total_net_pnl"] for s in daily_summary["asset_summaries"].values()
        )
        total_hours = sum(
            s["treatment_hours"] + s["control_hours"]
            for s in daily_summary["asset_summaries"].values()
        )

        overall["total_net_pnl_day"] = round(total_net_pnl, 2)
        overall["avg_net_pnl_per_hour"] = round(total_net_pnl / max(total_hours, 1), 2)
        overall["total_hours_collected"] = total_hours

        # Treatment/control balance
        treatment_hours = sum(
            s["treatment_hours"] for s in daily_summary["asset_summaries"].values()
        )
        control_hours = sum(
            s["control_hours"] for s in daily_summary["asset_summaries"].values()
        )
        overall["treatment_ratio"] = treatment_hours / max(total_hours, 1)
        overall["assignment_balance"] = (
            "GOOD" if 0.4 <= overall["treatment_ratio"] <= 0.6 else "UNBALANCED"
        )

        # Save updated summary
        with open(summary_file, "w") as f:
            json.dump(daily_summary, f, indent=2)

        # Update latest symlink
        latest_summary = self.artifacts_dir / "daily_summary_latest.json"
        if latest_summary.exists() or latest_summary.is_symlink():
            latest_summary.unlink()
        latest_summary.symlink_to(summary_file)

    def collect_current_hour(self) -> Dict[str, Any]:
        """Collect metrics for current hour."""

        now = datetime.datetime.utcnow()
        hour_start = now.replace(minute=0, second=0, microsecond=0)

        print(
            f"📊 Collecting metrics for {hour_start.strftime('%Y-%m-%d %H:00 UTC')}..."
        )

        # Collect metrics and covariates
        metrics = self.collect_trading_metrics(hour_start)
        covariates = self.collect_covariates(hour_start)

        # Display summary
        treatment_count = sum(
            1 for m in metrics.values() if m["assignment"] == "treatment"
        )
        control_count = len(metrics) - treatment_count
        total_net_pnl = sum(m["net_pnl_usd"] for m in metrics.values())

        print(
            f"   Assets: {len(metrics)} ({treatment_count} treatment, {control_count} control)"
        )
        print(f"   Total Net P&L: ${total_net_pnl:.2f}")

        for asset, asset_metrics in metrics.items():
            assignment_emoji = (
                "🧪" if asset_metrics["assignment"] == "treatment" else "🔧"
            )
            print(
                f"     {assignment_emoji} {asset}: ${asset_metrics['net_pnl_usd']:.2f} "
                + f"({asset_metrics['slip_bps_p95']:.1f}bp slip)"
            )

        # Save metrics
        metrics_file = self.save_hourly_metrics(hour_start, metrics, covariates)
        print(f"   Saved: {metrics_file}")

        return {
            "hour_start": hour_start.isoformat() + "Z",
            "metrics_collected": len(metrics),
            "treatment_assignments": treatment_count,
            "control_assignments": control_count,
            "total_net_pnl": total_net_pnl,
            "metrics_file": str(metrics_file),
        }


def main():
    """Main metrics collector function."""
    parser = argparse.ArgumentParser(description="Experiment Metrics Collector")
    parser.add_argument("-c", "--config", required=True, help="Experiment config file")
    parser.add_argument("--hour", help="Specific hour to collect (YYYY-MM-DD-HH)")
    parser.add_argument(
        "--simulate-day", action="store_true", help="Simulate full day collection"
    )
    args = parser.parse_args()

    try:
        collector = ExperimentMetricsCollector(args.config)

        if args.simulate_day:
            print("🔄 Simulating full day collection...")

            start_hour = datetime.datetime.utcnow().replace(
                minute=0, second=0, microsecond=0
            ) - datetime.timedelta(hours=23)

            for hour_offset in range(24):
                hour_start = start_hour + datetime.timedelta(hours=hour_offset)

                # Collect for this hour
                metrics = collector.collect_trading_metrics(hour_start)
                covariates = collector.collect_covariates(hour_start)
                collector.save_hourly_metrics(hour_start, metrics, covariates)

                print(f"   Collected: {hour_start.strftime('%H:00')}")

            print("✅ Day simulation complete")
            return 0

        elif args.hour:
            target_hour = datetime.datetime.strptime(args.hour, "%Y-%m-%d-%H")
            print(
                f"📊 Collecting metrics for {target_hour.strftime('%Y-%m-%d %H:00')}..."
            )

            metrics = collector.collect_trading_metrics(target_hour)
            covariates = collector.collect_covariates(target_hour)
            metrics_file = collector.save_hourly_metrics(
                target_hour, metrics, covariates
            )

            print(f"✅ Metrics collected: {metrics_file}")
            return 0

        else:
            # Collect current hour
            result = collector.collect_current_hour()
            print(f"✅ Current hour collection complete")
            return 0

    except Exception as e:
        print(f"❌ Metrics collection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
