#!/usr/bin/env python3
"""
EV Calibration: Cost-Aware Threshold Setting
Compute cost_per_active_hour and set dynamic EV threshold = cost + margin.
"""
import os
import sys
import json
import datetime
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


class EVCalibrator:
    def __init__(self, margin_usd: float = 5.0):
        self.margin_usd = margin_usd
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

    def load_cost_structure(self) -> Dict[str, float]:
        """Load current cost structure from various sources."""
        cost_structure = {
            "gpu_cost_per_hour": 0.0,
            "infrastructure_cost_per_hour": 0.0,
            "data_cost_per_hour": 0.0,
            "execution_cost_per_hour": 0.0,
            "monitoring_cost_per_hour": 0.0,
        }

        # Try to load from cost optimization artifacts
        try:
            # Check M10 quantization results
            quant_files = list(
                self.base_dir.glob("artifacts/cost/quant/*/quantization_report.json")
            )
            if quant_files:
                latest_quant = max(quant_files, key=lambda p: p.stat().st_mtime)
                with open(latest_quant, "r") as f:
                    quant_data = json.load(f)

                # Extract GPU cost from quantization analysis
                gpu_analysis = quant_data.get("gpu_analysis", {})
                cost_structure["gpu_cost_per_hour"] = gpu_analysis.get(
                    "cost_per_hour_optimized", 2.1
                )

            # Check CFO reports for operational costs
            cfo_files = list(self.base_dir.glob("artifacts/cfo/*/cfo_report.json"))
            if cfo_files:
                latest_cfo = max(cfo_files, key=lambda p: p.stat().st_mtime)
                with open(latest_cfo, "r") as f:
                    cfo_data = json.load(f)

                # Extract infrastructure costs
                cost_breakdown = cfo_data.get("cost_breakdown", {})
                cost_structure["infrastructure_cost_per_hour"] = cost_breakdown.get(
                    "infrastructure_hourly", 1.2
                )
                cost_structure["data_cost_per_hour"] = cost_breakdown.get(
                    "data_feeds_hourly", 0.8
                )
                cost_structure["execution_cost_per_hour"] = cost_breakdown.get(
                    "execution_fees_hourly", 1.5
                )
                cost_structure["monitoring_cost_per_hour"] = cost_breakdown.get(
                    "monitoring_hourly", 0.3
                )

        except Exception as e:
            print(f"⚠️ Could not load cost data: {e}")

        # Fallback to estimated costs if no data available
        if all(v == 0.0 for v in cost_structure.values()):
            cost_structure = {
                "gpu_cost_per_hour": 2.1,  # Optimized GPU cost from M10
                "infrastructure_cost_per_hour": 1.2,  # Redis, Kafka, monitoring
                "data_cost_per_hour": 0.8,  # Market data feeds
                "execution_cost_per_hour": 1.5,  # Average execution costs
                "monitoring_cost_per_hour": 0.3,  # Prometheus, Grafana, alerts
            }
            print("📊 Using estimated cost structure (no historical data found)")

        return cost_structure

    def load_historical_performance(self, window_days: int = 14) -> Dict[str, Any]:
        """Load historical P&L and activity data for cost calibration."""
        performance_data = {
            "total_hours_active": 0,
            "total_gross_pnl_usd": 0.0,
            "total_net_pnl_usd": 0.0,
            "avg_hourly_volume_usd": 0.0,
            "cost_ratio_current": 0.58,  # From M11/M13 optimization
        }

        try:
            # Try to load from economic close data
            econ_files = list(self.base_dir.glob("artifacts/econ/*/econ_close.json"))
            if econ_files:
                # Load recent economic data
                recent_econ = []
                cutoff_date = datetime.datetime.now() - datetime.timedelta(
                    days=window_days
                )

                for econ_file in econ_files:
                    try:
                        # Extract date from filename or file modification time
                        file_date = datetime.datetime.fromtimestamp(
                            econ_file.stat().st_mtime
                        )
                        if file_date >= cutoff_date:
                            with open(econ_file, "r") as f:
                                econ_data = json.load(f)
                            recent_econ.append(econ_data)
                    except:
                        continue

                if recent_econ:
                    # Aggregate performance data
                    total_gross = sum(
                        d.get("portfolio", {}).get("gross_pnl_usd", 0)
                        for d in recent_econ
                    )
                    total_net = sum(
                        d.get("portfolio", {}).get("net_pnl_usd", 0)
                        for d in recent_econ
                    )

                    performance_data.update(
                        {
                            "total_hours_active": len(recent_econ)
                            * 24,  # Assume 24h/day active
                            "total_gross_pnl_usd": total_gross,
                            "total_net_pnl_usd": total_net,
                            "avg_hourly_volume_usd": total_gross
                            / max(len(recent_econ) * 24, 1),
                        }
                    )

                    print(f"📊 Loaded {len(recent_econ)} days of economic data")

        except Exception as e:
            print(f"⚠️ Could not load historical performance: {e}")

        # Fallback to simulated recent performance
        if performance_data["total_hours_active"] == 0:
            # Simulate realistic performance based on M11/M13 results
            hours_active = window_days * 24
            performance_data.update(
                {
                    "total_hours_active": hours_active,
                    "total_gross_pnl_usd": hours_active * 12.5,  # $12.5/hour from M11
                    "total_net_pnl_usd": hours_active * 5.25,  # 58% cost ratio
                    "avg_hourly_volume_usd": 45000,  # Typical volume
                    "cost_ratio_current": 0.58,
                }
            )
            print(f"📊 Using simulated {window_days}-day performance data")

        return performance_data

    def calculate_cost_per_active_hour(
        self, cost_structure: Dict[str, float], performance_data: Dict[str, Any]
    ) -> float:
        """Calculate total cost per active trading hour."""

        # Base infrastructure costs (always running)
        base_hourly_cost = (
            cost_structure["gpu_cost_per_hour"]
            + cost_structure["infrastructure_cost_per_hour"]
            + cost_structure["data_cost_per_hour"]
            + cost_structure["monitoring_cost_per_hour"]
        )

        # Variable execution costs (proportional to activity)
        variable_hourly_cost = cost_structure["execution_cost_per_hour"]

        # Calculate activity factor (0-1) based on recent influence
        activity_factor = min(
            1.0, performance_data["avg_hourly_volume_usd"] / 50000
        )  # Normalize to $50k/hour

        # Total cost per active hour
        cost_per_active_hour = base_hourly_cost + (
            variable_hourly_cost * activity_factor
        )

        return cost_per_active_hour

    def calculate_break_even_threshold(self, cost_per_hour: float) -> float:
        """Calculate EV threshold for break-even + margin."""
        return cost_per_hour + self.margin_usd

    def validate_threshold(
        self, threshold: float, performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate threshold against historical performance."""
        validation = {
            "threshold_reasonable": False,
            "historical_beat_rate": 0.0,
            "expected_green_hours_per_day": 0.0,
            "cost_ratio_projection": 1.0,
        }

        # Check if threshold is reasonable (not too high/low)
        avg_gross_hourly = performance_data["total_gross_pnl_usd"] / max(
            performance_data["total_hours_active"], 1
        )

        validation["threshold_reasonable"] = (
            threshold <= avg_gross_hourly * 1.5
        )  # Within 150% of historical average

        # Estimate beat rate (simplified)
        if avg_gross_hourly > 0:
            # Assume normal distribution around historical average
            std_dev = avg_gross_hourly * 0.8  # 80% volatility

            # Probability of exceeding threshold
            z_score = (threshold - avg_gross_hourly) / std_dev
            beat_rate = max(
                0.1, 0.5 - (z_score * 0.2)
            )  # Simplified normal CDF approximation

            validation["historical_beat_rate"] = beat_rate
            validation["expected_green_hours_per_day"] = 24 * beat_rate

        # Project cost ratio improvement
        if validation["expected_green_hours_per_day"] > 0:
            # Assume 50% cost reduction during hibernation periods
            hibernation_factor = 0.5
            red_hours_per_day = 24 - validation["expected_green_hours_per_day"]

            # Current cost ratio with hibernation benefit
            cost_reduction = (red_hours_per_day / 24) * hibernation_factor
            validation["cost_ratio_projection"] = performance_data[
                "cost_ratio_current"
            ] * (1 - cost_reduction)

        return validation

    def generate_calibration_report(
        self, window_days: int = 14, output_dir: str = "artifacts/ev"
    ) -> Dict[str, Any]:
        """Generate comprehensive EV calibration report."""

        print("💰 EV Calibration: Cost-Aware Threshold Setting")
        print("=" * 55)
        print(f"Analysis Window: {window_days} days")
        print(f"Margin: ${self.margin_usd:.1f}/hour")
        print("=" * 55)

        # Load cost structure
        print("📊 Loading cost structure...")
        cost_structure = self.load_cost_structure()

        # Load historical performance
        print("📈 Loading historical performance...")
        performance_data = self.load_historical_performance(window_days)

        # Calculate cost per active hour
        print("💸 Calculating cost per active hour...")
        cost_per_hour = self.calculate_cost_per_active_hour(
            cost_structure, performance_data
        )

        # Calculate EV threshold
        print("🎯 Calculating EV threshold...")
        ev_threshold = self.calculate_break_even_threshold(cost_per_hour)

        # Validate threshold
        print("✅ Validating threshold...")
        validation = self.validate_threshold(ev_threshold, performance_data)

        # Create calibration report
        calibration_report = {
            "timestamp": datetime.datetime.now().isoformat() + "Z",
            "analysis_window_days": window_days,
            "margin_usd": self.margin_usd,
            "cost_structure": cost_structure,
            "performance_data": performance_data,
            "cost_per_active_hour_usd": cost_per_hour,
            "ev_threshold_usd": ev_threshold,
            "validation": validation,
            "recommendations": [],
        }

        # Add recommendations
        if validation["threshold_reasonable"]:
            calibration_report["recommendations"].append(
                "Threshold is reasonable based on historical performance"
            )
        else:
            calibration_report["recommendations"].append(
                "WARNING: Threshold may be too aggressive"
            )

        if validation["expected_green_hours_per_day"] >= 2:
            calibration_report["recommendations"].append(
                f"Expected {validation['expected_green_hours_per_day']:.1f} green hours/day"
            )
        else:
            calibration_report["recommendations"].append(
                "WARNING: Very few green hours expected - consider lowering threshold"
            )

        if validation["cost_ratio_projection"] <= 0.30:
            calibration_report["recommendations"].append(
                "✅ Projected cost ratio ≤30% with hibernation"
            )
        else:
            calibration_report["recommendations"].append(
                f"⚠️ Projected cost ratio {validation['cost_ratio_projection']:.1%} still above 30%"
            )

        # Save calibration report
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%SZ")
        report_file = output_path / f"ev_calibration_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(calibration_report, f, indent=2)

        # Create latest symlink
        latest_file = output_path / "ev_calibration.json"
        if latest_file.exists() or latest_file.is_symlink():
            latest_file.unlink()
        latest_file.symlink_to(report_file.name)

        # Print summary
        print(f"\n💰 EV Calibration Results:")
        print(f"  Cost per Active Hour: ${cost_per_hour:.2f}")
        print(f"  EV Threshold: ${ev_threshold:.2f}")
        print(
            f"  Expected Green Hours/Day: {validation['expected_green_hours_per_day']:.1f}"
        )
        print(f"  Projected Cost Ratio: {validation['cost_ratio_projection']:.1%}")
        print(f"  Report saved: {report_file}")

        if validation["cost_ratio_projection"] <= 0.30:
            print("✅ Calibration projects ≤30% cost ratio target achievable!")
        else:
            print("⚠️ Additional optimization needed to reach 30% cost ratio")

        return calibration_report


def main():
    """Main EV calibration function."""
    parser = argparse.ArgumentParser(
        description="EV Calibration: Cost-Aware Threshold Setting"
    )
    parser.add_argument("--window", default="14d", help="Analysis window (e.g., 14d)")
    parser.add_argument(
        "--margin", type=float, default=5.0, help="Margin above cost per hour"
    )
    parser.add_argument("--out", default="artifacts/ev", help="Output directory")
    args = parser.parse_args()

    # Parse window
    if args.window.endswith("d"):
        window_days = int(args.window[:-1])
    else:
        window_days = int(args.window)

    try:
        calibrator = EVCalibrator(margin_usd=args.margin)
        report = calibrator.generate_calibration_report(window_days, args.out)

        # Simple output for automation
        simple_result = {
            "cost_per_active_hour_usd": report["cost_per_active_hour_usd"],
            "ev_threshold_usd": report["ev_threshold_usd"],
            "expected_green_hours_per_day": report["validation"][
                "expected_green_hours_per_day"
            ],
            "cost_ratio_projection": report["validation"]["cost_ratio_projection"],
        }

        # Also save simple version for easy consumption
        simple_file = Path(args.out) / "ev_calibration_simple.json"
        with open(simple_file, "w") as f:
            json.dump(simple_result, f, indent=2)

        print(f"💡 Next: Run 'make ev-5m' to build 5-minute microwindow grid")
        return 0

    except Exception as e:
        print(f"❌ EV calibration failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
