#!/usr/bin/env python3
"""
M18: 48h Soak Gate for 15% Stability Verification
Verifies 15% ramp stability over 48h before 20% advancement.
"""
import os
import sys
import json
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional


class Soak15Gate:
    """48h soak gate for 15% ramp stability verification."""

    def __init__(self, window_hours: int = 48):
        self.window_hours = window_hours
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")
        self.soak_dir = self.base_dir / "artifacts" / "soak15"

        # Gate criteria (48h green-only)
        self.criteria = {
            "slip_p95": {"threshold": 12.0, "unit": "bps", "direction": "<="},
            "maker_ratio": {"threshold": 0.75, "unit": "%", "direction": ">="},
            "cancel_ratio": {"threshold": 0.40, "unit": "%", "direction": "<="},
            "decision_to_ack_p95_ms": {
                "threshold": 120,
                "unit": "ms",
                "direction": "<=",
            },
            "impact_bp_per_1k_p95": {
                "threshold": 8.0,
                "unit": "bp/$1k",
                "direction": "<=",
            },
            "net_pnl_green": {"threshold": 600, "unit": "USD", "direction": ">="},
            "cost_ratio_green": {"threshold": 0.30, "unit": "%", "direction": "<="},
            "drawdown_24h": {"threshold": 1.0, "unit": "%", "direction": "<="},
            "page_alerts": {"threshold": 0, "unit": "count", "direction": "=="},
        }

        # Gates directory
        self.gates_dir = self.base_dir / "artifacts" / "gates"
        self.gates_dir.mkdir(parents=True, exist_ok=True)

    def load_soak_history(self) -> List[Dict[str, Any]]:
        """Load 48h soak monitoring history."""

        snapshots = []

        try:
            # Load all soak snapshots from the window
            cutoff_time = datetime.datetime.now(
                datetime.timezone.utc
            ) - datetime.timedelta(hours=self.window_hours)

            for snapshot_file in self.soak_dir.glob("soak_snapshot_*.json"):
                try:
                    with open(snapshot_file, "r") as f:
                        snapshot = json.load(f)

                    # Check if within window
                    timestamp = datetime.datetime.fromisoformat(
                        snapshot["timestamp"].replace("Z", "+00:00")
                    )
                    if timestamp >= cutoff_time:
                        snapshots.append(snapshot)

                except Exception as e:
                    print(f"⚠️ Error loading {snapshot_file}: {e}")
                    continue

            # Sort by timestamp
            snapshots.sort(key=lambda x: x["timestamp"])

            print(
                f"📊 Loaded {len(snapshots)} soak snapshots from {self.window_hours}h window"
            )
            return snapshots

        except Exception as e:
            print(f"⚠️ Error loading soak history: {e}")
            return []

    def compute_window_metrics(self, snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregated metrics over the soak window."""

        if not snapshots:
            return {"error": "No snapshots available"}

        # Extract current metrics from each snapshot
        current_metrics = []
        for snapshot in snapshots:
            if snapshot.get("current_metrics", {}).get("collection_success", False):
                current_metrics.append(snapshot["current_metrics"])

        if not current_metrics:
            return {"error": "No valid metrics in snapshots"}

        # Compute window statistics
        window_stats = {}

        # Slippage metrics
        slip_values = [m.get("slip_p95_bps", 999) for m in current_metrics]
        window_stats["slip_p95"] = {
            "mean": np.mean(slip_values),
            "max": np.max(slip_values),
            "p95": np.percentile(slip_values, 95),
            "samples": len(slip_values),
        }

        # Maker ratio metrics
        maker_values = [m.get("maker_ratio", 0) for m in current_metrics]
        window_stats["maker_ratio"] = {
            "mean": np.mean(maker_values),
            "min": np.min(maker_values),
            "p05": np.percentile(maker_values, 5),
            "samples": len(maker_values),
        }

        # Cancel ratio metrics
        cancel_values = [m.get("cancel_ratio", 1) for m in current_metrics]
        window_stats["cancel_ratio"] = {
            "mean": np.mean(cancel_values),
            "max": np.max(cancel_values),
            "p95": np.percentile(cancel_values, 95),
            "samples": len(cancel_values),
        }

        # Latency metrics
        latency_values = [m.get("decision_to_ack_p95_ms", 999) for m in current_metrics]
        window_stats["decision_to_ack_p95_ms"] = {
            "mean": np.mean(latency_values),
            "max": np.max(latency_values),
            "p95": np.percentile(latency_values, 95),
            "samples": len(latency_values),
        }

        # Impact metrics
        impact_values = [m.get("impact_bp_per_1k", 999) for m in current_metrics]
        window_stats["impact_bp_per_1k_p95"] = {
            "mean": np.mean(impact_values),
            "max": np.max(impact_values),
            "p95": np.percentile(impact_values, 95),
            "samples": len(impact_values),
        }

        # Economic metrics (cumulative for window)
        pnl_values = [m.get("net_pnl_green_hourly", 0) for m in current_metrics]
        cost_values = [m.get("cost_ratio_green", 1) for m in current_metrics]

        window_stats["net_pnl_green"] = {
            "total": np.sum(pnl_values),  # Total P&L over window
            "hourly_mean": np.mean(pnl_values),
            "samples": len(pnl_values),
        }

        window_stats["cost_ratio_green"] = {
            "mean": np.mean(cost_values),
            "max": np.max(cost_values),
            "samples": len(cost_values),
        }

        # Risk metrics
        dd_values = [m.get("drawdown_24h_pct", 0) for m in current_metrics]
        window_stats["drawdown_24h"] = {
            "max": np.max(dd_values) if dd_values else 0,
            "mean": np.mean(dd_values) if dd_values else 0,
            "samples": len(dd_values),
        }

        # Alert metrics (cumulative)
        alert_values = [m.get("alert_count_24h", 0) for m in current_metrics]
        page_values = [m.get("page_count_24h", 0) for m in current_metrics]

        window_stats["page_alerts"] = {
            "total": sum(page_values),
            "max_24h": max(page_values) if page_values else 0,
            "samples": len(page_values),
        }

        window_stats["window_info"] = {
            "duration_hours": self.window_hours,
            "snapshots_count": len(snapshots),
            "valid_metrics_count": len(current_metrics),
            "start_time": snapshots[0]["timestamp"] if snapshots else None,
            "end_time": snapshots[-1]["timestamp"] if snapshots else None,
        }

        return window_stats

    def evaluate_gate_criteria(self, window_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all gate criteria against window statistics."""

        gate_results = {
            "passed": True,
            "failures": [],
            "warnings": [],
            "criteria_results": {},
            "overall_score": 0.0,
        }

        if "error" in window_stats:
            gate_results["passed"] = False
            gate_results["failures"].append(window_stats["error"])
            return gate_results

        passing_criteria = 0
        total_criteria = len(self.criteria)

        # Check each criterion
        for criterion_name, criterion_config in self.criteria.items():
            threshold = criterion_config["threshold"]
            direction = criterion_config["direction"]
            unit = criterion_config["unit"]

            result = {
                "criterion": criterion_name,
                "threshold": threshold,
                "direction": direction,
                "unit": unit,
                "status": "UNKNOWN",
            }

            # Get the appropriate metric value
            if criterion_name == "slip_p95":
                value = window_stats["slip_p95"]["p95"]
            elif criterion_name == "maker_ratio":
                value = window_stats["maker_ratio"][
                    "p05"
                ]  # Use 5th percentile (worst case)
            elif criterion_name == "cancel_ratio":
                value = window_stats["cancel_ratio"][
                    "p95"
                ]  # Use 95th percentile (worst case)
            elif criterion_name == "decision_to_ack_p95_ms":
                value = window_stats["decision_to_ack_p95_ms"]["p95"]
            elif criterion_name == "impact_bp_per_1k_p95":
                value = window_stats["impact_bp_per_1k_p95"]["p95"]
            elif criterion_name == "net_pnl_green":
                value = window_stats["net_pnl_green"]["total"]
            elif criterion_name == "cost_ratio_green":
                value = window_stats["cost_ratio_green"]["max"]  # Use worst case
            elif criterion_name == "drawdown_24h":
                value = window_stats["drawdown_24h"]["max"]
            elif criterion_name == "page_alerts":
                value = window_stats["page_alerts"]["total"]
            else:
                value = 0
                result["status"] = "ERROR"
                result["error"] = f"Unknown criterion: {criterion_name}"
                gate_results["failures"].append(f"Unknown criterion: {criterion_name}")
                continue

            result["value"] = value

            # Evaluate criterion
            if direction == "<=":
                passed = value <= threshold
            elif direction == ">=":
                passed = value >= threshold
            elif direction == "==":
                passed = value == threshold
            else:
                passed = False
                result["error"] = f"Unknown direction: {direction}"

            if passed:
                result["status"] = "PASS"
                passing_criteria += 1
            else:
                result["status"] = "FAIL"
                gate_results["failures"].append(
                    f"{criterion_name}: {value:.2f} {unit} {direction} {threshold} {unit}"
                )

            # Add warnings for close calls
            if not passed and direction == "<=" and value <= threshold * 1.1:
                gate_results["warnings"].append(
                    f"{criterion_name} close to threshold: {value:.2f} vs {threshold}"
                )
            elif not passed and direction == ">=" and value >= threshold * 0.9:
                gate_results["warnings"].append(
                    f"{criterion_name} close to threshold: {value:.2f} vs {threshold}"
                )

            gate_results["criteria_results"][criterion_name] = result

        # Overall assessment
        gate_results["overall_score"] = passing_criteria / total_criteria
        gate_results["passed"] = len(gate_results["failures"]) == 0

        return gate_results

    def create_gate_token(
        self, gate_results: Dict[str, Any], window_stats: Dict[str, Any]
    ) -> str:
        """Create gate token if criteria passed."""

        if not gate_results["passed"]:
            return ""

        token_data = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "gate": "soak15_ok",
            "window_hours": self.window_hours,
            "status": "PASS",
            "overall_score": gate_results["overall_score"],
            "criteria_results": gate_results["criteria_results"],
            "window_stats": window_stats,
            "valid_until": (
                datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(hours=72)
            ).isoformat(),  # 72h validity
        }

        try:
            token_file = self.gates_dir / "soak15_ok"
            with open(token_file, "w") as f:
                json.dump(token_data, f, indent=2)

            print(f"✅ Gate token created: {token_file}")
            return str(token_file)

        except Exception as e:
            print(f"⚠️ Failed to create gate token: {e}")
            return ""

    def run_gate_evaluation(self) -> Dict[str, Any]:
        """Run complete 48h soak gate evaluation."""

        print(f"🔒 M18: 48h Soak Gate Evaluation")
        print(f"   Window: {self.window_hours} hours")
        print(f"   Criteria: {len(self.criteria)} checks")
        print("=" * 40)

        # Load soak history
        print("📊 Loading soak monitoring history...")
        snapshots = self.load_soak_history()

        if not snapshots:
            return {
                "success": False,
                "reason": "no_soak_data",
                "message": "No soak monitoring data available",
            }

        # Compute window metrics
        print("📈 Computing window metrics...")
        window_stats = self.compute_window_metrics(snapshots)

        if "error" in window_stats:
            return {
                "success": False,
                "reason": "metrics_error",
                "message": window_stats["error"],
            }

        # Evaluate gate criteria
        print("🔍 Evaluating gate criteria...")
        gate_results = self.evaluate_gate_criteria(window_stats)

        # Display results
        print(f"\n🎯 Soak Gate Results:")
        print(f"  Window: {window_stats['window_info']['duration_hours']}h")
        print(f"  Snapshots: {window_stats['window_info']['snapshots_count']}")
        print(f"  Score: {gate_results['overall_score']:.1%}")
        print(f"  Status: {'✅ PASS' if gate_results['passed'] else '❌ FAIL'}")

        # Show criteria details
        for criterion_name, result in gate_results["criteria_results"].items():
            status_icon = "✅" if result["status"] == "PASS" else "❌"
            print(
                f"    {status_icon} {criterion_name}: {result['value']:.2f} {result['unit']} "
                f"({result['direction']} {result['threshold']} {result['unit']})"
            )

        # Show failures
        if gate_results["failures"]:
            print(f"\n❌ Gate Failures:")
            for failure in gate_results["failures"]:
                print(f"   • {failure}")

        # Show warnings
        if gate_results["warnings"]:
            print(f"\n⚠️ Warnings:")
            for warning in gate_results["warnings"]:
                print(f"   • {warning}")

        # Create token if passed
        token_file = ""
        if gate_results["passed"]:
            print(f"\n🎉 Creating gate token...")
            token_file = self.create_gate_token(gate_results, window_stats)

        result = {
            "success": gate_results["passed"],
            "gate_results": gate_results,
            "window_stats": window_stats,
            "token_file": token_file,
        }

        if gate_results["passed"]:
            print(f"\n✅ 48h SOAK GATE PASSED - Ready for 20% advancement!")
        else:
            print(f"\n❌ 48h soak gate failed - continue monitoring")

        return result


def main():
    """Main soak gate CLI."""
    parser = argparse.ArgumentParser(description="M18: 48h Soak Gate for 15% Stability")
    parser.add_argument(
        "--window", default="48h", help="Evaluation window (e.g., 48h, 72h)"
    )
    args = parser.parse_args()

    # Parse window
    try:
        if args.window.endswith("h"):
            window_hours = int(args.window[:-1])
        else:
            window_hours = int(args.window)
    except ValueError:
        print(f"❌ Invalid window format: {args.window}")
        return 1

    try:
        gate = Soak15Gate(window_hours)
        result = gate.run_gate_evaluation()

        if result["success"]:
            print(f"\n💡 Next: Run 'make ramp-20-green' to execute 20% advancement")
            return 0
        else:
            print(f"\n💡 Next: Continue 'make soak-15' monitoring")
            return 1

    except Exception as e:
        print(f"❌ Soak gate error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
