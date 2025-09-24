#!/usr/bin/env python3
"""
Sequential Testing with Always-Valid P-Values
Implement mixture-SPRT or alpha-spending sequential test for early stopping.
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
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import math


class SequentialTester:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.exp_config = self.config["experiment"]
        self.artifacts_dir = Path(self.config.get("artifacts_dir", "experiments/m11"))

        # Sequential testing parameters
        self.alpha = self.exp_config.get("alpha", 0.05)
        self.power_target = self.exp_config.get("power_target", 0.80)
        self.met_uplift = (
            self.exp_config.get("met_uplift_usd_per_day", 50) / 24
        )  # Hourly MET

    def load_config(self) -> Dict[str, Any]:
        """Load experiment configuration."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load experiment config: {e}")

    def load_latest_cuped_results(self) -> Optional[Dict[str, Any]]:
        """Load latest CUPED analysis results."""
        cuped_file = self.artifacts_dir / "cuped_analysis_latest.json"

        if not cuped_file.exists():
            # Try to find the most recent CUPED analysis file
            cuped_files = list(self.artifacts_dir.glob("cuped_analysis_*.json"))
            if cuped_files:
                latest_cuped = max(cuped_files, key=lambda f: f.stat().st_mtime)
                cuped_file = latest_cuped
                print(f"📊 Using latest CUPED analysis: {cuped_file.name}")
            else:
                print("⚠️ No CUPED analysis found")
                return None

        try:
            with open(cuped_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading CUPED results: {e}")
            return None

    def compute_sample_size_requirements(
        self, effect_size: float, variance: float
    ) -> Dict[str, int]:
        """Compute required sample sizes for different scenarios."""

        # Standard power calculation for two-sample t-test
        alpha = self.alpha
        beta = 1 - self.power_target

        # Effect size (Cohen's d)
        cohen_d = effect_size / np.sqrt(variance) if variance > 0 else 0

        # Required sample size per group (Welch's t-test approximation)
        if cohen_d > 0:
            # Using scipy's power calculation
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(self.power_target)

            n_per_group = ((z_alpha + z_beta) / cohen_d) ** 2 * 2
            n_total = int(np.ceil(n_per_group * 2))
        else:
            n_total = 1000  # Default large sample

        return {
            "n_per_group": int(np.ceil(n_per_group)) if cohen_d > 0 else 500,
            "n_total": n_total,
            "cohen_d": cohen_d,
            "power_calculation_valid": cohen_d > 0,
        }

    def alpha_spending_function(self, information_fraction: float) -> float:
        """O'Brien-Fleming alpha spending function."""

        if information_fraction <= 0:
            return 0
        elif information_fraction >= 1:
            return self.alpha
        else:
            # O'Brien-Fleming boundary
            return 2 * (
                1
                - stats.norm.cdf(
                    stats.norm.ppf(1 - self.alpha / 2) / np.sqrt(information_fraction)
                )
            )

    def compute_always_valid_p_value(
        self, z_score: float, information_fraction: float
    ) -> float:
        """Compute always-valid p-value using alpha-spending approach."""

        if information_fraction <= 0:
            return 1.0

        # Compute critical value at this information fraction
        alpha_spent = self.alpha_spending_function(information_fraction)
        critical_z = stats.norm.ppf(1 - alpha_spent / 2)

        # Always-valid p-value
        if abs(z_score) >= critical_z:
            p_value = alpha_spent
        else:
            # Interpolate p-value
            p_value = 2 * (
                1 - stats.norm.cdf(abs(z_score) / np.sqrt(information_fraction))
            )

        return min(p_value, 1.0)

    def sequential_test_analysis(self, cuped_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sequential test analysis."""

        print("📊 Running sequential test analysis...")

        # Extract primary results
        primary_results = cuped_results.get("primary_results", {})

        # Use CUPED results if available, otherwise raw
        if "cuped" in primary_results:
            results_to_use = primary_results["cuped"]
            analysis_type = "cuped"
        else:
            results_to_use = primary_results["raw"]
            analysis_type = "raw"

        effect = results_to_use["effect"]
        effect_se = results_to_use["effect_se"]
        n_treatment = results_to_use["n_treatment"]
        n_control = results_to_use["n_control"]
        n_total = n_treatment + n_control

        # Compute z-score
        z_score = effect / effect_se if effect_se > 0 else 0

        # Estimate required sample size
        variance_estimate = (effect_se**2) * n_total / 2  # Approximate pooled variance
        sample_size_req = self.compute_sample_size_requirements(
            self.met_uplift, variance_estimate
        )

        # Information fraction (how much of planned sample we have)
        information_fraction = (
            n_total / sample_size_req["n_total"]
            if sample_size_req["n_total"] > 0
            else 1.0
        )
        information_fraction = min(information_fraction, 1.0)

        # Compute always-valid p-value
        always_valid_p = self.compute_always_valid_p_value(
            z_score, information_fraction
        )

        # Sequential test decision
        min_days = self.exp_config.get("min_days", 7)
        max_days = self.exp_config.get("horizon_days", 14)

        # Estimate current experiment day (rough)
        current_day = max(1, n_total / (24 * len(self.exp_config["assets"])))

        # Decision logic
        if always_valid_p < self.alpha and effect > 0 and effect >= self.met_uplift:
            if current_day >= min_days:
                verdict = "PASS"
                reason = f"Significant positive effect (p={always_valid_p:.4f}, effect={effect:.3f} >= MET={self.met_uplift:.3f})"
            else:
                verdict = "EXTEND"
                reason = f"Effect significant but min days not reached ({current_day:.1f} < {min_days})"

        elif always_valid_p < self.alpha and effect < 0:
            verdict = "FAIL"
            reason = f"Significant negative effect (p={always_valid_p:.4f}, effect={effect:.3f})"

        elif information_fraction >= 1.0 or current_day >= max_days:
            if (
                effect >= self.met_uplift and always_valid_p < 0.1
            ):  # Relaxed threshold at end
                verdict = "PASS"
                reason = f"Max duration reached with promising effect (effect={effect:.3f}, p={always_valid_p:.4f})"
            else:
                verdict = "FAIL"
                reason = f"Max duration reached without sufficient evidence (effect={effect:.3f}, p={always_valid_p:.4f})"

        else:
            verdict = "EXTEND"
            reason = f"Insufficient evidence, continue experiment (p={always_valid_p:.4f}, info_frac={information_fraction:.2f})"

        # Confidence interval at current information fraction
        critical_z = stats.norm.ppf(
            1 - self.alpha_spending_function(information_fraction) / 2
        )
        ci_width = critical_z * effect_se
        ci_lower = effect - ci_width
        ci_upper = effect + ci_width

        return {
            "verdict": verdict,
            "reason": reason,
            "analysis_type": analysis_type,
            "effect": effect,
            "effect_se": effect_se,
            "z_score": z_score,
            "always_valid_p_value": always_valid_p,
            "information_fraction": information_fraction,
            "current_sample_size": n_total,
            "required_sample_size": sample_size_req["n_total"],
            "current_day_estimate": current_day,
            "min_days": min_days,
            "max_days": max_days,
            "confidence_interval": [ci_lower, ci_upper],
            "meets_met": effect >= self.met_uplift,
            "met_threshold": self.met_uplift,
            "alpha_spent": self.alpha_spending_function(information_fraction),
            "power_estimate": self.estimate_current_power(effect, effect_se, n_total),
        }

    def estimate_current_power(
        self, effect: float, effect_se: float, n_total: int
    ) -> float:
        """Estimate current statistical power."""

        if effect_se <= 0:
            return 0.0

        # Power to detect the MET
        cohen_d = self.met_uplift / (effect_se * np.sqrt(n_total / 2))

        # Approximate power calculation
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = cohen_d - z_alpha
        power = stats.norm.cdf(z_beta)

        return max(0.0, min(1.0, power))

    def run_sequential_test(self) -> Dict[str, Any]:
        """Run complete sequential test analysis."""

        print("🎯 Running sequential test...")

        # Load CUPED results
        cuped_results = self.load_latest_cuped_results()

        if not cuped_results:
            # Return default result
            return {
                "verdict": "EXTEND",
                "reason": "No data available for analysis",
                "always_valid_p_value": 1.0,
                "confidence_interval": [0, 0],
                "analysis_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "data_available": False,
            }

        # Perform sequential analysis
        sequential_results = self.sequential_test_analysis(cuped_results)

        # Add metadata
        sequential_results.update(
            {
                "analysis_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "experiment": self.exp_config["name"],
                "data_available": True,
                "alpha": self.alpha,
                "power_target": self.power_target,
            }
        )

        return sequential_results


def main():
    """Main sequential test function."""
    parser = argparse.ArgumentParser(description="Sequential Test Analysis")
    parser.add_argument("-c", "--config", required=True, help="Experiment config file")
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()

    try:
        tester = SequentialTester(args.config)

        # Run sequential test
        results = tester.run_sequential_test()

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
            output_path = tester.artifacts_dir / f"sequential_test_{timestamp}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Create latest symlink
        latest_path = tester.artifacts_dir / "sequential_test_latest.json"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(output_path.name)  # Use relative path

        # Display summary
        print(f"\n🎯 Sequential Test Results:")
        print(f"  Verdict: {results['verdict']}")
        print(f"  Reason: {results['reason']}")

        if results.get("data_available", False):
            print(f"  Effect: {results['effect']:.3f}")
            print(
                f"  95% CI: [{results['confidence_interval'][0]:.3f}, {results['confidence_interval'][1]:.3f}]"
            )
            print(f"  Always-Valid P-Value: {results['always_valid_p_value']:.4f}")
            print(f"  Information Fraction: {results['information_fraction']:.2f}")
            print(f"  Current Power: {results['power_estimate']:.1%}")
            print(f"  Meets MET: {'✅' if results['meets_met'] else '❌'}")

        print(f"\n📄 Results saved: {output_path}")

        # Print verdict for easy parsing
        print(results["verdict"])

        return 0

    except Exception as e:
        print(f"❌ Sequential test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
