#!/usr/bin/env python3
"""
CUPED (Controlled-experiment Using Pre-Experiment Data) Analysis
Reduce variance using pre-period covariates for more powerful experiment analysis.
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
from sklearn.linear_model import LinearRegression


class CUPEDAnalyzer:
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

    def load_experiment_data(self, num_days: int = None) -> pd.DataFrame:
        """Load experiment data from collected metrics."""

        # Find all daily directories
        daily_dirs = sorted(
            [
                d
                for d in self.artifacts_dir.iterdir()
                if d.is_dir() and d.name.count("-") == 2
            ]
        )

        if num_days:
            daily_dirs = daily_dirs[-num_days:]  # Take last N days

        all_data = []

        for daily_dir in daily_dirs:
            # Load all hourly metrics for this day
            metrics_files = sorted(daily_dir.glob("metrics_*.json"))

            for metrics_file in metrics_files:
                try:
                    with open(metrics_file, "r") as f:
                        metrics_data = json.load(f)

                    hour_start = pd.to_datetime(metrics_data["hour_start"])

                    # Extract data for each asset
                    for asset, asset_metrics in metrics_data["metrics"].items():
                        asset_covariates = metrics_data["covariates"].get(asset, {})

                        row = {
                            "timestamp": hour_start,
                            "date": hour_start.date(),
                            "hour": hour_start.hour,
                            "asset": asset,
                            "assignment": asset_metrics["assignment"],
                            # Primary and secondary metrics
                            "net_pnl_usd": asset_metrics["net_pnl_usd"],
                            "gross_pnl_usd": asset_metrics["gross_pnl_usd"],
                            "slip_bps_p95": asset_metrics["slip_bps_p95"],
                            "is_bps": asset_metrics["is_bps"],
                            "fill_ratio": asset_metrics["fill_ratio"],
                            "cost_ratio": asset_metrics["cost_ratio"],
                            # Covariates
                            "pre_pnl": asset_covariates.get("pre_pnl", 0),
                            "vol_5m": asset_covariates.get("vol_5m", 0),
                            "spread_bps": asset_covariates.get("spread_bps", 0),
                            "volume": asset_covariates.get("volume", 0),
                        }

                        all_data.append(row)

                except Exception as e:
                    print(f"⚠️ Error loading {metrics_file}: {e}")
                    continue

        if not all_data:
            raise ValueError("No experiment data found")

        df = pd.DataFrame(all_data)
        print(f"📊 Loaded {len(df)} observations from {len(daily_dirs)} days")
        print(f"   Assets: {df['asset'].nunique()}")
        print(f"   Treatment: {(df['assignment'] == 'treatment').sum()}")
        print(f"   Control: {(df['assignment'] == 'control').sum()}")

        return df

    def apply_cuped_adjustment(
        self, df: pd.DataFrame, outcome_col: str = "net_pnl_usd"
    ) -> Dict[str, Any]:
        """Apply CUPED adjustment to reduce variance."""

        print(f"🔧 Applying CUPED adjustment to {outcome_col}...")

        # Get covariates specified in config
        covariate_cols = self.exp_config.get("cuped_covariates", ["pre_pnl"])

        # Filter to available covariates
        available_covariates = [col for col in covariate_cols if col in df.columns]

        if not available_covariates:
            print("⚠️ No CUPED covariates available, using raw outcome")
            df[f"{outcome_col}_cuped"] = df[outcome_col]
            return {
                "variance_reduction": 0.0,
                "covariates_used": [],
                "adjustment_successful": False,
            }

        print(f"   Using covariates: {available_covariates}")

        # Prepare data
        X = df[available_covariates].fillna(0)
        y = df[outcome_col].fillna(0)

        # Fit linear regression to predict outcome from covariates
        model = LinearRegression()
        model.fit(X, y)

        # Generate predictions
        y_pred = model.predict(X)

        # CUPED adjustment: Y_cuped = Y - θ(X - E[X])
        # Where θ is the regression coefficient

        # Calculate covariate means
        X_means = X.mean()

        # Apply adjustment
        theta_weighted_covariates = (X - X_means).dot(model.coef_)
        y_cuped = y - theta_weighted_covariates

        # Add adjusted outcome to dataframe
        df[f"{outcome_col}_cuped"] = y_cuped

        # Calculate variance reduction
        var_original = y.var()
        var_cuped = y_cuped.var()
        variance_reduction = 1 - (var_cuped / var_original) if var_original > 0 else 0

        print(f"   Original variance: {var_original:.4f}")
        print(f"   CUPED variance: {var_cuped:.4f}")
        print(f"   Variance reduction: {variance_reduction:.1%}")

        return {
            "variance_reduction": variance_reduction,
            "covariates_used": available_covariates,
            "covariate_coefficients": dict(zip(available_covariates, model.coef_)),
            "model_r2": model.score(X, y),
            "adjustment_successful": True,
            "original_variance": var_original,
            "cuped_variance": var_cuped,
        }

    def compute_treatment_effects(
        self, df: pd.DataFrame, outcome_col: str = "net_pnl_usd"
    ) -> Dict[str, Any]:
        """Compute treatment effects with and without CUPED."""

        print(f"📈 Computing treatment effects for {outcome_col}...")

        results = {}

        # Raw analysis (without CUPED)
        treatment_data = df[df["assignment"] == "treatment"][outcome_col]
        control_data = df[df["assignment"] == "control"][outcome_col]

        if len(treatment_data) == 0 or len(control_data) == 0:
            raise ValueError("Insufficient data for treatment effect calculation")

        raw_effect = treatment_data.mean() - control_data.mean()
        raw_se = np.sqrt(
            treatment_data.var() / len(treatment_data)
            + control_data.var() / len(control_data)
        )
        raw_t_stat = raw_effect / raw_se if raw_se > 0 else 0
        raw_p_value = 2 * (
            1
            - stats.t.cdf(abs(raw_t_stat), len(treatment_data) + len(control_data) - 2)
        )

        results["raw"] = {
            "treatment_mean": treatment_data.mean(),
            "control_mean": control_data.mean(),
            "effect": raw_effect,
            "effect_se": raw_se,
            "t_statistic": raw_t_stat,
            "p_value": raw_p_value,
            "ci_95": [raw_effect - 1.96 * raw_se, raw_effect + 1.96 * raw_se],
            "n_treatment": len(treatment_data),
            "n_control": len(control_data),
        }

        # CUPED analysis (if available)
        cuped_col = f"{outcome_col}_cuped"
        if cuped_col in df.columns:
            treatment_cuped = df[df["assignment"] == "treatment"][cuped_col]
            control_cuped = df[df["assignment"] == "control"][cuped_col]

            cuped_effect = treatment_cuped.mean() - control_cuped.mean()
            cuped_se = np.sqrt(
                treatment_cuped.var() / len(treatment_cuped)
                + control_cuped.var() / len(control_cuped)
            )
            cuped_t_stat = cuped_effect / cuped_se if cuped_se > 0 else 0
            cuped_p_value = 2 * (
                1
                - stats.t.cdf(
                    abs(cuped_t_stat), len(treatment_cuped) + len(control_cuped) - 2
                )
            )

            results["cuped"] = {
                "treatment_mean": treatment_cuped.mean(),
                "control_mean": control_cuped.mean(),
                "effect": cuped_effect,
                "effect_se": cuped_se,
                "t_statistic": cuped_t_stat,
                "p_value": cuped_p_value,
                "ci_95": [
                    cuped_effect - 1.96 * cuped_se,
                    cuped_effect + 1.96 * cuped_se,
                ],
                "n_treatment": len(treatment_cuped),
                "n_control": len(control_cuped),
            }

            # Calculate power improvement
            power_improvement = (raw_se / cuped_se) ** 2 if cuped_se > 0 else 1
            results["power_improvement"] = power_improvement

        return results

    def analyze_secondary_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze secondary metrics."""

        secondary_metrics = self.exp_config["metrics"]["secondaries"]
        secondary_results = {}

        for metric in secondary_metrics:
            if metric in df.columns:
                print(f"   Analyzing {metric}...")

                # Apply CUPED if beneficial
                cuped_info = self.apply_cuped_adjustment(df, metric)
                treatment_effects = self.compute_treatment_effects(df, metric)

                secondary_results[metric] = {
                    "cuped_info": cuped_info,
                    "treatment_effects": treatment_effects,
                }

        return secondary_results

    def generate_cuped_summary(
        self,
        primary_results: Dict[str, Any],
        secondary_results: Dict[str, Any],
        cuped_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate summary of CUPED analysis."""

        primary_metric = self.exp_config["metrics"]["primary"]

        summary = {
            "analysis_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "experiment": self.exp_config["name"],
            "primary_metric": primary_metric,
            "cuped_adjustment": cuped_info,
            "primary_results": primary_results,
            "secondary_results": secondary_results,
        }

        # Key findings
        if "cuped" in primary_results:
            cuped_effect = primary_results["cuped"]["effect"]
            cuped_ci = primary_results["cuped"]["ci_95"]
            cuped_p = primary_results["cuped"]["p_value"]

            summary["key_findings"] = {
                "primary_effect_cuped": float(cuped_effect),
                "primary_ci_95": [float(cuped_ci[0]), float(cuped_ci[1])],
                "primary_p_value": float(cuped_p),
                "effect_significant": bool(cuped_p < 0.05),
                "ci_excludes_zero": bool(cuped_ci[0] > 0 or cuped_ci[1] < 0),
                "meets_met": bool(
                    cuped_effect
                    >= self.exp_config.get("met_uplift_usd_per_day", 50) / 24
                ),  # Hourly MET
                "variance_reduction": float(cuped_info.get("variance_reduction", 0)),
                "power_improvement": float(primary_results.get("power_improvement", 1)),
            }
        else:
            raw_effect = primary_results["raw"]["effect"]
            raw_ci = primary_results["raw"]["ci_95"]
            raw_p = primary_results["raw"]["p_value"]

            summary["key_findings"] = {
                "primary_effect_raw": float(raw_effect),
                "primary_ci_95": [float(raw_ci[0]), float(raw_ci[1])],
                "primary_p_value": float(raw_p),
                "effect_significant": bool(raw_p < 0.05),
                "ci_excludes_zero": bool(raw_ci[0] > 0 or raw_ci[1] < 0),
                "meets_met": bool(
                    raw_effect >= self.exp_config.get("met_uplift_usd_per_day", 50) / 24
                ),
                "variance_reduction": 0.0,
                "power_improvement": 1.0,
            }

        return summary

    def run_cuped_analysis(self, num_days: int = None) -> Dict[str, Any]:
        """Run complete CUPED analysis."""

        print("🧮 Running CUPED analysis...")

        # Load experiment data
        df = self.load_experiment_data(num_days)

        # Primary metric analysis
        primary_metric = self.exp_config["metrics"]["primary"]

        # Map primary metric name to actual column name in data
        if primary_metric == "net_pnl_usd_per_hour" and "net_pnl_usd" in df.columns:
            primary_metric_col = "net_pnl_usd"
        else:
            primary_metric_col = primary_metric

        # Apply CUPED to primary metric
        cuped_info = self.apply_cuped_adjustment(df, primary_metric_col)

        # Compute treatment effects
        primary_results = self.compute_treatment_effects(df, primary_metric_col)

        # Analyze secondary metrics
        print("🔍 Analyzing secondary metrics...")
        secondary_results = self.analyze_secondary_metrics(df)

        # Generate summary
        summary = self.generate_cuped_summary(
            primary_results, secondary_results, cuped_info
        )

        return summary


def main():
    """Main CUPED analysis function."""
    parser = argparse.ArgumentParser(description="CUPED Analysis")
    parser.add_argument("-c", "--config", required=True, help="Experiment config file")
    parser.add_argument("--days", type=int, help="Number of days to analyze")
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()

    try:
        analyzer = CUPEDAnalyzer(args.config)

        # Run analysis
        results = analyzer.run_cuped_analysis(args.days)

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
            output_path = analyzer.artifacts_dir / f"cuped_analysis_{timestamp}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Create latest symlink
        latest_path = analyzer.artifacts_dir / "cuped_analysis_latest.json"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(output_path)

        # Display summary
        findings = results["key_findings"]
        primary_metric = results["primary_metric"]

        print(f"\n🧮 CUPED Analysis Results:")
        print(f"  Primary Metric: {primary_metric}")

        if "primary_effect_cuped" in findings:
            effect = findings["primary_effect_cuped"]
            ci = findings["primary_ci_95"]
            print(f"  CUPED Effect: {effect:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")
            print(f"  Variance Reduction: {findings['variance_reduction']:.1%}")
            print(f"  Power Improvement: {findings['power_improvement']:.2f}x")
        else:
            effect = findings["primary_effect_raw"]
            ci = findings["primary_ci_95"]
            print(f"  Raw Effect: {effect:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")

        print(f"  P-value: {findings.get('primary_p_value', 0):.4f}")
        print(f"  Significant: {'✅' if findings['effect_significant'] else '❌'}")
        print(f"  Meets MET: {'✅' if findings['meets_met'] else '❌'}")

        print(f"\n📄 Results saved: {output_path}")

        return 0

    except Exception as e:
        print(f"❌ CUPED analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
