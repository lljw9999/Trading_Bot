#!/usr/bin/env python3
"""
EV Microwindow: 5-Minute EV Forecasting Model
Build fine-grained EV predictions with cost-aware thresholds for micro green windows.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class EVMicrowindowForecaster:
    def __init__(self, frequency: str = "5min"):
        self.frequency = frequency
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")
        self.assets = ["SOL-USD", "BTC-USD", "ETH-USD", "NVDA"]
        self.venues = ["coinbase", "binance", "alpaca"]

        # Load calibrated EV threshold
        self.ev_threshold = self.load_calibrated_threshold()

    def load_calibrated_threshold(self) -> float:
        """Load EV threshold from calibration."""
        try:
            calib_file = (
                self.base_dir / "artifacts" / "ev" / "ev_calibration_simple.json"
            )
            if calib_file.exists():
                with open(calib_file, "r") as f:
                    calib_data = json.load(f)
                return calib_data["ev_threshold_usd"]
        except Exception as e:
            print(f"⚠️ Could not load calibrated threshold: {e}")

        # Fallback to default
        return 9.40

    def generate_synthetic_microwindow_data(
        self, window_days: int = 14
    ) -> pd.DataFrame:
        """Generate synthetic 5-minute market microstructure data."""

        print(f"📊 Generating {window_days} days of 5-minute synthetic data...")

        # Create 5-minute timestamp grid
        start_time = datetime.datetime.now() - datetime.timedelta(days=window_days)
        end_time = datetime.datetime.now()

        timestamps = pd.date_range(start_time, end_time, freq="5min")

        data_rows = []

        for ts in timestamps:
            hour_of_day = ts.hour
            day_of_week = ts.weekday()
            minute_of_hour = ts.minute

            for asset in self.assets:
                for venue in self.venues:
                    # Skip invalid combinations
                    if venue == "binance" and asset == "NVDA":
                        continue
                    if venue == "alpaca" and asset != "NVDA":
                        continue

                    # Generate microstructure features

                    # Base volatility with time-of-day patterns
                    base_vol = (
                        0.002 if asset != "NVDA" else 0.0015
                    )  # Crypto vs stock volatility

                    # Market session effects
                    if asset == "NVDA":
                        # US market hours (9:30-16:00 ET, approx 14:30-21:00 UTC)
                        if 14 <= hour_of_day <= 21:
                            session_vol_mult = 1.5  # Higher vol during market hours
                        else:
                            session_vol_mult = 0.3  # Lower vol after hours
                    else:
                        # Crypto 24/7 with some daily patterns
                        session_vol_mult = 0.8 + 0.4 * np.sin(
                            2 * np.pi * hour_of_day / 24
                        )

                    # Microstructure features
                    vol_1m = base_vol * session_vol_mult * np.random.uniform(0.5, 2.0)
                    vol_5m = vol_1m * np.sqrt(5) * np.random.uniform(0.8, 1.2)

                    # Spread dynamics
                    base_spread = 2.0 if asset != "NVDA" else 1.5
                    spread_bps = (
                        base_spread * (1 + vol_5m * 100) * np.random.uniform(0.7, 1.5)
                    )

                    # Market depth (inverse relationship with volatility)
                    depth_factor = 1.0 / (1 + vol_5m * 50)
                    depth_ratio = depth_factor * np.random.uniform(0.8, 1.2)

                    # Slippage (related to depth and spread)
                    slip_p95 = (
                        spread_bps * (2.0 - depth_factor) * np.random.uniform(0.8, 1.3)
                    )

                    # Fee structure by venue
                    if venue == "binance" and asset != "NVDA":
                        maker_fee = -0.5  # Rebate
                        taker_fee = 1.0
                    elif venue == "coinbase" and asset != "NVDA":
                        maker_fee = -0.25  # Small rebate
                        taker_fee = 0.6
                    else:  # Alpaca stocks
                        maker_fee = -0.1
                        taker_fee = 0.35

                    # Cost structure (varies by venue and time)
                    if minute_of_hour == 0:  # Top of hour has higher costs
                        cost_mult = 1.2
                    else:
                        cost_mult = 1.0

                    cost_per_hour = 4.40 * cost_mult  # From calibration
                    cost_per_5min = (
                        cost_per_hour / 12
                    )  # 12 five-minute periods per hour

                    # Generate target EV with realistic patterns

                    # Time-based base EV
                    time_factor = np.sin(2 * np.pi * hour_of_day / 24) + 0.5 * np.sin(
                        2 * np.pi * minute_of_hour / 60
                    )

                    # Market condition factors
                    vol_penalty = -vol_5m * 500  # High vol reduces EV
                    spread_penalty = (
                        -max(0, spread_bps - 3.0) * 0.5
                    )  # Wide spreads reduce EV
                    depth_bonus = depth_ratio * 2.0  # Good depth improves EV
                    fee_impact = (
                        maker_fee - taker_fee
                    ) * 0.5  # Rebate venues have better EV

                    # Weekend effect for crypto
                    if asset != "NVDA" and day_of_week >= 5:  # Weekend
                        weekend_factor = 0.7  # Lower EV on weekends
                    else:
                        weekend_factor = 1.0

                    # Base EV calculation
                    base_ev = (
                        5.0  # Base level
                        + time_factor * 3.0
                        + vol_penalty
                        + spread_penalty
                        + depth_bonus
                        + fee_impact
                    ) * weekend_factor

                    # Add realistic noise
                    noise = np.random.normal(0, 2.0)
                    ev_usd_per_5min = base_ev + noise

                    # Convert to hourly rate
                    ev_usd_per_hour = ev_usd_per_5min * 12

                    # Store row
                    row = {
                        "timestamp": ts,
                        "asset": asset,
                        "venue": venue,
                        "vol_1m": vol_1m,
                        "vol_5m": vol_5m,
                        "spread_bps": spread_bps,
                        "depth_ratio": depth_ratio,
                        "slip_p95": slip_p95,
                        "maker_fee_bps": maker_fee,
                        "taker_fee_bps": taker_fee,
                        "cost_per_5min": cost_per_5min,
                        "hour_of_day": hour_of_day,
                        "minute_of_hour": minute_of_hour,
                        "day_of_week": day_of_week,
                        "ev_usd_per_hour": ev_usd_per_hour,
                    }

                    data_rows.append(row)

        df = pd.DataFrame(data_rows)
        print(
            f"   Generated {len(df)} 5-minute windows across {len(self.assets)} assets"
        )

        return df

    def train_microwindow_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train 5-minute EV forecasting model."""

        print("🧠 Training 5-minute EV forecasting model...")

        # Feature engineering
        feature_cols = [
            "vol_1m",
            "vol_5m",
            "spread_bps",
            "depth_ratio",
            "slip_p95",
            "maker_fee_bps",
            "taker_fee_bps",
            "cost_per_5min",
            "hour_of_day",
            "minute_of_hour",
            "day_of_week",
        ]

        # Add asset/venue encoding
        asset_dummies = pd.get_dummies(df["asset"], prefix="asset")
        venue_dummies = pd.get_dummies(df["venue"], prefix="venue")

        features_df = pd.concat(
            [df[feature_cols], asset_dummies, venue_dummies], axis=1
        )

        X = features_df.values
        y = df["ev_usd_per_hour"].values

        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )

        model.fit(X, y)

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        mae_scores = cross_val_score(
            model, X, y, cv=5, scoring="neg_mean_absolute_error"
        )

        model_info = {
            "model_type": "RandomForestRegressor",
            "n_features": X.shape[1],
            "feature_names": list(features_df.columns),
            "r2_cv_mean": float(np.mean(cv_scores)),
            "r2_cv_std": float(np.std(cv_scores)),
            "mae_cv_mean": float(-np.mean(mae_scores)),
            "mae_cv_std": float(np.std(mae_scores)),
            "ev_threshold_usd": self.ev_threshold,
        }

        print(
            f"   Model R² (CV): {model_info['r2_cv_mean']:.3f} ± {model_info['r2_cv_std']:.3f}"
        )
        print(
            f"   Model MAE (CV): {model_info['mae_cv_mean']:.1f} ± {model_info['mae_cv_std']:.1f} USD/hour"
        )

        return model, model_info, list(features_df.columns)

    def generate_5min_forecast_grid(
        self, model, feature_names: List[str], window_hours: int = 48
    ) -> pd.DataFrame:
        """Generate 5-minute EV forecast grid for next N hours."""

        print(f"📈 Generating 5-minute EV grid for next {window_hours} hours...")

        # Create future 5-minute timestamps
        start_time = datetime.datetime.now().replace(second=0, microsecond=0)
        # Round to next 5-minute boundary
        minutes = start_time.minute
        start_time = start_time.replace(minute=(minutes // 5) * 5)

        end_time = start_time + datetime.timedelta(hours=window_hours)
        timestamps = pd.date_range(start_time, end_time, freq="5min")

        forecast_rows = []

        for ts in timestamps:
            hour_of_day = ts.hour
            day_of_week = ts.weekday()
            minute_of_hour = ts.minute

            for asset in self.assets:
                for venue in self.venues:
                    # Skip invalid combinations
                    if venue == "binance" and asset == "NVDA":
                        continue
                    if venue == "alpaca" and asset != "NVDA":
                        continue

                    # Generate features for prediction (using recent patterns)
                    vol_1m = 0.002 if asset != "NVDA" else 0.0015
                    vol_5m = vol_1m * np.sqrt(5)
                    spread_bps = 2.0 if asset != "NVDA" else 1.5
                    depth_ratio = 0.8
                    slip_p95 = spread_bps * 1.5

                    if venue == "binance" and asset != "NVDA":
                        maker_fee, taker_fee = -0.5, 1.0
                    elif venue == "coinbase" and asset != "NVDA":
                        maker_fee, taker_fee = -0.25, 0.6
                    else:
                        maker_fee, taker_fee = -0.1, 0.35

                    cost_per_5min = 4.40 / 12  # From calibration

                    # Create feature vector
                    features = {
                        "vol_1m": vol_1m,
                        "vol_5m": vol_5m,
                        "spread_bps": spread_bps,
                        "depth_ratio": depth_ratio,
                        "slip_p95": slip_p95,
                        "maker_fee_bps": maker_fee,
                        "taker_fee_bps": taker_fee,
                        "cost_per_5min": cost_per_5min,
                        "hour_of_day": hour_of_day,
                        "minute_of_hour": minute_of_hour,
                        "day_of_week": day_of_week,
                    }

                    # Add asset/venue dummies
                    for feat_name in feature_names:
                        if feat_name.startswith("asset_"):
                            features[feat_name] = (
                                1 if feat_name == f"asset_{asset}" else 0
                            )
                        elif feat_name.startswith("venue_"):
                            features[feat_name] = (
                                1 if feat_name == f"venue_{venue}" else 0
                            )
                        elif feat_name not in features:
                            features[feat_name] = 0

                    # Create feature vector in correct order
                    feature_vector = [features.get(name, 0) for name in feature_names]

                    # Make prediction
                    ev_pred = model.predict([feature_vector])[0]

                    # Classify into bands
                    if ev_pred >= self.ev_threshold:
                        band = "green"
                    elif ev_pred >= 0:
                        band = "amber"
                    else:
                        band = "red"

                    forecast_rows.append(
                        {
                            "timestamp": ts,
                            "asset": asset,
                            "venue": venue,
                            "ev_usd_per_hour": ev_pred,
                            "band": band,
                        }
                    )

        forecast_df = pd.DataFrame(forecast_rows)

        # Calculate band distribution
        band_counts = forecast_df["band"].value_counts()
        total_windows = len(forecast_df)

        print(f"   Generated {total_windows} 5-minute predictions")
        print(
            f"   Band distribution: Green={band_counts.get('green', 0)}, Amber={band_counts.get('amber', 0)}, Red={band_counts.get('red', 0)}"
        )

        return forecast_df

    def generate_5min_calendar_markdown(self, forecast_df: pd.DataFrame) -> str:
        """Generate 5-minute trade calendar in markdown format."""

        band_counts = forecast_df["band"].value_counts()
        total_windows = len(forecast_df)

        green_count = band_counts.get("green", 0)
        amber_count = band_counts.get("amber", 0)
        red_count = band_counts.get("red", 0)

        green_pct = (green_count / total_windows) * 100
        amber_pct = (amber_count / total_windows) * 100
        red_pct = (red_count / total_windows) * 100

        calendar_md = f"""# 5-Minute Trade Calendar - EV Microwindows

**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
**EV Threshold:** ${self.ev_threshold:.1f}/hour (green ≥ threshold)
**Frequency:** 5-minute windows

## Summary

- **Total 5-Min Windows:** {total_windows:,}
- **🟢 Green (EV ≥ ${self.ev_threshold:.1f}):** {green_count} ({green_pct:.1f}%)
- **🟡 Amber (0 ≤ EV < ${self.ev_threshold:.1f}):** {amber_count} ({amber_pct:.1f}%)
- **🔴 Red (EV < 0):** {red_count} ({red_pct:.1f}%)

## Green Windows (Next 48 Hours)

"""

        # Find green windows
        green_windows = forecast_df[forecast_df["band"] == "green"].sort_values(
            "timestamp"
        )

        if len(green_windows) > 0:
            calendar_md += "| Time | Asset | Venue | EV ($/hour) |\n"
            calendar_md += "|------|-------|-------|-------------|\n"

            for _, window in green_windows.head(20).iterrows():  # Show first 20
                time_str = window["timestamp"].strftime("%m/%d %H:%M")
                calendar_md += f"| {time_str} | {window['asset']} | {window['venue']} | ${window['ev_usd_per_hour']:.1f} |\n"

            if len(green_windows) > 20:
                calendar_md += (
                    f"\n*... and {len(green_windows) - 20} more green windows*\n"
                )
        else:
            calendar_md += "*No green windows found in next 48 hours*\n"

        calendar_md += f"""

## Hourly Summary (Next 24 Hours)

"""

        # Hourly green window counts
        green_windows["hour"] = green_windows["timestamp"].dt.floor("h")
        hourly_green = (
            green_windows.groupby("hour").size().reset_index(name="green_count")
        )

        if len(hourly_green) > 0:
            calendar_md += "| Hour | Green Windows | Assets Active |\n"
            calendar_md += "|------|---------------|---------------|\n"

            for _, hour_data in hourly_green.head(24).iterrows():
                hour_str = hour_data["hour"].strftime("%m/%d %H:00")
                hour_windows = green_windows[green_windows["hour"] == hour_data["hour"]]
                active_assets = hour_windows["asset"].nunique()
                calendar_md += (
                    f"| {hour_str} | {hour_data['green_count']} | {active_assets} |\n"
                )
        else:
            calendar_md += "*No green hours in next 24 hours*\n"

        calendar_md += f"""

---
*Generated by EV Microwindow Forecaster - M14 Green Window Hunter*
"""

        return calendar_md

    def run_microwindow_analysis(
        self, window_days: int = 14, output_dir: str = "artifacts/ev"
    ) -> Dict[str, Any]:
        """Run complete 5-minute microwindow analysis."""

        print("📊 EV Microwindow: 5-Minute Forecasting")
        print("=" * 45)
        print(f"Training Window: {window_days} days")
        print(f"Frequency: {self.frequency}")
        print(f"EV Threshold: ${self.ev_threshold:.1f}/hour")
        print("=" * 45)

        # Generate training data
        df = self.generate_synthetic_microwindow_data(window_days)

        # Train model
        model, model_info, feature_names = self.train_microwindow_model(df)

        # Generate forecast grid
        forecast_df = self.generate_5min_forecast_grid(
            model, feature_names, window_hours=48
        )

        # Generate calendar
        calendar_md = self.generate_5min_calendar_markdown(forecast_df)

        # Save outputs
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%SZ")
        output_subdir = output_path / timestamp
        output_subdir.mkdir(exist_ok=True)

        # Save forecast grid
        forecast_file = output_subdir / "calendar_5m.parquet"
        forecast_df.to_parquet(forecast_file, index=False)

        # Save calendar markdown
        calendar_file = output_subdir / "calendar_5m.md"
        with open(calendar_file, "w") as f:
            f.write(calendar_md)

        # Save model metadata
        metadata_file = output_subdir / "microwindow_model_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(model_info, f, indent=2)

        # Create latest symlinks
        latest_forecast = output_path / "calendar_5m.parquet"
        latest_calendar = output_path / "calendar_5m.md"
        latest_metadata = output_path / "microwindow_model_metadata.json"

        for latest_file, target_file in [
            (latest_forecast, forecast_file),
            (latest_calendar, calendar_file),
            (latest_metadata, metadata_file),
        ]:
            if latest_file.exists() or latest_file.is_symlink():
                latest_file.unlink()
            latest_file.symlink_to(target_file.relative_to(output_path))

        # Summary stats
        band_counts = forecast_df["band"].value_counts()
        green_hours = (
            len(forecast_df[forecast_df["band"] == "green"]) / 12
        )  # Convert 5-min windows to hours

        summary = {
            "timestamp": datetime.datetime.now().isoformat() + "Z",
            "model_info": model_info,
            "forecast_summary": {
                "total_5min_windows": len(forecast_df),
                "green_windows": int(band_counts.get("green", 0)),
                "amber_windows": int(band_counts.get("amber", 0)),
                "red_windows": int(band_counts.get("red", 0)),
                "green_hours_per_day": green_hours / 2,  # 48 hours -> per day
                "green_percentage": (band_counts.get("green", 0) / len(forecast_df))
                * 100,
            },
            "output_files": {
                "forecast_grid": str(forecast_file),
                "calendar_markdown": str(calendar_file),
                "model_metadata": str(metadata_file),
            },
        }

        print(f"\n📊 5-Minute Forecast Results:")
        print(
            f"  Green Windows: {band_counts.get('green', 0)} ({summary['forecast_summary']['green_percentage']:.1f}%)"
        )
        print(
            f"  Green Hours/Day: {summary['forecast_summary']['green_hours_per_day']:.1f}"
        )
        print(f"  Forecast Grid: {forecast_file}")
        print(f"  Calendar: {calendar_file}")

        if summary["forecast_summary"]["green_hours_per_day"] >= 2:
            print("✅ Found ≥2 green hours/day target!")
        else:
            print("⚠️ Less than 2 green hours/day - consider lowering threshold")

        return summary


def main():
    """Main microwindow analysis function."""
    parser = argparse.ArgumentParser(description="EV Microwindow: 5-Minute Forecasting")
    parser.add_argument("--window", default="14d", help="Training window (e.g., 14d)")
    parser.add_argument("--freq", default="5min", help="Forecast frequency")
    parser.add_argument("--out", default="artifacts/ev", help="Output directory")
    args = parser.parse_args()

    # Parse window
    if args.window.endswith("d"):
        window_days = int(args.window[:-1])
    else:
        window_days = int(args.window)

    try:
        forecaster = EVMicrowindowForecaster(frequency=args.freq)
        summary = forecaster.run_microwindow_analysis(window_days, args.out)

        print(f"💡 Next: Run 'make event-gate' to enable signal spike overrides")
        return 0

    except Exception as e:
        print(f"❌ Microwindow analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
