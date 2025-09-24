#!/usr/bin/env python3
"""
Expected Value Forecaster & Trade Calendar
Generate hourly EV forecasts per asset/venue and emit green/amber/red trade calendar.
"""
import os
import sys
import json
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")


class EVForecaster:
    def __init__(self, window_days: int = 14, ev_threshold: float = 10.0):
        self.window_days = window_days
        self.ev_threshold = ev_threshold  # USD per hour threshold for green
        self.assets = ["SOL-USD", "BTC-USD", "ETH-USD", "NVDA"]
        self.venues = ["coinbase", "binance", "alpaca"]

    def generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic historical trading data for EV modeling."""

        print(f"📊 Generating {self.window_days} days of synthetic market data...")

        # Create hourly time series
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(days=self.window_days)
        hours = pd.date_range(start_time, end_time, freq="H")

        data = []

        for hour in hours:
            # Market regime features
            is_weekend = hour.weekday() >= 5
            is_night = hour.hour < 6 or hour.hour > 22
            is_lunch = 12 <= hour.hour <= 14

            for asset in self.assets:
                for venue in self.venues:
                    # Base market features
                    vol_base = np.random.gamma(2, 0.01)  # 5min volatility
                    spread_base = np.random.gamma(3, 2)  # spread in bps
                    depth_base = np.random.lognormal(10, 0.5)  # order book depth

                    # Regime adjustments
                    regime_mult = 1.0
                    if is_weekend:
                        regime_mult *= 0.7  # Lower volume/activity
                        spread_base *= 1.3  # Wider spreads
                    if is_night:
                        regime_mult *= 0.8
                        spread_base *= 1.2
                    if is_lunch:
                        regime_mult *= 0.9

                    vol_5m = vol_base * regime_mult
                    spread_bps = spread_base
                    depth = depth_base * regime_mult

                    # Trading costs (venue-specific)
                    if venue == "coinbase":
                        fees_bps = 6.0 if asset == "NVDA" else 3.5
                        rebate_bps = -1.0 if asset != "NVDA" else 0
                    elif venue == "binance":
                        fees_bps = 1.0 if asset != "NVDA" else 999  # No stocks
                        rebate_bps = -0.5 if asset != "NVDA" else 0
                    else:  # alpaca
                        fees_bps = 5.0 if asset == "NVDA" else 999  # Stocks only
                        rebate_bps = -0.2 if asset == "NVDA" else 0

                    # Skip invalid venue/asset combinations
                    if fees_bps > 500:
                        continue

                    # Slippage and impact costs
                    slip_p95 = spread_bps * 0.6 + np.random.gamma(2, 3)
                    is_bps = spread_bps * 0.4 + np.random.gamma(1.5, 2)

                    # Infrastructure costs (allocated)
                    cost_per_hour = 95.5 / 24 / len(self.assets)  # From CFO report

                    # Calculate expected net P&L
                    # Positive alpha in good regimes, negative in bad regimes
                    alpha_base = np.random.normal(12, 8)  # Base alpha

                    # Reduce alpha in bad regimes
                    if is_weekend or is_night:
                        alpha_base *= 0.6
                    if spread_bps > 8:  # Wide spread = poor conditions
                        alpha_base *= 0.7
                    if vol_5m < 0.005:  # Low vol = poor conditions
                        alpha_base *= 0.8

                    # Net P&L = Alpha - Trading Costs - Infrastructure
                    gross_pnl = alpha_base
                    trading_costs = (
                        fees_bps + max(0, slip_p95) + is_bps + abs(rebate_bps)
                    )
                    net_pnl = gross_pnl - trading_costs - cost_per_hour

                    # Add noise
                    net_pnl += np.random.normal(0, 5)

                    data.append(
                        {
                            "timestamp": hour,
                            "asset": asset,
                            "venue": venue,
                            "vol_5m": vol_5m,
                            "spread_bps": spread_bps,
                            "depth": depth,
                            "slip_p95": slip_p95,
                            "is_bps": is_bps,
                            "fees_bps": fees_bps,
                            "rebate_bps": rebate_bps,
                            "cost_per_hour": cost_per_hour,
                            "ev_usd_per_hour": net_pnl,
                            # Additional features for modeling
                            "hour": hour.hour,
                            "dow": hour.weekday(),
                            "is_weekend": is_weekend,
                            "is_night": is_night,
                            "is_lunch": is_lunch,
                        }
                    )

        df = pd.DataFrame(data)
        print(f"   Generated {len(df)} asset/venue/hour combinations")
        return df

    def train_ev_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train EV forecasting model."""

        print("🧠 Training EV forecasting model...")

        # Features for prediction
        feature_cols = [
            "vol_5m",
            "spread_bps",
            "depth",
            "slip_p95",
            "is_bps",
            "fees_bps",
            "rebate_bps",
            "cost_per_hour",
            "hour",
            "dow",
            "is_weekend",
            "is_night",
            "is_lunch",
        ]

        # Prepare training data
        X = df[feature_cols].fillna(0)
        y = df["ev_usd_per_hour"]

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)

        # Train model
        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        # Cross-validation scores
        cv_scores = []
        cv_maes = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            cv_scores.append(r2_score(y_val, y_pred))
            cv_maes.append(mean_absolute_error(y_val, y_pred))

        # Train final model on all data
        model.fit(X, y)

        # Feature importance
        feature_importance = dict(zip(feature_cols, model.feature_importances_))

        print(f"   Model R² (CV): {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        print(
            f"   Model MAE (CV): {np.mean(cv_maes):.1f} ± {np.std(cv_maes):.1f} USD/hour"
        )

        return {
            "model": model,
            "feature_cols": feature_cols,
            "cv_r2": np.mean(cv_scores),
            "cv_mae": np.mean(cv_maes),
            "feature_importance": feature_importance,
        }

    def generate_ev_grid(
        self, df: pd.DataFrame, model_info: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate EV predictions for next period."""

        print("📈 Generating EV grid for next 48 hours...")

        # Generate future timestamps
        start_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
        future_hours = pd.date_range(
            start_time, start_time + datetime.timedelta(hours=48), freq="H"
        )

        model = model_info["model"]
        feature_cols = model_info["feature_cols"]

        ev_grid = []

        for hour in future_hours:
            for asset in self.assets:
                for venue in self.venues:
                    # Get recent features as baseline (last week same hour)
                    recent_data = df[
                        (df["asset"] == asset)
                        & (df["venue"] == venue)
                        & (df["timestamp"].dt.hour == hour.hour)
                    ].tail(
                        7
                    )  # Last week

                    if recent_data.empty:
                        continue

                    # Use median recent features as forecast baseline
                    features = recent_data[feature_cols].median().to_dict()

                    # Update time features
                    features["hour"] = hour.hour
                    features["dow"] = hour.weekday()
                    features["is_weekend"] = hour.weekday() >= 5
                    features["is_night"] = hour.hour < 6 or hour.hour > 22
                    features["is_lunch"] = 12 <= hour.hour <= 14

                    # Predict EV
                    X_pred = pd.DataFrame([features])
                    ev_pred = model.predict(X_pred)[0]

                    # Classify into bands
                    if ev_pred >= self.ev_threshold:
                        band = "green"
                    elif ev_pred >= 0:
                        band = "amber"
                    else:
                        band = "red"

                    ev_grid.append(
                        {
                            "timestamp": hour,
                            "asset": asset,
                            "venue": venue,
                            "ev_usd_per_hour": ev_pred,
                            "band": band,
                            "date": hour.date(),
                            "hour": hour.hour,
                        }
                    )

        ev_df = pd.DataFrame(ev_grid)
        print(f"   Generated EV grid: {len(ev_df)} predictions")

        # Print band distribution
        band_counts = ev_df["band"].value_counts()
        print(
            f"   Band distribution: Green={band_counts.get('green', 0)}, "
            f"Amber={band_counts.get('amber', 0)}, Red={band_counts.get('red', 0)}"
        )

        return ev_df

    def generate_calendar_markdown(self, ev_df: pd.DataFrame) -> str:
        """Generate human-readable calendar markdown."""

        print("📅 Generating trade calendar markdown...")

        # Group by date and asset
        calendar_md = f"""# Trade Calendar - EV Forecasts\n\n"""
        calendar_md += f"""**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"""
        calendar_md += f"""**EV Threshold:** ${self.ev_threshold:.1f}/hour (green ≥ threshold)\n\n"""

        # Summary statistics
        total_windows = len(ev_df)
        green_windows = len(ev_df[ev_df["band"] == "green"])
        amber_windows = len(ev_df[ev_df["band"] == "amber"])
        red_windows = len(ev_df[ev_df["band"] == "red"])

        calendar_md += f"""## Summary\n\n"""
        calendar_md += f"""- **Total Trading Windows:** {total_windows}\n"""
        calendar_md += f"""- **🟢 Green (EV ≥ ${self.ev_threshold}):** {green_windows} ({green_windows/total_windows:.1%})\n"""
        calendar_md += f"""- **🟡 Amber (0 ≤ EV < ${self.ev_threshold}):** {amber_windows} ({amber_windows/total_windows:.1%})\n"""
        calendar_md += f"""- **🔴 Red (EV < 0):** {red_windows} ({red_windows/total_windows:.1%})\n\n"""

        # Daily breakdown
        calendar_md += f"""## Daily Schedule\n\n"""

        for date in sorted(ev_df["date"].unique()):
            daily_data = ev_df[ev_df["date"] == date]
            calendar_md += f"""### {date.strftime('%A, %B %d, %Y')}\n\n"""

            # Asset breakdown for this date
            for asset in self.assets:
                asset_data = daily_data[daily_data["asset"] == asset]
                if asset_data.empty:
                    continue

                green_hours = asset_data[asset_data["band"] == "green"]["hour"].tolist()
                amber_hours = asset_data[asset_data["band"] == "amber"]["hour"].tolist()
                red_hours = asset_data[asset_data["band"] == "red"]["hour"].tolist()

                calendar_md += f"""**{asset}:**\n"""
                if green_hours:
                    calendar_md += f"""- 🟢 Green: {', '.join(f'{h:02d}:00' for h in sorted(green_hours))}\n"""
                if amber_hours:
                    calendar_md += f"""- 🟡 Amber: {', '.join(f'{h:02d}:00' for h in sorted(amber_hours))}\n"""
                if red_hours:
                    calendar_md += f"""- 🔴 Red: {', '.join(f'{h:02d}:00' for h in sorted(red_hours))}\n"""
                calendar_md += f"""\n"""

            calendar_md += f"""\n"""

        # Best trading windows
        calendar_md += f"""## Best Trading Windows (Top 10)\n\n"""
        top_windows = ev_df.nlargest(10, "ev_usd_per_hour")
        calendar_md += f"""| Rank | Asset | Venue | Time | EV ($/hour) | Band |\n"""
        calendar_md += f"""|------|-------|-------|------|-------------|------|\n"""

        for i, (_, row) in enumerate(top_windows.iterrows(), 1):
            band_emoji = {"green": "🟢", "amber": "🟡", "red": "🔴"}[row["band"]]
            calendar_md += f"""| {i} | {row['asset']} | {row['venue']} | {row['timestamp'].strftime('%m/%d %H:%M')} | ${row['ev_usd_per_hour']:.1f} | {band_emoji} {row['band']} |\n"""

        calendar_md += (
            f"""\n---\n*Generated by EV Forecaster - M13 Market Selection*\n"""
        )

        return calendar_md

    def save_artifacts(
        self,
        ev_df: pd.DataFrame,
        calendar_md: str,
        model_info: Dict[str, Any],
        output_dir: str,
    ):
        """Save EV grid, calendar, and model artifacts."""

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%SZ")
        output_path = Path(output_dir) / timestamp
        output_path.mkdir(parents=True, exist_ok=True)

        # Save EV grid as parquet
        ev_file = output_path / "ev_grid.parquet"
        ev_df.to_parquet(ev_file, index=False)
        print(f"📊 EV grid saved: {ev_file}")

        # Save calendar markdown
        calendar_file = output_path / "calendar.md"
        with open(calendar_file, "w") as f:
            f.write(calendar_md)
        print(f"📅 Calendar saved: {calendar_file}")

        # Save model metadata
        model_meta = {
            "timestamp": timestamp,
            "window_days": self.window_days,
            "ev_threshold": self.ev_threshold,
            "assets": self.assets,
            "venues": self.venues,
            "cv_r2": model_info["cv_r2"],
            "cv_mae": model_info["cv_mae"],
            "feature_importance": model_info["feature_importance"],
            "ev_bands": ev_df["band"].value_counts().to_dict(),
        }

        meta_file = output_path / "model_metadata.json"
        with open(meta_file, "w") as f:
            json.dump(model_meta, f, indent=2, default=str)
        print(f"🧠 Model metadata saved: {meta_file}")

        # Create latest symlinks
        latest_ev = Path(output_dir) / "latest.parquet"
        latest_calendar = Path(output_dir) / "latest_calendar.md"
        latest_meta = Path(output_dir) / "latest_metadata.json"

        for latest, target in [
            (latest_ev, ev_file),
            (latest_calendar, calendar_file),
            (latest_meta, meta_file),
        ]:
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(target.relative_to(Path(output_dir)))

        print(f"🔗 Latest symlinks created")

        return {
            "ev_file": str(ev_file),
            "calendar_file": str(calendar_file),
            "meta_file": str(meta_file),
            "output_dir": str(output_path),
        }


def main():
    """Main EV forecaster function."""
    parser = argparse.ArgumentParser(
        description="Expected Value Forecaster & Trade Calendar"
    )
    parser.add_argument("--window", default="14d", help="Historical window (e.g. 14d)")
    parser.add_argument(
        "--ev-threshold",
        type=float,
        default=10.0,
        help="EV threshold for green band (USD/hour)",
    )
    parser.add_argument("--out", default="artifacts/ev", help="Output directory")
    args = parser.parse_args()

    # Parse window
    window_days = int(args.window.rstrip("d"))

    print("📈 EV Forecaster & Trade Calendar Generator")
    print("=" * 50)
    print(f"Window: {window_days} days")
    print(f"EV Threshold: ${args.ev_threshold}/hour")
    print(f"Output: {args.out}")
    print("=" * 50)

    try:
        forecaster = EVForecaster(window_days, args.ev_threshold)

        # Generate synthetic data (in production: load from database)
        df = forecaster.generate_synthetic_data()

        # Train EV model
        model_info = forecaster.train_ev_model(df)

        # Generate EV grid
        ev_df = forecaster.generate_ev_grid(df, model_info)

        # Generate calendar
        calendar_md = forecaster.generate_calendar_markdown(ev_df)

        # Save artifacts
        artifacts = forecaster.save_artifacts(ev_df, calendar_md, model_info, args.out)

        print(f"\n✅ EV Forecaster Complete!")
        print(f"📊 EV Grid: {artifacts['ev_file']}")
        print(f"📅 Calendar: {artifacts['calendar_file']}")
        print(f"🧠 Metadata: {artifacts['meta_file']}")

        # Show next actions
        green_count = len(ev_df[ev_df["band"] == "green"])
        print(f"\n🎯 Next 48 hours: {green_count} green windows available")
        print(f"💡 Run 'make duty-on' to enable duty-cycling")

        return 0

    except Exception as e:
        print(f"❌ EV forecaster failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
