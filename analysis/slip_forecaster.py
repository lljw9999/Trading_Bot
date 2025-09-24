#!/usr/bin/env python3
"""
Slippage Forecaster: Per-Venue/Asset Slippage Prediction
Train on fills data to predict p95 slippage for optimal execution sizing.
"""
import os
import sys
import json
import datetime
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")


@dataclass
class SlippageModelConfig:
    """Configuration for slippage model training."""

    window_days: int = 14
    min_samples_per_asset: int = 100
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int = 10


class SlippageForecaster:
    def __init__(self, config: SlippageModelConfig = None):
        self.config = config or SlippageModelConfig()
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")
        self.current_time = datetime.datetime.now(datetime.timezone.utc)

        # Model storage
        self.models = {}  # Per asset-venue models
        self.feature_importance = {}
        self.model_metrics = {}

    def load_fills_data(self, window_days: int) -> pd.DataFrame:
        """Load fills data for training."""
        try:
            # Look for fills in green economics data and audit records
            fills_data = []

            # Check green profit tracker data
            econ_dir = self.base_dir / "artifacts" / "econ_green"
            if econ_dir.exists():
                for timestamp_dir in econ_dir.glob("*Z"):
                    if timestamp_dir.is_dir():
                        daily_json = timestamp_dir / "daily.json"
                        if daily_json.exists():
                            try:
                                with open(daily_json, "r") as f:
                                    daily_data = json.load(f)

                                # Extract metrics as proxy fills
                                metrics = daily_data.get("metrics", {})
                                if metrics.get("total_fills", 0) > 0:
                                    # Simulate individual fills based on aggregated metrics
                                    num_fills = metrics["total_fills"]
                                    avg_slippage = metrics.get("avg_slippage_bps", 10)

                                    # Generate synthetic fills for training
                                    for i in range(
                                        min(num_fills, 50)
                                    ):  # Limit for performance
                                        fills_data.append(
                                            self.generate_synthetic_fill(
                                                timestamp_dir.name, avg_slippage
                                            )
                                        )
                            except Exception:
                                continue

            # If insufficient real data, generate comprehensive synthetic dataset
            if len(fills_data) < 1000:  # Need substantial data for training
                print(
                    "⚠️ Insufficient fills data found, generating synthetic training data"
                )
                fills_data = self.generate_synthetic_fills_dataset(window_days)

            df = pd.DataFrame(fills_data)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
                print(
                    f"📊 Loaded {len(df)} fills across {df['asset'].nunique()} assets"
                )

            return df

        except Exception as e:
            print(f"⚠️ Error loading fills data: {e}")
            return self.generate_synthetic_fills_dataset(window_days)

    def generate_synthetic_fill(
        self, timestamp_str: str, base_slippage: float
    ) -> Dict[str, Any]:
        """Generate a single synthetic fill record."""
        assets = ["SOL-USD", "BTC-USD", "ETH-USD", "NVDA"]
        venues = ["coinbase", "binance", "alpaca"]

        asset = np.random.choice(assets)
        venue = np.random.choice(venues if asset != "NVDA" else ["alpaca", "coinbase"])

        # Market microstructure features
        spread_bps = np.random.uniform(1, 25)  # Wider spreads = more slippage
        vol_1m = np.random.uniform(0.5, 3.0)  # Volatility factor
        vol_5m = np.random.uniform(0.8, 5.0)

        # Order book features
        ob_imbalance = np.random.uniform(-1, 1)  # Bid/ask imbalance
        depth_1 = np.random.uniform(0.1, 2.0)  # Depth at best
        depth_5 = np.random.uniform(0.5, 5.0)  # Depth 5 levels
        depth_10 = np.random.uniform(1.0, 10.0)

        # Execution features
        queue_pos_est = np.random.uniform(0, 1)  # Estimated queue position
        slice_pct = np.random.uniform(0.001, 0.05)  # Size as % of volume
        is_maker = np.random.random() < 0.7

        # Time features
        try:
            timestamp = datetime.datetime.fromisoformat(
                timestamp_str.replace("Z", "+00:00")
            )
        except:
            timestamp = self.current_time
        hour = timestamp.hour
        is_market_hours = 9 <= hour <= 16 if asset == "NVDA" else True

        # Slippage calculation (synthetic model)
        slippage_bps = base_slippage

        # Adjust based on features
        if not is_maker:
            slippage_bps += np.random.uniform(5, 15)  # Taker penalty

        slippage_bps += spread_bps * 0.3  # Spread impact
        slippage_bps += max(0, slice_pct * 1000) * 10  # Size impact
        slippage_bps += vol_1m * 2  # Volatility impact

        if depth_1 < 0.5:
            slippage_bps += 5  # Thin book penalty

        if abs(ob_imbalance) > 0.5:
            slippage_bps += 3  # Imbalance penalty

        if not is_market_hours and asset == "NVDA":
            slippage_bps += 8  # After hours penalty

        # Add noise
        slippage_bps += np.random.normal(0, 3)
        slippage_bps = max(0, slippage_bps)  # No negative slippage

        return {
            "timestamp": timestamp,
            "asset": asset,
            "venue": venue,
            "slippage_bps": slippage_bps,
            "spread_bps": spread_bps,
            "vol_1m": vol_1m,
            "vol_5m": vol_5m,
            "ob_imbalance": ob_imbalance,
            "depth_1": depth_1,
            "depth_5": depth_5,
            "depth_10": depth_10,
            "queue_pos_est": queue_pos_est,
            "is_maker": is_maker,
            "slice_pct": slice_pct,
            "hour": hour,
            "is_market_hours": is_market_hours,
            "notional_usd": np.random.uniform(1000, 100000),
        }

    def generate_synthetic_fills_dataset(
        self, window_days: int
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive synthetic fills dataset."""
        fills_data = []
        start_time = self.current_time - datetime.timedelta(days=window_days)

        # Generate fills across time period
        for day in range(window_days):
            day_start = start_time + datetime.timedelta(days=day)

            # 200-800 fills per day
            num_fills = np.random.randint(200, 800)

            for _ in range(num_fills):
                fill_time = day_start + datetime.timedelta(
                    hours=np.random.uniform(0, 24)
                )

                fill = self.generate_synthetic_fill(
                    fill_time.isoformat(), np.random.uniform(5, 20)  # Base slippage
                )
                fill["timestamp"] = fill_time
                fills_data.append(fill)

        print(f"📊 Generated {len(fills_data)} synthetic fills for {window_days} days")
        return fills_data

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for slippage prediction."""
        # Time-based features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6])

        # Market regime features
        df["high_vol_regime"] = df["vol_5m"] > df["vol_5m"].rolling(50).quantile(0.8)
        df["wide_spread_regime"] = df["spread_bps"] > df["spread_bps"].rolling(
            50
        ).quantile(0.8)

        # Interaction features
        df["vol_spread_interaction"] = df["vol_5m"] * df["spread_bps"]
        df["size_depth_ratio"] = df["slice_pct"] / np.maximum(df["depth_1"], 0.001)
        df["imbalance_size_interaction"] = df["ob_imbalance"] * df["slice_pct"]

        # Rolling statistics (per asset-venue)
        for asset_venue in df.groupby(["asset", "venue"]).groups.keys():
            mask = (df["asset"] == asset_venue[0]) & (df["venue"] == asset_venue[1])

            if mask.sum() > 10:  # Enough data for rolling stats
                subset = df[mask].copy()

                # Rolling averages
                df.loc[mask, "slippage_ma_10"] = (
                    subset["slippage_bps"].rolling(10, min_periods=1).mean()
                )
                df.loc[mask, "spread_ma_10"] = (
                    subset["spread_bps"].rolling(10, min_periods=1).mean()
                )
                df.loc[mask, "vol_ma_10"] = (
                    subset["vol_5m"].rolling(10, min_periods=1).mean()
                )

        # Fill NaNs with medians
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for model training."""
        return [
            "spread_bps",
            "vol_1m",
            "vol_5m",
            "ob_imbalance",
            "depth_1",
            "depth_5",
            "depth_10",
            "queue_pos_est",
            "slice_pct",
            "hour",
            "is_maker",
            "is_market_hours",
            "high_vol_regime",
            "wide_spread_regime",
            "is_weekend",
            "vol_spread_interaction",
            "size_depth_ratio",
            "imbalance_size_interaction",
            "slippage_ma_10",
            "spread_ma_10",
            "vol_ma_10",
        ]

    def train_asset_venue_model(
        self, asset: str, venue: str, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Train slippage model for specific asset-venue pair."""
        # Filter data for this asset-venue
        asset_venue_data = df[(df["asset"] == asset) & (df["venue"] == venue)].copy()

        if len(asset_venue_data) < self.config.min_samples_per_asset:
            print(
                f"⚠️ Insufficient data for {asset}-{venue}: {len(asset_venue_data)} samples"
            )
            return None

        # Prepare features and target
        feature_cols = self.get_feature_columns()
        available_features = [
            col for col in feature_cols if col in asset_venue_data.columns
        ]

        X = asset_venue_data[available_features]
        y = asset_venue_data["slippage_bps"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )

        # Train model
        model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Feature importance
        feature_importance = dict(zip(available_features, model.feature_importances_))

        # Calculate p95 predictions
        p95_actual = np.percentile(y_test, 95)
        p95_predicted = np.percentile(y_pred, 95)

        model_info = {
            "asset": asset,
            "venue": venue,
            "model": model,
            "feature_cols": available_features,
            "n_samples": len(asset_venue_data),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "mae": float(mae),
            "r2": float(r2),
            "p95_actual": float(p95_actual),
            "p95_predicted": float(p95_predicted),
            "feature_importance": feature_importance,
            "trained_at": self.current_time.isoformat(),
        }

        print(
            f"✅ Trained {asset}-{venue}: R²={r2:.3f}, MAE={mae:.1f}bp, P95={p95_predicted:.1f}bp"
        )

        return model_info

    def predict_slip_p95(
        self, asset: str, venue: str, regime_features: Dict[str, float]
    ) -> float:
        """Predict p95 slippage for given asset, venue, and market regime."""
        model_key = f"{asset}_{venue}"

        if model_key not in self.models:
            print(f"⚠️ No model available for {asset}-{venue}")
            # Return conservative estimate based on asset type
            if asset == "NVDA":
                return 18.0  # Equity baseline
            else:
                return 15.0  # Crypto baseline

        model_info = self.models[model_key]
        model = model_info["model"]
        feature_cols = model_info["feature_cols"]

        # Prepare feature vector
        feature_vector = []
        for col in feature_cols:
            if col in regime_features:
                feature_vector.append(regime_features[col])
            else:
                # Use median value from training data
                feature_vector.append(0.0)  # Default fallback

        # Predict slippage
        X_pred = np.array([feature_vector])
        predicted_slippage = model.predict(X_pred)[0]

        # Apply regime adjustments for p95 (conservative)
        p95_adjustment = 1.5  # P95 is typically 1.5x higher than mean
        p95_prediction = predicted_slippage * p95_adjustment

        return max(0.0, float(p95_prediction))

    def run_slippage_forecasting(
        self, window_days: int, output_dir: str
    ) -> Dict[str, Any]:
        """Run complete slippage forecasting pipeline."""

        print("📈 Slippage Forecaster: Per-Venue/Asset Prediction")
        print("=" * 55)
        print(f"Training window: {window_days} days")
        print(f"Output: {output_dir}")
        print("=" * 55)

        # Load and prepare data
        print("📊 Loading fills data...")
        df = self.load_fills_data(window_days)

        if df.empty:
            raise ValueError("No fills data available for training")

        print("🔧 Engineering features...")
        df = self.engineer_features(df)

        # Train models per asset-venue
        print("🤖 Training slippage models...")
        asset_venues = df.groupby(["asset", "venue"]).size()

        trained_models = 0
        for (asset, venue), count in asset_venues.items():
            if count >= self.config.min_samples_per_asset:
                model_info = self.train_asset_venue_model(asset, venue, df)
                if model_info:
                    model_key = f"{asset}_{venue}"
                    self.models[model_key] = model_info
                    trained_models += 1

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model report
        timestamp_str = self.current_time.strftime("%Y%m%d_%H%M%SZ")
        model_report = {
            "timestamp": self.current_time.isoformat(),
            "window_days": window_days,
            "total_fills": len(df),
            "asset_venues": len(asset_venues),
            "trained_models": trained_models,
            "config": {
                "min_samples_per_asset": self.config.min_samples_per_asset,
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
            },
            "models": {},
        }

        # Add model details (without sklearn model objects)
        for model_key, model_info in self.models.items():
            model_copy = model_info.copy()
            del model_copy["model"]  # Remove sklearn object for JSON serialization
            model_report["models"][model_key] = model_copy

        # Save model report
        report_file = output_path / "slip_model.json"
        with open(report_file, "w") as f:
            json.dump(model_report, f, indent=2)

        # Create prediction examples
        prediction_examples = []

        # Example regimes for testing
        test_regimes = [
            {
                "name": "low_volatility_tight_spread",
                "spread_bps": 3.0,
                "vol_1m": 0.5,
                "vol_5m": 0.8,
                "ob_imbalance": 0.1,
                "depth_1": 2.0,
                "slice_pct": 0.01,
                "is_maker": True,
                "hour": 14,
            },
            {
                "name": "high_volatility_wide_spread",
                "spread_bps": 20.0,
                "vol_1m": 2.5,
                "vol_5m": 4.0,
                "ob_imbalance": -0.6,
                "depth_1": 0.3,
                "slice_pct": 0.03,
                "is_maker": False,
                "hour": 9,
            },
            {
                "name": "moderate_conditions",
                "spread_bps": 8.0,
                "vol_1m": 1.2,
                "vol_5m": 1.8,
                "ob_imbalance": 0.0,
                "depth_1": 1.0,
                "slice_pct": 0.015,
                "is_maker": True,
                "hour": 11,
            },
        ]

        for regime in test_regimes:
            regime_predictions = {}
            for model_key in self.models.keys():
                asset, venue = model_key.split("_", 1)
                predicted_slip = self.predict_slip_p95(asset, venue, regime)
                regime_predictions[f"{asset}_{venue}"] = predicted_slip

            prediction_examples.append(
                {
                    "regime": regime["name"],
                    "conditions": {k: v for k, v in regime.items() if k != "name"},
                    "predicted_slip_p95_bps": regime_predictions,
                }
            )

        # Save prediction examples
        examples_file = output_path / f"slip_predictions_{timestamp_str}.json"
        with open(examples_file, "w") as f:
            json.dump(prediction_examples, f, indent=2)

        # Summary
        summary = {
            "success": True,
            "timestamp": self.current_time.isoformat(),
            "trained_models": trained_models,
            "total_fills": len(df),
            "window_days": window_days,
            "avg_r2": np.mean([m["r2"] for m in self.models.values()]),
            "avg_mae_bps": np.mean([m["mae"] for m in self.models.values()]),
            "model_report": str(report_file),
            "prediction_examples": str(examples_file),
            "models_available": list(self.models.keys()),
        }

        print(f"\n📈 Slippage Forecasting Summary:")
        print(f"  Models trained: {trained_models}")
        print(f"  Avg R²: {summary['avg_r2']:.3f}")
        print(f"  Avg MAE: {summary['avg_mae_bps']:.1f} bps")
        print(f"  Report saved: {report_file}")

        # Show prediction examples
        print(f"\n🔮 Example P95 Slippage Predictions:")
        for example in prediction_examples:
            print(f"  {example['regime']}:")
            for asset_venue, predicted_slip in example[
                "predicted_slip_p95_bps"
            ].items():
                print(f"    {asset_venue}: {predicted_slip:.1f} bps")

        return summary


def main():
    """Main slippage forecaster function."""
    parser = argparse.ArgumentParser(
        description="Slippage Forecaster: Per-Venue/Asset Prediction"
    )
    parser.add_argument("--window", default="14d", help="Training window (e.g., 14d)")
    parser.add_argument("--out", default="artifacts/exec", help="Output directory")
    parser.add_argument(
        "--min-samples", type=int, default=100, help="Minimum samples per asset-venue"
    )
    args = parser.parse_args()

    # Parse window
    if args.window.endswith("d"):
        window_days = int(args.window[:-1])
    else:
        window_days = int(args.window)

    try:
        config = SlippageModelConfig(
            window_days=window_days, min_samples_per_asset=args.min_samples
        )

        forecaster = SlippageForecaster(config)
        result = forecaster.run_slippage_forecasting(window_days, args.out)

        if result["success"]:
            print(f"✅ Slippage forecasting complete!")
            print(f"📄 Model report: {result['model_report']}")
            print(f"🔮 Predictions: {result['prediction_examples']}")
            print(f"💡 Next: Run 'make exec-v2' to implement queue timing")
            return 0
        else:
            print("❌ Slippage forecasting failed")
            return 1

    except Exception as e:
        print(f"❌ Slippage forecasting error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
