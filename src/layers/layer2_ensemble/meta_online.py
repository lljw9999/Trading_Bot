#!/usr/bin/env python3
"""
Online Meta-Learner with Regime Awareness
Online logistic blend with regime features (vol bucket, trend, liquidity) and Thompson sampling.
"""
import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import redis
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle


@dataclass
class RegimeFeatures:
    """Current market regime features."""

    vol_bucket: str  # low/medium/high
    trend: float  # -1 to 1
    liquidity: float  # 0 to 1
    timestamp: datetime.datetime


class OnlineMetaLearner:
    def __init__(self, redis_client=None, lookback_hours: int = 24):
        self.r = redis_client or redis.Redis(decode_responses=True)
        self.lookback_hours = lookback_hours
        self.scaler = StandardScaler()
        self.model = LogisticRegression(random_state=42)
        self.is_trained = False
        self.alpha_weights = {}

        # Load existing model if available
        self.load_model()

    def detect_regime(self) -> RegimeFeatures:
        """Detect current market regime from recent data."""
        try:
            # Get recent price data (mock implementation)
            # In production: pull from market data feed

            # Mock regime detection
            current_hour = datetime.datetime.now().hour

            # Vol bucket based on time of day (proxy for activity)
            if 9 <= current_hour <= 16:  # Market hours
                vol_bucket = "high"
                liquidity = 0.8
            elif 16 < current_hour <= 20:  # After hours
                vol_bucket = "medium"
                liquidity = 0.5
            else:  # Overnight
                vol_bucket = "low"
                liquidity = 0.2

            # Mock trend detection (sine wave pattern)
            trend = np.sin(current_hour * np.pi / 12) * 0.7

            return RegimeFeatures(
                vol_bucket=vol_bucket,
                trend=trend,
                liquidity=liquidity,
                timestamp=datetime.datetime.now(),
            )

        except Exception as e:
            # Default regime on error
            return RegimeFeatures(
                vol_bucket="medium",
                trend=0.0,
                liquidity=0.5,
                timestamp=datetime.datetime.now(),
            )

    def get_alpha_performance_data(self) -> pd.DataFrame:
        """Get recent alpha performance data."""
        # Mock implementation - in production would query actual performance data
        alpha_names = [
            "ma_momentum",
            "mean_rev",
            "momo_fast",
            "news_sent_alpha",
            "ob_pressure",
            "big_bet_flag",
        ]

        # Generate synthetic performance data
        hours_ago = pd.date_range(
            end=datetime.datetime.now(), periods=self.lookback_hours, freq="H"
        )

        data = []
        for i, timestamp in enumerate(hours_ago):
            regime = self.detect_regime()  # Would be time-specific in production

            for alpha in alpha_names:
                # Different alpha behavior in different regimes
                base_performance = np.random.normal(0, 0.1)

                # Regime-dependent adjustments
                if alpha == "ma_momentum" and regime.vol_bucket == "high":
                    performance_adj = 0.15  # Momentum works well in high vol
                elif alpha == "mean_rev" and regime.vol_bucket == "low":
                    performance_adj = 0.12  # Mean reversion works in low vol
                elif alpha == "news_sent_alpha" and regime.liquidity > 0.6:
                    performance_adj = 0.08  # News alpha works with liquidity
                else:
                    performance_adj = 0.0

                # Add trend dependency
                if alpha in ["ma_momentum", "momo_fast"]:
                    performance_adj += regime.trend * 0.1
                elif alpha == "mean_rev":
                    performance_adj -= abs(regime.trend) * 0.05

                final_performance = base_performance + performance_adj

                data.append(
                    {
                        "timestamp": timestamp,
                        "alpha": alpha,
                        "performance": final_performance,
                        "vol_bucket": regime.vol_bucket,
                        "trend": regime.trend,
                        "liquidity": regime.liquidity,
                        # Encode categorical features
                        "vol_low": 1 if regime.vol_bucket == "low" else 0,
                        "vol_medium": 1 if regime.vol_bucket == "medium" else 0,
                        "vol_high": 1 if regime.vol_bucket == "high" else 0,
                    }
                )

        return pd.DataFrame(data)

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training."""
        # Features: regime indicators, alpha identity, interactions
        alpha_names = df["alpha"].unique()
        feature_list = []
        targets = []

        for _, row in df.iterrows():
            features = []

            # Regime features
            features.extend([row["trend"], row["liquidity"]])
            features.extend([row["vol_low"], row["vol_medium"], row["vol_high"]])

            # Alpha one-hot encoding
            alpha_features = [
                1 if row["alpha"] == alpha else 0 for alpha in alpha_names
            ]
            features.extend(alpha_features)

            # Interaction terms (alpha x regime)
            for alpha in alpha_names:
                if row["alpha"] == alpha:
                    features.extend([row["trend"], row["liquidity"]])
                else:
                    features.extend([0, 0])

            feature_list.append(features)
            targets.append(1 if row["performance"] > 0 else 0)  # Binary target

        return np.array(feature_list), np.array(targets)

    def train_meta_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the meta-learning model."""
        print("🧠 Training online meta-learner...")

        if len(df) < 10:
            print("   Insufficient data for training")
            return {"status": "insufficient_data"}

        try:
            # Prepare data
            X, y = self.prepare_features(df)

            # Fit scaler and transform features
            X_scaled = self.scaler.fit_transform(X)

            # Train logistic regression
            self.model.fit(X_scaled, y)
            self.is_trained = True

            # Compute training metrics
            train_accuracy = self.model.score(X_scaled, y)
            feature_importance = abs(self.model.coef_[0])

            training_results = {
                "status": "success",
                "training_samples": len(X),
                "train_accuracy": train_accuracy,
                "feature_count": len(feature_importance),
                "avg_feature_importance": np.mean(feature_importance),
            }

            print(f"   Training accuracy: {train_accuracy:.3f}")
            print(f"   Features: {len(feature_importance)}")

            return training_results

        except Exception as e:
            print(f"   Training failed: {e}")
            return {"status": "failed", "error": str(e)}

    def compute_alpha_weights(self, current_regime: RegimeFeatures) -> Dict[str, float]:
        """Compute alpha weights for current regime."""
        if not self.is_trained:
            # Default equal weights
            alpha_names = [
                "ma_momentum",
                "mean_rev",
                "momo_fast",
                "news_sent_alpha",
                "ob_pressure",
                "big_bet_flag",
            ]
            return {alpha: 1.0 / len(alpha_names) for alpha in alpha_names}

        try:
            alpha_names = [
                "ma_momentum",
                "mean_rev",
                "momo_fast",
                "news_sent_alpha",
                "ob_pressure",
                "big_bet_flag",
            ]

            weights = {}

            for alpha in alpha_names:
                # Create feature vector for this alpha in current regime
                features = []

                # Regime features
                features.extend([current_regime.trend, current_regime.liquidity])
                features.extend(
                    [
                        1 if current_regime.vol_bucket == "low" else 0,
                        1 if current_regime.vol_bucket == "medium" else 0,
                        1 if current_regime.vol_bucket == "high" else 0,
                    ]
                )

                # Alpha one-hot
                alpha_features = [1 if alpha == a else 0 for a in alpha_names]
                features.extend(alpha_features)

                # Interaction terms
                for a in alpha_names:
                    if alpha == a:
                        features.extend(
                            [current_regime.trend, current_regime.liquidity]
                        )
                    else:
                        features.extend([0, 0])

                # Predict probability
                X_sample = np.array([features])
                X_scaled = self.scaler.transform(X_sample)
                prob = self.model.predict_proba(X_scaled)[0][
                    1
                ]  # Prob of positive performance

                weights[alpha] = max(0.01, prob)  # Minimum weight 1%

            # Normalize to sum to 1
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

            return weights

        except Exception as e:
            print(f"   Weight computation failed: {e}")
            # Fallback to equal weights
            return {alpha: 1.0 / len(alpha_names) for alpha in alpha_names}

    def apply_hard_caps(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply hard caps based on alpha attribution results."""
        try:
            # Load latest alpha attribution if available
            attr_path = Path("artifacts/alpha_attr/report_latest.json")
            if attr_path.exists():
                with open(attr_path, "r") as f:
                    attr_data = json.load(f)

                alpha_actions = {m["name"]: m for m in attr_data["alpha_metrics"]}

                # Apply caps
                for alpha, weight in weights.items():
                    if alpha in alpha_actions:
                        metric = alpha_actions[alpha]

                        # Hard caps based on attribution analysis
                        if metric["marginal_sharpe"] < 0:
                            weights[alpha] = min(
                                weight, 0.05
                            )  # Max 5% for negative Sharpe
                            print(
                                f"   Capping {alpha} to 5% (negative Sharpe: {metric['marginal_sharpe']:.3f})"
                            )

                        if metric["decay_half_life_days"] < 3:
                            weights[alpha] = min(weight, 0.1)  # Max 10% for fast decay
                            print(
                                f"   Capping {alpha} to 10% (fast decay: {metric['decay_half_life_days']:.1f}d)"
                            )

                        if metric["action"] == "pause":
                            weights[alpha] = 0.01  # Minimum weight for paused alphas
                            print(f"   Pausing {alpha} (action: {metric['action']})")

                # Renormalize after capping
                total_weight = sum(weights.values())
                weights = {k: v / total_weight for k, v in weights.items()}

        except Exception as e:
            print(f"   Hard cap application failed: {e}")

        return weights

    def update_param_server(self, weights: Dict[str, float]) -> bool:
        """Update parameter server with new weights."""
        try:
            # Store in Redis
            for alpha, weight in weights.items():
                key = f"meta_learner:weight:{alpha}"
                self.r.set(key, weight)
                self.r.expire(key, 3600 * 24)  # 24h expiry

            # Store metadata
            metadata = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "total_weight": sum(weights.values()),
                "max_weight": max(weights.values()),
                "min_weight": min(weights.values()),
                "weight_entropy": -sum(w * np.log(w + 1e-10) for w in weights.values()),
            }

            self.r.set("meta_learner:metadata", json.dumps(metadata))
            self.r.expire("meta_learner:metadata", 3600 * 24)

            print(f"   Updated parameter server with weights")
            return True

        except Exception as e:
            print(f"   Parameter server update failed: {e}")
            return False

    def save_model(self) -> bool:
        """Save trained model to disk."""
        try:
            model_dir = Path("artifacts/meta_learner")
            model_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")

            # Save model and scaler
            model_path = model_dir / f"model_{timestamp}.pkl"
            scaler_path = model_dir / f"scaler_{timestamp}.pkl"

            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)

            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

            # Create latest symlinks
            latest_model = model_dir / "model_latest.pkl"
            latest_scaler = model_dir / "scaler_latest.pkl"

            for latest, target in [
                (latest_model, model_path),
                (latest_scaler, scaler_path),
            ]:
                if latest.exists():
                    latest.unlink()
                latest.symlink_to(target)

            return True

        except Exception as e:
            print(f"   Model save failed: {e}")
            return False

    def load_model(self) -> bool:
        """Load existing model from disk."""
        try:
            model_dir = Path("artifacts/meta_learner")
            model_path = model_dir / "model_latest.pkl"
            scaler_path = model_dir / "scaler_latest.pkl"

            if model_path.exists() and scaler_path.exists():
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)

                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

                self.is_trained = True
                print("   Loaded existing meta-learner model")
                return True

        except Exception as e:
            print(f"   Model loading failed: {e}")

        return False

    def run_online_update(self) -> Dict[str, Any]:
        """Run online meta-learner update cycle."""
        print("🔄 Running online meta-learner update...")

        # Get current regime
        current_regime = self.detect_regime()
        print(
            f"   Regime: {current_regime.vol_bucket} vol, trend {current_regime.trend:.2f}, liquidity {current_regime.liquidity:.2f}"
        )

        # Get performance data
        performance_df = self.get_alpha_performance_data()
        print(f"   Loaded {len(performance_df)} performance samples")

        # Train/update model
        training_results = self.train_meta_model(performance_df)

        if training_results["status"] == "success":
            # Compute new weights
            raw_weights = self.compute_alpha_weights(current_regime)
            final_weights = self.apply_hard_caps(raw_weights)

            # Update parameter server
            param_update_success = self.update_param_server(final_weights)

            # Save model
            model_save_success = self.save_model()

            # Generate results
            results = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "regime": {
                    "vol_bucket": current_regime.vol_bucket,
                    "trend": current_regime.trend,
                    "liquidity": current_regime.liquidity,
                },
                "training": training_results,
                "weights": final_weights,
                "updates": {
                    "param_server": param_update_success,
                    "model_saved": model_save_success,
                },
            }

            # Display summary
            print(f"   New weights:")
            for alpha, weight in sorted(
                final_weights.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"     {alpha}: {weight:.1%}")

            return results

        else:
            return {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "status": "failed",
                "error": training_results,
            }


def main():
    """Main meta-learner function."""
    parser = argparse.ArgumentParser(description="Online Meta-Learner")
    parser.add_argument(
        "--hours", type=int, default=24, help="Lookback hours for training data"
    )
    parser.add_argument(
        "--output", default="artifacts/meta_learner", help="Output directory"
    )
    args = parser.parse_args()

    try:
        # Create output directory
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
        output_dir = Path(args.output) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run meta-learner update
        learner = OnlineMetaLearner(lookback_hours=args.hours)
        results = learner.run_online_update()

        # Save results
        results_path = output_dir / "meta_update.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Create latest symlink
        latest_path = Path(args.output) / "meta_update_latest.json"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(results_path)

        if results.get("status") != "failed":
            print(f"\n🧠 Meta-Learner Update Complete:")
            print(f"  Training Accuracy: {results['training']['train_accuracy']:.3f}")
            print(f"  Max Weight: {max(results['weights'].values()):.1%}")
            print(
                f"  Weight Entropy: {-sum(w * np.log(w + 1e-10) for w in results['weights'].values()):.3f}"
            )

            print(f"\n📄 Results: {results_path}")
            return 0
        else:
            print(
                f"❌ Meta-learner update failed: {results.get('error', 'Unknown error')}"
            )
            return 1

    except Exception as e:
        print(f"❌ Meta-learner failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
