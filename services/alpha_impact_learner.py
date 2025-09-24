#!/usr/bin/env python3
"""
Online Alpha Impact Learner
Continuously learns which features deserve weight today using River ML
"""

import os
import sys
import json
import time
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis
import numpy as np
from river import linear_model, optim, metrics, stats
from scipy import stats as scipy_stats

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("alpha_impact_learner")


class OnlineAlphaImpactLearner:
    """Online alpha impact learner using River ML."""

    def __init__(self):
        """Initialize alpha impact learner."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Initialize River models
        self.model = linear_model.LinearRegression(
            optimizer=optim.SGD(lr=0.01), l2=1e-4
        )

        # Performance tracker
        self.performance_tracker = metrics.MAE()
        self.feature_importance_tracker = {}

        # Feature definitions
        self.feature_keys = [
            "sent_bull",  # Sentiment bullishness
            "llm_impact",  # LLM impact score
            "basis_bps",  # Basis in bps
            "funding_ann",  # Annualized funding rate
            "vol_20",  # 20-period volatility
            "ob_imbalance",  # Order book imbalance
            "ma_momentum",  # Moving average momentum
            "vol_surface",  # Volatility surface skew
            "whale_flow",  # Whale flow indicator
            "news_sentiment",  # News sentiment score
            "onchain_flow",  # On-chain flow
            "tech_signal",  # Technical signal strength
        ]

        # Learning buffer and statistics
        self.learning_buffer = []
        self.feature_stats = {key: stats.Mean() for key in self.feature_keys}
        self.feature_volatility = {key: stats.Var() for key in self.feature_keys}

        # Configuration
        self.config = {
            "update_interval": 15,  # Update every 15 minutes
            "min_samples": 100,  # Minimum samples before making decisions
            "z_threshold": 2.0,  # Z-score threshold for feature gating
            "top_features": 3,  # Number of top features to tilt toward
            "max_tilt": 0.1,  # Maximum ensemble weight tilt
            "memory_decay": 0.99,  # Memory decay factor
        }

        # State tracking
        self.last_update = 0
        self.sample_count = 0
        self.feature_coefficients = {}
        self.feature_z_scores = {}
        self.gated_features = set()

        logger.info("🧠 Alpha Impact Learner initialized")
        logger.info(f"   Features: {len(self.feature_keys)}")
        logger.info(f"   Update interval: {self.config['update_interval']}min")
        logger.info(f"   Z-threshold: {self.config['z_threshold']}")

    def extract_features_from_state(self, state_data: Dict) -> Dict[str, float]:
        """Extract features from state data."""
        try:
            features = {}

            for key in self.feature_keys:
                # Try different possible key formats
                value = None

                # Direct key lookup
                if key in state_data:
                    value = float(state_data[key])

                # Try with prefixes
                elif f"feature:{key}" in state_data:
                    value = float(state_data[f"feature:{key}"])
                elif f"alpha:{key}" in state_data:
                    value = float(state_data[f"alpha:{key}"])

                # Generate synthetic features for demo
                else:
                    # Use deterministic but realistic synthetic data
                    seed_value = hash(key) % 1000
                    np.random.seed(int(time.time()) % 1000 + seed_value)

                    if "sentiment" in key or "bull" in key:
                        value = np.random.normal(0.5, 0.2)  # Sentiment around neutral
                    elif "bps" in key or "funding" in key:
                        value = np.random.normal(0, 10)  # Basis points around 0
                    elif "vol" in key:
                        value = max(0, np.random.normal(0.3, 0.1))  # Vol around 30%
                    elif "imbalance" in key or "flow" in key:
                        value = np.random.normal(0, 0.5)  # Flow around neutral
                    else:
                        value = np.random.normal(0, 1)  # Standard normal

                features[key] = float(value) if value is not None else 0.0

            return features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {key: 0.0 for key in self.feature_keys}

    def get_target_return(self) -> float:
        """Get target return (5-minute future return)."""
        try:
            # Try to get from Redis
            target = self.redis.get("target:future_ret_5m")
            if target:
                return float(target)

            # Generate synthetic target for demo
            # Use current time for some determinism
            seed = int(time.time()) % 10000
            np.random.seed(seed)

            # Realistic return distribution (basis points)
            return np.random.normal(0, 5) / 10000  # ~5bps std

        except Exception as e:
            logger.warning(f"Error getting target return: {e}")
            return 0.0

    def learn_one_sample(self, features: Dict[str, float], target: float) -> bool:
        """Learn from one sample."""
        try:
            # Validate features
            if not features or not any(features.values()):
                return False

            # Update feature statistics
            for key, value in features.items():
                if key in self.feature_stats:
                    self.feature_stats[key].update(value)
                    self.feature_volatility[key].update(value)

            # Learn with the model
            self.model.learn_one(features, target)
            self.performance_tracker.update(target, self.model.predict_one(features))

            # Add to buffer
            self.learning_buffer.append(
                {"features": features, "target": target, "timestamp": time.time()}
            )

            # Keep buffer manageable
            if len(self.learning_buffer) > 1000:
                self.learning_buffer = self.learning_buffer[-800:]

            self.sample_count += 1

            logger.debug(f"📚 Learned sample {self.sample_count}: target={target:.4f}")

            return True

        except Exception as e:
            logger.error(f"Error learning sample: {e}")
            return False

    def update_feature_importance(self):
        """Update feature importance and coefficients."""
        try:
            # Get model coefficients
            if hasattr(self.model, "weights") and self.model.weights:
                self.feature_coefficients = dict(self.model.weights)
            else:
                logger.warning("No model weights available")
                self.feature_coefficients = {key: 0.0 for key in self.feature_keys}

            # Calculate feature importance (absolute coefficients)
            importance_dict = {
                key: abs(coef) for key, coef in self.feature_coefficients.items()
            }

            # Calculate z-scores for coefficients
            if len(self.feature_coefficients) > 1:
                coef_values = list(self.feature_coefficients.values())
                coef_mean = np.mean(coef_values)
                coef_std = np.std(coef_values)

                if coef_std > 0:
                    self.feature_z_scores = {
                        key: (coef - coef_mean) / coef_std
                        for key, coef in self.feature_coefficients.items()
                    }
                else:
                    self.feature_z_scores = {
                        key: 0.0 for key in self.feature_coefficients
                    }
            else:
                self.feature_z_scores = {key: 0.0 for key in self.feature_coefficients}

            # Gate features with persistently negative z-scores
            self.gated_features = {
                key
                for key, z_score in self.feature_z_scores.items()
                if z_score < -self.config["z_threshold"]
            }

            # Store in Redis
            self.redis.hset("alpha:impact", mapping=self.feature_coefficients)
            self.redis.hset("alpha:importance", mapping=importance_dict)
            self.redis.hset("alpha:z_scores", mapping=self.feature_z_scores)

            # Set feature flags for gated features
            for feature in self.feature_keys:
                flag_key = f"feature_flag_{feature.lower()}"
                is_gated = feature in self.gated_features
                self.redis.hset("features:flags", flag_key, 0 if is_gated else 1)

            logger.info(
                f"📊 Updated feature importance: {len(self.gated_features)} features gated"
            )

        except Exception as e:
            logger.error(f"Error updating feature importance: {e}")

    def update_ensemble_weights(self):
        """Update ensemble weights with tilt toward top features."""
        try:
            if not self.feature_coefficients:
                return

            # Get top positive features
            positive_features = {
                key: coef for key, coef in self.feature_coefficients.items() if coef > 0
            }

            if not positive_features:
                return

            # Sort by coefficient value
            top_features = sorted(
                positive_features.items(), key=lambda x: x[1], reverse=True
            )[: self.config["top_features"]]

            # Calculate tilts
            total_positive_coef = sum(positive_features.values())
            ensemble_tilts = {}

            for feature, coef in top_features:
                tilt_weight = min(
                    self.config["max_tilt"],
                    (coef / total_positive_coef) * self.config["max_tilt"],
                )
                ensemble_tilts[feature] = tilt_weight

            # Store ensemble tilts
            self.redis.hset("ensemble:tilts", mapping=ensemble_tilts)

            # Update main ensemble weights
            current_weights = self.redis.hgetall("ensemble:weights")
            updated_weights = {}

            for feature in self.feature_keys:
                base_weight = float(current_weights.get(feature, 0.0))
                tilt = ensemble_tilts.get(feature, 0.0)

                # Apply tilt (additive)
                new_weight = base_weight + tilt
                updated_weights[feature] = max(0.0, min(1.0, new_weight))  # Clamp [0,1]

            # Normalize weights to sum to 1
            total_weight = sum(updated_weights.values())
            if total_weight > 0:
                updated_weights = {
                    key: weight / total_weight
                    for key, weight in updated_weights.items()
                }

                self.redis.hset("ensemble:weights", mapping=updated_weights)

                logger.info(
                    f"⚖️ Updated ensemble weights with tilts: {list(ensemble_tilts.keys())}"
                )

        except Exception as e:
            logger.error(f"Error updating ensemble weights: {e}")

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status."""
        try:
            status = {
                "service": "alpha_impact_learner",
                "timestamp": time.time(),
                "sample_count": self.sample_count,
                "last_update": self.last_update,
                "performance": {
                    "mae": (
                        self.performance_tracker.get() if self.sample_count > 0 else 0.0
                    ),
                },
                "feature_stats": {
                    "total_features": len(self.feature_keys),
                    "gated_features": len(self.gated_features),
                    "gated_list": list(self.gated_features),
                },
                "top_features": [],
                "config": self.config,
            }

            # Add top positive features
            if self.feature_coefficients:
                positive_features = [
                    (key, coef)
                    for key, coef in self.feature_coefficients.items()
                    if coef > 0
                ]
                positive_features.sort(key=lambda x: x[1], reverse=True)
                status["top_features"] = positive_features[:5]

            return status

        except Exception as e:
            return {
                "service": "alpha_impact_learner",
                "status": "error",
                "error": str(e),
            }

    async def run_learning_cycle(self) -> Dict[str, Any]:
        """Run one learning cycle."""
        try:
            cycle_start = time.time()

            # Get latest state data
            state_stream = self.redis.xrevrange("state:live", "+", "-", count=1)

            if not state_stream:
                logger.debug("No state data available")
                return {"status": "no_data"}

            # Extract features from latest state
            _, state_data = state_stream[0]
            features = self.extract_features_from_state(state_data)

            # Get target return
            target = self.get_target_return()

            # Learn from sample
            if self.learn_one_sample(features, target):
                # Update importance every N samples
                if self.sample_count % 50 == 0:  # Every 50 samples
                    self.update_feature_importance()
                    self.update_ensemble_weights()
                    self.last_update = time.time()

            cycle_duration = time.time() - cycle_start

            return {
                "status": "success",
                "cycle_duration": cycle_duration,
                "sample_count": self.sample_count,
                "target": target,
                "prediction": (
                    self.model.predict_one(features) if self.sample_count > 0 else 0.0
                ),
                "mae": self.performance_tracker.get() if self.sample_count > 0 else 0.0,
            }

        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")
            return {"status": "error", "error": str(e)}

    async def run_continuous_learning(self):
        """Run continuous learning loop."""
        logger.info("🚀 Starting continuous learning loop")

        try:
            while True:
                # Run learning cycle
                result = await self.run_learning_cycle()

                if result["status"] == "success":
                    if self.sample_count % 100 == 0:  # Log every 100 samples
                        logger.info(
                            f"📈 Sample {self.sample_count}: "
                            f"MAE={result.get('mae', 0):.4f}, "
                            f"Gated={len(self.gated_features)}"
                        )
                elif result["status"] == "error":
                    logger.error(f"Learning cycle error: {result.get('error')}")

                # Wait before next cycle
                await asyncio.sleep(1)  # 1 second between samples

        except asyncio.CancelledError:
            logger.info("Continuous learning stopped")
        except Exception as e:
            logger.error(f"Error in continuous learning: {e}")

    def run_scheduled_update(self):
        """Run scheduled update (called every 15 minutes)."""
        try:
            if self.sample_count < self.config["min_samples"]:
                logger.info(
                    f"Insufficient samples: {self.sample_count}/{self.config['min_samples']}"
                )
                return

            logger.info("⏰ Running scheduled update")

            # Update feature importance and ensemble weights
            self.update_feature_importance()
            self.update_ensemble_weights()

            # Log summary
            top_features = sorted(
                self.feature_coefficients.items(), key=lambda x: abs(x[1]), reverse=True
            )[:3]

            logger.info(
                f"📊 Top features: {[(k, f'{v:+.4f}') for k, v in top_features]}"
            )
            logger.info(f"🚫 Gated features: {list(self.gated_features)}")

            self.last_update = time.time()

        except Exception as e:
            logger.error(f"Error in scheduled update: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Online Alpha Impact Learner")
    parser.add_argument("--run", action="store_true", help="Run continuous learning")
    parser.add_argument("--update", action="store_true", help="Run one update cycle")
    parser.add_argument("--status", action="store_true", help="Show learning status")

    args = parser.parse_args()

    # Create learner
    learner = OnlineAlphaImpactLearner()

    if args.status:
        # Show status
        status = learner.get_learning_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.update:
        # Run one update
        learner.run_scheduled_update()
        return

    if args.run:
        # Run continuous learning
        try:
            asyncio.run(learner.run_continuous_learning())
        except KeyboardInterrupt:
            logger.info("Learning stopped by user")
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
