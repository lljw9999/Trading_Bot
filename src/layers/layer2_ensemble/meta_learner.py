"""
Ensemble Meta-Learner for Alpha Signal Combination

Uses RandomForest to combine signals from multiple alpha models and generate
final trading signals with confidence estimates.

Enhanced with logistic blending for OBP + MAM signals as per Future_instruction.txt A2-1.
"""

import numpy as np
import yaml
import os
import redis
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any
from decimal import Decimal
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from ..layer0_data_ingestion.schemas import FeatureSnapshot
from ...utils.logger import get_logger
from .bandit_blender import BanditBlender

try:
    from ...core.feature_flags import is_enabled
except ImportError:
    # Fallback for testing
    def is_enabled(flag_name: str) -> bool:
        return False


class MetaLearner:
    """
    RandomForest-based meta-learner for combining alpha signals.

    Takes signals from multiple alpha models and combines them into
    a single prediction with confidence estimate.

    Enhanced with logistic blending for specific alpha pairs.
    """

    def __init__(self, model_name: str = "meta_learner"):
        """
        Initialize the meta-learner.

        Args:
            model_name: Name identifier for the model
        """
        self.model_name = model_name
        self.logger = get_logger(f"ensemble.{model_name}")

        # Model components
        self.model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.scaler = StandardScaler()

        # Model state
        self.is_trained = False
        self.last_prediction_time: Optional[datetime] = None
        self.prediction_count = 0

        # Expected alpha models (will be populated dynamically)
        self.alpha_model_names = []
        self.feature_names = []

        # Signal history for training
        self.signal_history = []
        self.max_history = 10000

        # Load logistic blend weights (for fallback)
        self.logistic_weights = self._load_logistic_weights()

        # Initialize contextual bandit blender
        try:
            self.bandit_blender = BanditBlender()
            self.logger.info(
                "Meta-learner initialized with contextual bandit capability"
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize bandit blender: {e}")
            self.bandit_blender = None

        # Redis for state features
        try:
            self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        except Exception as e:
            self.logger.warning(f"Failed to connect to Redis: {e}")
            self.redis = None

    def _load_logistic_weights(self) -> Dict[str, float]:
        """Load logistic blend weights from configuration."""
        try:
            weights_file = "src/config/meta_weights.yaml"
            if os.path.exists(weights_file):
                with open(weights_file, "r") as f:
                    weights = yaml.safe_load(f) or {}
                    return weights.get("logistic_weights", {"w1": 1.0, "w2": 1.0})
            else:
                # Create default weights file
                default_weights = {
                    "logistic_weights": {
                        "w1": 1.0,  # OBP weight
                        "w2": 1.0,  # MAM weight
                    }
                }
                os.makedirs(os.path.dirname(weights_file), exist_ok=True)
                with open(weights_file, "w") as f:
                    yaml.dump(default_weights, f)

                self.logger.info(f"Created default weights file: {weights_file}")
                return default_weights["logistic_weights"]

        except Exception as e:
            self.logger.error(f"Error loading logistic weights: {e}")
            return {"w1": 1.0, "w2": 1.0}

    def _coerce_feature_snapshot(self, data: Any) -> Optional[FeatureSnapshot]:
        if isinstance(data, FeatureSnapshot):
            return data

        symbol = getattr(data, "symbol", "UNKNOWN")
        timestamp = getattr(data, "timestamp", datetime.now(timezone.utc))
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.now(timezone.utc)
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now(timezone.utc)

        mid_price = getattr(data, "mid_price", None)
        if mid_price is None:
            price = getattr(data, "price", None)
            if price is None:
                price = getattr(data, "last", None)
            if price is not None:
                try:
                    mid_price = Decimal(str(price))
                except (TypeError, ValueError):
                    mid_price = None

        return FeatureSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            mid_price=mid_price,
        )

    def predict(
        self,
        alpha_signals: Dict[str, Tuple[float, float]],
        market_features: Any,
    ) -> Tuple[float, float]:
        """
        Generate ensemble prediction from alpha signals.

        Args:
            alpha_signals: Dict of {model_name: (edge_bps, confidence)}
            market_features: Market feature snapshot for context

        Returns:
            Tuple of (ensemble_edge_bps, ensemble_confidence)
        """
        try:
            snapshot = self._coerce_feature_snapshot(market_features)
            if snapshot is None:
                return 0.0, 0.0

            # Use contextual bandit for ensemble weights if feature flag enabled
            if is_enabled("BANDIT_WEIGHTS") and self.bandit_blender is not None:
                edge_bps, confidence = self._bandit_ensemble(alpha_signals, snapshot)
                self.logger.debug(
                    f"Using bandit ensemble: edge={edge_bps:.2f}bps, conf={confidence:.3f}"
                )
            # Fall back to logistic blending for OBP + MAM
            elif self._can_use_logistic_blend(alpha_signals):
                edge_bps, confidence = self._logistic_blend(alpha_signals)
                self.logger.debug(
                    f"Using logistic blend: edge={edge_bps:.2f}bps, conf={confidence:.3f}"
                )
            else:
                # Extract features for prediction
                feature_vector = self._extract_features(alpha_signals, snapshot)

                if feature_vector is None:
                    return 0.0, 0.0

                # Make prediction if model is trained
                if self.is_trained:
                    edge_bps, confidence = self._predict_with_model(feature_vector)
                else:
                    # Use simple ensemble if model not trained yet
                    edge_bps, confidence = self._simple_ensemble(alpha_signals)

            self.prediction_count += 1
            self.last_prediction_time = datetime.now(timezone.utc)

            # Store signal for potential training
            self._store_signal(alpha_signals, market_features, edge_bps, confidence)

            self.logger.debug(
                f"Ensemble prediction for {market_features.symbol}: "
                f"edge={edge_bps:.2f}bps, confidence={confidence:.3f}"
            )

            return edge_bps, confidence

        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return 0.0, 0.0

    def predict_simple(self, alpha_signals: List[float]) -> float:
        """
        Simplified predict method for backward compatibility.

        Args:
            alpha_signals: List of alpha signal values

        Returns:
            Ensemble edge in basis points
        """
        try:
            if alpha_signals is None or len(alpha_signals) == 0:
                return 0.0

            # Simple weighted average for backward compatibility
            if len(alpha_signals) == 1:
                return alpha_signals[0]
            elif len(alpha_signals) == 2:
                # Use logistic blend for two signals
                w1 = self.logistic_weights.get("w1", 1.0)
                w2 = self.logistic_weights.get("w2", 1.0)

                # Scale and blend
                edge1_scaled = alpha_signals[0] / 100.0
                edge2_scaled = alpha_signals[1] / 100.0

                logit = w1 * edge1_scaled + w2 * edge2_scaled

                try:
                    prob = 1.0 / (1.0 + np.exp(-logit))
                except (OverflowError, FloatingPointError):
                    prob = 0.5

                blended_edge = (prob - 0.5) * 100
                return float(blended_edge)
            else:
                # Simple average for more than 2 signals
                return float(np.mean(alpha_signals))

        except Exception as e:
            self.logger.error(f"Error in simplified predict: {e}")
            return 0.0

    def _can_use_logistic_blend(
        self, alpha_signals: Dict[str, Tuple[float, float]]
    ) -> bool:
        """Check if we have the required signals for logistic blending."""
        required_models = ["ob_pressure", "ma_momentum"]  # OBP and MAM
        return all(model in alpha_signals for model in required_models)

    def _logistic_blend(
        self, alpha_signals: Dict[str, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Logistic blending of OBP + MAM signals as specified in Future_instruction.txt A2-1.

        Formula:
        logit = w1*obp_edge + w2*mam_edge
        prob = 1 / (1+exp(-logit))
        blended_edge = (prob-0.5)*100   # convert back to bp
        blended_conf = prob
        """
        try:
            # Get signals
            obp_edge, obp_conf = alpha_signals.get("ob_pressure", (0.0, 0.0))
            mam_edge, mam_conf = alpha_signals.get("ma_momentum", (0.0, 0.0))

            # Get weights
            w1 = self.logistic_weights.get("w1", 1.0)
            w2 = self.logistic_weights.get("w2", 1.0)

            # Scale edges to be in reasonable range for logistic function
            # Convert from basis points to decimal for logistic calculation
            obp_scaled = obp_edge / 100.0  # 25bp -> 0.25
            mam_scaled = mam_edge / 100.0  # 40bp -> 0.40

            # Calculate logit
            logit = w1 * obp_scaled + w2 * mam_scaled

            # Apply logistic function: prob = 1 / (1 + exp(-logit))
            try:
                if logit > 500:  # Prevent overflow
                    prob = 1.0
                elif logit < -500:
                    prob = 0.0
                else:
                    prob = 1.0 / (1.0 + np.exp(-logit))
            except (OverflowError, FloatingPointError):
                prob = 0.5  # Default to neutral

            # Convert back to basis points
            blended_edge = (prob - 0.5) * 100

            # Confidence is the probability itself, but ensure it's in [0, 1]
            blended_conf = max(0.0, min(1.0, prob))

            # Blend with input confidences while preserving directionality
            avg_input_conf = max(0.0, min(1.0, (obp_conf + mam_conf) / 2.0))
            blended_conf = blended_conf * avg_input_conf + 0.5 * (1 - avg_input_conf)

            self.logger.debug(
                f"Logistic blend: obp={obp_edge:.1f}bp, mam={mam_edge:.1f}bp, "
                f"logit={logit:.3f}, prob={prob:.3f}, "
                f"edge={blended_edge:.1f}bp, conf={blended_conf:.3f}"
            )

            return float(blended_edge), float(blended_conf)

        except Exception as e:
            self.logger.error(f"Error in logistic blend: {e}")
            return 0.0, 0.0

    def _bandit_ensemble(
        self,
        alpha_signals: Dict[str, Tuple[float, float]],
        market_features: FeatureSnapshot,
    ) -> Tuple[float, float]:
        """
        Use contextual bandit for ensemble weights as specified in Future_instruction.txt.

        Extracts context features and uses bandit to choose optimal weights,
        then computes weighted signal combination.
        """
        try:
            # Create state dict for feature extraction
            state = self._build_state_dict(market_features)

            # Extract context features for bandit
            context_features = self.bandit_blender.extract_context_features(state)

            # Get bandit-chosen weights
            weight_dict = self.bandit_blender.choose_weights(context_features)

            # Extract alpha signals into vector format
            alpha_vector = []
            arms = self.bandit_blender.arms

            for arm in arms:
                # Map arm names to signal keys
                signal_key = self._map_arm_to_signal_key(arm)
                if signal_key in alpha_signals:
                    edge, conf = alpha_signals[signal_key]
                    alpha_vector.append(edge)
                else:
                    alpha_vector.append(0.0)  # Missing signal

            alpha_vector = np.array(alpha_vector)
            weights = np.array([weight_dict.get(arm, 0.0) for arm in arms])

            # Compute weighted ensemble signal
            signal = np.dot(weights, alpha_vector)

            # Compute ensemble confidence (weighted average of individual confidences)
            conf_vector = []
            for arm in arms:
                signal_key = self._map_arm_to_signal_key(arm)
                if signal_key in alpha_signals:
                    _, conf = alpha_signals[signal_key]
                    conf_vector.append(conf)
                else:
                    conf_vector.append(0.0)

            conf_vector = np.array(conf_vector)
            ensemble_confidence = np.dot(weights, conf_vector)

            self.logger.debug(
                f"Bandit ensemble: weights={weights}, signal={signal:.2f}bp, conf={ensemble_confidence:.3f}"
            )

            return float(signal), float(ensemble_confidence)

        except Exception as e:
            self.logger.error(f"Error in bandit ensemble: {e}")
            return 0.0, 0.0

    def _build_state_dict(self, market_features: FeatureSnapshot) -> Dict[str, Any]:
        """Build state dictionary for bandit feature extraction."""
        try:
            # Base market features
            state = {
                "vol_20": getattr(market_features, "volatility_20m", 0.0),
                "rsi": getattr(market_features, "rsi", 50.0),
                "volume_ratio": getattr(market_features, "volume_ratio", 1.0),
                "spread_pct": getattr(market_features, "spread_bps", 0.0),
                "iv_slope": 0.0,  # Placeholder - would come from options data
            }

            # Add sentiment features from Redis if available
            if self.redis is not None:
                try:
                    sentiment = self.redis.hgetall("sentiment:latest")
                    state["sent_bull"] = float(sentiment.get("bull", 0))
                    state["sent_bear"] = float(sentiment.get("bear", 0))
                    state["llm_impact"] = float(sentiment.get("impact", 0.0))
                except Exception as e:
                    self.logger.debug(f"Could not fetch sentiment features: {e}")
                    state["sent_bull"] = 0.0
                    state["sent_bear"] = 0.0
                    state["llm_impact"] = 0.0

                # Add other Redis-based features
                try:
                    state["market_cap_flow"] = float(
                        self.redis.get("flow:market_cap_weighted") or 0.0
                    )
                    state["funding_rate"] = float(
                        self.redis.get("perp:funding_rate") or 0.0
                    )
                    state["oi_change"] = float(self.redis.get("oi:change_1h") or 0.0)
                except Exception:
                    state["market_cap_flow"] = 0.0
                    state["funding_rate"] = 0.0
                    state["oi_change"] = 0.0

            return state

        except Exception as e:
            self.logger.error(f"Error building state dict: {e}")
            return {}

    def _map_arm_to_signal_key(self, arm: str) -> str:
        """Map bandit arm names to alpha signal keys."""
        arm_mapping = {
            "obp": "ob_pressure",
            "mam": "ma_momentum",
            "lstm": "lstm_alpha",
            "news": "news_alpha",
            "onchain": "onchain_alpha",
        }
        return arm_mapping.get(arm, arm)

    def update_bandit_from_pnl(
        self, pnl: float, last_context: np.ndarray = None
    ) -> None:
        """Update contextual bandit with PnL feedback."""
        try:
            if not is_enabled("BANDIT_WEIGHTS") or self.bandit_blender is None:
                return

            if last_context is None:
                # Try to reconstruct context from latest state
                if self.redis is not None:
                    meta = self.redis.hgetall("ensemble:bandit_meta")
                    if meta:
                        # Use cached context or skip update
                        return

            self.bandit_blender.update_from_pnl_feedback(pnl, last_context)
            self.logger.debug(f"Updated bandit with PnL feedback: {pnl:.4f}")

        except Exception as e:
            self.logger.error(f"Error updating bandit from PnL: {e}")

    def _extract_features(
        self,
        alpha_signals: Dict[str, Tuple[float, float]],
        market_features: FeatureSnapshot,
    ) -> Optional[np.ndarray]:
        """Extract features for the meta-learner."""
        try:
            # Update model names if needed
            if set(alpha_signals.keys()) != set(self.alpha_model_names):
                self.alpha_model_names = sorted(alpha_signals.keys())
                self._update_feature_names()

            features = []

            # Alpha model signals (edge and confidence for each model)
            for model_name in self.alpha_model_names:
                if model_name in alpha_signals:
                    edge, conf = alpha_signals[model_name]
                    features.extend([edge, conf])
                else:
                    features.extend([0.0, 0.0])  # Missing model

            # Market context features
            features.extend(
                [
                    market_features.spread_bps or 0.0,
                    market_features.volatility_5m or 0.0,
                    market_features.volume_ratio or 1.0,
                    market_features.order_book_imbalance or 0.0,
                    market_features.return_1m or 0.0,
                ]
            )

            return np.array(features).reshape(1, -1)

        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None

    def _update_feature_names(self):
        """Update feature names based on alpha models."""
        self.feature_names = []

        # Alpha model features
        for model_name in self.alpha_model_names:
            self.feature_names.extend(
                [f"{model_name}_edge", f"{model_name}_confidence"]
            )

        # Market context features
        self.feature_names.extend(
            [
                "spread_bps",
                "volatility_5m",
                "volume_ratio",
                "order_book_imbalance",
                "return_1m",
            ]
        )

    def _predict_with_model(self, feature_vector: np.ndarray) -> Tuple[float, float]:
        """Make prediction using trained model."""
        try:
            # Scale features
            scaled_features = self.scaler.transform(feature_vector)

            # Get prediction from each tree
            tree_predictions = []
            for tree in self.model.estimators_:
                pred = tree.predict(scaled_features)[0]
                tree_predictions.append(pred)

            # Ensemble statistics
            ensemble_mean = np.mean(tree_predictions)
            ensemble_std = np.std(tree_predictions)

            # Edge is the mean prediction
            edge_bps = float(ensemble_mean)

            # Confidence is inverse of prediction variance (normalized)
            confidence = float(1.0 / (1.0 + ensemble_std))
            confidence = max(0.0, min(1.0, confidence))

            return edge_bps, confidence

        except Exception as e:
            self.logger.error(f"Error in model prediction: {e}")
            return 0.0, 0.0

    def _simple_ensemble(
        self, alpha_signals: Dict[str, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Simple ensemble when model is not trained."""
        try:
            if not alpha_signals:
                return 0.0, 0.0

            # Weighted average by confidence
            total_weighted_edge = 0.0
            total_weight = 0.0

            for model_name, (edge, conf) in alpha_signals.items():
                weight = conf if conf > 0 else 0.1  # Minimum weight
                total_weighted_edge += edge * weight
                total_weight += weight

            if total_weight > 0:
                ensemble_edge = total_weighted_edge / total_weight
                # Ensemble confidence is average of individual confidences
                ensemble_confidence = np.mean(
                    [conf for _, conf in alpha_signals.values()]
                )
            else:
                ensemble_edge = 0.0
                ensemble_confidence = 0.0

            return float(ensemble_edge), float(ensemble_confidence)

        except Exception as e:
            self.logger.error(f"Error in simple ensemble: {e}")
            return 0.0, 0.0

    def _store_signal(
        self,
        alpha_signals: Dict[str, Tuple[float, float]],
        market_features: FeatureSnapshot,
        ensemble_edge: float,
        ensemble_confidence: float,
    ):
        """Store signal for potential training."""
        try:
            signal_record = {
                "timestamp": datetime.now(timezone.utc),
                "symbol": market_features.symbol,
                "alpha_signals": alpha_signals.copy(),
                "market_features": market_features,
                "ensemble_edge": ensemble_edge,
                "ensemble_confidence": ensemble_confidence,
            }

            self.signal_history.append(signal_record)

            # Limit history size
            if len(self.signal_history) > self.max_history:
                self.signal_history.pop(0)

        except Exception as e:
            self.logger.error(f"Error storing signal: {e}")

    def train(
        self,
        training_data: List[
            Tuple[Dict[str, Tuple[float, float]], FeatureSnapshot, float]
        ],
    ) -> bool:
        """
        Train the meta-learner on historical data.

        Args:
            training_data: List of (alpha_signals, market_features, realized_return)

        Returns:
            True if training successful, False otherwise
        """
        try:
            if len(training_data) < 10:
                self.logger.warning("Insufficient training data")
                return False

            # Prepare training data
            X = []
            y = []

            for alpha_signals, market_features, realized_return in training_data:
                feature_vector = self._extract_features(alpha_signals, market_features)
                if feature_vector is not None:
                    X.append(feature_vector.flatten())
                    y.append(realized_return)

            if len(X) < 5:
                self.logger.warning("Insufficient valid training samples")
                return False

            X = np.array(X)
            y = np.array(y)

            # Fit scaler and model
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            self.model.fit(X_scaled, y)
            self.is_trained = True

            # Calculate training metrics
            train_score = self.model.score(X_scaled, y)

            self.logger.info(
                f"Meta-learner training completed: "
                f"samples={len(X)}, score={train_score:.3f}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error training meta-learner: {e}")
            return False

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained or not self.feature_names:
            return {}

        try:
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}

    def save_model(self, filepath: str) -> bool:
        """Save trained model to file."""
        try:
            if not self.is_trained:
                self.logger.warning("No trained model to save")
                return False

            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "alpha_model_names": self.alpha_model_names,
                "logistic_weights": self.logistic_weights,
            }

            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load trained model from file."""
        try:
            model_data = joblib.load(filepath)

            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data["feature_names"]
            self.alpha_model_names = model_data["alpha_model_names"]
            self.logistic_weights = model_data.get(
                "logistic_weights", {"w1": 1.0, "w2": 1.0}
            )
            self.is_trained = True

            self.logger.info(f"Model loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "prediction_count": self.prediction_count,
            "last_prediction_time": (
                self.last_prediction_time.isoformat()
                if self.last_prediction_time
                else None
            ),
            "alpha_model_names": self.alpha_model_names,
            "signal_history_size": len(self.signal_history),
            "logistic_weights": self.logistic_weights,
            "feature_importance": (
                self.get_feature_importance() if self.is_trained else {}
            ),
        }
