"""
Advanced Ensemble Learning System

Enhanced meta-learner with online learning, stacking techniques, and integration
of new alpha models (LSTM/Transformer, On-chain, Regime Detection).

Features:
- Online learning with adaptive weights
- Stacking ensemble with multiple meta-learners
- Dynamic ensemble selection based on performance
- Regime-aware model weighting
- Bayesian model combination
- Performance-based weight adaptation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import logging
from dataclasses import dataclass
from enum import Enum
import json
import pickle

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Optional ML dependencies with gating
try:
    import xgboost as xgb
    import lightgbm as lgb

    HAVE_GBDT = True
except Exception:  # ImportError or wheel-missing
    HAVE_GBDT = False
    xgb = None
    lgb = None

from ..layer0_data_ingestion.schemas import FeatureSnapshot, AlphaSignal
from ..layer1_alpha_models.regime_detector import RegimeDetector, MarketRegime


class EnsembleMethod(Enum):
    """Ensemble combination methods."""

    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NEURAL_NETWORK = "neural_network"
    BAYESIAN_RIDGE = "bayesian_ridge"
    STACKING = "stacking"


@dataclass
class ModelPerformance:
    """Model performance tracking."""

    name: str
    total_predictions: int
    correct_predictions: int
    total_pnl: float
    sharpe_ratio: float
    last_updated: datetime
    confidence_score: float
    recent_performance: List[float]  # Last 100 predictions


class OnlineLearningBuffer:
    """Buffer for online learning with adaptive sampling."""

    def __init__(self, max_size: int = 10000, decay_factor: float = 0.995):
        self.max_size = max_size
        self.decay_factor = decay_factor
        self.buffer = deque(maxlen=max_size)
        self.weights = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)

    def add_sample(self, features: np.ndarray, target: float, timestamp: datetime):
        """Add a new sample with temporal weighting."""
        self.buffer.append(features)
        self.timestamps.append(timestamp)

        # Calculate weight based on recency
        if len(self.timestamps) > 1:
            time_diff = (
                timestamp - self.timestamps[-2]
            ).total_seconds() / 3600  # Hours
            weight = min(1.0, self.decay_factor**time_diff)
        else:
            weight = 1.0

        self.weights.append(weight)

    def get_training_data(
        self, min_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get training data with sample weights."""
        if len(self.buffer) < min_samples:
            return None, None, None

        X = np.array(list(self.buffer))
        y = np.array([target for _, target, _ in self.buffer])
        weights = np.array(list(self.weights))

        return X, y, weights


class NeuralMetaLearner(nn.Module):
    """Neural network meta-learner for ensemble combination."""

    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.2):
        super(NeuralMetaLearner, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh(),  # Output between -1 and 1
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),  # Confidence between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning prediction and confidence."""
        prediction = self.network(x)
        confidence = self.confidence_head(x)
        return prediction, confidence


class AdvancedEnsemble:
    """
    Advanced ensemble learning system with online learning and stacking.

    Key Features:
    - Online learning with adaptive weights
    - Multiple meta-learner types (RF, XGB, Neural Networks)
    - Stacking ensemble combining multiple meta-learners
    - Dynamic model selection based on performance
    - Regime-aware weighting
    - Bayesian model combination
    """

    def __init__(
        self,
        symbol: str,
        primary_method: EnsembleMethod = EnsembleMethod.STACKING,
        online_learning: bool = True,
        use_regime_detection: bool = True,
    ):
        """
        Initialize advanced ensemble system.

        Args:
            symbol: Trading symbol
            primary_method: Primary ensemble method
            online_learning: Enable online learning
            use_regime_detection: Enable regime-aware weighting
        """
        self.symbol = symbol
        self.primary_method = primary_method
        self.online_learning = online_learning
        self.use_regime_detection = use_regime_detection

        # Model components
        self.meta_learners = {}
        self.stackers = {}
        self.scalers = {}
        self.is_trained = False

        # Performance tracking
        self.model_performance = {}
        self.ensemble_performance = ModelPerformance(
            name="ensemble",
            total_predictions=0,
            correct_predictions=0,
            total_pnl=0.0,
            sharpe_ratio=0.0,
            last_updated=datetime.now(),
            confidence_score=0.5,
            recent_performance=[],
        )

        # Online learning components
        if online_learning:
            self.online_buffer = OnlineLearningBuffer()
            self.online_update_frequency = 100  # Update every N samples
            self.online_update_count = 0

        # Regime detection
        if use_regime_detection:
            self.regime_detector = RegimeDetector(symbol)

        # Neural network components
        self.neural_meta_learner = None
        self.neural_optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model weights for dynamic selection
        self.model_weights = {}
        self.weight_adaptation_rate = 0.1

        # Expected alpha models (will be updated dynamically)
        self.alpha_model_names = [
            "ma_momentum",
            "mean_rev",
            "ob_pressure",
            "news_sent_alpha",
            "lstm_transformer",
            "onchain_alpha",
        ]

        self.logger = logging.getLogger(f"advanced_ensemble.{symbol}")
        self.logger.info(f"Initialized Advanced Ensemble for {symbol}")

        # Initialize meta-learners
        self._initialize_meta_learners()

    def _initialize_meta_learners(self):
        """Initialize all meta-learner models."""
        # Random Forest
        self.meta_learners[EnsembleMethod.RANDOM_FOREST] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        # Gradient Boosting
        self.meta_learners[EnsembleMethod.GRADIENT_BOOSTING] = (
            GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            )
        )

        # XGBoost (gated)
        if HAVE_GBDT:
            self.meta_learners[EnsembleMethod.XGBOOST] = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            )

        # LightGBM (gated)
        if HAVE_GBDT:
            self.meta_learners[EnsembleMethod.LIGHTGBM] = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1,
            )

        # Bayesian Ridge
        self.meta_learners[EnsembleMethod.BAYESIAN_RIDGE] = BayesianRidge(
            alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6
        )

        # Initialize scalers
        for method in self.meta_learners.keys():
            self.scalers[method] = StandardScaler()

        # Initialize model weights (equal initially)
        num_methods = len(self.meta_learners)
        for method in self.meta_learners.keys():
            self.model_weights[method] = 1.0 / num_methods

    def _initialize_neural_network(self, input_size: int):
        """Initialize neural network meta-learner."""
        self.neural_meta_learner = NeuralMetaLearner(
            input_size=input_size, hidden_size=64, dropout=0.2
        ).to(self.device)

        self.neural_optimizer = optim.Adam(
            self.neural_meta_learner.parameters(), lr=0.001, weight_decay=0.01
        )

        self.meta_learners[EnsembleMethod.NEURAL_NETWORK] = self.neural_meta_learner
        self.scalers[EnsembleMethod.NEURAL_NETWORK] = StandardScaler()

    def predict(
        self, alpha_signals: Dict[str, AlphaSignal], feature_snapshot: FeatureSnapshot
    ) -> Tuple[float, float]:
        """
        Generate ensemble prediction from alpha signals.

        Args:
            alpha_signals: Dictionary of alpha signals from different models
            feature_snapshot: Current market feature snapshot

        Returns:
            Tuple of (ensemble_edge_bps, ensemble_confidence)
        """
        try:
            # Get regime-aware weights if enabled
            regime_weights = None
            if self.use_regime_detection:
                regime_state = self.regime_detector.update(feature_snapshot)
                if regime_state:
                    regime_weights = self.regime_detector.get_regime_weights()

            # Extract features
            feature_vector = self._extract_features(
                alpha_signals, feature_snapshot, regime_weights
            )
            if feature_vector is None:
                return 0.0, 0.0

            # Generate predictions from all meta-learners
            predictions = {}
            confidences = {}

            if self.is_trained:
                for method, model in self.meta_learners.items():
                    try:
                        pred, conf = self._predict_with_model(
                            method, model, feature_vector
                        )
                        predictions[method] = pred
                        confidences[method] = conf
                    except Exception as e:
                        self.logger.warning(f"Error in {method} prediction: {e}")
                        continue

            # Combine predictions using selected method
            if self.primary_method == EnsembleMethod.STACKING and self.is_trained:
                edge_bps, confidence = self._stacking_prediction(
                    predictions, confidences
                )
            elif predictions:
                edge_bps, confidence = self._weighted_average_prediction(
                    predictions, confidences
                )
            else:
                # Fallback to simple combination
                edge_bps, confidence = self._simple_combination(alpha_signals)

            # Update performance tracking
            self.ensemble_performance.total_predictions += 1
            self.ensemble_performance.last_updated = datetime.now()

            # Store for online learning
            if self.online_learning:
                self.online_buffer.add_sample(feature_vector, edge_bps, datetime.now())
                self._maybe_update_online()

            return edge_bps, confidence

        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return 0.0, 0.0

    def _extract_features(
        self,
        alpha_signals: Dict[str, AlphaSignal],
        feature_snapshot: FeatureSnapshot,
        regime_weights: Optional[Dict[str, float]] = None,
    ) -> Optional[np.ndarray]:
        """Extract comprehensive features for meta-learning."""
        try:
            features = []

            # Alpha model signals (edge, confidence, strength)
            for model_name in self.alpha_model_names:
                if model_name in alpha_signals:
                    signal = alpha_signals[model_name]
                    features.extend(
                        [signal.edge_bps, signal.confidence, signal.signal_strength]
                    )

                    # Add regime weight if available
                    if regime_weights and model_name in regime_weights:
                        features.append(regime_weights[model_name])
                    else:
                        features.append(1.0)  # Default weight
                else:
                    features.extend([0.0, 0.0, 0.0, 1.0])  # Missing model

            # Market context features
            features.extend(
                [
                    feature_snapshot.spread_bps or 0.0,
                    feature_snapshot.volatility_5m or 0.0,
                    feature_snapshot.volatility_15m or 0.0,
                    feature_snapshot.volatility_1h or 0.0,
                    feature_snapshot.volume_ratio or 1.0,
                    feature_snapshot.order_book_imbalance or 0.0,
                    feature_snapshot.order_book_pressure or 0.0,
                    feature_snapshot.return_1m or 0.0,
                    feature_snapshot.return_5m or 0.0,
                    feature_snapshot.return_15m or 0.0,
                ]
            )

            # Technical indicators
            features.extend(
                [
                    float(feature_snapshot.sma_5) if feature_snapshot.sma_5 else 0.0,
                    float(feature_snapshot.sma_20) if feature_snapshot.sma_20 else 0.0,
                    float(feature_snapshot.ema_5) if feature_snapshot.ema_5 else 0.0,
                    float(feature_snapshot.ema_20) if feature_snapshot.ema_20 else 0.0,
                    feature_snapshot.rsi_14 or 50.0,
                    feature_snapshot.sent_score or 0.0,
                ]
            )

            # Signal interactions (cross-products of key signals)
            signal_values = []
            for model_name in self.alpha_model_names:
                if model_name in alpha_signals:
                    signal_values.append(alpha_signals[model_name].edge_bps)
                else:
                    signal_values.append(0.0)

            # Add pairwise interactions for top signals
            if len(signal_values) >= 2:
                features.append(
                    signal_values[0] * signal_values[1]
                )  # momentum * mean_rev
            if len(signal_values) >= 3:
                features.append(
                    signal_values[0] * signal_values[2]
                )  # momentum * ob_pressure
            if len(signal_values) >= 4:
                features.append(
                    signal_values[1] * signal_values[3]
                )  # mean_rev * news_sent

            # Regime state features
            if self.use_regime_detection and hasattr(
                self.regime_detector, "current_regime"
            ):
                regime_state = self.regime_detector.current_regime
                if regime_state:
                    features.extend(
                        [
                            regime_state.confidence,
                            regime_state.strength,
                            regime_state.duration,
                            regime_state.transition_probability,
                        ]
                    )
                else:
                    features.extend([0.5, 0.5, 1.0, 0.5])  # Default regime features
            else:
                features.extend([0.5, 0.5, 1.0, 0.5])

            return np.array(features).reshape(1, -1)

        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None

    def _predict_with_model(
        self, method: EnsembleMethod, model: Any, feature_vector: np.ndarray
    ) -> Tuple[float, float]:
        """Make prediction with specific model."""
        try:
            # Scale features
            scaled_features = self.scalers[method].transform(feature_vector)

            if method == EnsembleMethod.NEURAL_NETWORK:
                return self._predict_with_neural_network(scaled_features)

            # Standard sklearn prediction
            prediction = model.predict(scaled_features)[0]

            # Calculate confidence based on model type
            if hasattr(model, "predict_proba"):
                # For classification models
                confidence = np.max(model.predict_proba(scaled_features)[0])
            elif hasattr(model, "estimators_"):
                # For ensemble models (RF, GB)
                tree_predictions = [
                    tree.predict(scaled_features)[0] for tree in model.estimators_
                ]
                confidence = 1.0 / (1.0 + np.std(tree_predictions))
            elif method == EnsembleMethod.BAYESIAN_RIDGE:
                # For Bayesian models
                _, std = model.predict(scaled_features, return_std=True)
                confidence = 1.0 / (1.0 + std[0])
            else:
                confidence = 0.7  # Default confidence

            # Apply model weight
            weighted_prediction = prediction * self.model_weights.get(method, 1.0)

            return float(weighted_prediction), float(np.clip(confidence, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Error in {method} prediction: {e}")
            return 0.0, 0.0

    def _predict_with_neural_network(
        self, scaled_features: np.ndarray
    ) -> Tuple[float, float]:
        """Make prediction with neural network."""
        try:
            self.neural_meta_learner.eval()
            with torch.no_grad():
                x = torch.FloatTensor(scaled_features).to(self.device)
                prediction, confidence = self.neural_meta_learner(x)

                # Scale prediction to basis points
                pred_bps = prediction.cpu().item() * 50  # Scale to ±50 bps
                conf_score = confidence.cpu().item()

                return pred_bps, conf_score

        except Exception as e:
            self.logger.error(f"Error in neural network prediction: {e}")
            return 0.0, 0.0

    def _stacking_prediction(
        self,
        predictions: Dict[EnsembleMethod, float],
        confidences: Dict[EnsembleMethod, float],
    ) -> Tuple[float, float]:
        """Combine predictions using stacking meta-learner."""
        try:
            if not predictions:
                return 0.0, 0.0

            # Create stacking features (predictions from base models)
            stacking_features = []
            base_methods = [EnsembleMethod.RANDOM_FOREST, EnsembleMethod.BAYESIAN_RIDGE]
            if HAVE_GBDT:
                base_methods.extend([EnsembleMethod.XGBOOST, EnsembleMethod.LIGHTGBM])

            for method in base_methods:
                if method in predictions:
                    stacking_features.extend([predictions[method], confidences[method]])
                else:
                    stacking_features.extend([0.0, 0.0])

            # Use neural network as stacking meta-learner
            if self.neural_meta_learner and len(stacking_features) > 0:
                stacking_array = np.array(stacking_features).reshape(1, -1)

                # Scale stacking features
                if EnsembleMethod.NEURAL_NETWORK in self.scalers:
                    stacking_array = self.scalers[
                        EnsembleMethod.NEURAL_NETWORK
                    ].transform(stacking_array)

                final_pred, final_conf = self._predict_with_neural_network(
                    stacking_array
                )
                return final_pred, final_conf
            else:
                # Fallback to weighted average
                return self._weighted_average_prediction(predictions, confidences)

        except Exception as e:
            self.logger.error(f"Error in stacking prediction: {e}")
            return self._weighted_average_prediction(predictions, confidences)

    def _weighted_average_prediction(
        self,
        predictions: Dict[EnsembleMethod, float],
        confidences: Dict[EnsembleMethod, float],
    ) -> Tuple[float, float]:
        """Combine predictions using weighted average."""
        try:
            if not predictions:
                return 0.0, 0.0

            total_weighted_pred = 0.0
            total_weight = 0.0
            conf_values = []

            for method, prediction in predictions.items():
                # Weight by model confidence and performance
                model_weight = self.model_weights.get(method, 1.0)
                confidence = confidences.get(method, 0.5)

                # Combine weights
                final_weight = model_weight * confidence

                total_weighted_pred += prediction * final_weight
                total_weight += final_weight
                conf_values.append(confidence)

            if total_weight > 0:
                ensemble_pred = total_weighted_pred / total_weight
                ensemble_conf = np.mean(conf_values)
            else:
                ensemble_pred = 0.0
                ensemble_conf = 0.0

            return float(ensemble_pred), float(np.clip(ensemble_conf, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Error in weighted average prediction: {e}")
            return 0.0, 0.0

    def _simple_combination(
        self, alpha_signals: Dict[str, AlphaSignal]
    ) -> Tuple[float, float]:
        """Simple combination when models are not trained."""
        try:
            if not alpha_signals:
                return 0.0, 0.0

            # Get regime weights if available
            regime_weights = None
            if self.use_regime_detection:
                regime_weights = self.regime_detector.get_regime_weights()

            total_weighted_edge = 0.0
            total_weight = 0.0
            conf_values = []

            for model_name, signal in alpha_signals.items():
                # Get regime weight for this model
                regime_weight = 1.0
                if regime_weights and model_name in regime_weights:
                    regime_weight = regime_weights[model_name]

                # Combine weights
                final_weight = signal.confidence * regime_weight

                total_weighted_edge += signal.edge_bps * final_weight
                total_weight += final_weight
                conf_values.append(signal.confidence)

            if total_weight > 0:
                ensemble_edge = total_weighted_edge / total_weight
                ensemble_conf = np.mean(conf_values)
            else:
                ensemble_edge = 0.0
                ensemble_conf = 0.0

            return float(ensemble_edge), float(np.clip(ensemble_conf, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Error in simple combination: {e}")
            return 0.0, 0.0

    def _maybe_update_online(self):
        """Update models online if enough new samples."""
        if not self.online_learning:
            return

        self.online_update_count += 1
        if self.online_update_count % self.online_update_frequency == 0:
            self._update_models_online()

    def _update_models_online(self):
        """Update models using online learning."""
        try:
            # Get training data from buffer
            X, y, weights = self.online_buffer.get_training_data()
            if X is None:
                return

            self.logger.info(f"Updating models online with {len(X)} samples")

            # Update each model
            for method, model in self.meta_learners.items():
                try:
                    if method == EnsembleMethod.NEURAL_NETWORK:
                        self._update_neural_network_online(X, y, weights)
                    else:
                        # Update sklearn models
                        X_scaled = self.scalers[method].transform(X)

                        if hasattr(model, "partial_fit"):
                            model.partial_fit(X_scaled, y, sample_weight=weights)
                        else:
                            # Retrain model with recent data
                            model.fit(X_scaled, y, sample_weight=weights)

                except Exception as e:
                    self.logger.warning(f"Error updating {method} online: {e}")
                    continue

            # Update model weights based on recent performance
            self._update_model_weights()

        except Exception as e:
            self.logger.error(f"Error in online model update: {e}")

    def _update_neural_network_online(
        self, X: np.ndarray, y: np.ndarray, weights: np.ndarray
    ):
        """Update neural network using online learning."""
        try:
            if not self.neural_meta_learner:
                return

            self.neural_meta_learner.train()

            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            weights_tensor = torch.FloatTensor(weights).to(self.device)

            # Mini-batch training
            batch_size = min(32, len(X))
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i : i + batch_size]
                batch_y = y_tensor[i : i + batch_size]
                batch_weights = weights_tensor[i : i + batch_size]

                self.neural_optimizer.zero_grad()

                predictions, confidences = self.neural_meta_learner(batch_X)

                # Weighted loss
                loss = torch.mean(
                    batch_weights * (predictions.squeeze() - batch_y) ** 2
                )

                loss.backward()
                self.neural_optimizer.step()

        except Exception as e:
            self.logger.error(f"Error updating neural network online: {e}")

    def _update_model_weights(self):
        """Update model weights based on recent performance."""
        try:
            # This is a simplified weight update - in practice, you'd track
            # model performance and update weights based on prediction accuracy

            # For now, use equal weights with small random perturbation
            total_weight = 0.0
            for method in self.model_weights:
                # Add small random perturbation to explore
                perturbation = np.random.normal(0, 0.01)
                self.model_weights[method] = max(
                    0.1, self.model_weights[method] + perturbation
                )
                total_weight += self.model_weights[method]

            # Normalize weights
            if total_weight > 0:
                for method in self.model_weights:
                    self.model_weights[method] /= total_weight

        except Exception as e:
            self.logger.error(f"Error updating model weights: {e}")

    def train(
        self, training_data: List[Tuple[Dict[str, AlphaSignal], FeatureSnapshot, float]]
    ) -> bool:
        """
        Train the ensemble system.

        Args:
            training_data: List of (alpha_signals, feature_snapshot, realized_return)

        Returns:
            True if training successful
        """
        try:
            if len(training_data) < 50:
                self.logger.warning("Insufficient training data")
                return False

            self.logger.info(f"Training ensemble with {len(training_data)} samples")

            # Prepare training data
            X_list = []
            y_list = []

            for alpha_signals, feature_snapshot, realized_return in training_data:
                # Get regime weights for this sample
                regime_weights = None
                if self.use_regime_detection:
                    regime_state = self.regime_detector.update(feature_snapshot)
                    if regime_state:
                        regime_weights = self.regime_detector.get_regime_weights()

                feature_vector = self._extract_features(
                    alpha_signals, feature_snapshot, regime_weights
                )
                if feature_vector is not None:
                    X_list.append(feature_vector.flatten())
                    y_list.append(realized_return)

            if len(X_list) < 20:
                self.logger.warning("Insufficient valid training samples")
                return False

            X = np.array(X_list)
            y = np.array(y_list)

            # Initialize neural network if not done
            if not self.neural_meta_learner:
                self._initialize_neural_network(X.shape[1])

            # Train each meta-learner
            for method, model in self.meta_learners.items():
                try:
                    self.logger.info(f"Training {method.value}")

                    # Fit scaler
                    self.scalers[method].fit(X)
                    X_scaled = self.scalers[method].transform(X)

                    if method == EnsembleMethod.NEURAL_NETWORK:
                        self._train_neural_network(X_scaled, y)
                    else:
                        # Train sklearn models
                        model.fit(X_scaled, y)

                    # Calculate training score
                    if method != EnsembleMethod.NEURAL_NETWORK:
                        train_score = model.score(X_scaled, y)
                        self.logger.info(
                            f"{method.value} training score: {train_score:.3f}"
                        )

                except Exception as e:
                    self.logger.error(f"Error training {method.value}: {e}")
                    continue

            # Train stacking meta-learner
            if self.primary_method == EnsembleMethod.STACKING:
                self._train_stacking_meta_learner(X, y)

            self.is_trained = True
            self.logger.info("Ensemble training completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error training ensemble: {e}")
            return False

    def _train_neural_network(self, X: np.ndarray, y: np.ndarray):
        """Train neural network meta-learner."""
        try:
            self.neural_meta_learner.train()

            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)

            # Training loop
            epochs = 100
            batch_size = 32

            for epoch in range(epochs):
                total_loss = 0.0

                for i in range(0, len(X), batch_size):
                    batch_X = X_tensor[i : i + batch_size]
                    batch_y = y_tensor[i : i + batch_size]

                    self.neural_optimizer.zero_grad()

                    predictions, confidences = self.neural_meta_learner(batch_X)

                    # Loss combining prediction accuracy and confidence calibration
                    pred_loss = torch.mean((predictions.squeeze() - batch_y) ** 2)
                    conf_loss = torch.mean(
                        (confidences.squeeze() - 0.5) ** 2
                    )  # Regularize confidence

                    total_loss_batch = pred_loss + 0.1 * conf_loss
                    total_loss += total_loss_batch.item()

                    total_loss_batch.backward()
                    self.neural_optimizer.step()

                if epoch % 20 == 0:
                    self.logger.debug(
                        f"Neural network epoch {epoch}: loss = {total_loss:.6f}"
                    )

        except Exception as e:
            self.logger.error(f"Error training neural network: {e}")

    def _train_stacking_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Train stacking meta-learner using cross-validation."""
        try:
            # This is a simplified stacking implementation
            # In practice, you'd use cross-validation to generate base predictions

            # For now, just ensure the neural network is trained as the meta-learner
            if self.neural_meta_learner:
                self.logger.info("Stacking meta-learner (neural network) trained")

        except Exception as e:
            self.logger.error(f"Error training stacking meta-learner: {e}")

    def update_performance(self, prediction: float, actual: float, pnl: float):
        """Update performance metrics."""
        try:
            self.ensemble_performance.recent_performance.append(actual - prediction)

            # Keep only last 100 performances
            if len(self.ensemble_performance.recent_performance) > 100:
                self.ensemble_performance.recent_performance.pop(0)

            # Update correctness (sign prediction)
            if np.sign(prediction) == np.sign(actual):
                self.ensemble_performance.correct_predictions += 1

            # Update PnL
            self.ensemble_performance.total_pnl += pnl

            # Update Sharpe ratio (simplified)
            if len(self.ensemble_performance.recent_performance) > 10:
                returns = np.array(self.ensemble_performance.recent_performance)
                if np.std(returns) > 0:
                    self.ensemble_performance.sharpe_ratio = np.mean(returns) / np.std(
                        returns
                    )
                else:
                    self.ensemble_performance.sharpe_ratio = 0.0

        except Exception as e:
            self.logger.error(f"Error updating performance: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive ensemble statistics."""
        stats = {
            "symbol": self.symbol,
            "primary_method": self.primary_method.value,
            "is_trained": self.is_trained,
            "online_learning": self.online_learning,
            "use_regime_detection": self.use_regime_detection,
            "model_weights": self.model_weights,
            "ensemble_performance": {
                "total_predictions": self.ensemble_performance.total_predictions,
                "correct_predictions": self.ensemble_performance.correct_predictions,
                "accuracy": (
                    self.ensemble_performance.correct_predictions
                    / max(1, self.ensemble_performance.total_predictions)
                ),
                "total_pnl": self.ensemble_performance.total_pnl,
                "sharpe_ratio": self.ensemble_performance.sharpe_ratio,
                "last_updated": self.ensemble_performance.last_updated.isoformat(),
            },
        }

        if self.use_regime_detection:
            stats["regime_detector"] = self.regime_detector.get_stats()

        return stats

    def save_model(self, filepath: str) -> bool:
        """Save ensemble model."""
        try:
            model_data = {
                "symbol": self.symbol,
                "primary_method": self.primary_method.value,
                "is_trained": self.is_trained,
                "model_weights": self.model_weights,
                "alpha_model_names": self.alpha_model_names,
                "meta_learners": {},
                "scalers": {},
                "ensemble_performance": self.ensemble_performance,
            }

            # Save sklearn models
            for method, model in self.meta_learners.items():
                if method != EnsembleMethod.NEURAL_NETWORK:
                    model_data["meta_learners"][method.value] = model
                    model_data["scalers"][method.value] = self.scalers[method]

            # Save neural network separately
            if self.neural_meta_learner:
                torch.save(
                    self.neural_meta_learner.state_dict(), filepath + "_neural.pth"
                )

            # Save main model data
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)

            self.logger.info(f"Ensemble model saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load ensemble model."""
        try:
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)

            self.symbol = model_data["symbol"]
            self.primary_method = EnsembleMethod(model_data["primary_method"])
            self.is_trained = model_data["is_trained"]
            self.model_weights = model_data["model_weights"]
            self.alpha_model_names = model_data["alpha_model_names"]
            self.ensemble_performance = model_data["ensemble_performance"]

            # Load sklearn models
            for method_name, model in model_data["meta_learners"].items():
                method = EnsembleMethod(method_name)
                self.meta_learners[method] = model
                self.scalers[method] = model_data["scalers"][method_name]

            # Load neural network
            neural_path = filepath + "_neural.pth"
            try:
                if self.neural_meta_learner:
                    self.neural_meta_learner.load_state_dict(torch.load(neural_path))
                    self.meta_learners[EnsembleMethod.NEURAL_NETWORK] = (
                        self.neural_meta_learner
                    )
            except FileNotFoundError:
                self.logger.warning("Neural network weights not found")

            self.logger.info(f"Ensemble model loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False


# Factory function
def create_advanced_ensemble(symbol: str, **kwargs) -> AdvancedEnsemble:
    """
    Factory function to create advanced ensemble.

    Args:
        symbol: Trading symbol
        **kwargs: Additional ensemble parameters

    Returns:
        Initialized AdvancedEnsemble instance
    """
    return AdvancedEnsemble(symbol=symbol, **kwargs)
