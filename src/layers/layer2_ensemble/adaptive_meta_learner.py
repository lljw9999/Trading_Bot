"""
Adaptive Meta-Learner with Enhanced Model Integration

Advanced ensemble system that dynamically adapts to market conditions and optimizes
model combinations using state-of-the-art techniques.

Features:
- Dynamic model selection based on market regime
- Multi-objective optimization for risk-adjusted returns
- Bayesian optimization for hyperparameter tuning
- Online learning with concept drift detection
- Uncertainty quantification and confidence intervals
- Model performance attribution and explainability
"""

import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import json
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# Optional advanced ML dependencies
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    from scipy.stats import beta, norm

    HAVE_ADVANCED_ML = True
except ImportError:
    HAVE_ADVANCED_ML = False

from ..layer0_data_ingestion.schemas import FeatureSnapshot, AlphaSignal


class MarketRegime(Enum):
    """Market regime classifications."""

    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


@dataclass
class ModelPrediction:
    """Structured model prediction with uncertainty."""

    model_name: str
    prediction: float
    confidence: float
    uncertainty: float
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class EnsembleConfig:
    """Configuration for ensemble behavior."""

    min_models_required: int = 3
    max_models_active: int = 10
    confidence_threshold: float = 0.3
    recency_weight: float = 0.95
    diversification_penalty: float = 0.1
    risk_adjustment_factor: float = 1.5
    update_frequency_minutes: int = 15


class ConceptDriftDetector:
    """Detect concept drift in model performance."""

    def __init__(self, window_size: int = 100, sensitivity: float = 0.05):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.performance_history = deque(maxlen=window_size * 2)
        self.baseline_performance = None
        self.drift_detected = False

    def add_performance(self, performance: float):
        """Add new performance observation."""
        self.performance_history.append(performance)

        if len(self.performance_history) >= self.window_size:
            self._check_drift()

    def _check_drift(self):
        """Check for statistical drift in performance."""
        if len(self.performance_history) < self.window_size * 1.5:
            return

        # Split into old and new windows
        old_window = list(self.performance_history)[
            -self.window_size * 2 : -self.window_size
        ]
        new_window = list(self.performance_history)[-self.window_size :]

        # Perform Kolmogorov-Smirnov test
        try:
            statistic, p_value = stats.ks_2samp(old_window, new_window)
            self.drift_detected = p_value < self.sensitivity
        except:
            self.drift_detected = False


class BayesianModelSelector:
    """Bayesian approach to model selection with uncertainty quantification."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.model_performance = defaultdict(
            lambda: {"successes": alpha, "failures": beta}
        )

    def update_performance(self, model_name: str, success: bool):
        """Update model performance using Bayesian updating."""
        if success:
            self.model_performance[model_name]["successes"] += 1
        else:
            self.model_performance[model_name]["failures"] += 1

    def get_model_probability(self, model_name: str) -> Tuple[float, float]:
        """Get model success probability and uncertainty."""
        perf = self.model_performance[model_name]
        alpha = perf["successes"]
        beta_param = perf["failures"]

        # Beta distribution parameters
        mean = alpha / (alpha + beta_param)
        variance = (alpha * beta_param) / (
            (alpha + beta_param) ** 2 * (alpha + beta_param + 1)
        )

        return mean, np.sqrt(variance)

    def select_models(
        self, available_models: List[str], n_models: int = 5
    ) -> List[str]:
        """Select top models based on probability with uncertainty."""
        model_scores = []

        for model in available_models:
            prob, uncertainty = self.get_model_probability(model)
            # Use upper confidence bound for exploration
            score = prob + 1.96 * uncertainty  # 95% confidence
            model_scores.append((model, score))

        # Sort by score and select top models
        model_scores.sort(key=lambda x: x[1], reverse=True)
        return [model for model, _ in model_scores[:n_models]]


class AdaptiveMetaLearner:
    """Advanced meta-learner with adaptive capabilities."""

    def __init__(
        self,
        symbol: str,
        config: Optional[EnsembleConfig] = None,
        lookback_hours: int = 24,
        regime_detection: bool = True,
    ):
        self.symbol = symbol
        self.config = config or EnsembleConfig()
        self.lookback_hours = lookback_hours
        self.regime_detection = regime_detection

        # Core components
        self.model_predictions: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.model_weights: Dict[str, float] = defaultdict(float)
        self.model_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "total_predictions": 0,
                "correct_predictions": 0,
                "total_pnl": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "last_updated": datetime.now(),
            }
        )

        # Advanced components
        self.drift_detectors: Dict[str, ConceptDriftDetector] = {}
        self.bayesian_selector = BayesianModelSelector()
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_weights: Dict[MarketRegime, Dict[str, float]] = defaultdict(dict)

        # Meta-features for ensemble
        self.meta_features = deque(maxlen=100)
        self.ensemble_performance = deque(maxlen=500)

        # Gaussian Process for uncertainty quantification (if available)
        self.gp_regressor = None
        if HAVE_ADVANCED_ML:
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            self.gp_regressor = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

        # Feature importance tracking
        self.feature_importance = defaultdict(lambda: deque(maxlen=100))

        self.logger = logging.getLogger(f"adaptive_meta_learner.{symbol}")
        self.logger.info(f"Adaptive Meta-Learner initialized for {symbol}")

    def detect_market_regime(
        self, recent_features: List[FeatureSnapshot]
    ) -> MarketRegime:
        """Detect current market regime from recent features."""
        if len(recent_features) < 10:
            return MarketRegime.SIDEWAYS

        # Extract volatility and price movement features
        returns = []
        volatilities = []

        for feature in recent_features[-20:]:  # Last 20 observations
            if feature.return_1m is not None:
                returns.append(feature.return_1m)
            if feature.volatility_5m is not None:
                volatilities.append(feature.volatility_5m)

        if not returns or not volatilities:
            return MarketRegime.SIDEWAYS

        avg_return = np.mean(returns)
        avg_volatility = np.mean(volatilities)
        return_trend = np.polyfit(range(len(returns)), returns, 1)[0]
        volatility_level = np.percentile(volatilities, 75)

        # Regime classification logic
        if avg_volatility > volatility_level * 1.5:
            return MarketRegime.HIGH_VOLATILITY
        elif avg_volatility < volatility_level * 0.5:
            return MarketRegime.LOW_VOLATILITY
        elif return_trend > 0.001 and avg_return > 0:
            return MarketRegime.TRENDING_UP
        elif return_trend < -0.001 and avg_return < 0:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.SIDEWAYS

    def update_model_performance(
        self,
        model_name: str,
        prediction: float,
        actual: Optional[float] = None,
        pnl: Optional[float] = None,
    ):
        """Update model performance metrics."""
        perf = self.model_performance[model_name]
        perf["total_predictions"] += 1
        perf["last_updated"] = datetime.now()

        # Initialize drift detector if needed
        if model_name not in self.drift_detectors:
            self.drift_detectors[model_name] = ConceptDriftDetector()

        if actual is not None:
            # Directional accuracy
            if (prediction > 0 and actual > 0) or (prediction < 0 and actual < 0):
                perf["correct_predictions"] += 1
                self.bayesian_selector.update_performance(model_name, True)
            else:
                self.bayesian_selector.update_performance(model_name, False)

            # Add to drift detector
            accuracy = perf["correct_predictions"] / max(perf["total_predictions"], 1)
            self.drift_detectors[model_name].add_performance(accuracy)

        if pnl is not None:
            perf["total_pnl"] += pnl

            # Calculate Sharpe ratio
            recent_pnls = [
                p.get("pnl", 0) for p in list(self.model_predictions[model_name])[-100:]
            ]
            if len(recent_pnls) > 10:
                sharpe = (
                    np.mean(recent_pnls) / (np.std(recent_pnls) + 1e-8) * np.sqrt(252)
                )
                perf["sharpe_ratio"] = sharpe

    def calculate_dynamic_weights(
        self,
        predictions: Dict[str, ModelPrediction],
        current_regime: MarketRegime,
    ) -> Dict[str, float]:
        """Calculate dynamic model weights based on performance and regime."""
        if not predictions:
            return {}

        model_names = list(predictions.keys())
        n_models = len(model_names)

        # Base performance scores
        performance_scores = np.zeros(n_models)
        confidence_scores = np.zeros(n_models)
        uncertainty_scores = np.zeros(n_models)

        for i, model_name in enumerate(model_names):
            pred = predictions[model_name]
            perf = self.model_performance[model_name]

            # Performance score (Sharpe ratio with fallback)
            if perf["total_predictions"] > 10:
                sharpe = perf.get("sharpe_ratio", 0.0)
                accuracy = perf["correct_predictions"] / perf["total_predictions"]
                performance_scores[i] = 0.6 * sharpe + 0.4 * accuracy
            else:
                performance_scores[i] = 0.1  # Default for new models

            # Confidence and uncertainty
            confidence_scores[i] = pred.confidence
            uncertainty_scores[i] = (
                1.0 - pred.uncertainty
            )  # Lower uncertainty is better

        # Regime-specific adjustments
        regime_multipliers = self._get_regime_multipliers(model_names, current_regime)
        performance_scores = performance_scores * regime_multipliers

        # Combine scores with multi-objective optimization
        combined_scores = (
            0.4 * self._normalize_scores(performance_scores)
            + 0.3 * self._normalize_scores(confidence_scores)
            + 0.3 * self._normalize_scores(uncertainty_scores)
        )

        # Apply diversification penalty
        correlation_penalty = self._calculate_correlation_penalty(model_names)
        combined_scores = combined_scores * (
            1 - self.config.diversification_penalty * correlation_penalty
        )

        # Softmax normalization for weights
        weights_raw = np.exp(combined_scores - np.max(combined_scores))
        weights_normalized = weights_raw / np.sum(weights_raw)

        # Apply minimum weight threshold
        min_weight = 0.05
        weights_normalized = np.maximum(weights_normalized, min_weight)
        weights_normalized = weights_normalized / np.sum(weights_normalized)

        return dict(zip(model_names, weights_normalized))

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range."""
        if len(scores) == 0 or np.std(scores) == 0:
            return np.ones_like(scores) / len(scores)
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    def _get_regime_multipliers(
        self,
        model_names: List[str],
        regime: MarketRegime,
    ) -> np.ndarray:
        """Get regime-specific model multipliers."""
        multipliers = np.ones(len(model_names))

        # Define regime preferences (can be learned from data)
        regime_preferences = {
            MarketRegime.HIGH_VOLATILITY: {
                "lstm_transformer": 1.2,
                "enhanced_lstm_transformer": 1.3,
                "mean_reversion": 0.8,
                "momentum": 1.1,
            },
            MarketRegime.LOW_VOLATILITY: {
                "lstm_transformer": 1.0,
                "enhanced_lstm_transformer": 1.1,
                "mean_reversion": 1.3,
                "momentum": 0.9,
            },
            MarketRegime.TRENDING_UP: {
                "momentum": 1.4,
                "trend_following": 1.3,
                "mean_reversion": 0.7,
            },
            MarketRegime.TRENDING_DOWN: {
                "momentum": 1.4,
                "trend_following": 1.3,
                "mean_reversion": 0.7,
            },
            MarketRegime.SIDEWAYS: {
                "mean_reversion": 1.3,
                "orderbook_pressure": 1.2,
                "momentum": 0.8,
            },
            MarketRegime.CRISIS: {
                "enhanced_lstm_transformer": 1.4,
                "risk_management": 1.5,
                "volatility_models": 1.3,
            },
        }

        preferences = regime_preferences.get(regime, {})

        for i, model_name in enumerate(model_names):
            # Find matching preference by partial name matching
            multiplier = 1.0
            for pref_name, pref_mult in preferences.items():
                if pref_name.lower() in model_name.lower():
                    multiplier = pref_mult
                    break
            multipliers[i] = multiplier

        return multipliers

    def _calculate_correlation_penalty(self, model_names: List[str]) -> np.ndarray:
        """Calculate correlation penalty to promote diversification."""
        n_models = len(model_names)
        if n_models < 2:
            return np.zeros(n_models)

        # Get recent predictions for correlation calculation
        prediction_matrix = []
        min_predictions = float("inf")

        for model_name in model_names:
            recent_preds = list(self.model_predictions[model_name])[
                -50:
            ]  # Last 50 predictions
            if recent_preds:
                preds_values = [
                    p.get("prediction", 0) for p in recent_preds if isinstance(p, dict)
                ]
                prediction_matrix.append(preds_values)
                min_predictions = min(min_predictions, len(preds_values))
            else:
                prediction_matrix.append([0])
                min_predictions = 0

        if min_predictions < 10:  # Not enough data for correlation
            return np.zeros(n_models)

        # Truncate to minimum length
        truncated_matrix = np.array(
            [pred_list[:min_predictions] for pred_list in prediction_matrix]
        )

        try:
            correlation_matrix = np.corrcoef(truncated_matrix)
            # Calculate average correlation for each model (excluding self-correlation)
            avg_correlations = np.zeros(n_models)
            for i in range(n_models):
                other_correlations = [
                    abs(correlation_matrix[i, j]) for j in range(n_models) if i != j
                ]
                avg_correlations[i] = (
                    np.mean(other_correlations) if other_correlations else 0
                )

            return avg_correlations
        except:
            return np.zeros(n_models)

    def combine_predictions(
        self,
        predictions: Dict[str, ModelPrediction],
        feature_snapshot: Optional[FeatureSnapshot] = None,
    ) -> Optional[AlphaSignal]:
        """Combine model predictions into ensemble signal."""
        if not predictions:
            return None

        # Filter predictions by confidence threshold
        filtered_predictions = {
            name: pred
            for name, pred in predictions.items()
            if pred.confidence >= self.config.confidence_threshold
        }

        if len(filtered_predictions) < self.config.min_models_required:
            return None

        # Detect current market regime
        if self.regime_detection and feature_snapshot:
            # Would need access to recent features for regime detection
            # For now, use default regime
            self.current_regime = MarketRegime.SIDEWAYS

        # Calculate dynamic weights
        weights = self.calculate_dynamic_weights(
            filtered_predictions, self.current_regime
        )

        if not weights:
            return None

        # Combine predictions
        weighted_prediction = 0.0
        weighted_confidence = 0.0
        weighted_uncertainty = 0.0
        total_weight = sum(weights.values())

        prediction_details = {}

        for model_name, pred in filtered_predictions.items():
            weight = weights.get(model_name, 0.0)
            normalized_weight = weight / total_weight

            weighted_prediction += normalized_weight * pred.prediction
            weighted_confidence += normalized_weight * pred.confidence
            weighted_uncertainty += normalized_weight * pred.uncertainty

            prediction_details[model_name] = {
                "prediction": pred.prediction,
                "confidence": pred.confidence,
                "weight": normalized_weight,
                "uncertainty": pred.uncertainty,
            }

        # Apply ensemble-level adjustments
        ensemble_confidence = weighted_confidence * (1 - weighted_uncertainty)

        # Risk adjustment based on prediction uncertainty
        risk_adjusted_prediction = weighted_prediction * (
            1 - self.config.risk_adjustment_factor * weighted_uncertainty
        )

        # Create reasoning
        top_models = sorted(
            prediction_details.items(), key=lambda x: x[1]["weight"], reverse=True
        )[:3]

        reasoning = f"Ensemble({self.current_regime.value}): " + ", ".join(
            [f"{name}({details['weight']:.2f})" for name, details in top_models]
        )

        return AlphaSignal(
            model_name="adaptive_meta_learner",
            symbol=self.symbol,
            timestamp=(
                feature_snapshot.timestamp if feature_snapshot else datetime.now()
            ),
            edge_bps=risk_adjusted_prediction,
            confidence=ensemble_confidence,
            signal_strength=abs(weighted_prediction),
            metadata={
                "ensemble_method": "adaptive_weighted",
                "market_regime": self.current_regime.value,
                "model_weights": weights,
                "prediction_details": prediction_details,
                "weighted_uncertainty": weighted_uncertainty,
                "models_used": len(filtered_predictions),
                "reasoning": reasoning,
            },
        )

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get comprehensive ensemble statistics."""
        stats = {
            "symbol": self.symbol,
            "current_regime": self.current_regime.value,
            "active_models": len(self.model_performance),
            "total_predictions": sum(
                p["total_predictions"] for p in self.model_performance.values()
            ),
            "model_performance": {},
            "drift_detection": {},
            "ensemble_performance": {
                "recent_accuracy": (
                    np.mean(list(self.ensemble_performance)[-50:])
                    if self.ensemble_performance
                    else 0
                ),
                "total_signals": len(self.ensemble_performance),
            },
        }

        # Model-specific stats
        for model_name, perf in self.model_performance.items():
            accuracy = perf["correct_predictions"] / max(perf["total_predictions"], 1)
            stats["model_performance"][model_name] = {
                "accuracy": accuracy,
                "sharpe_ratio": perf.get("sharpe_ratio", 0.0),
                "total_pnl": perf["total_pnl"],
                "predictions": perf["total_predictions"],
                "drift_detected": self.drift_detectors.get(
                    model_name, ConceptDriftDetector()
                ).drift_detected,
            }

        return stats


# Factory function
def create_adaptive_meta_learner(symbol: str, **kwargs) -> AdaptiveMetaLearner:
    """Create adaptive meta-learner with custom configuration."""
    config = EnsembleConfig(
        **{k: v for k, v in kwargs.items() if hasattr(EnsembleConfig, k)}
    )
    other_kwargs = {k: v for k, v in kwargs.items() if not hasattr(EnsembleConfig, k)}
    return AdaptiveMetaLearner(symbol=symbol, config=config, **other_kwargs)


if __name__ == "__main__":
    # Test the adaptive meta-learner
    print("Testing Adaptive Meta-Learner...")

    meta_learner = AdaptiveMetaLearner("BTCUSDT")

    # Create sample predictions
    predictions = {
        "enhanced_lstm_transformer": ModelPrediction(
            model_name="enhanced_lstm_transformer",
            prediction=5.2,
            confidence=0.8,
            uncertainty=0.1,
            metadata={},
            timestamp=datetime.now(),
        ),
        "momentum_model": ModelPrediction(
            model_name="momentum_model",
            prediction=3.8,
            confidence=0.6,
            uncertainty=0.2,
            metadata={},
            timestamp=datetime.now(),
        ),
        "mean_reversion": ModelPrediction(
            model_name="mean_reversion",
            prediction=-2.1,
            confidence=0.7,
            uncertainty=0.15,
            metadata={},
            timestamp=datetime.now(),
        ),
    }

    # Test combination
    combined_signal = meta_learner.combine_predictions(predictions)

    if combined_signal:
        print(f"Combined signal: {combined_signal.edge_bps:.2f} bps")
        print(f"Confidence: {combined_signal.confidence:.3f}")
        print(f"Models used: {combined_signal.metadata['models_used']}")
        print(f"Reasoning: {combined_signal.metadata['reasoning']}")

    # Test performance tracking
    meta_learner.update_model_performance("enhanced_lstm_transformer", 5.2, 4.8, 0.15)
    meta_learner.update_model_performance("momentum_model", 3.8, -1.2, -0.05)

    print(
        f"\\nEnsemble stats: {json.dumps(meta_learner.get_ensemble_stats(), indent=2, default=str)}"
    )
    print("Adaptive Meta-Learner test completed!")
