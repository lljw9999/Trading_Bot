"""
Ensemble Integration Module

Integrates all alpha models (existing and new) with the advanced ensemble system.
Handles signal collection, preprocessing, and routing to the appropriate ensemble method.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from collections import deque

from ..layer0_data_ingestion.schemas import FeatureSnapshot, AlphaSignal
from ..layer1_alpha_models.ma_momentum import MovingAverageMomentumAlpha
from ..layer1_alpha_models.mean_rev import MeanReversionAlpha
from ..layer1_alpha_models.ob_pressure import OrderBookPressureAlpha
from ..layer1_alpha_models.news_sent_alpha import NewsSentimentAlpha
from ..layer1_alpha_models.lstm_transformer_alpha import LSTMTransformerAlpha
from ..layer1_alpha_models.onchain_alpha import OnChainAlpha
from ..layer1_alpha_models.regime_detector import RegimeDetector
from .advanced_ensemble import AdvancedEnsemble, EnsembleMethod
from .meta_learner import MetaLearner


@dataclass
class EnsembleConfig:
    """Configuration for ensemble integration."""

    symbol: str
    use_advanced_ensemble: bool = True
    ensemble_method: EnsembleMethod = EnsembleMethod.STACKING
    enable_online_learning: bool = True
    enable_regime_detection: bool = True
    update_frequency: int = 100
    max_signal_age: int = 300  # seconds
    min_confidence_threshold: float = 0.1
    enable_lstm_transformer: bool = True
    enable_onchain: bool = True
    enable_regime_weighting: bool = True


class EnsembleIntegrator:
    """
    Integrates all alpha models with advanced ensemble learning.

    Responsibilities:
    - Collect signals from all alpha models
    - Preprocess and validate signals
    - Route signals to appropriate ensemble method
    - Handle fallback scenarios
    - Track model performance
    """

    def __init__(self, config: EnsembleConfig):
        """
        Initialize ensemble integrator.

        Args:
            config: Ensemble configuration
        """
        self.config = config
        self.symbol = config.symbol

        # Initialize alpha models
        self.alpha_models = {}
        self.model_status = {}

        # Initialize ensemble systems
        self.advanced_ensemble = None
        self.fallback_ensemble = None

        # Signal management
        self.signal_cache = {}
        self.signal_history = deque(maxlen=1000)

        # Performance tracking
        self.model_performance = {}
        self.ensemble_performance = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "average_confidence": 0.0,
            "last_prediction_time": None,
        }

        self.logger = logging.getLogger(f"ensemble_integrator.{config.symbol}")

        # Initialize components
        self._initialize_alpha_models()
        self._initialize_ensembles()

        self.logger.info(f"Ensemble integrator initialized for {config.symbol}")

    def _initialize_alpha_models(self):
        """Initialize all alpha models."""
        try:
            # Core alpha models (always enabled)
            self.alpha_models["ma_momentum"] = MovingAverageMomentumAlpha(
                symbol=self.symbol
            )
            self.model_status["ma_momentum"] = {"enabled": True, "last_signal": None}

            self.alpha_models["mean_rev"] = MeanReversionAlpha(symbol=self.symbol)
            self.model_status["mean_rev"] = {"enabled": True, "last_signal": None}

            self.alpha_models["ob_pressure"] = OrderBookPressureAlpha(
                symbol=self.symbol
            )
            self.model_status["ob_pressure"] = {"enabled": True, "last_signal": None}

            self.alpha_models["news_sent_alpha"] = NewsSentimentAlpha(
                symbol=self.symbol
            )
            self.model_status["news_sent_alpha"] = {
                "enabled": True,
                "last_signal": None,
            }

            # Advanced alpha models (conditionally enabled)
            if self.config.enable_lstm_transformer:
                self.alpha_models["lstm_transformer"] = LSTMTransformerAlpha(
                    symbol=self.symbol
                )
                self.model_status["lstm_transformer"] = {
                    "enabled": True,
                    "last_signal": None,
                }

            if self.config.enable_onchain:
                self.alpha_models["onchain_alpha"] = OnChainAlpha(symbol=self.symbol)
                self.model_status["onchain_alpha"] = {
                    "enabled": True,
                    "last_signal": None,
                }

            self.logger.info(f"Initialized {len(self.alpha_models)} alpha models")

        except Exception as e:
            self.logger.error(f"Error initializing alpha models: {e}")

    def _initialize_ensembles(self):
        """Initialize ensemble systems."""
        try:
            # Advanced ensemble (primary)
            if self.config.use_advanced_ensemble:
                self.advanced_ensemble = AdvancedEnsemble(
                    symbol=self.symbol,
                    primary_method=self.config.ensemble_method,
                    online_learning=self.config.enable_online_learning,
                    use_regime_detection=self.config.enable_regime_detection,
                )

            # Fallback ensemble (simple meta-learner)
            self.fallback_ensemble = MetaLearner(model_name=f"fallback_{self.symbol}")

            self.logger.info("Ensemble systems initialized")

        except Exception as e:
            self.logger.error(f"Error initializing ensembles: {e}")

    async def generate_ensemble_signal(
        self, feature_snapshot: FeatureSnapshot
    ) -> Optional[AlphaSignal]:
        """
        Generate ensemble signal from all alpha models.

        Args:
            feature_snapshot: Current market feature snapshot

        Returns:
            Ensemble alpha signal or None if failed
        """
        try:
            # Collect signals from all alpha models
            alpha_signals = await self._collect_alpha_signals(feature_snapshot)

            if not alpha_signals:
                self.logger.warning("No alpha signals collected")
                return None

            # Validate signals
            valid_signals = self._validate_signals(alpha_signals)
            if not valid_signals:
                self.logger.warning("No valid signals after validation")
                return None

            # Generate ensemble prediction
            ensemble_edge, ensemble_confidence = (
                await self._generate_ensemble_prediction(
                    valid_signals, feature_snapshot
                )
            )

            # Create ensemble signal
            ensemble_signal = AlphaSignal(
                model_name="ensemble",
                symbol=self.symbol,
                timestamp=feature_snapshot.timestamp,
                edge_bps=ensemble_edge,
                confidence=ensemble_confidence,
                signal_strength=abs(ensemble_edge) / 100.0,  # Normalize to 0-1
                metadata={
                    "num_models": len(valid_signals),
                    "model_names": list(valid_signals.keys()),
                    "ensemble_method": (
                        self.config.ensemble_method.value
                        if self.config.use_advanced_ensemble
                        else "fallback"
                    ),
                    "regime_detection": self.config.enable_regime_detection,
                },
            )

            # Update performance tracking
            self._update_performance_tracking(ensemble_signal, valid_signals)

            # Store signal in history
            self.signal_history.append(
                {
                    "timestamp": datetime.now(),
                    "ensemble_signal": ensemble_signal,
                    "alpha_signals": valid_signals,
                    "feature_snapshot": feature_snapshot,
                }
            )

            self.logger.debug(
                f"Generated ensemble signal: {ensemble_edge:.2f} bps, conf: {ensemble_confidence:.3f}"
            )

            return ensemble_signal

        except Exception as e:
            self.logger.error(f"Error generating ensemble signal: {e}")
            return None

    async def _collect_alpha_signals(
        self, feature_snapshot: FeatureSnapshot
    ) -> Dict[str, AlphaSignal]:
        """Collect signals from all alpha models."""
        alpha_signals = {}

        # Collect signals concurrently
        signal_tasks = []
        for model_name, model in self.alpha_models.items():
            if self.model_status[model_name]["enabled"]:
                task = self._get_model_signal(model_name, model, feature_snapshot)
                signal_tasks.append(task)

        # Wait for all signals
        if signal_tasks:
            signal_results = await asyncio.gather(*signal_tasks, return_exceptions=True)

            for i, result in enumerate(signal_results):
                if isinstance(result, Exception):
                    model_name = list(self.alpha_models.keys())[i]
                    self.logger.warning(
                        f"Error getting signal from {model_name}: {result}"
                    )
                    continue

                if result:
                    model_name, signal = result
                    alpha_signals[model_name] = signal
                    self.model_status[model_name]["last_signal"] = datetime.now()

        return alpha_signals

    async def _get_model_signal(
        self, model_name: str, model: Any, feature_snapshot: FeatureSnapshot
    ) -> Optional[Tuple[str, AlphaSignal]]:
        """Get signal from a specific model."""
        try:
            signal = None

            # Handle different model types
            if model_name == "ma_momentum":
                signal = model.generate_signal(feature_snapshot)
            elif model_name == "mean_rev":
                signal = model.generate_signal(feature_snapshot)
            elif model_name == "ob_pressure":
                signal = model.generate_signal(feature_snapshot)
            elif model_name == "news_sent_alpha":
                signal = await model.generate_signal(feature_snapshot)
            elif model_name == "lstm_transformer":
                signal = model.update_features(feature_snapshot)
            elif model_name == "onchain_alpha":
                signal = await model.generate_signal(feature_snapshot)

            if signal:
                return model_name, signal
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error getting signal from {model_name}: {e}")
            return None

    def _validate_signals(
        self, alpha_signals: Dict[str, AlphaSignal]
    ) -> Dict[str, AlphaSignal]:
        """Validate and filter alpha signals."""
        valid_signals = {}
        current_time = datetime.now()

        for model_name, signal in alpha_signals.items():
            try:
                # Check signal age
                if signal.timestamp:
                    age_seconds = (current_time - signal.timestamp).total_seconds()
                    if age_seconds > self.config.max_signal_age:
                        self.logger.warning(
                            f"Signal from {model_name} is too old: {age_seconds:.1f}s"
                        )
                        continue

                # Check confidence threshold
                if signal.confidence < self.config.min_confidence_threshold:
                    self.logger.warning(
                        f"Signal from {model_name} has low confidence: {signal.confidence:.3f}"
                    )
                    continue

                # Check for reasonable edge values
                if abs(signal.edge_bps) > 200:  # ±200 bps max
                    self.logger.warning(
                        f"Signal from {model_name} has extreme edge: {signal.edge_bps:.2f} bps"
                    )
                    continue

                # Check for NaN or infinite values
                if not np.isfinite(signal.edge_bps) or not np.isfinite(
                    signal.confidence
                ):
                    self.logger.warning(f"Signal from {model_name} has invalid values")
                    continue

                valid_signals[model_name] = signal

            except Exception as e:
                self.logger.error(f"Error validating signal from {model_name}: {e}")
                continue

        return valid_signals

    async def _generate_ensemble_prediction(
        self, alpha_signals: Dict[str, AlphaSignal], feature_snapshot: FeatureSnapshot
    ) -> Tuple[float, float]:
        """Generate ensemble prediction from validated signals."""
        try:
            # Try advanced ensemble first
            if self.advanced_ensemble:
                edge_bps, confidence = self.advanced_ensemble.predict(
                    alpha_signals, feature_snapshot
                )

                # Check if prediction is valid
                if np.isfinite(edge_bps) and np.isfinite(confidence) and confidence > 0:
                    return edge_bps, confidence
                else:
                    self.logger.warning(
                        "Advanced ensemble returned invalid prediction, falling back"
                    )

            # Fallback to simple ensemble
            return self._fallback_ensemble_prediction(alpha_signals, feature_snapshot)

        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return self._fallback_ensemble_prediction(alpha_signals, feature_snapshot)

    def _fallback_ensemble_prediction(
        self, alpha_signals: Dict[str, AlphaSignal], feature_snapshot: FeatureSnapshot
    ) -> Tuple[float, float]:
        """Generate fallback ensemble prediction."""
        try:
            # Convert to format expected by fallback ensemble
            signal_dict = {}
            for model_name, signal in alpha_signals.items():
                signal_dict[model_name] = (signal.edge_bps, signal.confidence)

            edge_bps, confidence = self.fallback_ensemble.predict(
                signal_dict, feature_snapshot
            )

            # Ensure valid values
            if not np.isfinite(edge_bps):
                edge_bps = 0.0
            if not np.isfinite(confidence):
                confidence = 0.0

            return edge_bps, max(0.0, min(1.0, confidence))

        except Exception as e:
            self.logger.error(f"Error in fallback prediction: {e}")

            # Last resort: simple weighted average
            if alpha_signals:
                total_weighted_edge = 0.0
                total_weight = 0.0

                for signal in alpha_signals.values():
                    weight = signal.confidence
                    total_weighted_edge += signal.edge_bps * weight
                    total_weight += weight

                if total_weight > 0:
                    return total_weighted_edge / total_weight, total_weight / len(
                        alpha_signals
                    )

            return 0.0, 0.0

    def _update_performance_tracking(
        self, ensemble_signal: AlphaSignal, alpha_signals: Dict[str, AlphaSignal]
    ):
        """Update performance tracking metrics."""
        try:
            # Update ensemble performance
            self.ensemble_performance["total_predictions"] += 1
            self.ensemble_performance["last_prediction_time"] = datetime.now()

            if ensemble_signal.confidence > self.config.min_confidence_threshold:
                self.ensemble_performance["successful_predictions"] += 1
            else:
                self.ensemble_performance["failed_predictions"] += 1

            # Update average confidence
            total_preds = self.ensemble_performance["total_predictions"]
            prev_avg = self.ensemble_performance["average_confidence"]
            new_avg = (
                prev_avg * (total_preds - 1) + ensemble_signal.confidence
            ) / total_preds
            self.ensemble_performance["average_confidence"] = new_avg

            # Update individual model performance
            for model_name, signal in alpha_signals.items():
                if model_name not in self.model_performance:
                    self.model_performance[model_name] = {
                        "total_signals": 0,
                        "average_confidence": 0.0,
                        "average_edge": 0.0,
                        "last_signal_time": None,
                    }

                perf = self.model_performance[model_name]
                perf["total_signals"] += 1
                perf["last_signal_time"] = datetime.now()

                # Update averages
                n = perf["total_signals"]
                perf["average_confidence"] = (
                    perf["average_confidence"] * (n - 1) + signal.confidence
                ) / n
                perf["average_edge"] = (
                    perf["average_edge"] * (n - 1) + abs(signal.edge_bps)
                ) / n

        except Exception as e:
            self.logger.error(f"Error updating performance tracking: {e}")

    def train_ensemble(
        self, historical_data: List[Tuple[FeatureSnapshot, float]]
    ) -> bool:
        """
        Train ensemble systems on historical data.

        Args:
            historical_data: List of (feature_snapshot, realized_return) tuples

        Returns:
            True if training successful
        """
        try:
            if len(historical_data) < 100:
                self.logger.warning("Insufficient historical data for training")
                return False

            self.logger.info(f"Training ensemble on {len(historical_data)} samples")

            # Prepare training data
            training_data = []

            for feature_snapshot, realized_return in historical_data:
                # Get alpha signals for this snapshot (simulate)
                alpha_signals = self._simulate_alpha_signals(feature_snapshot)

                if alpha_signals:
                    training_data.append(
                        (alpha_signals, feature_snapshot, realized_return)
                    )

            if len(training_data) < 50:
                self.logger.warning("Insufficient valid training samples")
                return False

            # Train advanced ensemble
            success = True
            if self.advanced_ensemble:
                success = self.advanced_ensemble.train(training_data)
                if success:
                    self.logger.info("Advanced ensemble training successful")
                else:
                    self.logger.warning("Advanced ensemble training failed")

            # Train fallback ensemble
            fallback_data = []
            for alpha_signals, feature_snapshot, realized_return in training_data:
                fallback_data.append((alpha_signals, feature_snapshot, realized_return))

            if self.fallback_ensemble.train(fallback_data):
                self.logger.info("Fallback ensemble training successful")
            else:
                self.logger.warning("Fallback ensemble training failed")
                success = False

            return success

        except Exception as e:
            self.logger.error(f"Error training ensemble: {e}")
            return False

    def _simulate_alpha_signals(
        self, feature_snapshot: FeatureSnapshot
    ) -> Dict[str, AlphaSignal]:
        """Simulate alpha signals for training data (simplified)."""
        # This is a placeholder - in practice, you'd replay historical signals
        # or use a more sophisticated simulation

        signals = {}

        # Simple signal simulation based on feature snapshot
        if feature_snapshot.return_1m is not None:
            signals["ma_momentum"] = AlphaSignal(
                model_name="ma_momentum",
                symbol=self.symbol,
                timestamp=feature_snapshot.timestamp,
                edge_bps=feature_snapshot.return_1m * 1000,  # Scale return to bps
                confidence=0.6,
                signal_strength=0.5,
            )

        return signals

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        status = {
            "ensemble_config": {
                "symbol": self.config.symbol,
                "use_advanced_ensemble": self.config.use_advanced_ensemble,
                "ensemble_method": self.config.ensemble_method.value,
                "enable_online_learning": self.config.enable_online_learning,
                "enable_regime_detection": self.config.enable_regime_detection,
            },
            "alpha_models": {},
            "ensemble_performance": self.ensemble_performance,
            "model_performance": self.model_performance,
        }

        # Add alpha model status
        for model_name, model_status in self.model_status.items():
            status["alpha_models"][model_name] = {
                "enabled": model_status["enabled"],
                "last_signal": (
                    model_status["last_signal"].isoformat()
                    if model_status["last_signal"]
                    else None
                ),
                "has_model": model_name in self.alpha_models,
            }

        # Add ensemble system status
        if self.advanced_ensemble:
            status["advanced_ensemble"] = self.advanced_ensemble.get_stats()

        if self.fallback_ensemble:
            status["fallback_ensemble"] = self.fallback_ensemble.get_stats()

        return status

    def enable_model(self, model_name: str) -> bool:
        """Enable a specific alpha model."""
        if model_name in self.model_status:
            self.model_status[model_name]["enabled"] = True
            self.logger.info(f"Enabled model: {model_name}")
            return True
        return False

    def disable_model(self, model_name: str) -> bool:
        """Disable a specific alpha model."""
        if model_name in self.model_status:
            self.model_status[model_name]["enabled"] = False
            self.logger.info(f"Disabled model: {model_name}")
            return True
        return False

    def get_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent ensemble signals."""
        return list(self.signal_history)[-limit:]

    async def shutdown(self):
        """Shutdown ensemble integrator."""
        try:
            self.logger.info("Shutting down ensemble integrator")

            # Save models if needed
            if self.advanced_ensemble:
                try:
                    self.advanced_ensemble.save_model(
                        f"models/advanced_ensemble_{self.symbol}.pkl"
                    )
                except Exception as e:
                    self.logger.error(f"Error saving advanced ensemble: {e}")

            if self.fallback_ensemble:
                try:
                    self.fallback_ensemble.save_model(
                        f"models/fallback_ensemble_{self.symbol}.pkl"
                    )
                except Exception as e:
                    self.logger.error(f"Error saving fallback ensemble: {e}")

            self.logger.info("Ensemble integrator shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Factory function
def create_ensemble_integrator(symbol: str, **kwargs) -> EnsembleIntegrator:
    """
    Create ensemble integrator with default configuration.

    Args:
        symbol: Trading symbol
        **kwargs: Configuration overrides

    Returns:
        Initialized EnsembleIntegrator instance
    """
    config = EnsembleConfig(symbol=symbol, **kwargs)
    return EnsembleIntegrator(config)
