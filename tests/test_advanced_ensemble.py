"""
Tests for Advanced Ensemble Learning System
"""

import os
import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

try:
    import xgboost  # noqa: F401
    import lightgbm  # noqa: F401

    HAS_GBDT = True
except Exception:  # noqa: BLE001 - optional dependency
    HAS_GBDT = False

RUN_ML_TESTS = os.getenv("RUN_ML_TESTS", "0") == "1"

if not (HAS_GBDT and RUN_ML_TESTS):
    pytest.skip(
        "requires RUN_ML_TESTS=1 with xgboost/lightgbm extras installed",
        allow_module_level=True,
    )

pytestmark = [pytest.mark.ml]

from src.layers.layer0_data_ingestion.schemas import FeatureSnapshot, AlphaSignal
from src.layers.layer2_ensemble.advanced_ensemble import (
    AdvancedEnsemble,
    EnsembleMethod,
)
from src.layers.layer2_ensemble.ensemble_integrator import (
    EnsembleIntegrator,
    EnsembleConfig,
)


class TestAdvancedEnsemble:
    """Test cases for AdvancedEnsemble class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.symbol = "BTCUSDT"
        self.ensemble = AdvancedEnsemble(
            symbol=self.symbol,
            primary_method=EnsembleMethod.STACKING,
            online_learning=True,
            use_regime_detection=True,
        )

    def create_test_feature_snapshot(self, price: float = 50000.0) -> FeatureSnapshot:
        """Create a test feature snapshot."""
        return FeatureSnapshot(
            symbol=self.symbol,
            timestamp=datetime.now(),
            mid_price=Decimal(str(price)),
            spread_bps=2.5,
            return_1m=0.001,
            return_5m=0.003,
            return_15m=0.005,
            volatility_5m=0.02,
            volatility_15m=0.025,
            volatility_1h=0.03,
            volume_ratio=1.2,
            order_book_imbalance=0.1,
            order_book_pressure=0.05,
            sma_5=Decimal("50005"),
            sma_20=Decimal("49995"),
            ema_5=Decimal("50010"),
            ema_20=Decimal("49990"),
            rsi_14=55.0,
            sent_score=0.1,
            volume_1m=Decimal("100"),
        )

    def create_test_alpha_signals(self) -> dict:
        """Create test alpha signals."""
        return {
            "ma_momentum": AlphaSignal(
                model_name="ma_momentum",
                symbol=self.symbol,
                timestamp=datetime.now(),
                edge_bps=15.0,
                confidence=0.7,
                signal_strength=0.6,
            ),
            "mean_rev": AlphaSignal(
                model_name="mean_rev",
                symbol=self.symbol,
                timestamp=datetime.now(),
                edge_bps=-8.0,
                confidence=0.6,
                signal_strength=0.4,
            ),
            "ob_pressure": AlphaSignal(
                model_name="ob_pressure",
                symbol=self.symbol,
                timestamp=datetime.now(),
                edge_bps=5.0,
                confidence=0.8,
                signal_strength=0.3,
            ),
            "news_sent_alpha": AlphaSignal(
                model_name="news_sent_alpha",
                symbol=self.symbol,
                timestamp=datetime.now(),
                edge_bps=3.0,
                confidence=0.5,
                signal_strength=0.2,
            ),
        }

    def test_initialization(self):
        """Test ensemble initialization."""
        assert self.ensemble.symbol == self.symbol
        assert self.ensemble.primary_method == EnsembleMethod.STACKING
        assert self.ensemble.online_learning is True
        assert self.ensemble.use_regime_detection is True
        assert not self.ensemble.is_trained
        assert len(self.ensemble.meta_learners) > 0
        assert len(self.ensemble.model_weights) > 0

    def test_predict_untrained(self):
        """Test prediction before training."""
        alpha_signals = self.create_test_alpha_signals()
        feature_snapshot = self.create_test_feature_snapshot()

        edge_bps, confidence = self.ensemble.predict(alpha_signals, feature_snapshot)

        assert isinstance(edge_bps, float)
        assert isinstance(confidence, float)
        assert -200 <= edge_bps <= 200  # Reasonable range
        assert 0.0 <= confidence <= 1.0

    def test_training(self):
        """Test ensemble training."""
        # Create training data
        training_data = []
        for i in range(100):
            alpha_signals = self.create_test_alpha_signals()
            feature_snapshot = self.create_test_feature_snapshot(price=50000 + i * 10)
            realized_return = np.random.randn() * 0.01  # Random return
            training_data.append((alpha_signals, feature_snapshot, realized_return))

        # Train ensemble
        success = self.ensemble.train(training_data)

        assert success is True
        assert self.ensemble.is_trained is True

    def test_predict_trained(self):
        """Test prediction after training."""
        # Train first
        training_data = []
        for i in range(100):
            alpha_signals = self.create_test_alpha_signals()
            feature_snapshot = self.create_test_feature_snapshot(price=50000 + i * 10)
            realized_return = np.random.randn() * 0.01
            training_data.append((alpha_signals, feature_snapshot, realized_return))

        self.ensemble.train(training_data)

        # Test prediction
        alpha_signals = self.create_test_alpha_signals()
        feature_snapshot = self.create_test_feature_snapshot()

        edge_bps, confidence = self.ensemble.predict(alpha_signals, feature_snapshot)

        assert isinstance(edge_bps, float)
        assert isinstance(confidence, float)
        assert -200 <= edge_bps <= 200
        assert 0.0 <= confidence <= 1.0

    def test_feature_extraction(self):
        """Test feature extraction."""
        alpha_signals = self.create_test_alpha_signals()
        feature_snapshot = self.create_test_feature_snapshot()

        features = self.ensemble._extract_features(alpha_signals, feature_snapshot)

        assert features is not None
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 1  # Single sample
        assert features.shape[1] > 0  # Has features

    def test_model_weights_update(self):
        """Test model weights update."""
        initial_weights = self.ensemble.model_weights.copy()

        # Update weights
        self.ensemble._update_model_weights()

        # Check weights are still normalized
        total_weight = sum(self.ensemble.model_weights.values())
        assert abs(total_weight - 1.0) < 1e-6

    def test_performance_tracking(self):
        """Test performance tracking."""
        initial_predictions = self.ensemble.ensemble_performance.total_predictions

        # Make a prediction
        alpha_signals = self.create_test_alpha_signals()
        feature_snapshot = self.create_test_feature_snapshot()

        self.ensemble.predict(alpha_signals, feature_snapshot)

        # Check performance was updated
        assert (
            self.ensemble.ensemble_performance.total_predictions
            == initial_predictions + 1
        )
        assert self.ensemble.ensemble_performance.last_updated is not None

    def test_stats(self):
        """Test statistics reporting."""
        stats = self.ensemble.get_stats()

        assert isinstance(stats, dict)
        assert "symbol" in stats
        assert "primary_method" in stats
        assert "is_trained" in stats
        assert "ensemble_performance" in stats
        assert stats["symbol"] == self.symbol

    def test_save_load_model(self, tmp_path):
        """Test model saving and loading."""
        # Train model first
        training_data = []
        for i in range(50):
            alpha_signals = self.create_test_alpha_signals()
            feature_snapshot = self.create_test_feature_snapshot(price=50000 + i * 10)
            realized_return = np.random.randn() * 0.01
            training_data.append((alpha_signals, feature_snapshot, realized_return))

        self.ensemble.train(training_data)

        # Save model
        model_path = tmp_path / "test_ensemble.pkl"
        success = self.ensemble.save_model(str(model_path))
        assert success is True

        # Create new ensemble and load
        new_ensemble = AdvancedEnsemble(
            symbol=self.symbol, primary_method=EnsembleMethod.STACKING
        )

        success = new_ensemble.load_model(str(model_path))
        assert success is True
        assert new_ensemble.is_trained is True
        assert new_ensemble.symbol == self.symbol


class TestEnsembleIntegrator:
    """Test cases for EnsembleIntegrator class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.symbol = "BTCUSDT"
        self.config = EnsembleConfig(
            symbol=self.symbol,
            use_advanced_ensemble=True,
            ensemble_method=EnsembleMethod.STACKING,
            enable_online_learning=True,
            enable_regime_detection=True,
        )

    def create_test_feature_snapshot(self, price: float = 50000.0) -> FeatureSnapshot:
        """Create a test feature snapshot."""
        return FeatureSnapshot(
            symbol=self.symbol,
            timestamp=datetime.now(),
            mid_price=Decimal(str(price)),
            spread_bps=2.5,
            return_1m=0.001,
            return_5m=0.003,
            return_15m=0.005,
            volatility_5m=0.02,
            volatility_15m=0.025,
            volatility_1h=0.03,
            volume_ratio=1.2,
            order_book_imbalance=0.1,
            order_book_pressure=0.05,
            sma_5=Decimal("50005"),
            sma_20=Decimal("49995"),
            ema_5=Decimal("50010"),
            ema_20=Decimal("49990"),
            rsi_14=55.0,
            sent_score=0.1,
            volume_1m=Decimal("100"),
        )

    @patch("src.layers.layer2_ensemble.ensemble_integrator.MovingAverageMomentumAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.MeanReversionAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.OrderBookPressureAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.NewsSentimentAlpha")
    def test_initialization(self, mock_news, mock_obp, mock_mean_rev, mock_momentum):
        """Test integrator initialization."""
        integrator = EnsembleIntegrator(self.config)

        assert integrator.symbol == self.symbol
        assert integrator.config == self.config
        assert len(integrator.alpha_models) > 0
        assert len(integrator.model_status) > 0
        assert integrator.advanced_ensemble is not None
        assert integrator.fallback_ensemble is not None

    @patch("src.layers.layer2_ensemble.ensemble_integrator.MovingAverageMomentumAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.MeanReversionAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.OrderBookPressureAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.NewsSentimentAlpha")
    def test_signal_validation(self, mock_news, mock_obp, mock_mean_rev, mock_momentum):
        """Test signal validation."""
        integrator = EnsembleIntegrator(self.config)

        # Create test signals
        valid_signal = AlphaSignal(
            model_name="test_model",
            symbol=self.symbol,
            timestamp=datetime.now(),
            edge_bps=10.0,
            confidence=0.7,
            signal_strength=0.5,
        )

        invalid_signal = AlphaSignal(
            model_name="test_model",
            symbol=self.symbol,
            timestamp=datetime.now() - timedelta(seconds=500),  # Too old
            edge_bps=10.0,
            confidence=0.05,  # Too low confidence
            signal_strength=0.5,
        )

        alpha_signals = {"valid": valid_signal, "invalid": invalid_signal}

        valid_signals = integrator._validate_signals(alpha_signals)

        assert "valid" in valid_signals
        assert "invalid" not in valid_signals

    @patch("src.layers.layer2_ensemble.ensemble_integrator.MovingAverageMomentumAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.MeanReversionAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.OrderBookPressureAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.NewsSentimentAlpha")
    def test_model_enable_disable(
        self, mock_news, mock_obp, mock_mean_rev, mock_momentum
    ):
        """Test model enable/disable functionality."""
        integrator = EnsembleIntegrator(self.config)

        # Test enabling
        success = integrator.enable_model("ma_momentum")
        assert success is True
        assert integrator.model_status["ma_momentum"]["enabled"] is True

        # Test disabling
        success = integrator.disable_model("ma_momentum")
        assert success is True
        assert integrator.model_status["ma_momentum"]["enabled"] is False

        # Test invalid model
        success = integrator.enable_model("invalid_model")
        assert success is False

    @patch("src.layers.layer2_ensemble.ensemble_integrator.MovingAverageMomentumAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.MeanReversionAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.OrderBookPressureAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.NewsSentimentAlpha")
    def test_model_status(self, mock_news, mock_obp, mock_mean_rev, mock_momentum):
        """Test model status reporting."""
        integrator = EnsembleIntegrator(self.config)

        status = integrator.get_model_status()

        assert isinstance(status, dict)
        assert "ensemble_config" in status
        assert "alpha_models" in status
        assert "ensemble_performance" in status
        assert "model_performance" in status
        assert "advanced_ensemble" in status
        assert "fallback_ensemble" in status

    @patch("src.layers.layer2_ensemble.ensemble_integrator.MovingAverageMomentumAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.MeanReversionAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.OrderBookPressureAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.NewsSentimentAlpha")
    def test_fallback_prediction(
        self, mock_news, mock_obp, mock_mean_rev, mock_momentum
    ):
        """Test fallback prediction."""
        integrator = EnsembleIntegrator(self.config)

        # Create test signals
        alpha_signals = {
            "ma_momentum": AlphaSignal(
                model_name="ma_momentum",
                symbol=self.symbol,
                timestamp=datetime.now(),
                edge_bps=15.0,
                confidence=0.7,
                signal_strength=0.6,
            ),
            "mean_rev": AlphaSignal(
                model_name="mean_rev",
                symbol=self.symbol,
                timestamp=datetime.now(),
                edge_bps=-8.0,
                confidence=0.6,
                signal_strength=0.4,
            ),
        }

        feature_snapshot = self.create_test_feature_snapshot()

        edge_bps, confidence = integrator._fallback_ensemble_prediction(
            alpha_signals, feature_snapshot
        )

        assert isinstance(edge_bps, float)
        assert isinstance(confidence, float)
        assert -200 <= edge_bps <= 200
        assert 0.0 <= confidence <= 1.0


class TestEnsembleFactory:
    """Test factory functions."""

    def test_create_advanced_ensemble(self):
        """Test advanced ensemble factory."""
        from src.layers.layer2_ensemble.advanced_ensemble import (
            create_advanced_ensemble,
        )

        ensemble = create_advanced_ensemble("BTCUSDT")

        assert isinstance(ensemble, AdvancedEnsemble)
        assert ensemble.symbol == "BTCUSDT"

    def test_create_ensemble_integrator(self):
        """Test ensemble integrator factory."""
        from src.layers.layer2_ensemble.ensemble_integrator import (
            create_ensemble_integrator,
        )

        with (
            patch(
                "src.layers.layer2_ensemble.ensemble_integrator.MovingAverageMomentumAlpha"
            ),
            patch("src.layers.layer2_ensemble.ensemble_integrator.MeanReversionAlpha"),
            patch(
                "src.layers.layer2_ensemble.ensemble_integrator.OrderBookPressureAlpha"
            ),
            patch("src.layers.layer2_ensemble.ensemble_integrator.NewsSentimentAlpha"),
        ):

            integrator = create_ensemble_integrator("BTCUSDT")

            assert isinstance(integrator, EnsembleIntegrator)
            assert integrator.symbol == "BTCUSDT"


# Integration tests
class TestEnsembleIntegration:
    """Integration tests for ensemble system."""

    def setup_method(self):
        """Setup test fixtures."""
        self.symbol = "BTCUSDT"

    def create_test_feature_snapshot(self, price: float = 50000.0) -> FeatureSnapshot:
        """Create a test feature snapshot."""
        return FeatureSnapshot(
            symbol=self.symbol,
            timestamp=datetime.now(),
            mid_price=Decimal(str(price)),
            spread_bps=2.5,
            return_1m=0.001,
            return_5m=0.003,
            return_15m=0.005,
            volatility_5m=0.02,
            volatility_15m=0.025,
            volatility_1h=0.03,
            volume_ratio=1.2,
            order_book_imbalance=0.1,
            order_book_pressure=0.05,
            sma_5=Decimal("50005"),
            sma_20=Decimal("49995"),
            ema_5=Decimal("50010"),
            ema_20=Decimal("49990"),
            rsi_14=55.0,
            sent_score=0.1,
            volume_1m=Decimal("100"),
        )

    @patch("src.layers.layer2_ensemble.ensemble_integrator.MovingAverageMomentumAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.MeanReversionAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.OrderBookPressureAlpha")
    @patch("src.layers.layer2_ensemble.ensemble_integrator.NewsSentimentAlpha")
    @pytest.mark.asyncio
    async def test_full_ensemble_pipeline(
        self, mock_news, mock_obp, mock_mean_rev, mock_momentum
    ):
        """Test complete ensemble pipeline."""
        # Mock alpha models to return test signals
        mock_momentum.return_value.generate_signal.return_value = AlphaSignal(
            model_name="ma_momentum",
            symbol=self.symbol,
            timestamp=datetime.now(),
            edge_bps=15.0,
            confidence=0.7,
            signal_strength=0.6,
        )

        mock_mean_rev.return_value.generate_signal.return_value = AlphaSignal(
            model_name="mean_rev",
            symbol=self.symbol,
            timestamp=datetime.now(),
            edge_bps=-8.0,
            confidence=0.6,
            signal_strength=0.4,
        )

        mock_obp.return_value.generate_signal.return_value = AlphaSignal(
            model_name="ob_pressure",
            symbol=self.symbol,
            timestamp=datetime.now(),
            edge_bps=5.0,
            confidence=0.8,
            signal_strength=0.3,
        )

        mock_news.return_value.generate_signal.return_value = AlphaSignal(
            model_name="news_sent_alpha",
            symbol=self.symbol,
            timestamp=datetime.now(),
            edge_bps=3.0,
            confidence=0.5,
            signal_strength=0.2,
        )

        # Create integrator
        config = EnsembleConfig(
            symbol=self.symbol,
            use_advanced_ensemble=False,  # Use simpler ensemble for testing
            enable_lstm_transformer=False,
            enable_onchain=False,
        )

        integrator = EnsembleIntegrator(config)

        # Generate ensemble signal
        feature_snapshot = self.create_test_feature_snapshot()
        ensemble_signal = await integrator.generate_ensemble_signal(feature_snapshot)

        # Verify ensemble signal
        assert ensemble_signal is not None
        assert isinstance(ensemble_signal, AlphaSignal)
        assert ensemble_signal.model_name == "ensemble"
        assert ensemble_signal.symbol == self.symbol
        assert isinstance(ensemble_signal.edge_bps, float)
        assert isinstance(ensemble_signal.confidence, float)
        assert 0.0 <= ensemble_signal.confidence <= 1.0

        # Check metadata
        assert "num_models" in ensemble_signal.metadata
        assert "model_names" in ensemble_signal.metadata
        assert "ensemble_method" in ensemble_signal.metadata

        # Test performance tracking
        status = integrator.get_model_status()
        assert status["ensemble_performance"]["total_predictions"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
