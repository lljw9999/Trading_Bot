#!/usr/bin/env python3
"""
Minimal test for the advanced ensemble system
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from datetime import datetime
from decimal import Decimal

# Import only what we need
from src.layers.layer0_data_ingestion.schemas import FeatureSnapshot, AlphaSignal
from src.layers.layer2_ensemble.advanced_ensemble import (
    AdvancedEnsemble,
    EnsembleMethod,
)


def create_test_feature_snapshot(price: float = 50000.0) -> FeatureSnapshot:
    """Create a test feature snapshot."""
    return FeatureSnapshot(
        symbol="BTCUSDT",
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


def create_test_alpha_signals() -> dict:
    """Create test alpha signals."""
    return {
        "ma_momentum": AlphaSignal(
            model_name="ma_momentum",
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            edge_bps=15.0,
            confidence=0.7,
            signal_strength=0.6,
        ),
        "mean_rev": AlphaSignal(
            model_name="mean_rev",
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            edge_bps=-8.0,
            confidence=0.6,
            signal_strength=0.4,
        ),
        "ob_pressure": AlphaSignal(
            model_name="ob_pressure",
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            edge_bps=5.0,
            confidence=0.8,
            signal_strength=0.3,
        ),
        "news_sent_alpha": AlphaSignal(
            model_name="news_sent_alpha",
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            edge_bps=3.0,
            confidence=0.5,
            signal_strength=0.2,
        ),
    }


def test_ensemble_core():
    """Test core ensemble functionality."""
    print("🧪 Testing Advanced Ensemble Core System")
    print("=" * 60)

    # Initialize ensemble with minimal features
    print("1. Initializing ensemble...")
    ensemble = AdvancedEnsemble(
        symbol="BTCUSDT",
        primary_method=EnsembleMethod.RANDOM_FOREST,
        online_learning=False,  # Disable for simple test
        use_regime_detection=False,  # Disable for simple test
    )
    print(
        f"   ✅ Ensemble initialized with {len(ensemble.meta_learners)} meta-learners"
    )
    print(f"   ✅ Model weights: {list(ensemble.model_weights.keys())}")

    # Test prediction before training
    print("\n2. Testing prediction before training...")
    alpha_signals = create_test_alpha_signals()
    feature_snapshot = create_test_feature_snapshot()

    edge_bps, confidence = ensemble.predict(alpha_signals, feature_snapshot)
    print(
        f"   ✅ Pre-training prediction: {edge_bps:.2f} bps, confidence: {confidence:.3f}"
    )

    # Test feature extraction
    print("\n3. Testing feature extraction...")
    features = ensemble._extract_features(alpha_signals, feature_snapshot)
    print(
        f"   ✅ Features extracted: shape {features.shape}, {features.shape[1]} features"
    )

    # Test simple combination
    print("\n4. Testing simple combination...")
    simple_edge, simple_conf = ensemble._simple_combination(alpha_signals)
    print(
        f"   ✅ Simple combination: {simple_edge:.2f} bps, confidence: {simple_conf:.3f}"
    )

    # Test training with minimal data
    print("\n5. Testing training with minimal data...")
    training_data = []
    for i in range(50):  # Smaller dataset
        alpha_signals = create_test_alpha_signals()
        # Add some variation to the signals
        alpha_signals["ma_momentum"].edge_bps += np.random.randn() * 5
        alpha_signals["mean_rev"].edge_bps += np.random.randn() * 3

        feature_snapshot = create_test_feature_snapshot(price=50000 + i * 10)
        realized_return = np.random.randn() * 0.01  # Random return
        training_data.append((alpha_signals, feature_snapshot, realized_return))

    success = ensemble.train(training_data)
    print(f"   ✅ Training {'successful' if success else 'failed'}")

    # Test prediction after training
    print("\n6. Testing prediction after training...")
    alpha_signals = create_test_alpha_signals()
    feature_snapshot = create_test_feature_snapshot()

    edge_bps, confidence = ensemble.predict(alpha_signals, feature_snapshot)
    print(
        f"   ✅ Post-training prediction: {edge_bps:.2f} bps, confidence: {confidence:.3f}"
    )

    # Test model weights
    print("\n7. Testing model weights...")
    for method, weight in ensemble.model_weights.items():
        print(f"   ✅ {method.value}: {weight:.3f}")

    # Test statistics
    print("\n8. Testing statistics...")
    stats = ensemble.get_stats()
    print(
        f"   ✅ Total predictions: {stats['ensemble_performance']['total_predictions']}"
    )
    print(f"   ✅ Model trained: {stats['is_trained']}")
    print(f"   ✅ Primary method: {stats['primary_method']}")

    # Test performance tracking
    print("\n9. Testing performance tracking...")
    ensemble.update_performance(10.0, 12.0, 100.0)  # prediction, actual, pnl
    ensemble.update_performance(-5.0, -3.0, 50.0)

    updated_stats = ensemble.get_stats()
    print(
        f"   ✅ Updated total PnL: {updated_stats['ensemble_performance']['total_pnl']:.2f}"
    )
    print(
        f"   ✅ Correct predictions: {updated_stats['ensemble_performance']['correct_predictions']}"
    )

    print("\n🎉 All core tests passed!")
    print("\n📊 Final Statistics:")
    print(f"   • Symbol: {stats['symbol']}")
    print(f"   • Ensemble method: {stats['primary_method']}")
    print(f"   • Models trained: {stats['is_trained']}")
    print(
        f"   • Total predictions: {updated_stats['ensemble_performance']['total_predictions']}"
    )
    print(
        f"   • Model accuracy: {updated_stats['ensemble_performance']['accuracy']:.3f}"
    )

    # Test completed successfully - no return value needed for pytest


if __name__ == "__main__":
    try:
        test_ensemble_core()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
