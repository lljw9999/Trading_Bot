#!/usr/bin/env python3
"""
Unit tests for alpha models (TST-α)

Tests for OBP (Order-Book-Pressure) and MAM (Moving-Average Momentum) alpha models
as specified in Future_instruction.txt.
"""

import unittest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

# Import the alpha models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.layers.layer1_alpha_models.ob_pressure import OrderBookPressureAlpha
from src.layers.layer1_alpha_models.ma_momentum import MovingAverageMomentumAlpha
from src.layers.layer2_ensemble.meta_learner import MetaLearner
from src.layers.layer0_data_ingestion.schemas import FeatureSnapshot


class TestOrderBookPressureAlpha(unittest.TestCase):
    """Test cases for Order Book Pressure Alpha model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.alpha = OrderBookPressureAlpha(edge_scaling=25.0)
        self.timestamp = "2025-01-15T10:00:00Z"
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.alpha.edge_scaling, 25.0)
        self.assertEqual(self.alpha.min_confidence, 0.50)
        self.assertEqual(self.alpha.max_confidence, 1.0)
        self.assertEqual(self.alpha.signal_count, 0)
        self.assertEqual(self.alpha.hit_count, 0)
    
    def test_generate_signal_buy_pressure(self):
        """Test signal generation with buy pressure (bid_size > ask_size)."""
        signal = self.alpha.generate_signal(
            symbol="BTC-USD",
            bid_size=10.0,
            ask_size=2.0,
            timestamp=self.timestamp
        )
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.symbol, "BTC-USD")
        self.assertGreater(signal.edge_bps, 0)  # Positive edge for buy pressure
        self.assertGreaterEqual(signal.confidence, 0.50)
        self.assertLessEqual(signal.confidence, 1.0)
        
        # Check formula: pressure = (10-2)/(10+2+1e-9) = 8/12 = 0.667
        # edge = 25 * 0.667 = 16.67 bps
        expected_pressure = (10.0 - 2.0) / (10.0 + 2.0 + 1e-9)
        expected_edge = 25.0 * expected_pressure
        self.assertAlmostEqual(signal.edge_bps, expected_edge, places=2)
    
    def test_generate_signal_sell_pressure(self):
        """Test signal generation with sell pressure (ask_size > bid_size)."""
        signal = self.alpha.generate_signal(
            symbol="ETH-USD",
            bid_size=3.0,
            ask_size=12.0,
            timestamp=self.timestamp
        )
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.symbol, "ETH-USD")
        self.assertLess(signal.edge_bps, 0)  # Negative edge for sell pressure
        self.assertGreaterEqual(signal.confidence, 0.50)
        self.assertLessEqual(signal.confidence, 1.0)
    
    def test_generate_signal_balanced_book(self):
        """Test signal generation with balanced order book."""
        signal = self.alpha.generate_signal(
            symbol="SOL-USD",
            bid_size=5.0,
            ask_size=5.0,
            timestamp=self.timestamp
        )
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.symbol, "SOL-USD")
        self.assertAlmostEqual(signal.edge_bps, 0.0, places=2)  # Near zero edge
        self.assertAlmostEqual(signal.confidence, 0.50, places=2)  # Min confidence
    
    def test_generate_signal_invalid_inputs(self):
        """Test signal generation with invalid inputs."""
        # Zero bid size
        signal = self.alpha.generate_signal("BTC-USD", 0.0, 5.0, self.timestamp)
        self.assertIsNone(signal)
        
        # Negative ask size
        signal = self.alpha.generate_signal("BTC-USD", 5.0, -1.0, self.timestamp)
        self.assertIsNone(signal)
    
    def test_update_from_feature_snapshot(self):
        """Test signal generation from feature snapshot."""
        # Mock feature snapshot
        feature_snapshot = Mock()
        feature_snapshot.symbol = "BTC-USD"
        feature_snapshot.timestamp = datetime.fromisoformat("2025-01-15T10:00:00")
        feature_snapshot.order_book_pressure = 0.5  # 50% buy pressure
        
        signal = self.alpha.update_from_feature_snapshot(feature_snapshot)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.symbol, "BTC-USD")
        self.assertAlmostEqual(signal.edge_bps, 25.0 * 0.5, places=2)  # 12.5 bps
    
    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        # Generate a signal first
        signal = self.alpha.generate_signal("BTC-USD", 10.0, 2.0, self.timestamp)
        self.assertEqual(self.alpha.signal_count, 1)
        
        # Update performance with correct prediction
        self.alpha.update_performance("BTC-USD", 10.0, 15.0)  # Both positive
        self.assertEqual(self.alpha.hit_count, 1)
        self.assertEqual(self.alpha.get_hit_rate(), 1.0)
        
        # Update performance with incorrect prediction
        self.alpha.update_performance("BTC-USD", -5.0, 10.0)  # Opposite signs
        self.assertEqual(self.alpha.hit_count, 1)  # No change
        self.assertEqual(self.alpha.get_hit_rate(), 0.5)  # 1/2 = 0.5
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        stats = self.alpha.get_stats()
        
        self.assertEqual(stats['model_name'], 'ob_pressure_v0')
        self.assertEqual(stats['signal_count'], 0)
        self.assertEqual(stats['hit_count'], 0)
        self.assertEqual(stats['edge_scaling'], 25.0)
        self.assertEqual(stats['min_confidence'], 0.50)


class TestMovingAverageMomentumAlpha(unittest.TestCase):
    """Test cases for Moving Average Momentum Alpha model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.alpha = MovingAverageMomentumAlpha(
            short_period=5,
            long_period=30,
            edge_scaling=40.0
        )
        self.timestamp = "2025-01-15T10:00:00Z"
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.alpha.short_period, 5)
        self.assertEqual(self.alpha.long_period, 30)
        self.assertEqual(self.alpha.edge_scaling, 40.0)
        self.assertEqual(self.alpha.min_confidence, 0.55)
        self.assertEqual(self.alpha.max_confidence, 0.9)
        self.assertEqual(self.alpha.signal_count, 0)
    
    def test_update_price_insufficient_data(self):
        """Test that no signal is generated with insufficient data."""
        # Add only a few prices (less than min_samples=30)
        for i in range(10):
            signal = self.alpha.update_price("BTC-USD", 50000 + i, f"2025-01-15T10:{i:02d}:00Z")
            self.assertIsNone(signal)
    
    def test_update_price_upward_momentum(self):
        """Test signal generation with upward momentum."""
        base_price = 50000
        
        # Add 30 prices with upward trend
        for i in range(30):
            price = base_price * (1 + i * 0.01)  # 1% increase per step
            signal = self.alpha.update_price("BTC-USD", price, f"2025-01-15T10:{i:02d}:00Z")
            
            if i >= 29:  # Should generate signal on 30th update
                self.assertIsNotNone(signal)
                self.assertEqual(signal.symbol, "BTC-USD")
                self.assertGreater(signal.edge_bps, 0)  # Positive edge for upward momentum
                self.assertGreaterEqual(signal.confidence, 0.55)
                self.assertLessEqual(signal.confidence, 0.9)
    
    def test_update_price_downward_momentum(self):
        """Test signal generation with downward momentum."""
        base_price = 50000
        
        # Add 30 prices with downward trend
        for i in range(30):
            price = base_price * (1 - i * 0.01)  # 1% decrease per step
            signal = self.alpha.update_price("BTC-USD", price, f"2025-01-15T10:{i:02d}:00Z")
            
            if i >= 29:  # Should generate signal on 30th update
                self.assertIsNotNone(signal)
                self.assertEqual(signal.symbol, "BTC-USD")
                self.assertLess(signal.edge_bps, 0)  # Negative edge for downward momentum
                self.assertGreaterEqual(signal.confidence, 0.55)
                self.assertLessEqual(signal.confidence, 0.9)
    
    def test_update_price_sideways_market(self):
        """Test signal generation with sideways market (no momentum)."""
        base_price = 50000
        
        # Add 30 prices with no trend (small random noise)
        np.random.seed(42)  # For reproducible results
        for i in range(30):
            noise = np.random.normal(0, 0.001)  # 0.1% noise
            price = base_price * (1 + noise)
            signal = self.alpha.update_price("BTC-USD", price, f"2025-01-15T10:{i:02d}:00Z")
            
            if i >= 29:  # Should generate signal on 30th update
                if signal:  # Might be None if edge is too small
                    self.assertLess(abs(signal.edge_bps), 5.0)  # Very small edge
    
    def test_update_from_feature_snapshot(self):
        """Test signal generation from feature snapshot."""
        # Mock feature snapshot
        feature_snapshot = Mock()
        feature_snapshot.symbol = "BTC-USD"
        feature_snapshot.timestamp = datetime.fromisoformat("2025-01-15T10:00:00")
        feature_snapshot.mid_price = 50000.0
        
        # First call should return None (insufficient data)
        signal = self.alpha.update_from_feature_snapshot(feature_snapshot)
        self.assertIsNone(signal)
        
        # Add enough data points
        for i in range(30):
            feature_snapshot.mid_price = 50000.0 + i * 100  # Upward trend
            signal = self.alpha.update_from_feature_snapshot(feature_snapshot)
            
            if i >= 29:
                self.assertIsNotNone(signal)
                self.assertGreater(signal.edge_bps, 0)
    
    def test_edge_capping(self):
        """Test that edge is capped at ±40 bp."""
        base_price = 50000
        
        # Add 30 prices with very strong upward trend
        for i in range(30):
            price = base_price * (1 + i * 0.05)  # 5% increase per step (very strong)
            signal = self.alpha.update_price("BTC-USD", price, f"2025-01-15T10:{i:02d}:00Z")
            
            if signal:
                self.assertLessEqual(abs(signal.edge_bps), 40.0)  # Should be capped at 40
    
    def test_confidence_bounds(self):
        """Test that confidence is within specified bounds."""
        base_price = 50000
        
        # Add 30 prices with moderate trend
        for i in range(30):
            price = base_price * (1 + i * 0.02)  # 2% increase per step
            signal = self.alpha.update_price("BTC-USD", price, f"2025-01-15T10:{i:02d}:00Z")
            
            if signal:
                self.assertGreaterEqual(signal.confidence, 0.55)
                self.assertLessEqual(signal.confidence, 0.9)
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        stats = self.alpha.get_stats()
        
        self.assertEqual(stats['model_name'], 'ma_momentum_v0')
        self.assertEqual(stats['signal_count'], 0)
        self.assertEqual(stats['hit_count'], 0)
        self.assertEqual(stats['short_period'], 5)
        self.assertEqual(stats['long_period'], 30)
        self.assertEqual(stats['active_symbols'], 0)


class TestMetaLearnerLogisticBlending(unittest.TestCase):
    """Test cases for MetaLearner logistic blending functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.meta_learner = MetaLearner()
        
        # Mock feature snapshot
        self.feature_snapshot = Mock()
        self.feature_snapshot.symbol = "BTC-USD"
        self.feature_snapshot.timestamp = datetime.fromisoformat("2025-01-15T10:00:00")
        self.feature_snapshot.spread_bps = 5.0
        self.feature_snapshot.volatility_5m = 0.02
        self.feature_snapshot.volume_ratio = 1.2
        self.feature_snapshot.order_book_imbalance = 0.1
        self.feature_snapshot.return_1m = 0.001
    
    def test_can_use_logistic_blend(self):
        """Test detection of required signals for logistic blending."""
        # Test with OBP + MAM signals
        alpha_signals = {
            'ob_pressure': (15.0, 0.7),
            'ma_momentum': (20.0, 0.8)
        }
        self.assertTrue(self.meta_learner._can_use_logistic_blend(alpha_signals))
        
        # Test with missing signals
        alpha_signals = {
            'ob_pressure': (15.0, 0.7),
            'other_model': (10.0, 0.6)
        }
        self.assertFalse(self.meta_learner._can_use_logistic_blend(alpha_signals))
    
    def test_logistic_blend_positive_signals(self):
        """Test logistic blending with positive signals."""
        alpha_signals = {
            'ob_pressure': (15.0, 0.7),
            'ma_momentum': (20.0, 0.8)
        }
        
        edge, confidence = self.meta_learner._logistic_blend(alpha_signals)
        
        self.assertGreater(edge, 0)  # Should be positive
        self.assertGreater(confidence, 0.5)  # Should be above neutral
        self.assertLessEqual(confidence, 1.0)
    
    def test_logistic_blend_negative_signals(self):
        """Test logistic blending with negative signals."""
        alpha_signals = {
            'ob_pressure': (-15.0, 0.7),
            'ma_momentum': (-20.0, 0.8)
        }
        
        edge, confidence = self.meta_learner._logistic_blend(alpha_signals)
        
        self.assertLess(edge, 0)  # Should be negative
        self.assertLess(confidence, 0.5)  # Should be below neutral
        self.assertGreaterEqual(confidence, 0.0)
    
    def test_logistic_blend_mixed_signals(self):
        """Test logistic blending with mixed (positive/negative) signals."""
        alpha_signals = {
            'ob_pressure': (15.0, 0.7),
            'ma_momentum': (-10.0, 0.6)
        }
        
        edge, confidence = self.meta_learner._logistic_blend(alpha_signals)
        
        # Result should be somewhere in between
        self.assertLess(abs(edge), 15.0)  # Should be less than stronger signal
        self.assertGreater(confidence, 0.0)
        self.assertLess(confidence, 1.0)
    
    def test_predict_with_logistic_blend(self):
        """Test full prediction pipeline with logistic blending."""
        alpha_signals = {
            'ob_pressure': (15.0, 0.7),
            'ma_momentum': (20.0, 0.8)
        }
        
        edge, confidence = self.meta_learner.predict(alpha_signals, self.feature_snapshot)
        
        self.assertIsInstance(edge, float)
        self.assertIsInstance(confidence, float)
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Should use logistic blend (not simple ensemble)
        self.assertGreater(edge, 0)  # Both signals positive
    
    def test_predict_simple_with_two_signals(self):
        """Test simplified predict method with two signals."""
        alpha_signals = [15.0, 20.0]  # Two positive signals
        
        edge = self.meta_learner.predict_simple(alpha_signals)
        
        self.assertIsInstance(edge, float)
        self.assertGreater(edge, 0)  # Should be positive
    
    def test_predict_simple_with_one_signal(self):
        """Test simplified predict method with one signal."""
        alpha_signals = [15.0]
        
        edge = self.meta_learner.predict_simple(alpha_signals)
        
        self.assertEqual(edge, 15.0)  # Should return the signal as-is
    
    def test_predict_simple_with_multiple_signals(self):
        """Test simplified predict method with multiple signals."""
        alpha_signals = [10.0, 20.0, 30.0]
        
        edge = self.meta_learner.predict_simple(alpha_signals)
        
        self.assertEqual(edge, 20.0)  # Should return the average
    
    def test_logistic_weights_loading(self):
        """Test loading of logistic weights."""
        # Test that weights are loaded
        self.assertIn('w1', self.meta_learner.logistic_weights)
        self.assertIn('w2', self.meta_learner.logistic_weights)
        self.assertIsInstance(self.meta_learner.logistic_weights['w1'], float)
        self.assertIsInstance(self.meta_learner.logistic_weights['w2'], float)


class TestSyntheticSmokeRun(unittest.TestCase):
    """Synthetic smoke-run test for the complete alpha pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline: OBP + MAM → MetaLearner → Final Signal."""
        # Initialize models
        obp_alpha = OrderBookPressureAlpha(edge_scaling=25.0)
        mam_alpha = MovingAverageMomentumAlpha(short_period=5, long_period=30)
        meta_learner = MetaLearner()
        
        # Generate test data
        symbol = "BTC-USD"
        base_price = 50000
        
        # 1. Generate MA momentum signal (need 30 data points)
        mam_signal = None
        for i in range(30):
            price = base_price * (1 + i * 0.01)  # Upward trend
            timestamp = f"2025-01-15T10:{i:02d}:00Z"
            mam_signal = mam_alpha.update_price(symbol, price, timestamp)
        
        # 2. Generate OBP signal
        obp_signal = obp_alpha.generate_signal(
            symbol=symbol,
            bid_size=10.0,
            ask_size=3.0,
            timestamp="2025-01-15T10:30:00Z"
        )
        
        # 3. Combine signals in meta-learner
        if mam_signal and obp_signal:
            alpha_signals = {
                'ma_momentum': (mam_signal.edge_bps, mam_signal.confidence),
                'ob_pressure': (obp_signal.edge_bps, obp_signal.confidence)
            }
            
            # Mock feature snapshot
            feature_snapshot = Mock()
            feature_snapshot.symbol = symbol
            feature_snapshot.timestamp = datetime.fromisoformat("2025-01-15T10:30:00")
            feature_snapshot.spread_bps = 5.0
            feature_snapshot.volatility_5m = 0.02
            feature_snapshot.volume_ratio = 1.2
            feature_snapshot.order_book_imbalance = 0.1
            feature_snapshot.return_1m = 0.001
            
            ensemble_edge, ensemble_confidence = meta_learner.predict(alpha_signals, feature_snapshot)
            
            # Verify pipeline worked
            self.assertIsInstance(ensemble_edge, float)
            self.assertIsInstance(ensemble_confidence, float)
            self.assertGreater(ensemble_confidence, 0.0)
            self.assertLessEqual(ensemble_confidence, 1.0)
            
            # Both signals should be positive (upward momentum + buy pressure)
            self.assertGreater(mam_signal.edge_bps, 0)
            self.assertGreater(obp_signal.edge_bps, 0)
            self.assertGreater(ensemble_edge, 0)
            
            print(f"✅ End-to-end pipeline test passed:")
            print(f"   MAM signal: {mam_signal.edge_bps:.1f}bps (conf: {mam_signal.confidence:.2f})")
            print(f"   OBP signal: {obp_signal.edge_bps:.1f}bps (conf: {obp_signal.confidence:.2f})")
            print(f"   Ensemble:   {ensemble_edge:.1f}bps (conf: {ensemble_confidence:.2f})")
        else:
            self.fail("Failed to generate required signals for pipeline test")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 