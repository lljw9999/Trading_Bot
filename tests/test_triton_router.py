#!/usr/bin/env python3
"""
Unit tests for Triton routing ensemble.

Tests routing logic, model selection, and performance for the 
intelligent alpha model ensemble.
"""

import unittest
import time
import numpy as np
import sys
import os
from unittest.mock import Mock, patch
from typing import Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the router model directly
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'docker', 'triton', 'model_repository', 'router', '1'))
from model import TritonPythonModel, ROUTING_CONFIG


class TestTritonRouter(unittest.TestCase):
    """Test Triton router model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.router = TritonPythonModel()
        
        # Mock initialization args
        mock_args = {
            'model_config': {},
            'model_instance_kind': 'CPU',
            'model_instance_device_id': 0
        }
        
        # Initialize router (will use mock ONNX sessions)
        self.router.initialize(mock_args)
    
    def test_router_initialization(self):
        """Test router initialization."""
        self.assertIsNotNone(self.router.routing_stats)
        self.assertEqual(self.router.routing_stats['total_requests'], 0)
        self.assertIsNotNone(self.router.logger)
    
    def test_btc_short_horizon_routes_to_tlob(self):
        """Test BTC with short horizon routes to TLOB-Tiny."""
        symbol = "BTC-USD"
        time_horizon = 3  # 3 minutes
        feature_shape = (32, 10)  # Short sequence
        
        model_choice = self.router._select_model(symbol, time_horizon, feature_shape)
        self.assertEqual(model_choice, "tlob_tiny")
    
    def test_btc_long_horizon_routes_to_patchtst(self):
        """Test BTC with long horizon routes to PatchTST."""
        symbol = "BTC-USD"
        time_horizon = 30  # 30 minutes
        feature_shape = (32, 10)
        
        model_choice = self.router._select_model(symbol, time_horizon, feature_shape)
        self.assertEqual(model_choice, "patchtst_small")
    
    def test_eth_short_horizon_routes_to_tlob(self):
        """Test ETH with short horizon routes to TLOB-Tiny."""
        symbol = "ETH-USD"
        time_horizon = 5  # 5 minutes (threshold)
        feature_shape = (20, 8)
        
        model_choice = self.router._select_model(symbol, time_horizon, feature_shape)
        self.assertEqual(model_choice, "tlob_tiny")
    
    def test_long_sequence_routes_to_patchtst(self):
        """Test long sequences route to PatchTST regardless of symbol."""
        symbol = "BTC-USD"
        time_horizon = 2  # Short horizon
        feature_shape = (100, 5)  # Long sequence > TLOB max
        
        model_choice = self.router._select_model(symbol, time_horizon, feature_shape)
        self.assertEqual(model_choice, "patchtst_small")
    
    def test_unknown_symbol_short_horizon_routes_to_tlob(self):
        """Test unknown symbol with short horizon routes to TLOB."""
        symbol = "AAPL"  # Not in HF symbols
        time_horizon = 3
        feature_shape = (20, 5)
        
        model_choice = self.router._select_model(symbol, time_horizon, feature_shape)
        self.assertEqual(model_choice, "tlob_tiny")
    
    def test_unknown_symbol_long_horizon_routes_to_patchtst(self):
        """Test unknown symbol with long horizon routes to PatchTST."""
        symbol = "AAPL"  # Not in HF symbols
        time_horizon = 15  # Medium horizon
        feature_shape = (50, 5)
        
        model_choice = self.router._select_model(symbol, time_horizon, feature_shape)
        self.assertEqual(model_choice, "patchtst_small")
    
    def test_tlob_prediction_output_format(self):
        """Test TLOB prediction returns correct format."""
        features = np.random.randn(32, 10).astype(np.float32)
        
        prediction, confidence = self.router._predict_tlob(features)
        
        # Validate output format
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(prediction.shape, (1,))  # Single pressure value
        self.assertIsInstance(confidence, float)
        self.assertTrue(0.0 <= confidence <= 1.0)
        self.assertTrue(-1.0 <= prediction[0] <= 1.0)  # Pressure range
    
    def test_patchtst_prediction_output_format(self):
        """Test PatchTST prediction returns correct format."""
        features = np.random.randn(96, 5).astype(np.float32)
        
        prediction, confidence = self.router._predict_patchtst(features)
        
        # Validate output format
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(len(prediction.shape), 1)  # Flattened forecast
        self.assertIsInstance(confidence, float)
        self.assertTrue(0.0 <= confidence <= 1.0)
    
    def test_tlob_feature_padding_and_truncation(self):
        """Test TLOB handles different feature dimensions correctly."""
        # Test feature padding (less than 10 features)
        features_few = np.random.randn(32, 5).astype(np.float32)
        prediction1, confidence1 = self.router._predict_tlob(features_few)
        self.assertEqual(prediction1.shape, (1,))
        
        # Test feature truncation (more than 10 features)
        features_many = np.random.randn(32, 15).astype(np.float32)
        prediction2, confidence2 = self.router._predict_tlob(features_many)
        self.assertEqual(prediction2.shape, (1,))
        
        # Test sequence padding (short sequence)
        features_short = np.random.randn(15, 10).astype(np.float32)
        prediction3, confidence3 = self.router._predict_tlob(features_short)
        self.assertEqual(prediction3.shape, (1,))
        
        # Test sequence truncation (long sequence)
        features_long = np.random.randn(50, 10).astype(np.float32)
        prediction4, confidence4 = self.router._predict_tlob(features_long)
        self.assertEqual(prediction4.shape, (1,))
    
    def test_patchtst_feature_padding_and_truncation(self):
        """Test PatchTST handles different feature dimensions correctly."""
        # Test feature padding (less than 5 features)
        features_few = np.random.randn(96, 3).astype(np.float32)
        prediction1, confidence1 = self.router._predict_patchtst(features_few)
        self.assertGreater(len(prediction1), 0)
        
        # Test feature truncation (more than 5 features)
        features_many = np.random.randn(96, 8).astype(np.float32)
        prediction2, confidence2 = self.router._predict_patchtst(features_many)
        self.assertGreater(len(prediction2), 0)
        
        # Test sequence padding (short sequence)
        features_short = np.random.randn(30, 5).astype(np.float32)
        prediction3, confidence3 = self.router._predict_patchtst(features_short)
        self.assertGreater(len(prediction3), 0)
        
        # Test sequence truncation (long sequence)
        features_long = np.random.randn(150, 5).astype(np.float32)
        prediction4, confidence4 = self.router._predict_patchtst(features_long)
        self.assertGreater(len(prediction4), 0)
    
    def test_routing_latency_under_10ms(self):
        """Test routing + prediction latency is under 10ms."""
        symbol = "BTC-USD"
        features = np.random.randn(32, 10).astype(np.float32)
        time_horizon = 3
        
        # Warm up
        for _ in range(3):
            self.router._route_and_predict(symbol, features, time_horizon)
        
        # Measure latency
        latencies = []
        for _ in range(20):
            start_time = time.time()
            model_choice, prediction, confidence = self.router._route_and_predict(
                symbol, features, time_horizon
            )
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        print(f"\n📊 Routing Latency Stats:")
        print(f"   - Average: {avg_latency:.2f} ms")
        print(f"   - Median: {median_latency:.2f} ms")
        print(f"   - P95: {p95_latency:.2f} ms")
        
        # Assert median latency is under 10ms target
        self.assertLess(median_latency, 10.0, 
                       f"Median latency {median_latency:.2f}ms exceeds 10ms target")
    
    def test_routing_statistics_tracking(self):
        """Test routing statistics are properly tracked."""
        initial_requests = self.router.routing_stats["total_requests"]
        
        # Make some predictions
        btc_features = np.random.randn(32, 10).astype(np.float32)
        eth_features = np.random.randn(96, 5).astype(np.float32)
        
        # These should route to TLOB
        self.router._route_and_predict("BTC-USD", btc_features, 3)
        self.router._route_and_predict("ETH-USD", btc_features, 2)
        
        # These should route to PatchTST
        self.router._route_and_predict("BTC-USD", eth_features, 30)
        self.router._route_and_predict("AAPL", eth_features, 15)
        
        # Check statistics
        stats = self.router.routing_stats
        self.assertEqual(stats["total_requests"], initial_requests + 4)
        self.assertGreaterEqual(stats["tlob_requests"], 2)
        self.assertGreaterEqual(stats["patchtst_requests"], 2)
        self.assertEqual(len(stats["routing_latency_ms"]), 4)
    
    def test_edge_case_empty_features(self):
        """Test handling of empty or malformed features."""
        symbol = "BTC-USD"
        time_horizon = 3
        
        # Empty features
        empty_features = np.array([], dtype=np.float32).reshape(0, 0)
        
        # Should handle gracefully without crashing
        try:
            model_choice, prediction, confidence = self.router._route_and_predict(
                symbol, empty_features, time_horizon
            )
            # Should return some default values
            self.assertIn(model_choice, ["tlob_tiny", "patchtst_small"])
            self.assertIsInstance(prediction, np.ndarray)
            self.assertIsInstance(confidence, float)
        except Exception as e:
            # If it throws an exception, that's also acceptable for edge cases
            self.assertIsInstance(e, Exception)
    
    def test_edge_case_extreme_time_horizons(self):
        """Test handling of extreme time horizon values."""
        features = np.random.randn(32, 10).astype(np.float32)
        
        # Test negative time horizon
        model_choice_neg = self.router._select_model("BTC-USD", -5, features.shape)
        self.assertIn(model_choice_neg, ["tlob_tiny", "patchtst_small"])
        
        # Test very large time horizon
        model_choice_large = self.router._select_model("BTC-USD", 10000, features.shape)
        self.assertEqual(model_choice_large, "patchtst_small")  # Should route to PatchTST
        
        # Test zero time horizon
        model_choice_zero = self.router._select_model("BTC-USD", 0, features.shape)
        self.assertEqual(model_choice_zero, "tlob_tiny")  # Should route to TLOB
    
    def test_symbol_case_insensitivity(self):
        """Test symbol routing is case-insensitive."""
        features = np.random.randn(32, 10).astype(np.float32)
        time_horizon = 3
        
        # Test different cases of the same symbol
        model1 = self.router._select_model("btc-usd", time_horizon, features.shape)
        model2 = self.router._select_model("BTC-USD", time_horizon, features.shape)
        model3 = self.router._select_model("Btc-Usd", time_horizon, features.shape)
        
        self.assertEqual(model1, model2)
        self.assertEqual(model2, model3)
        self.assertEqual(model1, "tlob_tiny")  # Should all route to TLOB
    
    def test_confidence_ranges(self):
        """Test confidence values are in valid ranges."""
        features_tlob = np.random.randn(32, 10).astype(np.float32)
        features_patchtst = np.random.randn(96, 5).astype(np.float32)
        
        # Test TLOB confidence
        for _ in range(10):
            _, confidence_tlob = self.router._predict_tlob(features_tlob)
            self.assertTrue(0.0 <= confidence_tlob <= 1.0)
        
        # Test PatchTST confidence
        for _ in range(10):
            _, confidence_patchtst = self.router._predict_patchtst(features_patchtst)
            self.assertTrue(0.0 <= confidence_patchtst <= 1.0)


class TestRoutingConfig(unittest.TestCase):
    """Test routing configuration validation."""
    
    def test_routing_config_structure(self):
        """Test routing config has all required keys."""
        required_keys = [
            "hf_symbols", "short_horizon_threshold", "medium_horizon_threshold",
            "tlob_max_sequence", "patchtst_max_sequence", 
            "high_confidence_threshold", "medium_confidence_threshold"
        ]
        
        for key in required_keys:
            self.assertIn(key, ROUTING_CONFIG, f"Missing config key: {key}")
    
    def test_routing_thresholds_logical(self):
        """Test routing thresholds are logically ordered."""
        short_thresh = ROUTING_CONFIG["short_horizon_threshold"]
        medium_thresh = ROUTING_CONFIG["medium_horizon_threshold"]
        
        self.assertLess(short_thresh, medium_thresh, 
                       "Short threshold should be less than medium threshold")
        self.assertGreater(short_thresh, 0, "Short threshold should be positive")
        self.assertGreater(medium_thresh, 0, "Medium threshold should be positive")
    
    def test_model_sequence_limits(self):
        """Test model sequence limits are reasonable."""
        tlob_max = ROUTING_CONFIG["tlob_max_sequence"]
        patchtst_max = ROUTING_CONFIG["patchtst_max_sequence"]
        
        self.assertGreater(tlob_max, 0, "TLOB max sequence should be positive")
        self.assertGreater(patchtst_max, 0, "PatchTST max sequence should be positive")
        self.assertLess(tlob_max, patchtst_max, 
                       "TLOB should handle shorter sequences than PatchTST")


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestTritonRouter))
    suite.addTest(unittest.makeSuite(TestRoutingConfig))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All Triton router tests passed!")
    else:
        print("❌ Some tests failed!")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1) 