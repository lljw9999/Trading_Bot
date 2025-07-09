#!/usr/bin/env python3
"""
Unit tests for ONNX model runner.

Tests the ONNX inference pipeline for transformer models with
latency validation and functional correctness.
"""

import unittest
import tempfile
import os
import time
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.onnx_runner import ONNXRunner, ModelManager, get_model_manager


class TestONNXRunner(unittest.TestCase):
    """Test ONNX runner functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = ONNXRunner()
        self.models_dir = "models"
        
        # Expected model files
        self.tlob_model_path = os.path.join(self.models_dir, "tlob_tiny_int8.onnx")
        self.patchtst_model_path = os.path.join(self.models_dir, "patchtst_small_int8.onnx")
    
    def test_load_tlob_tiny_model(self):
        """Test loading TLOB-Tiny ONNX model."""
        if not os.path.exists(self.tlob_model_path):
            self.skipTest(f"Model file not found: {self.tlob_model_path}")
        
        success = self.runner.load_model(self.tlob_model_path, "tlob_tiny")
        self.assertTrue(success, "Failed to load TLOB-Tiny model")
        
        # Check model is in loaded models
        self.assertIn("tlob_tiny", self.runner.models)
        
        # Check metadata
        metadata = self.runner.get_model_info("tlob_tiny")
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['input_name'], "orderbook_features")
        self.assertEqual(len(metadata['input_shape']), 3)  # [batch, seq_len, features]
    
    def test_load_patchtst_small_model(self):
        """Test loading PatchTST-Small ONNX model."""
        if not os.path.exists(self.patchtst_model_path):
            self.skipTest(f"Model file not found: {self.patchtst_model_path}")
        
        success = self.runner.load_model(self.patchtst_model_path, "patchtst_small")
        self.assertTrue(success, "Failed to load PatchTST-Small model")
        
        # Check model is in loaded models
        self.assertIn("patchtst_small", self.runner.models)
        
        # Check metadata
        metadata = self.runner.get_model_info("patchtst_small")
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['input_name'], "timeseries_features")
        self.assertEqual(len(metadata['input_shape']), 3)  # [batch, seq_len, features]
    
    def test_tlob_tiny_inference(self):
        """Test TLOB-Tiny model inference."""
        if not os.path.exists(self.tlob_model_path):
            self.skipTest(f"Model file not found: {self.tlob_model_path}")
        
        # Load model
        success = self.runner.load_model(self.tlob_model_path, "tlob_tiny")
        self.assertTrue(success)
        
        # Create dummy input: (batch_size, seq_len, n_features) = (1, 32, 10)
        input_data = np.random.randn(1, 32, 10).astype(np.float32)
        
        # Run inference
        output = self.runner.predict("tlob_tiny", input_data)
        
        # Validate output shape and range
        self.assertEqual(output.shape, (1, 1), "Unexpected output shape")
        self.assertTrue(-1.0 <= output[0, 0] <= 1.0, "Output should be in [-1, 1] range")
    
    def test_patchtst_small_inference(self):
        """Test PatchTST-Small model inference."""
        if not os.path.exists(self.patchtst_model_path):
            self.skipTest(f"Model file not found: {self.patchtst_model_path}")
        
        # Load model
        success = self.runner.load_model(self.patchtst_model_path, "patchtst_small")
        self.assertTrue(success)
        
        # Create dummy input: (batch_size, seq_len, n_features) = (1, 96, 5)
        input_data = np.random.randn(1, 96, 5).astype(np.float32)
        
        # Run inference
        output = self.runner.predict("patchtst_small", input_data)
        
        # Validate output shape: (batch_size, pred_len, n_features) = (1, 8, 5)
        self.assertEqual(output.shape, (1, 8, 5), "Unexpected output shape")
    
    def test_inference_latency_under_3ms(self):
        """Test that inference latency is under 3ms target."""
        if not os.path.exists(self.tlob_model_path):
            self.skipTest(f"Model file not found: {self.tlob_model_path}")
        
        # Load model
        success = self.runner.load_model(self.tlob_model_path, "tlob_tiny")
        self.assertTrue(success)
        
        # Create dummy input
        input_data = np.random.randn(1, 32, 10).astype(np.float32)
        
        # Warm up model (ignore these timings)
        for _ in range(5):
            self.runner.predict("tlob_tiny", input_data)
        
        # Measure inference latency
        latencies = []
        for _ in range(20):
            start_time = time.time()
            self.runner.predict("tlob_tiny", input_data)
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        print(f"\n📊 TLOB-Tiny Latency Stats:")
        print(f"   - Average: {avg_latency:.2f} ms")
        print(f"   - Median: {median_latency:.2f} ms")
        print(f"   - P95: {p95_latency:.2f} ms")
        
        # Assert median latency is under 3ms target
        self.assertLess(median_latency, 3.0, 
                       f"Median latency {median_latency:.2f}ms exceeds 3ms target")
    
    def test_batch_inference(self):
        """Test batch inference functionality."""
        if not os.path.exists(self.tlob_model_path):
            self.skipTest(f"Model file not found: {self.tlob_model_path}")
        
        # Load model
        success = self.runner.load_model(self.tlob_model_path, "tlob_tiny")
        self.assertTrue(success)
        
        # Create batch input
        batch_inputs = [
            np.random.randn(32, 10).astype(np.float32),
            np.random.randn(32, 10).astype(np.float32),
            np.random.randn(32, 10).astype(np.float32)
        ]
        
        # Run batch inference
        batch_outputs = self.runner.predict_batch("tlob_tiny", batch_inputs)
        
        # Validate batch outputs
        self.assertEqual(len(batch_outputs), 3, "Unexpected batch size")
        for output in batch_outputs:
            self.assertEqual(output.shape, (1,), "Unexpected output shape")
    
    def test_model_benchmarking(self):
        """Test model benchmarking functionality."""
        if not os.path.exists(self.tlob_model_path):
            self.skipTest(f"Model file not found: {self.tlob_model_path}")
        
        # Load model
        success = self.runner.load_model(self.tlob_model_path, "tlob_tiny")
        self.assertTrue(success)
        
        # Run benchmark
        stats = self.runner.benchmark_model("tlob_tiny", num_iterations=50)
        
        # Validate benchmark results
        required_keys = ['mean_ms', 'median_ms', 'p95_ms', 'p99_ms', 'min_ms', 'max_ms']
        for key in required_keys:
            self.assertIn(key, stats, f"Missing benchmark stat: {key}")
            self.assertIsInstance(stats[key], (int, float), f"Invalid type for {key}")
        
        # Sanity checks
        self.assertGreater(stats['mean_ms'], 0, "Mean latency should be positive")
        self.assertLessEqual(stats['min_ms'], stats['median_ms'], "Min should be <= median")
        self.assertLessEqual(stats['median_ms'], stats['max_ms'], "Median should be <= max")
    
    def test_model_unloading(self):
        """Test model unloading functionality."""
        if not os.path.exists(self.tlob_model_path):
            self.skipTest(f"Model file not found: {self.tlob_model_path}")
        
        # Load model
        success = self.runner.load_model(self.tlob_model_path, "tlob_tiny")
        self.assertTrue(success)
        self.assertIn("tlob_tiny", self.runner.models)
        
        # Unload model
        success = self.runner.unload_model("tlob_tiny")
        self.assertTrue(success)
        self.assertNotIn("tlob_tiny", self.runner.models)
    
    def test_invalid_model_path(self):
        """Test loading non-existent model file."""
        success = self.runner.load_model("nonexistent/model.onnx", "invalid_model")
        self.assertFalse(success, "Should fail to load non-existent model")
    
    def test_inference_on_unloaded_model(self):
        """Test inference on unloaded model raises error."""
        input_data = np.random.randn(1, 32, 10).astype(np.float32)
        
        with self.assertRaises(ValueError):
            self.runner.predict("unloaded_model", input_data)


class TestModelManager(unittest.TestCase):
    """Test high-level model manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.models_dir = "models"
        self.manager = ModelManager(self.models_dir)
    
    def test_model_manager_initialization(self):
        """Test model manager initialization."""
        self.assertEqual(str(self.manager.models_dir), self.models_dir)
        self.assertEqual(len(self.manager.loaded_models), 0)
    
    def test_load_alpha_models_quantized(self):
        """Test loading quantized alpha models."""
        # Check if model files exist
        tlob_path = Path(self.models_dir) / "tlob_tiny_int8.onnx"
        patchtst_path = Path(self.models_dir) / "patchtst_small_int8.onnx"
        
        if not (tlob_path.exists() and patchtst_path.exists()):
            self.skipTest("Quantized model files not found")
        
        success = self.manager.load_alpha_models(quantized=True)
        self.assertTrue(success, "Failed to load alpha models")
        self.assertEqual(len(self.manager.loaded_models), 2)
        self.assertIn("tlob_tiny", self.manager.loaded_models)
        self.assertIn("patchtst_small", self.manager.loaded_models)
    
    def test_order_book_pressure_prediction(self):
        """Test order book pressure prediction."""
        # Check if model files exist
        tlob_path = Path(self.models_dir) / "tlob_tiny_int8.onnx"
        
        if not tlob_path.exists():
            self.skipTest("TLOB-Tiny model file not found")
        
        # Load models
        success = self.manager.load_alpha_models(quantized=True)
        if not success:
            self.skipTest("Failed to load alpha models")
        
        # Create dummy order book features
        features = np.random.randn(32, 10).astype(np.float32)
        
        # Predict pressure
        pressure = self.manager.predict_order_book_pressure(features)
        
        # Validate prediction
        self.assertIsInstance(pressure, float)
        self.assertTrue(-1.0 <= pressure <= 1.0, "Pressure should be in [-1, 1] range")
    
    def test_price_forecast_prediction(self):
        """Test price forecast prediction."""
        # Check if model files exist
        patchtst_path = Path(self.models_dir) / "patchtst_small_int8.onnx"
        
        if not patchtst_path.exists():
            self.skipTest("PatchTST-Small model file not found")
        
        # Load models
        success = self.manager.load_alpha_models(quantized=True)
        if not success:
            self.skipTest("Failed to load alpha models")
        
        # Create dummy time series
        timeseries = np.random.randn(96, 5).astype(np.float32)
        
        # Predict forecast
        forecast = self.manager.predict_price_forecast(timeseries)
        
        # Validate forecast
        self.assertIsInstance(forecast, np.ndarray)
        self.assertEqual(forecast.shape, (8, 5), "Unexpected forecast shape")
    
    def test_get_model_manager_singleton(self):
        """Test global model manager singleton."""
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        
        # Should return the same instance
        self.assertIs(manager1, manager2, "Should return singleton instance")
    
    def test_manager_status(self):
        """Test model manager status reporting."""
        status = self.manager.get_status()
        
        # Validate status structure
        required_keys = ['loaded_models', 'total_models', 'models_dir', 'model_info']
        for key in required_keys:
            self.assertIn(key, status, f"Missing status key: {key}")
        
        self.assertIsInstance(status['loaded_models'], list)
        self.assertIsInstance(status['total_models'], int)
        self.assertIsInstance(status['model_info'], dict)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestONNXRunner))
    suite.addTest(unittest.makeSuite(TestModelManager))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All ONNX runner tests passed!")
    else:
        print("❌ Some tests failed!")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1) 