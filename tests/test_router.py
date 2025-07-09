#!/usr/bin/env python3
"""
Tests for Model Router (Task A)

Validates routing logic for 8 parameter combinations and sub-50µs latency.
"""

import unittest
import time
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.router import (
    ModelRouter, AssetClass, ModelFamily, InstrumentInfo, 
    RoutingRule, ModelRouterConfig, create_model_router
)


class TestModelRouter(unittest.TestCase):
    """Test Model Router implementation."""
    
    def setUp(self):
        """Set up test environment with mocked Redis."""
        self.mock_redis = Mock()
        self.router = ModelRouter(redis_client=self.mock_redis)
        
        # Mock instrument data
        self.instrument_data = {
            "BTC-USD": {
                "asset_class": "crypto",
                "exchange": "coinbase",
                "tick_size": "0.01",
                "lot_size": "0.001",
                "market_hours": "{}"
            },
            "ETH-USD": {
                "asset_class": "crypto", 
                "exchange": "coinbase",
                "tick_size": "0.01",
                "lot_size": "0.001",
                "market_hours": "{}"
            },
            "AAPL": {
                "asset_class": "us_stocks",
                "exchange": "nasdaq",
                "tick_size": "0.01",
                "lot_size": "1",
                "market_hours": '{"open": "09:30", "close": "16:00"}'
            },
            "TSLA": {
                "asset_class": "us_stocks",
                "exchange": "nasdaq", 
                "tick_size": "0.01",
                "lot_size": "1",
                "market_hours": '{"open": "09:30", "close": "16:00"}'
            },
            "000001.SZ": {
                "asset_class": "a_shares",
                "exchange": "shenzhen",
                "tick_size": "0.01",
                "lot_size": "100", 
                "market_hours": '{"open": "09:30", "close": "15:00"}'
            }
        }
    
    def _setup_redis_mock(self, symbol: str):
        """Setup Redis mock to return instrument data."""
        def mock_hgetall(key):
            if key == f"instrument:info:{symbol}":
                return {k.encode(): v.encode() for k, v in self.instrument_data[symbol].items()}
            return {}
        
        self.mock_redis.hgetall.side_effect = mock_hgetall
    
    def test_crypto_high_frequency_routing(self):
        """Test Case 1: Crypto + High Frequency (< 1 min) → TLOB-Tiny."""
        self._setup_redis_mock("BTC-USD")
        
        model_id = self.router.select_model("BTC-USD", horizon_ms=30000)  # 30 seconds
        self.assertEqual(model_id, "tlob_tiny")
        
        print("✅ Test 1: Crypto high-frequency → tlob_tiny")
    
    def test_crypto_medium_frequency_routing(self):
        """Test Case 2: Crypto + Medium Frequency (1 min - 2h) → PatchTST-Small."""
        self._setup_redis_mock("ETH-USD")
        
        model_id = self.router.select_model("ETH-USD", horizon_ms=300000)  # 5 minutes
        self.assertEqual(model_id, "patchtst_small")
        
        print("✅ Test 2: Crypto medium-frequency → patchtst_small")
    
    def test_crypto_long_term_routing(self):
        """Test Case 3: Crypto + Long Term (> 2h) → MambaTS-Small."""
        self._setup_redis_mock("BTC-USD")
        
        model_id = self.router.select_model("BTC-USD", horizon_ms=14400000)  # 4 hours
        self.assertEqual(model_id, "mamba_ts_small")
        
        print("✅ Test 3: Crypto long-term → mamba_ts_small")
    
    def test_us_stocks_intraday_routing(self):
        """Test Case 4: US Stocks + Intraday (< 4h) → TimesNet-Base."""
        self._setup_redis_mock("AAPL")
        
        model_id = self.router.select_model("AAPL", horizon_ms=3600000)  # 1 hour
        self.assertEqual(model_id, "timesnet_base")
        
        print("✅ Test 4: US stocks intraday → timesnet_base")
    
    def test_us_stocks_overnight_routing(self):
        """Test Case 5: US Stocks + Overnight (> 4h) → MambaTS-Small."""
        self._setup_redis_mock("TSLA")
        
        model_id = self.router.select_model("TSLA", horizon_ms=28800000)  # 8 hours
        self.assertEqual(model_id, "mamba_ts_small")
        
        print("✅ Test 5: US stocks overnight → mamba_ts_small")
    
    def test_a_shares_intraday_routing(self):
        """Test Case 6: A-shares + Intraday (< 4h) → TimesNet-Base."""
        self._setup_redis_mock("000001.SZ")
        
        model_id = self.router.select_model("000001.SZ", horizon_ms=7200000)  # 2 hours
        self.assertEqual(model_id, "timesnet_base")
        
        print("✅ Test 6: A-shares intraday → timesnet_base")
    
    def test_a_shares_overnight_routing(self):
        """Test Case 7: A-shares + Overnight (> 4h) → Chronos-Bolt-Base."""
        self._setup_redis_mock("000001.SZ")
        
        model_id = self.router.select_model("000001.SZ", horizon_ms=21600000)  # 6 hours
        self.assertEqual(model_id, "chronos_bolt_base")
        
        print("✅ Test 7: A-shares overnight → chronos_bolt_base")
    
    def test_fallback_routing(self):
        """Test Case 8: Unknown instrument → Default fallback."""
        # Setup empty Redis response
        self.mock_redis.hgetall.return_value = {}
        
        model_id = self.router.select_model("UNKNOWN", horizon_ms=60000)
        self.assertEqual(model_id, "tlob_tiny")  # Default fallback
        
        print("✅ Test 8: Unknown instrument → tlob_tiny (fallback)")
    
    def test_latency_requirement(self):
        """Test that router latency is ≤ 50µs per call."""
        self._setup_redis_mock("BTC-USD")
        
        # Warm up the router (first call may be slower due to cache)
        self.router.select_model("BTC-USD", horizon_ms=30000)
        
        # Measure latency over multiple calls
        latencies = []
        num_calls = 100
        
        for _ in range(num_calls):
            start_time = time.perf_counter()
            self.router.select_model("BTC-USD", horizon_ms=30000)
            end_time = time.perf_counter()
            
            latency_us = (end_time - start_time) * 1_000_000
            latencies.append(latency_us)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        print(f"✅ Latency stats over {num_calls} calls:")
        print(f"   Average: {avg_latency:.1f}µs")
        print(f"   Min: {min_latency:.1f}µs")
        print(f"   Max: {max_latency:.1f}µs")
        
        # Check acceptance criteria
        self.assertLessEqual(avg_latency, 50.0, f"Average latency {avg_latency:.1f}µs exceeds 50µs")
        self.assertLessEqual(max_latency, 100.0, f"Max latency {max_latency:.1f}µs too high")
    
    def test_instrument_info_caching(self):
        """Test that instrument info is properly cached."""
        self._setup_redis_mock("BTC-USD")
        
        # First call should hit Redis
        self.router.select_model("BTC-USD", horizon_ms=30000)
        self.assertEqual(self.mock_redis.hgetall.call_count, 1)
        
        # Second call should use cache
        self.router.select_model("BTC-USD", horizon_ms=30000)
        self.assertEqual(self.mock_redis.hgetall.call_count, 1)  # No additional calls
        
        print("✅ Instrument info caching works correctly")
    
    def test_symbol_inference(self):
        """Test asset class inference from symbol patterns."""
        # Mock Redis to return empty data (forcing inference)
        self.mock_redis.hgetall.return_value = {}
        
        test_cases = [
            ("BTC-USD", "crypto"),
            ("ETH-USDT", "crypto"), 
            ("AAPL", "us_stocks"),
            ("MSFT", "us_stocks"),
            ("000001.SZ", "a_shares"),
            ("600000.SS", "a_shares")
        ]
        
        for symbol, expected_asset_class in test_cases:
            instrument_info = self.router._get_instrument_info(symbol)
            self.assertIsNotNone(instrument_info)
            self.assertEqual(instrument_info.asset_class.value, expected_asset_class)
        
        print("✅ Symbol inference works for all asset classes")
    
    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        self._setup_redis_mock("BTC-USD")
        
        # Reset stats
        self.router.reset_performance_stats()
        
        # Make some calls
        for _ in range(5):
            self.router.select_model("BTC-USD", horizon_ms=30000)
        
        stats = self.router.get_performance_stats()
        self.assertEqual(stats["call_count"], 5)
        self.assertGreater(stats["avg_latency_us"], 0)
        self.assertGreater(stats["total_latency_us"], 0)
        
        print("✅ Performance metrics tracking works correctly")
    
    def test_model_selection_publish(self):
        """Test publishing model selection to Redis."""
        self.router.publish_selection("BTC-USD", "tlob_tiny", {"confidence": 0.85})
        
        # Verify Redis publish was called
        self.mock_redis.publish.assert_called_once()
        call_args = self.mock_redis.publish.call_args
        
        channel = call_args[0][0]
        message = call_args[0][1]
        
        self.assertEqual(channel, "alpha.selected.BTC-USD")
        self.assertIn("tlob_tiny", message)
        self.assertIn("confidence", message)
        
        print("✅ Model selection publishing works correctly")


class TestModelRouterConfig(unittest.TestCase):
    """Test Model Router configuration and rule compilation."""
    
    def test_default_config_loading(self):
        """Test default configuration loading."""
        router = ModelRouter(redis_client=Mock())
        
        # Should have loaded default rules
        self.assertGreater(len(router._compiled_rules), 0)
        self.assertEqual(router.config.default_model, "tlob_tiny")
        
        print("✅ Default configuration loads correctly")
    
    def test_rule_priority_sorting(self):
        """Test that rules are sorted by priority."""
        router = ModelRouter(redis_client=Mock())
        
        # Rules should be sorted by priority (ascending)
        for i in range(len(router._compiled_rules) - 1):
            current_priority = router._compiled_rules[i].priority
            next_priority = router._compiled_rules[i + 1].priority
            self.assertLessEqual(current_priority, next_priority)
        
        print("✅ Rules are properly sorted by priority")
    
    def test_config_hot_reload(self):
        """Test configuration hot reload functionality."""
        router = ModelRouter(redis_client=Mock())
        original_rule_count = len(router._compiled_rules)
        
        # Create new config with additional rule
        new_config = ModelRouterConfig(
            rules=[
                {
                    "asset_class": "forex",
                    "horizon_min_ms": 0,
                    "horizon_max_ms": 3600000,
                    "model_id": "tlob_tiny",
                    "priority": 5
                }
            ]
        )
        
        router.reload_config(new_config)
        
        # Should have reloaded with new rules
        self.assertEqual(len(router._compiled_rules), 1)
        self.assertNotEqual(len(router._compiled_rules), original_rule_count)
        
        print("✅ Configuration hot reload works correctly")


class TestModelRouterFactory(unittest.TestCase):
    """Test Model Router factory function."""
    
    @patch('src.core.router.redis.Redis')
    def test_factory_creation(self, mock_redis_class):
        """Test factory function creates router correctly."""
        mock_redis_instance = Mock()
        mock_redis_class.from_url.return_value = mock_redis_instance
        
        router = create_model_router("redis://localhost:6379/1")
        
        # Verify Redis client creation
        mock_redis_class.from_url.assert_called_once_with("redis://localhost:6379/1")
        self.assertIsInstance(router, ModelRouter)
        
        print("✅ Factory function works correctly")


if __name__ == '__main__':
    print("🧪 Running Model Router Tests (Task A)...")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases for 8 parameter combinations + performance
    suite.addTest(unittest.makeSuite(TestModelRouter))
    suite.addTest(unittest.makeSuite(TestModelRouterConfig))
    suite.addTest(unittest.makeSuite(TestModelRouterFactory))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 60)
    print(f"📊 Model Router Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All Model Router tests passed!")
        print("🎯 Task A: Model Router - COMPLETED")
        print()
        print("✅ Implementation Summary:")
        print("   • 8 parameter combinations validated")
        print("   • Sub-50µs latency achieved")
        print("   • Factory + strategy pattern implemented")
        print("   • Redis instrument metadata integration")
        print("   • Performance metrics tracking")
        print("   • Hot-reload configuration support")
        print()
        print("🚀 Ready for Task B: Signal Mux/Demux!")
    else:
        print("❌ Some Model Router tests failed!")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1) 