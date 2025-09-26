#!/usr/bin/env python3
"""
Unit tests for sentiment bus functionality.

Tests the ±90 second lookup window for sentiment data integration
with the feature bus system.
"""

import unittest
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.layers.layer0_data_ingestion.feature_bus import FeatureBus
from src.layers.layer0_data_ingestion.schemas import FeatureSnapshot


class TestSentimentBus(unittest.TestCase):
    """Test sentiment data lookup functionality in the feature bus."""

    def setUp(self):
        """Set up test fixtures."""
        self.feature_bus = FeatureBus()
        self.test_symbol = "BTC-USD"
        self.base_time = datetime.now(timezone.utc)

    def test_sentiment_lookup_within_window(self):
        """Test that sentiment lookup returns correct data within ±90s window."""
        # Add sentiment data at base time
        sentiment_score = 0.75
        asyncio.run(
            self.feature_bus.update_sentiment(
                self.test_symbol, sentiment_score, self.base_time
            )
        )

        # Test lookup 60 seconds later (within window)
        lookup_time = self.base_time + timedelta(seconds=60)
        result = self.feature_bus._get_latest_sentiment(self.test_symbol, lookup_time)

        self.assertIsNotNone(result)
        self.assertEqual(result, sentiment_score)

    def test_sentiment_lookup_at_window_edge(self):
        """Test sentiment lookup exactly at ±90s boundary."""
        sentiment_score = -0.85
        asyncio.run(
            self.feature_bus.update_sentiment(
                self.test_symbol, sentiment_score, self.base_time
            )
        )

        # Test lookup exactly 90 seconds later (at boundary)
        lookup_time = self.base_time + timedelta(seconds=90)
        result = self.feature_bus._get_latest_sentiment(self.test_symbol, lookup_time)

        self.assertIsNotNone(result)
        self.assertEqual(result, sentiment_score)

        # Test lookup exactly 90 seconds before (at boundary)
        lookup_time = self.base_time - timedelta(seconds=90)
        result = self.feature_bus._get_latest_sentiment(self.test_symbol, lookup_time)

        self.assertIsNotNone(result)
        self.assertEqual(result, sentiment_score)

    def test_sentiment_lookup_outside_window(self):
        """Test that sentiment lookup returns None outside ±90s window."""
        sentiment_score = 0.45
        asyncio.run(
            self.feature_bus.update_sentiment(
                self.test_symbol, sentiment_score, self.base_time
            )
        )

        # Test lookup 91 seconds later (outside window)
        lookup_time = self.base_time + timedelta(seconds=91)
        result = self.feature_bus._get_latest_sentiment(self.test_symbol, lookup_time)

        self.assertIsNone(result)

        # Test lookup 91 seconds before (outside window)
        lookup_time = self.base_time - timedelta(seconds=91)
        result = self.feature_bus._get_latest_sentiment(self.test_symbol, lookup_time)

        self.assertIsNone(result)

    def test_sentiment_lookup_no_data(self):
        """Test sentiment lookup when no data exists for symbol."""
        result = self.feature_bus._get_latest_sentiment(
            "UNKNOWN-SYMBOL", self.base_time
        )
        self.assertIsNone(result)

    def test_sentiment_lookup_multiple_records(self):
        """Test that sentiment lookup returns the most recent record within window."""
        # Add multiple sentiment records
        old_sentiment = 0.3
        new_sentiment = 0.8

        old_time = self.base_time - timedelta(seconds=30)
        new_time = self.base_time + timedelta(seconds=30)

        asyncio.run(
            self.feature_bus.update_sentiment(self.test_symbol, old_sentiment, old_time)
        )
        asyncio.run(
            self.feature_bus.update_sentiment(self.test_symbol, new_sentiment, new_time)
        )

        # Lookup should return the newer sentiment
        lookup_time = self.base_time + timedelta(seconds=45)
        result = self.feature_bus._get_latest_sentiment(self.test_symbol, lookup_time)

        self.assertIsNotNone(result)
        self.assertEqual(result, new_sentiment)

    def test_fundamental_lookup_within_window(self):
        """Test fundamental P/E ratio lookup within ±90s window."""
        pe_ratio = 25.5
        asyncio.run(
            self.feature_bus.update_fundamental(
                self.test_symbol, pe_ratio, self.base_time
            )
        )

        # Test lookup 60 seconds later (within window)
        lookup_time = self.base_time + timedelta(seconds=60)
        result = self.feature_bus._get_latest_fundamental(self.test_symbol, lookup_time)

        self.assertIsNotNone(result)
        self.assertEqual(result, pe_ratio)

    def test_fundamental_lookup_outside_window(self):
        """Test fundamental lookup returns None outside ±90s window."""
        pe_ratio = 18.3
        asyncio.run(
            self.feature_bus.update_fundamental(
                self.test_symbol, pe_ratio, self.base_time
            )
        )

        # Test lookup 100 seconds later (outside window)
        lookup_time = self.base_time + timedelta(seconds=100)
        result = self.feature_bus._get_latest_fundamental(self.test_symbol, lookup_time)

        self.assertIsNone(result)

    def test_feature_computation_with_sentiment(self):
        """Test feature computation includes sentiment data when available."""
        # Add sentiment data
        sentiment_score = 0.6
        asyncio.run(
            self.feature_bus.update_sentiment(
                self.test_symbol, sentiment_score, self.base_time
            )
        )

        # Mock a market tick
        mock_tick = Mock()
        mock_tick.symbol = self.test_symbol
        mock_tick.timestamp = self.base_time + timedelta(seconds=30)
        mock_tick.mid = 50000.0
        mock_tick.spread = 10.0
        mock_tick.spread_bps = 2.0
        mock_tick.bid_size = 1.0
        mock_tick.ask_size = 1.5
        mock_tick.volume = 100.0

        # Process tick and check if sentiment is included
        features = asyncio.run(self.feature_bus.process_tick(mock_tick))

        self.assertIsNotNone(features)
        self.assertEqual(features.sent_score, sentiment_score)

    def test_feature_computation_without_sentiment(self):
        """Test feature computation with default neutral sentiment when no data."""
        # Mock a market tick without any sentiment data
        mock_tick = Mock()
        mock_tick.symbol = "NEW-SYMBOL"
        mock_tick.timestamp = self.base_time
        mock_tick.mid = 50000.0
        mock_tick.spread = 10.0
        mock_tick.spread_bps = 2.0
        mock_tick.bid_size = 1.0
        mock_tick.ask_size = 1.5
        mock_tick.volume = 100.0

        # Process tick and check default sentiment
        features = asyncio.run(self.feature_bus.process_tick(mock_tick))

        self.assertIsNotNone(features)
        self.assertEqual(features.sent_score, 0.0)  # Default neutral sentiment


class TestSentimentBusAsync(unittest.TestCase):
    """Test async functionality of sentiment bus."""

    def setUp(self):
        """Set up test fixtures."""
        self.feature_bus = FeatureBus()
        self.test_symbol = "ETH-USD"

    def test_async_sentiment_update(self):
        """Test async sentiment update functionality."""

        async def run_test():
            sentiment_score = 0.9
            timestamp = datetime.now(timezone.utc)

            # Update sentiment asynchronously
            await self.feature_bus.update_sentiment(
                self.test_symbol, sentiment_score, timestamp
            )

            # Verify data was stored
            result = self.feature_bus._get_latest_sentiment(self.test_symbol, timestamp)
            self.assertEqual(result, sentiment_score)

        asyncio.run(run_test())

    def test_async_fundamental_update(self):
        """Test async fundamental update functionality."""

        async def run_test():
            pe_ratio = 22.4
            timestamp = datetime.now(timezone.utc)

            # Update fundamental asynchronously
            await self.feature_bus.update_fundamental(
                self.test_symbol, pe_ratio, timestamp
            )

            # Verify data was stored
            result = self.feature_bus._get_latest_fundamental(
                self.test_symbol, timestamp
            )
            self.assertEqual(result, pe_ratio)

        asyncio.run(run_test())


if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestSentimentBus))
    suite.addTest(unittest.makeSuite(TestSentimentBusAsync))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("✅ All sentiment bus tests passed!")
    else:
        print("❌ Some tests failed!")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
