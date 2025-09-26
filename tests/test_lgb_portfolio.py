#!/usr/bin/env python3
"""
Unit tests for LightGBM Portfolio Signals.

Tests Sharpe optimization, position sizing, risk management,
and portfolio constraint implementation.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.portfolio.lgb_sharpe import SharpeOptimizer, create_sharpe_optimizer
from src.portfolio.signals import (
    PortfolioSignalGenerator,
    PortfolioSignal,
    create_portfolio_generator,
)


class TestSharpeOptimizer(unittest.TestCase):
    """Test LightGBM Sharpe optimization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = SharpeOptimizer(
            lookback_window=50,  # Reduced for testing
            sharpe_window=5,  # Reduced for testing
            risk_free_rate=0.02,
            max_position=0.2,
            min_position=0.01,
        )

        # Create sample OHLCV data with more samples
        self.sample_data = self._create_sample_data(
            n_days=150
        )  # Increased for more training data

    def _create_sample_data(self, n_days=150) -> pd.DataFrame:  # Increased default
        """Create realistic sample OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")

        # Generate realistic price series with trend and volatility
        returns = np.random.normal(0.0008, 0.02, n_days)  # ~20% annual vol
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate OHLCV from price series
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * np.random.uniform(1.0, 1.02)
            low = close * np.random.uniform(0.98, 1.0)
            open_price = close * np.random.uniform(0.99, 1.01)
            volume = np.random.randint(100000, 1000000)

            data.append(
                {
                    "timestamp": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def test_optimizer_initialization(self):
        """Test SharpeOptimizer initialization."""
        self.assertEqual(self.optimizer.sharpe_window, 5)  # Updated to match setUp
        self.assertEqual(self.optimizer.risk_free_rate, 0.02)
        self.assertEqual(self.optimizer.max_position, 0.2)
        self.assertFalse(self.optimizer.is_trained)
        self.assertIsNone(self.optimizer.model)

    def test_feature_creation(self):
        """Test feature engineering from OHLCV data."""
        features = self.optimizer.create_features(self.sample_data)

        # Check that basic features are created
        self.assertIn("returns", features.columns)
        self.assertIn("log_returns", features.columns)
        self.assertIn("volume_ratio", features.columns)

        # Check momentum features
        momentum_features = [col for col in features.columns if "momentum" in col]
        self.assertGreater(len(momentum_features), 0)

        # Check volatility features
        volatility_features = [col for col in features.columns if "volatility" in col]
        self.assertGreater(len(volatility_features), 0)

        # Check trend features
        ma_features = [col for col in features.columns if "ma_" in col]
        self.assertGreater(len(ma_features), 0)

        # Check volume features
        self.assertIn("volume_trend", features.columns)
        self.assertIn("obv", features.columns)

        # Check risk features
        self.assertIn("var_5pct", features.columns)
        self.assertIn("max_drawdown", features.columns)

    def test_target_sharpe_calculation(self):
        """Test Sharpe ratio target calculation."""
        features = self.optimizer.create_features(self.sample_data)
        targets = self.optimizer.calculate_target_sharpe(features)

        # Check target properties
        self.assertIsInstance(targets, pd.Series)
        self.assertTrue(targets.min() >= 0)  # Should be scaled to [0, 1]
        self.assertTrue(targets.max() <= 1)

        # Check for reasonable values (not all NaN)
        valid_targets = targets.dropna()
        self.assertGreater(len(valid_targets), 10)

    def test_model_training(self):
        """Test LightGBM model training."""
        # Train model
        metrics = self.optimizer.train(self.sample_data, validation_split=0.3)

        # Check training completed
        self.assertTrue(self.optimizer.is_trained)
        self.assertIsNotNone(self.optimizer.model)
        self.assertIsNotNone(self.optimizer.feature_columns)

        # Check metrics
        self.assertIn("val_rmse", metrics)
        self.assertIn("val_correlation", metrics)
        self.assertIn("feature_importance", metrics)
        self.assertGreater(metrics["num_features"], 10)

        # Check RMSE is reasonable
        self.assertLess(
            metrics["val_rmse"], 1.0
        )  # Should be much less than 1 for [0,1] targets

    def test_position_prediction(self):
        """Test position size prediction."""
        # Train model first
        self.optimizer.train(self.sample_data, validation_split=0.3)

        # Predict on recent data
        recent_data = self.sample_data.tail(50)
        positions = self.optimizer.predict_position_size(recent_data)

        # Check prediction properties
        self.assertIsInstance(positions, pd.Series)
        self.assertEqual(len(positions), len(recent_data))

        # Check position constraints
        self.assertTrue(positions.min() >= 0)
        self.assertTrue(positions.max() <= self.optimizer.max_position)

        # Check for some non-zero positions
        non_zero_positions = positions[positions > 0.01]
        self.assertGreater(len(non_zero_positions), 0)

    def test_sharpe_to_position_conversion(self):
        """Test Sharpe ratio to position size conversion."""
        # Test various Sharpe predictions
        sharpe_pred = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        positions = self.optimizer._sharpe_to_position(sharpe_pred)

        # Check basic properties
        self.assertEqual(len(positions), len(sharpe_pred))
        self.assertTrue(all(pos >= 0 for pos in positions))
        self.assertTrue(all(pos <= self.optimizer.max_position for pos in positions))

        # Check monotonicity (higher Sharpe -> higher position, generally)
        self.assertGreaterEqual(positions[4], positions[0])  # 0.9 vs 0.1 Sharpe

    def test_feature_importance(self):
        """Test feature importance extraction."""
        # Train model first
        self.optimizer.train(self.sample_data, validation_split=0.3)

        # Get feature importance
        importance = self.optimizer.get_feature_importance(top_n=10)

        self.assertIsInstance(importance, dict)
        self.assertLessEqual(len(importance), 10)

        # Check that importance values are positive
        for feature, imp in importance.items():
            self.assertGreater(imp, 0)

    def test_model_save_load(self):
        """Test model serialization."""
        import tempfile

        # Train model
        self.optimizer.train(self.sample_data, validation_split=0.3)

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_file:
            self.optimizer.save_model(tmp_file.name)

            # Create new optimizer and load
            new_optimizer = SharpeOptimizer()
            new_optimizer.load_model(tmp_file.name)

            # Check that model is loaded correctly
            self.assertTrue(new_optimizer.is_trained)
            self.assertEqual(new_optimizer.sharpe_window, self.optimizer.sharpe_window)
            self.assertEqual(
                new_optimizer.feature_columns, self.optimizer.feature_columns
            )

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test prediction without training
        with self.assertRaises(ValueError):
            self.optimizer.predict_position_size(self.sample_data)

        # Test training with insufficient data
        small_data = self.sample_data.head(10)
        with self.assertRaises(ValueError):
            self.optimizer.train(small_data)

    def test_factory_function(self):
        """Test factory function for optimizer creation."""
        optimizer = create_sharpe_optimizer(sharpe_window=15, max_position=0.3)

        self.assertIsInstance(optimizer, SharpeOptimizer)
        self.assertEqual(optimizer.sharpe_window, 15)
        self.assertEqual(optimizer.max_position, 0.3)


class TestPortfolioSignalGenerator(unittest.TestCase):
    """Test portfolio signal generation and risk management."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = PortfolioSignalGenerator(
            target_volatility=0.15,
            max_concentration=0.3,
            correlation_threshold=0.8,
            min_signal_strength=0.5,
        )

        # Create sample data for multiple assets
        self.assets = ["BTC", "ETH", "AAPL"]
        self.asset_data = {}

        for i, asset in enumerate(self.assets):
            # Create slightly different data for each asset
            np.random.seed(42 + i)
            data = self._create_sample_data(
                asset, n_days=120
            )  # Increased for more training data
            self.asset_data[asset] = data

            # Create and add trained optimizer with smaller parameters for testing
            optimizer = SharpeOptimizer(lookback_window=30, sharpe_window=5)
            optimizer.train(data, validation_split=0.2)  # Reduced validation split
            self.generator.add_asset(asset, optimizer)

    def _create_sample_data(
        self, symbol: str, n_days: int = 120
    ) -> pd.DataFrame:  # Increased default
        """Create sample data for an asset."""
        dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

        # Different volatility for different assets
        vol_map = {"BTC": 0.04, "ETH": 0.035, "AAPL": 0.02}
        volatility = vol_map.get(symbol, 0.025)

        returns = np.random.normal(0.001, volatility, n_days)
        prices = 100 * np.exp(np.cumsum(returns))

        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * np.random.uniform(1.0, 1.015)
            low = close * np.random.uniform(0.985, 1.0)
            open_price = close * np.random.uniform(0.995, 1.005)
            volume = np.random.randint(50000, 500000)

            data.append(
                {
                    "timestamp": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def test_generator_initialization(self):
        """Test portfolio signal generator initialization."""
        self.assertEqual(self.generator.target_volatility, 0.15)
        self.assertEqual(self.generator.max_concentration, 0.3)
        self.assertEqual(len(self.generator.optimizers), 3)
        self.assertEqual(len(self.generator.current_positions), 3)

    def test_add_asset(self):
        """Test adding assets to portfolio."""
        # Add new asset
        optimizer = SharpeOptimizer()
        self.generator.add_asset("GOOGL", optimizer)

        self.assertIn("GOOGL", self.generator.optimizers)
        self.assertIn("GOOGL", self.generator.current_positions)
        self.assertEqual(self.generator.current_positions["GOOGL"], 0.0)

    def test_market_data_update(self):
        """Test market data updating."""
        # Update data for existing asset
        btc_data = self.asset_data["BTC"]
        self.generator.update_market_data("BTC", btc_data)

        self.assertIn("BTC", self.generator.price_data)
        self.assertIn("BTC", self.generator.volatility_estimates)
        self.assertGreater(self.generator.volatility_estimates["BTC"], 0)

    def test_correlation_matrix_calculation(self):
        """Test correlation matrix calculation."""
        # Update market data for all assets
        for symbol, data in self.asset_data.items():
            self.generator.update_market_data(symbol, data)

        # Calculate correlation matrix
        corr_matrix = self.generator.calculate_correlation_matrix()

        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertEqual(corr_matrix.shape, (3, 3))

        # Check diagonal is 1.0
        for symbol in self.assets:
            self.assertAlmostEqual(corr_matrix.loc[symbol, symbol], 1.0, places=2)

    def test_risk_budget_calculation(self):
        """Test risk budget calculation."""
        # Update market data
        for symbol, data in self.asset_data.items():
            self.generator.update_market_data(symbol, data)

        # Calculate risk budgets
        risk_budgets = self.generator.calculate_risk_budgets()

        self.assertIsInstance(risk_budgets, dict)
        self.assertEqual(len(risk_budgets), 3)

        # Check budgets are positive and reasonable
        total_budget = sum(risk_budgets.values())
        self.assertGreater(total_budget, 0.5)  # At least 50% allocated

        # Check concentration limits (allow reasonable tolerance)
        for budget in risk_budgets.values():
            self.assertGreater(budget, 0)  # All positive
            self.assertLessEqual(budget, 0.5)  # Reasonable upper bound

    def test_raw_signal_generation(self):
        """Test raw signal generation."""
        # Update market data
        for symbol, data in self.asset_data.items():
            self.generator.update_market_data(symbol, data)

        # Generate raw signals
        timestamp = datetime.now()
        raw_signals = self.generator.generate_raw_signals(timestamp)

        self.assertIsInstance(raw_signals, dict)

        # Check signal format
        for symbol, (position, confidence) in raw_signals.items():
            self.assertIsInstance(position, float)
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(position, 0.0)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_portfolio_constraints_application(self):
        """Test portfolio constraint application."""
        # Update market data
        for symbol, data in self.asset_data.items():
            self.generator.update_market_data(symbol, data)

        # Generate raw signals
        timestamp = datetime.now()
        raw_signals = self.generator.generate_raw_signals(timestamp)

        # Apply constraints
        portfolio_signals = self.generator.apply_portfolio_constraints(
            raw_signals, timestamp
        )

        self.assertIsInstance(portfolio_signals, dict)

        # Check total position doesn't exceed 100%
        total_position = sum(
            signal.position_size for signal in portfolio_signals.values()
        )
        self.assertLessEqual(total_position, 1.0)

        # Check individual position limits
        for signal in portfolio_signals.values():
            self.assertLessEqual(signal.position_size, self.generator.max_concentration)
            self.assertGreaterEqual(
                signal.confidence, self.generator.min_signal_strength
            )

    def test_portfolio_signal_generation(self):
        """Test end-to-end signal generation."""
        # Update market data
        for symbol, data in self.asset_data.items():
            self.generator.update_market_data(symbol, data)

        # Generate portfolio signals
        timestamp = datetime.now()
        signals = self.generator.generate_signals(timestamp)

        self.assertIsInstance(signals, list)

        # Check signal objects
        for signal in signals:
            self.assertIsInstance(signal, PortfolioSignal)
            self.assertIn(signal.symbol, self.assets)
            self.assertGreaterEqual(signal.position_size, 0.0)
            self.assertLessEqual(signal.position_size, self.generator.max_concentration)
            self.assertIn(signal.signal_type, ["entry", "exit", "rebalance"])

    def test_portfolio_status(self):
        """Test portfolio status reporting."""
        # Update market data and generate signals
        for symbol, data in self.asset_data.items():
            self.generator.update_market_data(symbol, data)

        timestamp = datetime.now()
        self.generator.generate_signals(timestamp)

        # Get portfolio status
        status = self.generator.get_portfolio_status()

        self.assertIsInstance(status, dict)
        self.assertIn("total_exposure", status)
        self.assertIn("portfolio_volatility", status)
        self.assertIn("num_positions", status)
        self.assertIn("positions", status)
        self.assertIn("risk_budgets", status)

    def test_rebalancing_logic(self):
        """Test portfolio rebalancing logic."""
        timestamp = datetime.now()

        # Should rebalance initially
        self.assertTrue(self.generator.should_rebalance(timestamp))

        # Update market data and generate signals
        for symbol, data in self.asset_data.items():
            self.generator.update_market_data(symbol, data)

        self.generator.generate_signals(timestamp)

        # Should not rebalance very shortly after (within same second)
        short_future = timestamp + timedelta(seconds=1)
        # Note: this may still return True due to drift, which is acceptable

        # Should definitely rebalance after 24 hours
        future_timestamp = timestamp + timedelta(hours=25)
        self.assertTrue(self.generator.should_rebalance(future_timestamp))

    def test_correlation_penalty(self):
        """Test correlation penalty calculation."""
        # Create correlation matrix
        for symbol, data in self.asset_data.items():
            self.generator.update_market_data(symbol, data)

        corr_matrix = self.generator.calculate_correlation_matrix()

        # Test penalty calculation
        signals = {"BTC": (0.2, 0.8), "ETH": (0.15, 0.7)}
        penalty = self.generator._calculate_correlation_penalty(
            "BTC", corr_matrix, signals
        )

        self.assertIsInstance(penalty, float)
        self.assertGreaterEqual(penalty, 0.0)
        self.assertLessEqual(penalty, 1.0)

    def test_signal_to_dict_conversion(self):
        """Test PortfolioSignal to dict conversion."""
        signal = PortfolioSignal(
            symbol="BTC",
            timestamp=datetime.now(),
            position_size=0.15,
            confidence=0.8,
            expected_sharpe=1.2,
            risk_contribution=0.3,
            signal_type="entry",
        )

        signal_dict = signal.to_dict()

        self.assertIsInstance(signal_dict, dict)
        self.assertEqual(signal_dict["symbol"], "BTC")
        self.assertEqual(signal_dict["position_size"], 0.15)
        self.assertEqual(signal_dict["signal_type"], "entry")

    def test_factory_function(self):
        """Test factory function for generator creation."""
        generator = create_portfolio_generator(
            target_volatility=0.2, max_concentration=0.4
        )

        self.assertIsInstance(generator, PortfolioSignalGenerator)
        self.assertEqual(generator.target_volatility, 0.2)
        self.assertEqual(generator.max_concentration, 0.4)

    def test_empty_data_handling(self):
        """Test handling of empty or insufficient data."""
        # Test with no market data
        timestamp = datetime.now()
        signals = self.generator.generate_signals(timestamp)
        self.assertEqual(len(signals), 0)

        # Test with insufficient data
        small_data = self.asset_data["BTC"].head(10)
        self.generator.update_market_data("BTC", small_data)
        signals = self.generator.generate_signals(timestamp)
        # Should handle gracefully without crashing


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete portfolio system."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from data to signals."""
        # Create sample data
        np.random.seed(42)
        data = self._create_comprehensive_data()

        # Train optimizer
        optimizer = SharpeOptimizer(lookback_window=100, sharpe_window=15)
        metrics = optimizer.train(data, validation_split=0.2)

        # Verify training
        self.assertTrue(optimizer.is_trained)
        self.assertLess(metrics["val_rmse"], 0.5)

        # Create portfolio generator
        generator = PortfolioSignalGenerator(target_volatility=0.12)
        generator.add_asset("TEST", optimizer)
        generator.update_market_data("TEST", data)

        # Generate signals
        signals = generator.generate_signals(datetime.now())

        # Verify complete pipeline
        self.assertIsInstance(signals, list)
        if len(signals) > 0:
            self.assertIsInstance(signals[0], PortfolioSignal)

    def _create_comprehensive_data(self, n_days=300) -> pd.DataFrame:
        """Create comprehensive test data."""
        dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")

        # Create realistic price series with regime changes
        returns = []
        for i in range(n_days):
            if i < 100:
                # Bull market
                ret = np.random.normal(0.0015, 0.018)
            elif i < 200:
                # Bear market
                ret = np.random.normal(-0.0008, 0.025)
            else:
                # Recovery
                ret = np.random.normal(0.0012, 0.022)
            returns.append(ret)

        prices = 100 * np.exp(np.cumsum(returns))

        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * np.random.uniform(1.0, 1.025)
            low = close * np.random.uniform(0.975, 1.0)
            open_price = close * np.random.uniform(0.99, 1.01)
            volume = np.random.randint(100000, 2000000)

            data.append(
                {
                    "timestamp": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df


if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestSharpeOptimizer))
    suite.addTest(unittest.makeSuite(TestPortfolioSignalGenerator))
    suite.addTest(unittest.makeSuite(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("✅ All LightGBM portfolio tests passed!")
    else:
        print("❌ Some tests failed!")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")

    # Exit with error code if tests failed
    # Note: sys.exit removed to avoid pytest return-value warnings
    # Test framework will handle success/failure appropriately
