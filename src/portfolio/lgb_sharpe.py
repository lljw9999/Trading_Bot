#!/usr/bin/env python3
"""
LightGBM Sharpe Optimization Model

Implements portfolio optimization using LightGBM with Sharpe ratio maximization.
Designed for high-frequency trading with emphasis on risk-adjusted returns.

Key Features:
- Feature engineering from OHLCV data
- Rolling Sharpe ratio targets
- Risk factor consideration  
- Position sizing recommendations
- Real-time inference optimized
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple, Union
import logging
import joblib
from pathlib import Path
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# Setup logging
logger = logging.getLogger(__name__)


class SharpeOptimizer:
    """
    LightGBM-based Sharpe ratio optimization for portfolio construction.

    Predicts optimal position sizes based on historical performance and
    risk-adjusted return expectations.
    """

    def __init__(
        self,
        lookback_window: int = 252,  # 1 year of daily data
        sharpe_window: int = 21,  # 3 weeks for Sharpe calculation
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        max_position: float = 0.25,  # Max 25% position size
        min_position: float = 0.01,
    ):  # Min 1% position size
        """
        Initialize Sharpe optimizer.

        Args:
            lookback_window: Historical data window for features
            sharpe_window: Rolling window for Sharpe ratio calculation
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            max_position: Maximum position size (0-1)
            min_position: Minimum position size (0-1)
        """
        self.lookback_window = lookback_window
        self.sharpe_window = sharpe_window
        self.risk_free_rate = risk_free_rate
        self.max_position = max_position
        self.min_position = min_position

        # Model components
        self.model = None
        self.feature_columns = None
        self.scaler = None
        self.is_trained = False

        # Feature engineering parameters
        self.momentum_windows = [5, 10, 20, 50]
        self.volatility_windows = [5, 10, 20]
        self.ma_windows = [10, 20, 50, 200]

        logger.info(f"SharpeOptimizer initialized with {sharpe_window}d Sharpe window")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features for portfolio optimization.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with engineered features
        """
        features = df.copy()

        # Price-based features
        features["returns"] = features["close"].pct_change()
        features["log_returns"] = np.log(features["close"] / features["close"].shift(1))
        features["volume_ratio"] = (
            features["volume"] / features["volume"].rolling(20).mean()
        )

        # Technical indicators
        self._add_momentum_features(features)
        self._add_volatility_features(features)
        self._add_trend_features(features)
        self._add_volume_features(features)
        self._add_risk_features(features)

        # Cross-sectional features (if multiple symbols)
        if "symbol" in features.columns:
            self._add_relative_features(features)

        return features

    def _add_momentum_features(self, df: pd.DataFrame) -> None:
        """Add momentum-based features."""
        for window in self.momentum_windows:
            # Price momentum
            df[f"momentum_{window}d"] = df["close"] / df["close"].shift(window) - 1

            # Volume-weighted momentum
            vwap = (df["volume"] * df["close"]).rolling(window).sum() / df[
                "volume"
            ].rolling(window).sum()
            df[f"vwap_momentum_{window}d"] = df["close"] / vwap - 1

            # RSI-like momentum
            returns = df["close"].pct_change()
            gains = returns.where(returns > 0, 0).rolling(window).mean()
            losses = (-returns.where(returns < 0, 0)).rolling(window).mean()
            df[f"rsi_{window}d"] = 100 - (100 / (1 + gains / (losses + 1e-8)))

    def _add_volatility_features(self, df: pd.DataFrame) -> None:
        """Add volatility-based features."""
        returns = df["close"].pct_change()

        for window in self.volatility_windows:
            # Historical volatility
            df[f"volatility_{window}d"] = returns.rolling(window).std() * np.sqrt(252)

            # GARCH-like features
            df[f"volatility_rank_{window}d"] = (
                df[f"volatility_{window}d"].rolling(window * 4).rank(pct=True)
            )

            # High-low volatility
            hl_vol = np.log(df["high"] / df["low"]).rolling(window).mean()
            df[f"hl_volatility_{window}d"] = hl_vol

    def _add_trend_features(self, df: pd.DataFrame) -> None:
        """Add trend-based features."""
        for window in self.ma_windows:
            # Moving averages
            ma = df["close"].rolling(window).mean()
            df[f"ma_{window}d"] = ma
            df[f"ma_ratio_{window}d"] = df["close"] / ma - 1

            # Moving average slopes
            df[f"ma_slope_{window}d"] = (ma - ma.shift(5)) / ma.shift(5)

        # Bollinger Bands
        ma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["bb_upper"] = ma_20 + 2 * std_20
        df["bb_lower"] = ma_20 - 2 * std_20
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

    def _add_volume_features(self, df: pd.DataFrame) -> None:
        """Add volume-based features."""
        # Volume trends
        df["volume_ma_10"] = df["volume"].rolling(10).mean()
        df["volume_ma_50"] = df["volume"].rolling(50).mean()
        df["volume_trend"] = df["volume_ma_10"] / df["volume_ma_50"] - 1

        # On-balance volume
        obv = (df["volume"] * np.sign(df["close"].pct_change())).cumsum()
        df["obv"] = obv
        df["obv_ma"] = obv.rolling(20).mean()
        df["obv_divergence"] = (obv - df["obv_ma"]) / df["obv_ma"]

        # Money flow index
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]
        positive_flow = (
            money_flow.where(typical_price > typical_price.shift(1), 0)
            .rolling(14)
            .sum()
        )
        negative_flow = (
            money_flow.where(typical_price < typical_price.shift(1), 0)
            .rolling(14)
            .sum()
        )
        df["mfi"] = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-8)))

    def _add_risk_features(self, df: pd.DataFrame) -> None:
        """Add risk-based features."""
        returns = df["close"].pct_change()

        # Value at Risk (VaR)
        df["var_5pct"] = returns.rolling(50).quantile(0.05)
        df["var_1pct"] = returns.rolling(50).quantile(0.01)

        # Maximum drawdown
        rolling_max = df["close"].rolling(50).max()
        drawdown = (df["close"] - rolling_max) / rolling_max
        df["max_drawdown"] = drawdown.rolling(50).min()

        # Skewness and kurtosis
        df["returns_skew"] = returns.rolling(50).skew()
        df["returns_kurtosis"] = returns.rolling(50).kurt()

        # Beta (if market data available)
        if hasattr(self, "market_returns"):
            covariance = returns.rolling(50).cov(self.market_returns.rolling(50))
            market_variance = self.market_returns.rolling(50).var()
            df["beta"] = covariance / market_variance

    def _add_relative_features(self, df: pd.DataFrame) -> None:
        """Add cross-sectional relative features."""
        # Group by time and calculate relative metrics
        time_groups = df.groupby(df.index)

        # Relative performance
        df["relative_return"] = time_groups["returns"].rank(pct=True)
        df["relative_volume"] = time_groups["volume_ratio"].rank(pct=True)

        # Check if volatility_5d exists before using it
        if "volatility_5d" in df.columns:
            df["relative_volatility"] = time_groups["volatility_5d"].rank(pct=True)

        # Sector/correlation features (simplified)
        if "sector" in df.columns:
            sector_return = df.groupby(["sector", df.index])["returns"].mean()
            df["sector_relative"] = df["returns"] - df.apply(
                lambda row: sector_return.get((row["sector"], row.name), 0), axis=1
            )

    def calculate_target_sharpe(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate forward-looking Sharpe ratio targets.

        Args:
            df: DataFrame with return data

        Returns:
            Series of target Sharpe ratios
        """
        returns = df["returns"].fillna(0)

        # Forward-looking returns for target
        forward_returns = returns.shift(-self.sharpe_window)

        # Rolling Sharpe calculation
        rolling_mean = forward_returns.rolling(
            self.sharpe_window, min_periods=int(self.sharpe_window / 2)
        ).mean()
        rolling_std = forward_returns.rolling(
            self.sharpe_window, min_periods=int(self.sharpe_window / 2)
        ).std()

        # Annualized Sharpe ratio
        daily_rf = (1 + self.risk_free_rate) ** (1 / 252) - 1
        sharpe_ratio = (rolling_mean - daily_rf) / (rolling_std + 1e-8)

        # Normalize Sharpe ratio for ML (scale to 0-1)
        # Use a more robust normalization
        sharpe_clipped = np.clip(sharpe_ratio, -3, 3)  # Clip extreme values
        sharpe_normalized = np.tanh(sharpe_clipped / 2)  # Bounded between -1 and 1
        sharpe_scaled = (sharpe_normalized + 1) / 2  # Scale to 0-1

        return sharpe_scaled

    def train(
        self,
        df: pd.DataFrame,
        validation_split: float = 0.2,
        early_stopping_rounds: int = 50,
    ) -> Dict:
        """
        Train the LightGBM Sharpe optimization model.

        Args:
            df: Training data with OHLCV
            validation_split: Fraction for validation
            early_stopping_rounds: Early stopping patience

        Returns:
            Training metrics dictionary
        """
        logger.info("Starting LightGBM Sharpe optimizer training...")

        # Feature engineering
        features_df = self.create_features(df)

        # Calculate targets
        targets = self.calculate_target_sharpe(features_df)

        # Select feature columns (exclude non-predictive columns)
        exclude_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "returns",
            "log_returns",
        ]
        if "symbol" in features_df.columns:
            exclude_cols.append("symbol")
        if "timestamp" in features_df.columns:
            exclude_cols.append("timestamp")

        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols

        # Prepare training data
        X = features_df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # More robust imputation
        # Fill NaN with median for numerical stability
        for col in X.columns:
            if X[col].dtype in ["float64", "float32", "int64", "int32"]:
                median_val = X[col].median()
                if pd.isna(median_val):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)

        y = targets.fillna(0.5)  # Neutral Sharpe for missing values

        # Remove rows with insufficient data (more lenient)
        # Only remove if ALL features are NaN or target is NaN
        valid_mask = ~(y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        # Additional check: remove rows where too many features are NaN
        nan_ratio = X.isna().sum(axis=1) / len(X.columns)
        valid_rows = nan_ratio < 0.5  # Allow up to 50% NaN features per row
        X = X[valid_rows]
        y = y[valid_rows]

        if len(X) < 50:  # Need minimum samples for training
            raise ValueError(
                f"Insufficient training data: only {len(X)} valid samples. Need at least 50."
            )

        logger.info(f"Training with {len(X)} samples and {len(feature_cols)} features")

        # Train/validation split (time-based)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # LightGBM parameters optimized for Sharpe prediction
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }

        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ],
        )

        # Mark as trained
        self.is_trained = True

        # Calculate metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        metrics = {
            "train_rmse": np.sqrt(np.mean((y_train - train_pred) ** 2)),
            "val_rmse": np.sqrt(np.mean((y_val - val_pred) ** 2)),
            "train_correlation": np.corrcoef(y_train, train_pred)[0, 1],
            "val_correlation": np.corrcoef(y_val, val_pred)[0, 1],
            "feature_importance": dict(
                zip(self.feature_columns, self.model.feature_importance())
            ),
            "num_features": len(self.feature_columns),
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
        }

        logger.info(f"Training complete:")
        logger.info(f"  - Validation RMSE: {metrics['val_rmse']:.4f}")
        logger.info(f"  - Validation Correlation: {metrics['val_correlation']:.4f}")
        logger.info(f"  - Features: {metrics['num_features']}")

        return metrics

    def predict_position_size(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict optimal position sizes based on Sharpe predictions.

        Args:
            df: DataFrame with current market data

        Returns:
            Series of recommended position sizes (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Feature engineering
        features_df = self.create_features(df)

        # Prepare features
        X = features_df[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())

        # Predict Sharpe ratios
        sharpe_pred = self.model.predict(X)

        # Convert Sharpe predictions to position sizes
        # Higher Sharpe -> Larger position (with risk limits)
        position_sizes = self._sharpe_to_position(sharpe_pred)

        return pd.Series(position_sizes, index=df.index)

    def _sharpe_to_position(self, sharpe_pred: np.ndarray) -> np.ndarray:
        """
        Convert Sharpe ratio predictions to position sizes.

        Args:
            sharpe_pred: Predicted Sharpe ratios (0-1 scaled)

        Returns:
            Position sizes constrained by risk limits
        """
        # Scale from [0, 1] back to meaningful Sharpe range
        sharpe_rescaled = (sharpe_pred - 0.5) * 4  # Roughly [-2, 2] Sharpe range

        # Apply Kelly criterion-inspired sizing
        # Position = max(0, min(max_pos, Sharpe / 2))  # Conservative Kelly
        kelly_fraction = np.maximum(0, np.minimum(sharpe_rescaled / 2, 1))

        # Apply position limits
        position_sizes = np.clip(
            kelly_fraction * self.max_position, self.min_position, self.max_position
        )

        # Set very low Sharpe predictions to zero position
        position_sizes = np.where(sharpe_pred < 0.4, 0, position_sizes)

        return position_sizes

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N most important features."""
        if not self.is_trained:
            return {}

        importance = self.model.feature_importance()
        feature_importance = dict(zip(self.feature_columns, importance))

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        return dict(sorted_features[:top_n])

    def save_model(self, path: str) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")

        model_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "params": {
                "lookback_window": self.lookback_window,
                "sharpe_window": self.sharpe_window,
                "risk_free_rate": self.risk_free_rate,
                "max_position": self.max_position,
                "min_position": self.min_position,
            },
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load trained model from disk."""
        model_data = joblib.load(path)

        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]

        # Update parameters
        params = model_data["params"]
        self.lookback_window = params["lookback_window"]
        self.sharpe_window = params["sharpe_window"]
        self.risk_free_rate = params["risk_free_rate"]
        self.max_position = params["max_position"]
        self.min_position = params["min_position"]

        self.is_trained = True
        logger.info(f"Model loaded from {path}")


# Factory function for easy model creation
def create_sharpe_optimizer(**kwargs) -> SharpeOptimizer:
    """Create a SharpeOptimizer with default or custom parameters."""
    return SharpeOptimizer(**kwargs)
