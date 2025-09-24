"""
Market Regime Detection System

Implements multiple regime detection algorithms:
- Hidden Markov Models (HMM) for probabilistic state detection
- K-means clustering for volatility regimes
- Rule-based filters for trend/range detection
- Volatility regime classification

Enables adaptive model weighting based on market conditions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from enum import Enum
import logging

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

from ..layer0_data_ingestion.schemas import FeatureSnapshot


class MarketRegime(Enum):
    """Market regime classification."""

    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class RegimeState:
    """Current market regime state."""

    regime: MarketRegime
    confidence: float
    duration: int  # Number of periods in this regime
    strength: float  # How strong the regime signal is
    transition_probability: float  # Probability of regime change
    metadata: Dict[str, Any]


class HMMRegimeDetector:
    """
    Hidden Markov Model for regime detection.

    Uses market features to identify hidden market states:
    - State 0: Low volatility, trending
    - State 1: High volatility, ranging
    - State 2: Crisis/extreme volatility
    """

    def __init__(self, n_states: int = 3, lookback_periods: int = 100):
        """
        Initialize HMM regime detector.

        Args:
            n_states: Number of hidden states
            lookback_periods: Lookback period for training
        """
        self.n_states = n_states
        self.lookback_periods = lookback_periods

        # HMM model
        self.model = hmm.GaussianHMM(
            n_components=n_states, covariance_type="full", random_state=42
        )

        # Data storage
        self.feature_history = deque(maxlen=lookback_periods)
        self.state_history = deque(maxlen=lookback_periods)

        # Training state
        self.is_trained = False
        self.last_retrain = None
        self.retrain_frequency = 50  # Retrain every 50 observations

        # State mapping
        self.state_mapping = {
            0: MarketRegime.LOW_VOLATILITY,
            1: MarketRegime.HIGH_VOLATILITY,
            2: MarketRegime.CRISIS,
        }

        self.logger = logging.getLogger("hmm_regime_detector")

    def _extract_features(self, feature_snapshot: FeatureSnapshot) -> np.ndarray:
        """Extract relevant features for regime detection."""
        features = []

        # Volatility features
        features.extend(
            [
                feature_snapshot.volatility_5m or 0.0,
                feature_snapshot.volatility_15m or 0.0,
                feature_snapshot.volatility_1h or 0.0,
            ]
        )

        # Return features
        features.extend(
            [
                feature_snapshot.return_1m or 0.0,
                feature_snapshot.return_5m or 0.0,
                feature_snapshot.return_15m or 0.0,
            ]
        )

        # Spread and liquidity
        features.extend(
            [
                feature_snapshot.spread_bps or 0.0,
                feature_snapshot.order_book_imbalance or 0.0,
            ]
        )

        # Volume
        features.extend(
            [
                (
                    np.log(float(feature_snapshot.volume_1m) + 1)
                    if feature_snapshot.volume_1m
                    else 0.0
                ),
                feature_snapshot.volume_ratio or 0.0,
            ]
        )

        return np.array(features)

    def update(self, feature_snapshot: FeatureSnapshot) -> Optional[RegimeState]:
        """
        Update regime detector with new data.

        Args:
            feature_snapshot: New market data

        Returns:
            Current regime state if available
        """
        # Extract features
        features = self._extract_features(feature_snapshot)
        self.feature_history.append(features)

        # Check if we need to retrain
        if len(self.feature_history) >= 30 and (
            not self.is_trained
            or len(self.feature_history) % self.retrain_frequency == 0
        ):
            self._train_model()

        # Predict current state
        if self.is_trained and len(self.feature_history) >= 2:
            return self._predict_regime()

        return None

    def _train_model(self):
        """Train the HMM model on historical data."""
        if len(self.feature_history) < 30:
            return

        try:
            # Prepare training data
            X = np.array(list(self.feature_history))

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train HMM
            self.model.fit(X_scaled)

            # Store scaler for future use
            self.scaler = scaler

            self.is_trained = True
            self.last_retrain = datetime.now()

            self.logger.info(f"HMM model retrained on {len(X)} samples")

        except Exception as e:
            self.logger.error(f"Error training HMM model: {e}")

    def _predict_regime(self) -> RegimeState:
        """Predict current market regime."""
        try:
            # Get recent features
            recent_features = np.array(list(self.feature_history)[-10:])
            X_scaled = self.scaler.transform(recent_features)

            # Predict states
            states = self.model.predict(X_scaled)
            current_state = states[-1]

            # Get state probabilities
            log_probs = self.model.score_samples(X_scaled)
            state_probs = self.model.predict_proba(X_scaled)
            current_prob = state_probs[-1, current_state]

            # Calculate regime duration
            duration = 1
            for i in range(len(states) - 2, -1, -1):
                if states[i] == current_state:
                    duration += 1
                else:
                    break

            # Map to regime
            regime = self.state_mapping.get(current_state, MarketRegime.SIDEWAYS)

            # Calculate transition probability
            transition_prob = 1.0 - current_prob

            # Calculate regime strength
            strength = current_prob

            return RegimeState(
                regime=regime,
                confidence=current_prob,
                duration=duration,
                strength=strength,
                transition_probability=transition_prob,
                metadata={
                    "hmm_state": int(current_state),
                    "state_probabilities": state_probs[-1].tolist(),
                    "log_likelihood": log_probs[-1],
                },
            )

        except Exception as e:
            self.logger.error(f"Error predicting regime: {e}")
            return None


class VolatilityRegimeDetector:
    """
    Volatility-based regime detector using rolling statistics.

    Classifies market into:
    - Low volatility (calm markets)
    - Normal volatility
    - High volatility (stressed markets)
    """

    def __init__(self, lookback_periods: int = 100):
        """
        Initialize volatility regime detector.

        Args:
            lookback_periods: Lookback period for volatility calculation
        """
        self.lookback_periods = lookback_periods
        self.volatility_history = deque(maxlen=lookback_periods)

        # Volatility thresholds (will be adaptive)
        self.low_vol_threshold = 0.01  # 1% volatility
        self.high_vol_threshold = 0.05  # 5% volatility

        self.logger = logging.getLogger("volatility_regime_detector")

    def update(self, feature_snapshot: FeatureSnapshot) -> Optional[RegimeState]:
        """Update with new volatility data."""
        # Use 1-hour volatility as main metric
        current_vol = feature_snapshot.volatility_1h
        if current_vol is None:
            return None

        self.volatility_history.append(current_vol)

        if len(self.volatility_history) < 10:
            return None

        # Calculate adaptive thresholds
        vol_array = np.array(self.volatility_history)
        percentile_25 = np.percentile(vol_array, 25)
        percentile_75 = np.percentile(vol_array, 75)

        # Classify regime
        if current_vol < percentile_25:
            regime = MarketRegime.LOW_VOLATILITY
            strength = (percentile_25 - current_vol) / percentile_25
        elif current_vol > percentile_75:
            regime = MarketRegime.HIGH_VOLATILITY
            strength = (current_vol - percentile_75) / percentile_75
        else:
            regime = MarketRegime.SIDEWAYS
            strength = 0.5

        # Calculate confidence
        confidence = min(strength + 0.3, 0.9)

        # Calculate duration
        duration = 1
        target_regime = regime
        for i in range(len(self.volatility_history) - 2, -1, -1):
            vol = self.volatility_history[i]
            if vol < percentile_25 and target_regime == MarketRegime.LOW_VOLATILITY:
                duration += 1
            elif vol > percentile_75 and target_regime == MarketRegime.HIGH_VOLATILITY:
                duration += 1
            elif (
                percentile_25 <= vol <= percentile_75
                and target_regime == MarketRegime.SIDEWAYS
            ):
                duration += 1
            else:
                break

        return RegimeState(
            regime=regime,
            confidence=confidence,
            duration=duration,
            strength=strength,
            transition_probability=1.0 - confidence,
            metadata={
                "current_volatility": current_vol,
                "vol_percentile_25": percentile_25,
                "vol_percentile_75": percentile_75,
                "vol_rank": np.percentile(vol_array <= current_vol, 100),
            },
        )


class TrendRegimeDetector:
    """
    Trend-based regime detector using moving averages and momentum.

    Classifies market into:
    - Bull trend (sustained upward movement)
    - Bear trend (sustained downward movement)
    - Sideways (range-bound)
    """

    def __init__(self, lookback_periods: int = 50):
        """
        Initialize trend regime detector.

        Args:
            lookback_periods: Lookback period for trend calculation
        """
        self.lookback_periods = lookback_periods
        self.price_history = deque(maxlen=lookback_periods)
        self.return_history = deque(maxlen=lookback_periods)

        self.logger = logging.getLogger("trend_regime_detector")

    def update(self, feature_snapshot: FeatureSnapshot) -> Optional[RegimeState]:
        """Update with new price data."""
        if not feature_snapshot.mid_price:
            return None

        current_price = float(feature_snapshot.mid_price)
        self.price_history.append(current_price)

        # Calculate return
        if len(self.price_history) >= 2:
            prev_price = self.price_history[-2]
            return_1tick = (current_price - prev_price) / prev_price
            self.return_history.append(return_1tick)

        if len(self.price_history) < 20:
            return None

        # Calculate moving averages
        prices = np.array(self.price_history)
        ma_short = np.mean(prices[-5:])  # 5-period MA
        ma_long = np.mean(prices[-20:])  # 20-period MA

        # Calculate trend strength
        trend_strength = (ma_short - ma_long) / ma_long

        # Calculate momentum
        if len(self.return_history) >= 10:
            returns = np.array(self.return_history)
            momentum = np.mean(returns[-10:])  # 10-period momentum

            # Combine trend and momentum
            trend_signal = trend_strength + momentum * 10
        else:
            trend_signal = trend_strength

        # Classify regime
        if trend_signal > 0.02:  # 2% threshold
            regime = MarketRegime.BULL_TREND
            strength = min(trend_signal / 0.05, 1.0)
        elif trend_signal < -0.02:
            regime = MarketRegime.BEAR_TREND
            strength = min(abs(trend_signal) / 0.05, 1.0)
        else:
            regime = MarketRegime.SIDEWAYS
            strength = 0.5

        # Calculate confidence
        confidence = min(strength + 0.3, 0.9)

        # Calculate duration (simplified)
        duration = 1

        return RegimeState(
            regime=regime,
            confidence=confidence,
            duration=duration,
            strength=strength,
            transition_probability=1.0 - confidence,
            metadata={
                "trend_strength": trend_strength,
                "momentum": momentum if len(self.return_history) >= 10 else 0,
                "ma_short": ma_short,
                "ma_long": ma_long,
                "trend_signal": trend_signal,
            },
        )


class RegimeDetector:
    """
    Master regime detector that combines multiple detection methods.

    Provides ensemble regime detection by combining:
    - HMM-based probabilistic detection
    - Volatility-based classification
    - Trend-based classification
    """

    def __init__(self, symbol: str):
        """
        Initialize master regime detector.

        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol

        # Individual detectors
        self.hmm_detector = HMMRegimeDetector()
        self.volatility_detector = VolatilityRegimeDetector()
        self.trend_detector = TrendRegimeDetector()

        # Ensemble weights
        self.weights = {"hmm": 0.4, "volatility": 0.3, "trend": 0.3}

        # State tracking
        self.current_regime = None
        self.regime_history = deque(maxlen=100)

        self.logger = logging.getLogger(f"regime_detector.{symbol}")
        self.logger.info(f"Initialized regime detector for {symbol}")

    def update(self, feature_snapshot: FeatureSnapshot) -> Optional[RegimeState]:
        """
        Update regime detector with new data.

        Args:
            feature_snapshot: New market data

        Returns:
            Ensemble regime state
        """
        # Update individual detectors
        hmm_state = self.hmm_detector.update(feature_snapshot)
        vol_state = self.volatility_detector.update(feature_snapshot)
        trend_state = self.trend_detector.update(feature_snapshot)

        # Store individual states
        states = {"hmm": hmm_state, "volatility": vol_state, "trend": trend_state}

        # Filter None states
        valid_states = {k: v for k, v in states.items() if v is not None}

        if not valid_states:
            return None

        # Combine states using weighted voting
        regime_votes = {}
        confidence_sum = 0
        strength_sum = 0
        total_weight = 0

        for detector_name, state in valid_states.items():
            weight = self.weights[detector_name]
            regime = state.regime

            # Accumulate votes
            if regime not in regime_votes:
                regime_votes[regime] = 0
            regime_votes[regime] += weight * state.confidence

            # Accumulate metrics
            confidence_sum += weight * state.confidence
            strength_sum += weight * state.strength
            total_weight += weight

        # Determine winning regime
        if regime_votes:
            winning_regime = max(regime_votes, key=regime_votes.get)
            winning_score = regime_votes[winning_regime]
        else:
            winning_regime = MarketRegime.SIDEWAYS
            winning_score = 0.5

        # Calculate ensemble metrics
        ensemble_confidence = confidence_sum / total_weight if total_weight > 0 else 0.5
        ensemble_strength = strength_sum / total_weight if total_weight > 0 else 0.5

        # Calculate duration
        duration = 1
        if self.regime_history:
            for prev_state in reversed(self.regime_history):
                if prev_state.regime == winning_regime:
                    duration += 1
                else:
                    break

        # Calculate transition probability
        transition_prob = 1.0 - ensemble_confidence

        # Create ensemble state
        ensemble_state = RegimeState(
            regime=winning_regime,
            confidence=ensemble_confidence,
            duration=duration,
            strength=ensemble_strength,
            transition_probability=transition_prob,
            metadata={
                "regime_votes": regime_votes,
                "individual_states": {
                    k: v.regime.value for k, v in valid_states.items()
                },
                "detector_confidences": {
                    k: v.confidence for k, v in valid_states.items()
                },
                "ensemble_score": winning_score,
                "active_detectors": list(valid_states.keys()),
            },
        )

        # Update history
        self.regime_history.append(ensemble_state)
        self.current_regime = ensemble_state

        return ensemble_state

    def get_regime_weights(self) -> Dict[str, float]:
        """
        Get model weights for the current regime.

        Returns:
            Dictionary of model weights based on current regime
        """
        if not self.current_regime:
            # Default weights when no regime detected
            return {
                "ma_momentum": 0.25,
                "mean_rev": 0.25,
                "ob_pressure": 0.25,
                "news_sent_alpha": 0.25,
                "lstm_transformer": 0.0,  # Need time to train
                "onchain_alpha": 0.0,  # Need data to be reliable
            }

        regime = self.current_regime.regime
        confidence = self.current_regime.confidence

        # Regime-specific weights
        if regime == MarketRegime.BULL_TREND:
            weights = {
                "ma_momentum": 0.4,  # Strong in trends
                "mean_rev": 0.1,  # Weak in trends
                "ob_pressure": 0.2,  # Moderate
                "news_sent_alpha": 0.15,  # Moderate
                "lstm_transformer": 0.1,  # Learning
                "onchain_alpha": 0.05,  # Supporting
            }
        elif regime == MarketRegime.BEAR_TREND:
            weights = {
                "ma_momentum": 0.4,  # Strong in trends
                "mean_rev": 0.1,  # Weak in trends
                "ob_pressure": 0.2,  # Moderate
                "news_sent_alpha": 0.2,  # Important in downturns
                "lstm_transformer": 0.05,  # Reduced
                "onchain_alpha": 0.05,  # Supporting
            }
        elif regime == MarketRegime.SIDEWAYS:
            weights = {
                "ma_momentum": 0.1,  # Weak in ranging
                "mean_rev": 0.4,  # Strong in ranging
                "ob_pressure": 0.3,  # Strong in ranging
                "news_sent_alpha": 0.1,  # Moderate
                "lstm_transformer": 0.05,  # Learning
                "onchain_alpha": 0.05,  # Supporting
            }
        elif regime == MarketRegime.HIGH_VOLATILITY:
            weights = {
                "ma_momentum": 0.3,  # Moderate
                "mean_rev": 0.2,  # Risky in high vol
                "ob_pressure": 0.1,  # Noisy in high vol
                "news_sent_alpha": 0.3,  # Important in volatility
                "lstm_transformer": 0.05,  # Reduced
                "onchain_alpha": 0.05,  # Supporting
            }
        elif regime == MarketRegime.LOW_VOLATILITY:
            weights = {
                "ma_momentum": 0.2,  # Moderate
                "mean_rev": 0.3,  # Good in low vol
                "ob_pressure": 0.3,  # Good in low vol
                "news_sent_alpha": 0.1,  # Less important
                "lstm_transformer": 0.05,  # Learning
                "onchain_alpha": 0.05,  # Supporting
            }
        else:
            # Default fallback
            weights = {
                "ma_momentum": 0.25,
                "mean_rev": 0.25,
                "ob_pressure": 0.25,
                "news_sent_alpha": 0.25,
                "lstm_transformer": 0.0,
                "onchain_alpha": 0.0,
            }

        # Adjust weights based on confidence
        # Higher confidence = more extreme weights
        # Lower confidence = more balanced weights
        if confidence < 0.6:
            # Low confidence - move toward equal weights
            equal_weight = 1.0 / len(weights)
            for key in weights:
                weights[key] = 0.7 * weights[key] + 0.3 * equal_weight

        return weights

    def get_stats(self) -> Dict[str, Any]:
        """Get regime detector statistics."""
        return {
            "symbol": self.symbol,
            "current_regime": (
                self.current_regime.regime.value if self.current_regime else None
            ),
            "regime_confidence": (
                self.current_regime.confidence if self.current_regime else 0
            ),
            "regime_duration": (
                self.current_regime.duration if self.current_regime else 0
            ),
            "regime_history_length": len(self.regime_history),
            "detector_weights": self.weights,
            "hmm_trained": self.hmm_detector.is_trained,
            "model_weights": self.get_regime_weights(),
        }


# Factory function
def create_regime_detector(symbol: str) -> RegimeDetector:
    """
    Factory function to create regime detector.

    Args:
        symbol: Trading symbol

    Returns:
        Initialized RegimeDetector instance
    """
    return RegimeDetector(symbol)


# Example usage
if __name__ == "__main__":
    from decimal import Decimal

    # Test regime detector
    detector = RegimeDetector("BTCUSDT")

    # Simulate market data
    base_price = 50000
    for i in range(100):
        # Simulate different market conditions
        if i < 30:
            # Low volatility trending
            price = base_price + i * 50 + np.random.randn() * 100
            volatility = 0.01 + np.random.randn() * 0.002
        elif i < 60:
            # High volatility ranging
            price = base_price + 1500 + np.random.randn() * 500
            volatility = 0.05 + np.random.randn() * 0.01
        else:
            # Bear trend
            price = base_price + 1500 - (i - 60) * 30 + np.random.randn() * 200
            volatility = 0.03 + np.random.randn() * 0.005

        # Create feature snapshot
        feature_snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            mid_price=Decimal(str(max(price, 1000))),
            volatility_1h=max(volatility, 0.001),
            volatility_5m=max(volatility * 0.8, 0.001),
            volatility_15m=max(volatility * 0.9, 0.001),
            return_1m=np.random.randn() * 0.002,
            return_5m=np.random.randn() * 0.005,
            return_15m=np.random.randn() * 0.008,
            spread_bps=2.0 + np.random.randn() * 0.5,
            order_book_imbalance=np.random.randn() * 0.1,
            volume_1m=Decimal(str(100 + np.random.randn() * 20)),
        )

        # Update detector
        state = detector.update(feature_snapshot)

        if state and i % 10 == 0:
            print(
                f"Step {i}: Regime = {state.regime.value}, "
                f"Confidence = {state.confidence:.3f}, "
                f"Duration = {state.duration}"
            )

            # Show model weights
            weights = detector.get_regime_weights()
            print(f"  Model weights: {weights}")

    print("\nFinal Stats:")
    print(detector.get_stats())
