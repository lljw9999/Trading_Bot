#!/usr/bin/env python3
"""
News Sentiment Alpha Model (Layer 1)

News sentiment strategy:
- Edge = 40 bp × sent_score (clipped to ±40 bp)
- Confidence based on sentiment strength and source quality
- Uses sentiment scores from GPT-4o enriched news data
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AlphaSignal:
    """Alpha signal output"""

    symbol: str
    edge_bps: float
    confidence: float
    timestamp: str
    reasoning: str


class NewsSentimentAlpha:
    """
    News sentiment alpha model.

    Generates trading signals based on GPT-4o analyzed sentiment scores
    from financial news, Reddit, and other sources.
    """

    def __init__(
        self,
        edge_scaling: float = 40.0,  # 40 bp max edge
        min_confidence: float = 0.3,
        max_confidence: float = 0.9,
    ):
        """
        Initialize news sentiment alpha model.

        Args:
            edge_scaling: Maximum edge in basis points (±40 bp)
            min_confidence: Minimum confidence level
            max_confidence: Maximum confidence level
        """
        self.edge_scaling = edge_scaling
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence

        # Performance tracking
        self.signal_count = 0
        self.hit_count = 0

        # Sentiment thresholds
        self.neutral_threshold = 0.1  # ±0.1 considered neutral
        self.strong_sentiment_threshold = 0.7  # >0.7 considered strong

        logger.info(
            f"News Sentiment Alpha initialized with edge_scaling={edge_scaling}bp"
        )

    def update_from_feature_snapshot(self, feature_snapshot) -> Optional[AlphaSignal]:
        """
        Generate signal from FeatureBus snapshot with sentiment data.

        Args:
            feature_snapshot: FeatureSnapshot object with sentiment data

        Returns:
            AlphaSignal if signal generated, None otherwise
        """
        try:
            # Extract sentiment score from feature snapshot
            if (
                not hasattr(feature_snapshot, "sent_score")
                or feature_snapshot.sent_score is None
            ):
                # No sentiment data available
                return None

            # Clip sentiment score to valid range (-1.0 to 1.0)
            sent_score = max(-1.0, min(1.0, feature_snapshot.sent_score))

            # Log if clipping occurred
            if sent_score != feature_snapshot.sent_score:
                logger.debug(
                    f"Clipped sentiment score from {feature_snapshot.sent_score:.3f} to {sent_score:.3f} for {feature_snapshot.symbol}"
                )

            # Skip neutral sentiment (low signal)
            if abs(sent_score) < self.neutral_threshold:
                return None

            # Calculate edge: 40 bp × sent_score
            edge_bps = self.edge_scaling * sent_score

            # Calculate confidence based on sentiment strength
            sentiment_strength = abs(sent_score)

            # Base confidence from sentiment strength
            base_confidence = self.min_confidence + (sentiment_strength * 0.6)

            # Boost confidence for strong sentiment
            if sentiment_strength >= self.strong_sentiment_threshold:
                confidence_boost = 0.2
            else:
                confidence_boost = 0.0

            confidence = min(base_confidence + confidence_boost, self.max_confidence)

            # Create reasoning string
            sentiment_desc = "positive" if sent_score > 0 else "negative"
            strength_desc = (
                "strong"
                if sentiment_strength >= self.strong_sentiment_threshold
                else "moderate"
            )

            reasoning = (
                f"sent_score={sent_score:.3f} ({strength_desc} {sentiment_desc})"
            )

            signal = AlphaSignal(
                symbol=feature_snapshot.symbol,
                edge_bps=edge_bps,
                confidence=confidence,
                timestamp=feature_snapshot.timestamp.isoformat(),
                reasoning=reasoning,
            )

            self.signal_count += 1

            logger.debug(
                f"News sentiment signal for {feature_snapshot.symbol}: "
                f"edge={edge_bps:.1f}bp, confidence={confidence:.2f}, {reasoning}"
            )

            return signal

        except Exception as e:
            logger.error(
                f"Error generating sentiment signal from feature snapshot: {e}"
            )
            return None

    def update_price(
        self, symbol: str, price: float, timestamp: str, sentiment_score: float
    ) -> Optional[AlphaSignal]:
        """
        Direct update with price and sentiment data (alternative interface).

        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: ISO timestamp
            sentiment_score: Sentiment score (-1.0 to 1.0)

        Returns:
            AlphaSignal if generated, None otherwise
        """
        try:
            # Clip sentiment score to valid range (-1.0 to 1.0)
            sent_score = max(-1.0, min(1.0, sentiment_score))

            # Log if clipping occurred
            if sent_score != sentiment_score:
                logger.debug(
                    f"Clipped sentiment score from {sentiment_score:.3f} to {sent_score:.3f} for {symbol}"
                )

            # Skip neutral sentiment
            if abs(sent_score) < self.neutral_threshold:
                return None

            # Calculate edge and confidence (same logic as feature snapshot method)
            edge_bps = self.edge_scaling * sent_score

            sentiment_strength = abs(sent_score)
            confidence = min(
                self.min_confidence
                + (sentiment_strength * 0.6)
                + (
                    0.2
                    if sentiment_strength >= self.strong_sentiment_threshold
                    else 0.0
                ),
                self.max_confidence,
            )

            sentiment_desc = "positive" if sent_score > 0 else "negative"
            strength_desc = (
                "strong"
                if sentiment_strength >= self.strong_sentiment_threshold
                else "moderate"
            )
            reasoning = (
                f"sent_score={sent_score:.3f} ({strength_desc} {sentiment_desc})"
            )

            signal = AlphaSignal(
                symbol=symbol,
                edge_bps=edge_bps,
                confidence=confidence,
                timestamp=timestamp,
                reasoning=reasoning,
            )

            self.signal_count += 1

            logger.debug(
                f"News sentiment signal for {symbol}: edge={edge_bps:.1f}bp, "
                f"confidence={confidence:.2f}"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating sentiment signal: {e}")
            return None

    def update_performance(
        self, symbol: str, realized_return_bps: float, predicted_edge_bps: float
    ):
        """Update performance tracking based on realized returns."""
        try:
            # Check if prediction was correct (same direction)
            if (predicted_edge_bps > 0 and realized_return_bps > 0) or (
                predicted_edge_bps < 0 and realized_return_bps < 0
            ):
                self.hit_count += 1

            logger.debug(
                f"Performance update for {symbol}: predicted={predicted_edge_bps:.1f}bp, "
                f"realized={realized_return_bps:.1f}bp, hit_rate={self.get_hit_rate():.2%}"
            )

        except Exception as e:
            logger.error(f"Error updating performance: {e}")

    def get_hit_rate(self) -> float:
        """Get current hit rate."""
        if self.signal_count == 0:
            return 0.60  # Default optimistic for sentiment
        return self.hit_count / self.signal_count

    def get_stats(self) -> Dict[str, float]:
        """Get model statistics."""
        return {
            "signal_count": self.signal_count,
            "hit_count": self.hit_count,
            "hit_rate": self.get_hit_rate(),
            "edge_scaling": self.edge_scaling,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
        }


# Unit tests for the sentiment alpha model
def test_sentiment_alpha():
    """Unit test for sentiment alpha model."""
    alpha = NewsSentimentAlpha()

    # Test cases based on the specification
    test_cases = [
        # (sent_score, expected_edge_bps, description)
        (0.8, 32.0, "Strong positive sentiment"),
        (-0.6, -24.0, "Moderate negative sentiment"),
        (2.0, 40.0, "Extreme positive (clipped to +40)"),
        (-1.5, -40.0, "Extreme negative (clipped to -40)"),
        (0.05, None, "Neutral sentiment (should return None)"),
        (0.0, None, "Zero sentiment (should return None)"),
    ]

    print("🧪 Testing News Sentiment Alpha Model")
    print("=" * 50)

    for sent_score, expected_edge, description in test_cases:
        signal = alpha.update_price(
            symbol="TEST",
            price=100.0,
            timestamp=datetime.now().isoformat(),
            sentiment_score=sent_score,
        )

        if expected_edge is None:
            assert signal is None, f"Expected None for {description}"
            print(f"✅ {description}: correctly returned None")
        else:
            assert signal is not None, f"Expected signal for {description}"
            assert (
                abs(signal.edge_bps - expected_edge) < 0.1
            ), f"Expected {expected_edge}bp, got {signal.edge_bps}bp for {description}"
            print(
                f"✅ {description}: edge={signal.edge_bps:.1f}bp, confidence={signal.confidence:.2f}"
            )

    print(f"\n📊 Model Stats: {alpha.get_stats()}")
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_sentiment_alpha()
