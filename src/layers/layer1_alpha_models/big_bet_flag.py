#!/usr/bin/env python3
"""
Big-Bet Detector (Layer 1)

Detects high-confidence trading opportunities based on:
- Strong sentiment: abs(sent_score) > 0.75 
- Fundamental surprise: abs(earnings_surprise) > 1σ
- Combined signal strength

Emits big_bet_flag=True when both conditions are met.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BigBetSignal:
    """Big bet signal output with flag and reasoning."""
    symbol: str
    edge_bps: float
    confidence: float
    timestamp: str
    big_bet_flag: bool
    reasoning: str

class BigBetDetector:
    """
    Big-bet detector for high-confidence trading opportunities.
    
    Flags trades when multiple strong signals align:
    - Extreme sentiment (abs > 0.75)
    - Fundamental surprise (abs > 1 standard deviation)
    """
    
    def __init__(self, 
                 sentiment_threshold: float = 0.75,
                 earnings_sigma_threshold: float = 1.0,
                 base_edge_scaling: float = 30.0):
        """
        Initialize big-bet detector.
        
        Args:
            sentiment_threshold: Minimum abs(sentiment) for big bet
            earnings_sigma_threshold: Minimum earnings surprise in std devs
            base_edge_scaling: Base edge scaling for normal signals
        """
        self.sentiment_threshold = sentiment_threshold
        self.earnings_sigma_threshold = earnings_sigma_threshold
        self.base_edge_scaling = base_edge_scaling
        
        # Historical earnings data for volatility calculation
        self.earnings_history: Dict[str, list] = {}
        
        # Performance tracking
        self.big_bet_count = 0
        self.big_bet_hits = 0
        self.total_signals = 0
        
        logger.info(f"Big-Bet Detector initialized: sent_thresh={sentiment_threshold}, "
                   f"earnings_thresh={earnings_sigma_threshold}σ")
    
    def update_earnings_data(self, symbol: str, earnings_surprise: float):
        """Update earnings surprise data for volatility calculation."""
        if symbol not in self.earnings_history:
            self.earnings_history[symbol] = []
        
        self.earnings_history[symbol].append(earnings_surprise)
        
        # Keep only last 20 earnings reports for rolling volatility
        if len(self.earnings_history[symbol]) > 20:
            self.earnings_history[symbol] = self.earnings_history[symbol][-20:]
    
    def _calculate_earnings_sigma(self, symbol: str) -> float:
        """Calculate standard deviation of earnings surprises for symbol."""
        if symbol not in self.earnings_history or len(self.earnings_history[symbol]) < 3:
            return 1.0  # Default sigma if insufficient data
        
        earnings_data = np.array(self.earnings_history[symbol])
        return np.std(earnings_data)
    
    def detect_big_bet(self, symbol: str, sent_score: float, 
                      earnings_surprise: Optional[float] = None,
                      timestamp: Optional[str] = None) -> BigBetSignal:
        """
        Detect big-bet opportunity from sentiment and fundamental data.
        
        Args:
            symbol: Trading symbol
            sent_score: Sentiment score (-1.0 to 1.0)
            earnings_surprise: Earnings surprise (actual - expected)
            timestamp: ISO timestamp
            
        Returns:
            BigBetSignal with flag and reasoning
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        try:
            # Clip sentiment to valid range
            sent_score = max(-1.0, min(1.0, sent_score))
            
            # Check sentiment condition
            sentiment_extreme = abs(sent_score) > self.sentiment_threshold
            
            # Check earnings condition
            earnings_extreme = False
            earnings_sigma_multiple = 0.0
            
            if earnings_surprise is not None:
                # Update earnings history
                self.update_earnings_data(symbol, earnings_surprise)
                
                # Calculate sigma multiple
                earnings_sigma = self._calculate_earnings_sigma(symbol)
                if earnings_sigma > 0:
                    earnings_sigma_multiple = abs(earnings_surprise) / earnings_sigma
                    earnings_extreme = earnings_sigma_multiple > self.earnings_sigma_threshold
            
            # Determine big bet flag
            big_bet_flag = sentiment_extreme and earnings_extreme
            
            # Calculate edge based on signal strength
            if big_bet_flag:
                # Big bet: amplified edge
                sentiment_component = sent_score * 40  # 40bp max from sentiment
                earnings_component = np.sign(earnings_surprise) * min(20, earnings_sigma_multiple * 10)  # Up to 20bp from earnings
                edge_bps = sentiment_component + earnings_component
                confidence = 0.85  # High confidence for big bets
                
                reasoning = f"BIG_BET: sent={sent_score:.3f} (>{self.sentiment_threshold}), " \
                           f"earnings={earnings_sigma_multiple:.2f}σ (>{self.earnings_sigma_threshold}σ)"
                
                self.big_bet_count += 1
                
            elif sentiment_extreme:
                # Sentiment-only signal
                edge_bps = sent_score * self.base_edge_scaling
                confidence = 0.6 + 0.2 * abs(sent_score)
                
                if earnings_surprise is not None:
                    reasoning = f"SENTIMENT: sent={sent_score:.3f} (>{self.sentiment_threshold}), " \
                               f"earnings={earnings_sigma_multiple:.2f}σ (<{self.earnings_sigma_threshold}σ)"
                else:
                    reasoning = f"SENTIMENT: sent={sent_score:.3f} (>{self.sentiment_threshold}), no_earnings"
                
            elif earnings_extreme:
                # Earnings-only signal  
                edge_bps = np.sign(earnings_surprise) * min(25, earnings_sigma_multiple * 12)
                confidence = 0.5 + 0.3 * min(1.0, earnings_sigma_multiple / 2.0)
                
                reasoning = f"EARNINGS: sent={sent_score:.3f} (<={self.sentiment_threshold}), " \
                           f"earnings={earnings_sigma_multiple:.2f}σ (>{self.earnings_sigma_threshold}σ)"
                
            else:
                # Weak signal
                edge_bps = sent_score * 15  # Reduced edge for weak signals
                confidence = 0.3 + 0.2 * abs(sent_score)
                
                if earnings_surprise is not None:
                    reasoning = f"WEAK: sent={sent_score:.3f} (<={self.sentiment_threshold}), " \
                               f"earnings={earnings_sigma_multiple:.2f}σ (<={self.earnings_sigma_threshold}σ)"
                else:
                    reasoning = f"WEAK: sent={sent_score:.3f} (<={self.sentiment_threshold}), no_earnings"
            
            # Clamp edge to reasonable bounds
            edge_bps = max(-100, min(100, edge_bps))
            confidence = max(0.1, min(0.95, confidence))
            
            signal = BigBetSignal(
                symbol=symbol,
                edge_bps=edge_bps,
                confidence=confidence,
                timestamp=timestamp,
                big_bet_flag=big_bet_flag,
                reasoning=reasoning
            )
            
            self.total_signals += 1
            
            logger.debug(f"Big-bet detector for {symbol}: flag={big_bet_flag}, "
                        f"edge={edge_bps:.1f}bp, {reasoning}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in big-bet detection: {e}")
            
            # Return neutral signal on error
            return BigBetSignal(
                symbol=symbol,
                edge_bps=0.0,
                confidence=0.1,
                timestamp=timestamp,
                big_bet_flag=False,
                reasoning=f"ERROR: {str(e)[:50]}"
            )
    
    def update_from_feature_snapshot(self, feature_snapshot, 
                                   earnings_surprise: Optional[float] = None) -> BigBetSignal:
        """
        Generate big-bet signal from FeatureBus snapshot.
        
        Args:
            feature_snapshot: FeatureSnapshot with sentiment data
            earnings_surprise: Optional earnings surprise data
            
        Returns:
            BigBetSignal with flag and reasoning
        """
        try:
            # Extract sentiment from feature snapshot
            sent_score = getattr(feature_snapshot, 'sent_score', 0.0) or 0.0
            
            return self.detect_big_bet(
                symbol=feature_snapshot.symbol,
                sent_score=sent_score,
                earnings_surprise=earnings_surprise,
                timestamp=feature_snapshot.timestamp.isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error processing feature snapshot: {e}")
            return BigBetSignal(
                symbol=getattr(feature_snapshot, 'symbol', 'UNKNOWN'),
                edge_bps=0.0,
                confidence=0.1,
                timestamp=datetime.now().isoformat(),
                big_bet_flag=False,
                reasoning=f"SNAPSHOT_ERROR: {str(e)[:50]}"
            )
    
    def update_performance(self, symbol: str, realized_return_bps: float, 
                          predicted_edge_bps: float, was_big_bet: bool):
        """Update performance tracking."""
        try:
            # Check if prediction was correct
            prediction_correct = (predicted_edge_bps > 0 and realized_return_bps > 0) or \
                               (predicted_edge_bps < 0 and realized_return_bps < 0)
            
            if was_big_bet and prediction_correct:
                self.big_bet_hits += 1
            
            logger.debug(f"Performance update for {symbol}: predicted={predicted_edge_bps:.1f}bp, "
                        f"realized={realized_return_bps:.1f}bp, big_bet={was_big_bet}, "
                        f"correct={prediction_correct}")
                        
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def get_stats(self) -> Dict[str, float]:
        """Get detector statistics."""
        big_bet_hit_rate = (self.big_bet_hits / self.big_bet_count) if self.big_bet_count > 0 else 0.0
        big_bet_rate = (self.big_bet_count / self.total_signals) if self.total_signals > 0 else 0.0
        
        return {
            'total_signals': self.total_signals,
            'big_bet_count': self.big_bet_count,
            'big_bet_hits': self.big_bet_hits,
            'big_bet_rate': big_bet_rate,
            'big_bet_hit_rate': big_bet_hit_rate,
            'sentiment_threshold': self.sentiment_threshold,
            'earnings_threshold': self.earnings_sigma_threshold
        }

# Unit tests for the big-bet detector
def test_big_bet_detector():
    """Unit test for big-bet detector."""
    detector = BigBetDetector()
    
    print("🧪 Testing Big-Bet Detector")
    print("=" * 50)
    
    # Test case 1: Big bet (strong sentiment + earnings surprise)
    signal1 = detector.detect_big_bet("TEST", sent_score=0.85, earnings_surprise=2.5)
    print(f"✅ Strong sentiment + earnings: flag={signal1.big_bet_flag}, edge={signal1.edge_bps:.1f}bp")
    assert signal1.big_bet_flag == True, "Should flag big bet"
    
    # Test case 2: Sentiment only
    signal2 = detector.detect_big_bet("TEST", sent_score=0.8, earnings_surprise=0.5)  
    print(f"✅ Sentiment only: flag={signal2.big_bet_flag}, edge={signal2.edge_bps:.1f}bp")
    assert signal2.big_bet_flag == False, "Should not flag big bet"
    
    # Test case 3: Neither condition met
    signal3 = detector.detect_big_bet("TEST", sent_score=0.5, earnings_surprise=0.3)
    print(f"✅ Weak signals: flag={signal3.big_bet_flag}, edge={signal3.edge_bps:.1f}bp")
    assert signal3.big_bet_flag == False, "Should not flag big bet"
    
    # Test case 4: Negative sentiment + negative earnings
    signal4 = detector.detect_big_bet("TEST", sent_score=-0.9, earnings_surprise=-3.0)
    print(f"✅ Negative big bet: flag={signal4.big_bet_flag}, edge={signal4.edge_bps:.1f}bp")
    assert signal4.big_bet_flag == True, "Should flag negative big bet"
    assert signal4.edge_bps < 0, "Should have negative edge"
    
    print(f"\n📊 Detector Stats: {detector.get_stats()}")
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_big_bet_detector() 