#!/usr/bin/env python3
"""
Order-Book-Pressure Alpha Model (Layer 1)

Order book pressure strategy:
- Edge = 25 * pressure where pressure = (bid_size - ask_size) / (bid_size + ask_size + 1e-9)
- Confidence = 0.50 + 0.5 * abs(pressure)
- Uses best bid/ask sizes from FeatureBus snapshot
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

class OrderBookPressureAlpha:
    """
    Order Book Pressure Alpha Model
    
    Generates alpha signals based on order book imbalance:
    - When bid size > ask size, expect upward pressure (positive edge)
    - When ask size > bid size, expect downward pressure (negative edge)
    - Edge = 25 * pressure where pressure = (bid_size - ask_size) / (bid_size + ask_size + 1e-9)
    - Confidence increases with absolute pressure
    """
    
    def __init__(self, 
                 edge_scaling: float = 25.0,
                 min_confidence: float = 0.50,
                 max_confidence: float = 1.0):
        """
        Initialize order book pressure model.
        
        Args:
            edge_scaling: Scaling factor to convert pressure to basis points
            min_confidence: Minimum confidence level
            max_confidence: Maximum confidence level
        """
        self.edge_scaling = edge_scaling
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        
        # Performance tracking
        self.signal_count = 0
        self.hit_count = 0
        
        logger.info(f"Order book pressure alpha initialized: scaling={edge_scaling}, "
                   f"min_conf={min_confidence}, max_conf={max_confidence}")
    
    def generate_signal(self, symbol: str, bid_size: float, ask_size: float, timestamp: str) -> Optional[AlphaSignal]:
        """
        Generate alpha signal from order book data.
        
        Args:
            symbol: Trading symbol
            bid_size: Best bid size
            ask_size: Best ask size
            timestamp: Timestamp string
            
        Returns:
            AlphaSignal if signal generated, None otherwise
        """
        try:
            # Validate inputs
            if bid_size <= 0 or ask_size <= 0:
                return None
                
            # Calculate pressure
            pressure = (bid_size - ask_size) / (bid_size + ask_size + 1e-9)
            
            # Calculate edge in basis points
            edge_bps = self.edge_scaling * pressure
            
            # Calculate confidence
            confidence = self.min_confidence + 0.5 * abs(pressure)
            confidence = min(confidence, self.max_confidence)
            
            # Create reasoning
            reasoning = (f"bid_size={bid_size:.2f}, ask_size={ask_size:.2f}, "
                        f"pressure={pressure:.4f}, imbalance={'BUY' if pressure > 0 else 'SELL'}")
            
            signal = AlphaSignal(
                symbol=symbol,
                edge_bps=edge_bps,
                confidence=confidence,
                timestamp=timestamp,
                reasoning=reasoning
            )
            
            self.signal_count += 1
            
            logger.debug(f"OB pressure signal for {symbol}: edge={edge_bps:.1f}bps, "
                        f"confidence={confidence:.2f}, pressure={pressure:.4f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating OB pressure signal for {symbol}: {e}")
            return None
    
    def update_from_feature_snapshot(self, feature_snapshot) -> Optional[AlphaSignal]:
        """
        Generate signal from FeatureBus snapshot.
        
        Args:
            feature_snapshot: FeatureSnapshot object with order book data
            
        Returns:
            AlphaSignal if signal generated, None otherwise
        """
        try:
            # Extract bid/ask sizes from feature snapshot
            # For this implementation, we'll use order_book_pressure if available
            # or simulate from order book imbalance
            
            if hasattr(feature_snapshot, 'order_book_pressure') and feature_snapshot.order_book_pressure is not None:
                # Use pre-calculated pressure
                pressure = feature_snapshot.order_book_pressure
                edge_bps = self.edge_scaling * pressure
                confidence = self.min_confidence + 0.5 * abs(pressure)
                confidence = min(confidence, self.max_confidence)
                
                reasoning = f"ob_pressure={pressure:.4f}, from_feature_bus=True"
                
            elif hasattr(feature_snapshot, 'order_book_imbalance') and feature_snapshot.order_book_imbalance is not None:
                # Use order book imbalance as proxy
                imbalance = feature_snapshot.order_book_imbalance
                
                # Convert imbalance to pressure-like metric
                pressure = imbalance  # Assuming imbalance is already normalized
                edge_bps = self.edge_scaling * pressure
                confidence = self.min_confidence + 0.5 * abs(pressure)
                confidence = min(confidence, self.max_confidence)
                
                reasoning = f"ob_imbalance={imbalance:.4f}, proxy_pressure={pressure:.4f}"
                
            else:
                # No order book data available
                return None
            
            signal = AlphaSignal(
                symbol=feature_snapshot.symbol,
                edge_bps=edge_bps,
                confidence=confidence,
                timestamp=feature_snapshot.timestamp.isoformat(),
                reasoning=reasoning
            )
            
            self.signal_count += 1
            
            logger.debug(f"OB pressure signal for {feature_snapshot.symbol}: edge={edge_bps:.1f}bps, "
                        f"confidence={confidence:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal from feature snapshot: {e}")
            return None
    
    def update_performance(self, symbol: str, realized_return_bps: float, predicted_edge_bps: float):
        """Update performance tracking based on realized returns."""
        try:
            # Check if prediction was correct
            if (predicted_edge_bps > 0 and realized_return_bps > 0) or \
               (predicted_edge_bps < 0 and realized_return_bps < 0):
                self.hit_count += 1
                
            logger.debug(f"Performance update for {symbol}: predicted={predicted_edge_bps:.1f}bps, "
                        f"realized={realized_return_bps:.1f}bps, hit_rate={self.get_hit_rate():.2%}")
                        
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def get_hit_rate(self) -> float:
        """Get current hit rate."""
        if self.signal_count == 0:
            return 0.55  # Default optimistic for order book pressure
        return self.hit_count / self.signal_count
    
    def get_stats(self) -> Dict[str, any]:
        """Get model statistics."""
        return {
            'model_name': 'ob_pressure_v0',
            'signal_count': self.signal_count,
            'hit_count': self.hit_count,
            'hit_rate': self.get_hit_rate(),
            'edge_scaling': self.edge_scaling,
            'min_confidence': self.min_confidence
        }
    
    def reset(self):
        """Reset model state."""
        self.signal_count = 0
        self.hit_count = 0
        logger.info("Order book pressure alpha model reset")


def create_ob_pressure_alpha(**kwargs) -> OrderBookPressureAlpha:
    """Factory function to create order book pressure alpha."""
    return OrderBookPressureAlpha(**kwargs)


def main():
    """Test the order book pressure alpha model."""
    logger.info("🚀 Testing Order Book Pressure Alpha Model...")
    
    # Initialize model
    alpha = OrderBookPressureAlpha(edge_scaling=25.0)
    
    # Test cases with different order book scenarios
    test_cases = [
        # Strong buy pressure
        {'symbol': 'BTC-USD', 'bid_size': 10.0, 'ask_size': 2.0, 'expected_edge': 'positive'},
        # Strong sell pressure  
        {'symbol': 'ETH-USD', 'bid_size': 3.0, 'ask_size': 12.0, 'expected_edge': 'negative'},
        # Balanced book
        {'symbol': 'SOL-USD', 'bid_size': 5.0, 'ask_size': 5.0, 'expected_edge': 'zero'},
        # Moderate buy pressure
        {'symbol': 'ADA-USD', 'bid_size': 8.0, 'ask_size': 6.0, 'expected_edge': 'positive'},
    ]
    
    signals = []
    
    for i, case in enumerate(test_cases):
        timestamp = f"2025-01-15T10:{i:02d}:00Z"
        signal = alpha.generate_signal(
            symbol=case['symbol'],
            bid_size=case['bid_size'],
            ask_size=case['ask_size'],
            timestamp=timestamp
        )
        
        if signal:
            signals.append(signal)
            pressure = (case['bid_size'] - case['ask_size']) / (case['bid_size'] + case['ask_size'] + 1e-9)
            
            print(f"📊 {case['symbol']} Signal: edge={signal.edge_bps:.1f}bps "
                  f"conf={signal.confidence:.2f} pressure={pressure:.3f}")
            
            # Verify edge direction matches expectation
            if case['expected_edge'] == 'positive' and signal.edge_bps > 0:
                print(f"✅ Correct positive edge for buy pressure")
            elif case['expected_edge'] == 'negative' and signal.edge_bps < 0:
                print(f"✅ Correct negative edge for sell pressure")
            elif case['expected_edge'] == 'zero' and abs(signal.edge_bps) < 1.0:
                print(f"✅ Correct near-zero edge for balanced book")
        else:
            print(f"❌ No signal generated for {case['symbol']}")
    
    # Test performance tracking
    if signals:
        # Simulate some performance updates
        for signal in signals[:2]:
            realized_return = signal.edge_bps * 0.6  # 60% of predicted edge realized
            alpha.update_performance(signal.symbol, realized_return, signal.edge_bps)
    
    # Print stats
    stats = alpha.get_stats()
    print(f"\n📈 Model Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Generated {len(signals)} signals")
    print(f"📊 Hit rate: {alpha.get_hit_rate():.1%}")
    
    # Test edge bounds
    edge_values = [abs(s.edge_bps) for s in signals]
    if edge_values:
        print(f"🎯 Edge range: {min(edge_values):.1f} to {max(edge_values):.1f} bps")
        print(f"📊 Max edge within ±25bp: {max(edge_values) <= 25}")
    
    # Test confidence bounds
    conf_values = [s.confidence for s in signals]
    if conf_values:
        print(f"🎯 Confidence range: {min(conf_values):.2f} to {max(conf_values):.2f}")
        print(f"📊 Confidence in [0.5,1]: {all(0.5 <= c <= 1.0 for c in conf_values)}")
    
    print("🎉 Order Book Pressure Alpha test completed!")


if __name__ == "__main__":
    main() 