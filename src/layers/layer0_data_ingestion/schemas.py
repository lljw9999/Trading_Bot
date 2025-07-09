"""
Data schemas for market data normalization
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from decimal import Decimal


@dataclass
class MarketTick:
    """Unified market tick data structure for all asset classes."""
    
    # Identity
    symbol: str
    exchange: str
    asset_type: str  # 'crypto' or 'stock'
    
    # Timing
    timestamp: datetime
    exchange_timestamp: Optional[datetime] = None
    
    # Price data
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    mid: Optional[Decimal] = None
    last: Optional[Decimal] = None
    
    # Volume
    bid_size: Optional[Decimal] = None
    ask_size: Optional[Decimal] = None
    volume: Optional[Decimal] = None
    
    # Order book (top 5 levels)
    bids: Optional[List[Tuple[Decimal, Decimal]]] = None  # [(price, size), ...]
    asks: Optional[List[Tuple[Decimal, Decimal]]] = None  # [(price, size), ...]
    
    # Derived fields
    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.ask and self.bid:
            return self.ask - self.bid
        return None
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Calculate spread in basis points."""
        if self.spread and self.mid and self.mid > 0:
            return float(self.spread / self.mid * 10000)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'asset_type': self.asset_type,
            'timestamp': self.timestamp.isoformat(),
            'exchange_timestamp': self.exchange_timestamp.isoformat() if self.exchange_timestamp else None,
            'bid': float(self.bid) if self.bid else None,
            'ask': float(self.ask) if self.ask else None,
            'mid': float(self.mid) if self.mid else None,
            'last': float(self.last) if self.last else None,
            'bid_size': float(self.bid_size) if self.bid_size else None,
            'ask_size': float(self.ask_size) if self.ask_size else None,
            'volume': float(self.volume) if self.volume else None,
            'spread': float(self.spread) if self.spread else None,
            'spread_bps': self.spread_bps
        }


@dataclass
class FeatureSnapshot:
    """Computed features from market data for alpha models."""
    
    # Identity and timing
    symbol: str
    timestamp: datetime
    
    # Basic price features
    mid_price: Optional[Decimal] = None
    spread: Optional[Decimal] = None
    spread_bps: Optional[float] = None
    
    # Returns
    return_1m: Optional[float] = None
    return_5m: Optional[float] = None
    return_15m: Optional[float] = None
    return_1h: Optional[float] = None
    
    # Volatility
    volatility_5m: Optional[float] = None
    volatility_15m: Optional[float] = None
    volatility_1h: Optional[float] = None
    
    # Order book features
    order_book_imbalance: Optional[float] = None
    order_book_pressure: Optional[float] = None
    
    # Volume features
    volume_1m: Optional[Decimal] = None
    volume_5m: Optional[Decimal] = None
    volume_ratio: Optional[float] = None  # 1m/5m volume ratio
    
    # Technical indicators
    sma_5: Optional[Decimal] = None
    sma_20: Optional[Decimal] = None
    sma_50: Optional[Decimal] = None
    ema_5: Optional[Decimal] = None
    ema_20: Optional[Decimal] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[Decimal] = None
    bollinger_lower: Optional[Decimal] = None
    
    # Market microstructure
    tick_direction: Optional[int] = None  # 1 for uptick, -1 for downtick
    effective_spread: Optional[float] = None
    realized_spread: Optional[float] = None
    price_impact: Optional[float] = None
    
    # Soft Information Features (Sentiment & Fundamentals)
    sent_score: Optional[float] = None  # -1.0 to 1.0, sentiment score from GPT-4o
    fund_pe: Optional[float] = None     # Fundamental P/E ratio for context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'mid_price': float(self.mid_price) if self.mid_price else None,
            'spread': float(self.spread) if self.spread else None,
            'spread_bps': self.spread_bps,
            'return_1m': self.return_1m,
            'return_5m': self.return_5m,
            'return_15m': self.return_15m,
            'return_1h': self.return_1h,
            'volatility_5m': self.volatility_5m,
            'volatility_15m': self.volatility_15m,
            'volatility_1h': self.volatility_1h,
            'order_book_imbalance': self.order_book_imbalance,
            'order_book_pressure': self.order_book_pressure,
            'volume_1m': float(self.volume_1m) if self.volume_1m else None,
            'volume_5m': float(self.volume_5m) if self.volume_5m else None,
            'volume_ratio': self.volume_ratio,
            'sma_5': float(self.sma_5) if self.sma_5 else None,
            'sma_20': float(self.sma_20) if self.sma_20 else None,
            'sma_50': float(self.sma_50) if self.sma_50 else None,
            'ema_5': float(self.ema_5) if self.ema_5 else None,
            'ema_20': float(self.ema_20) if self.ema_20 else None,
            'rsi_14': self.rsi_14,
            'macd': self.macd,
            'macd_signal': self.macd_signal,
            'bollinger_upper': float(self.bollinger_upper) if self.bollinger_upper else None,
            'bollinger_lower': float(self.bollinger_lower) if self.bollinger_lower else None,
            'tick_direction': self.tick_direction,
            'effective_spread': self.effective_spread,
            'realized_spread': self.realized_spread,
            'price_impact': self.price_impact,
            'sent_score': self.sent_score,
            'fund_pe': self.fund_pe
        }


@dataclass
class AlphaSignal:
    """Alpha signal from individual models."""
    
    model_name: str
    symbol: str
    timestamp: datetime
    edge_bps: float  # Expected edge in basis points
    confidence: float  # Model confidence [0, 1]
    signal_strength: float  # Raw signal strength
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'edge_bps': self.edge_bps,
            'confidence': self.confidence,
            'signal_strength': self.signal_strength,
            'metadata': self.metadata or {}
        }


@dataclass
class TradingDecision:
    """Final trading decision from the ensemble."""
    
    symbol: str
    timestamp: datetime
    target_position: Decimal  # Target position in dollars
    current_position: Decimal  # Current position in dollars
    edge_bps: float  # Ensemble edge estimate
    confidence: float  # Ensemble confidence
    position_delta: Decimal  # Required position change
    reasoning: str  # Human-readable reasoning
    risk_adjusted: bool  # Whether position was risk-adjusted
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'target_position': float(self.target_position),
            'current_position': float(self.current_position),
            'edge_bps': self.edge_bps,
            'confidence': self.confidence,
            'position_delta': float(self.position_delta),
            'reasoning': self.reasoning,
            'risk_adjusted': self.risk_adjusted,
            'metadata': self.metadata or {}
        } 