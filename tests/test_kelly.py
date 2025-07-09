#!/usr/bin/env python3
"""
Test Kelly Position Sizing with Stocks Support
"""

from decimal import Decimal
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from layers.layer3_position_sizing.kelly_sizing import KellySizing


class TestKellySizing:
    """Test Kelly position sizing with stocks support"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.kelly = KellySizing(
            max_position_size=0.1,  # 10% max position
            kelly_fraction=0.25,    # Quarter Kelly
            min_edge_threshold=1.0  # 1bp minimum edge
        )
        
        self.portfolio_value = Decimal('100000')  # $100k portfolio
        self.stock_price = Decimal('150')  # $150 per share
        
    def test_stocks_position_sizing(self):
        """Test position sizing for stocks with Reg-T constraints"""
        # Test AAPL position sizing
        position, reasoning = self.kelly.calculate_position_size(
            symbol="AAPL",
            edge_bps=10.0,  # 10 basis points edge
            confidence=0.8,
            current_price=self.stock_price,
            portfolio_value=self.portfolio_value,
            instrument_type="stocks"
        )
        
        # Should return a position
        assert position != Decimal('0')
        assert abs(position) <= self.portfolio_value * Decimal('0.25')  # Max 25% for stocks
        assert "instrument=stocks" in reasoning
        assert "max_pos=25.0%" in reasoning
        
    def test_crypto_position_sizing(self):
        """Test position sizing for crypto"""
        position, reasoning = self.kelly.calculate_position_size(
            symbol="BTC-USD",
            edge_bps=15.0,  # 15 basis points edge
            confidence=0.9,
            current_price=Decimal('50000'),
            portfolio_value=self.portfolio_value,
            instrument_type="crypto"
        )
        
        # Should return a position
        assert position != Decimal('0')
        assert abs(position) <= self.portfolio_value * Decimal('0.20')  # Max 20% for crypto
        assert "instrument=crypto" in reasoning
        assert "max_pos=20.0%" in reasoning
        
    def test_leverage_constraints_stocks(self):
        """Test that stocks respect 4:1 Reg-T leverage limits"""
        # Add some existing positions to test leverage
        self.kelly.current_positions = {
            'AAPL': 50000,  # $50k position
            'TSLA': 30000,  # $30k position
            'MSFT': 20000   # $20k position
        }
        
        # Try to add another large position
        position, reasoning = self.kelly.calculate_position_size(
            symbol="GOOGL",
            edge_bps=20.0,
            confidence=0.9,
            current_price=Decimal('2500'),
            portfolio_value=self.portfolio_value,
            instrument_type="stocks"
        )
        
        # Should be limited by leverage constraints
        # Total positions would be $100k + new position
        # 4:1 leverage means max $400k total positions on $100k equity
        # Current positions = $100k, so max additional should be $300k
        # But individual position limit is 25% = $25k
        assert abs(position) <= Decimal('25000')
        
    def test_minimum_edge_threshold(self):
        """Test minimum edge threshold"""
        position, reasoning = self.kelly.calculate_position_size(
            symbol="AAPL",
            edge_bps=0.5,  # Below 1bp threshold
            confidence=0.8,
            current_price=self.stock_price,
            portfolio_value=self.portfolio_value,
            instrument_type="stocks"
        )
        
        assert position == Decimal('0')
        assert "below threshold" in reasoning
        
    def test_negative_edge_short_position(self):
        """Test negative edge results in short position"""
        position, reasoning = self.kelly.calculate_position_size(
            symbol="AAPL",
            edge_bps=-10.0,  # Negative edge (short signal)
            confidence=0.8,
            current_price=self.stock_price,
            portfolio_value=self.portfolio_value,
            instrument_type="stocks"
        )
        
        # Should be negative (short position)
        assert position < Decimal('0')
        assert abs(position) <= self.portfolio_value * Decimal('0.25')
        
    def test_drawdown_limit(self):
        """Test drawdown limit prevents new positions"""
        # Set high drawdown
        self.kelly.current_drawdown = 0.06  # 6% drawdown (above 5% limit)
        
        position, reasoning = self.kelly.calculate_position_size(
            symbol="AAPL",
            edge_bps=10.0,
            confidence=0.8,
            current_price=self.stock_price,
            portfolio_value=self.portfolio_value,
            instrument_type="stocks"
        )
        
        assert position == Decimal('0')
        assert "exceeds limit" in reasoning
        
    def test_instrument_constraints(self):
        """Test that instrument constraints are properly applied"""
        # Test stocks constraints
        stocks_constraints = self.kelly.instrument_constraints['stocks']
        assert stocks_constraints['max_leverage'] == 4.0
        assert stocks_constraints['max_position_pct'] == 0.25
        
        # Test crypto constraints
        crypto_constraints = self.kelly.instrument_constraints['crypto']
        assert crypto_constraints['max_leverage'] == 3.0
        assert crypto_constraints['max_position_pct'] == 0.20


if __name__ == "__main__":
    # Run tests
    test = TestKellySizing()
    test.setup_method()
    
    print("🧪 Running Kelly sizing tests...")
    
    try:
        test.test_stocks_position_sizing()
        print("✅ Stocks position sizing test passed")
        
        test.test_crypto_position_sizing()
        print("✅ Crypto position sizing test passed")
        
        test.test_leverage_constraints_stocks()
        print("✅ Leverage constraints test passed")
        
        test.test_minimum_edge_threshold()
        print("✅ Minimum edge threshold test passed")
        
        test.test_negative_edge_short_position()
        print("✅ Negative edge short position test passed")
        
        test.test_drawdown_limit()
        print("✅ Drawdown limit test passed")
        
        test.test_instrument_constraints()
        print("✅ Instrument constraints test passed")
        
        print("🎉 All Kelly sizing tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise 