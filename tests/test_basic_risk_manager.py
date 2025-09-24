"""Unit tests for the basic risk manager."""

import unittest
from unittest.mock import Mock

import pytest

pytest.importorskip(
    "aiofiles", reason="aiofiles optional dependency required for compliance WORM"
)

from src.layers.layer5_risk.basic_risk_manager import BasicRiskManager


class TestBasicRiskManager(unittest.TestCase):
    """Test the basic risk manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.risk_mgr = BasicRiskManager(
            max_position_pct=0.25,  # 25% max position
            max_drawdown_pct=0.03,  # 3% max drawdown
            volatility_multiplier=4.0,  # 4x vol circuit breaker
            min_trade_notional=10.0,  # $10 min trade size
        )

    def test_position_limit(self):
        """Test position size limit enforcement."""
        # Mock portfolio state
        portfolio = Mock()
        portfolio.total_value = 100000  # $100k portfolio
        portfolio.positions = {"AAPL": 24000}  # $24k position (24%)

        # Test trade that would exceed limit
        allowed = self.risk_mgr.check_position_limit(
            symbol="AAPL", new_notional=2000, portfolio=portfolio  # Would push to 26%
        )
        self.assertFalse(allowed)

        # Test trade within limit
        allowed = self.risk_mgr.check_position_limit(
            symbol="AAPL", new_notional=500, portfolio=portfolio  # Only to 24.5%
        )
        self.assertTrue(allowed)

    def test_drawdown_stop(self):
        """Test drawdown limit enforcement."""
        # Mock portfolio state
        portfolio = Mock()
        portfolio.total_value = 97000  # Down 3% from 100k
        portfolio.starting_value = 100000

        # Test trade when at drawdown limit
        allowed = self.risk_mgr.check_drawdown_limit(portfolio)
        self.assertFalse(allowed)

        # Test trade within drawdown limit
        portfolio.total_value = 98000  # Only down 2%
        allowed = self.risk_mgr.check_drawdown_limit(portfolio)
        self.assertTrue(allowed)

    def test_min_trade_filter(self):
        """Test minimum trade size filter."""
        # Test trade below minimum
        allowed = self.risk_mgr.check_min_trade_size(notional=5.0)
        self.assertFalse(allowed)

        # Test trade at minimum
        allowed = self.risk_mgr.check_min_trade_size(notional=10.0)
        self.assertTrue(allowed)

        # Test trade above minimum
        allowed = self.risk_mgr.check_min_trade_size(notional=20.0)
        self.assertTrue(allowed)
