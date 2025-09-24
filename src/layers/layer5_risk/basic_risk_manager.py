"""
Basic Risk Manager for Trading System

Implements essential risk checks:
- Position size limits
- Portfolio concentration limits
- Drawdown monitoring
- Volatility-based circuit breakers
"""

import logging
from typing import Dict, Tuple, Optional
from decimal import Decimal
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class BasicRiskManager:
    """Basic risk management implementation."""

    def __init__(
        self,
        max_position_pct: float = 0.25,  # Max 25% in single position
        max_drawdown_pct: float = 0.03,  # 3% max drawdown
        vol_multiplier_limit: float = 4.0,  # 4x normal vol triggers halt
        min_trade_size: float = 10.0,
        volatility_multiplier: Optional[float] = None,
        min_trade_notional: Optional[float] = None,
    ):  # $10 minimum trade
        """Initialize risk parameters."""
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        if volatility_multiplier is not None:
            vol_multiplier_limit = volatility_multiplier
        self.vol_multiplier_limit = vol_multiplier_limit
        if min_trade_notional is not None:
            min_trade_size = min_trade_notional
        self.min_trade_size = min_trade_size

        # State tracking
        self.positions: Dict[str, Decimal] = {}  # Current positions by symbol
        self.peak_equity = Decimal("0")  # High water mark
        self.current_equity = Decimal("0")  # Current equity
        self.vol_by_symbol: Dict[str, float] = {}  # Volatility tracking
        self.trading_halted = False
        self.total_checks = 0
        self.total_rejections = 0

        logger.info(
            f"🛡️  Risk Manager initialized with:"
            f"\n    Max position: {self.max_position_pct:.1%}"
            f"\n    Max drawdown: {self.max_drawdown_pct:.1%}"
            f"\n    Vol limit: {self.vol_multiplier_limit}x"
            f"\n    Min trade: ${self.min_trade_size:.2f}"
        )

    def check_position_risk(
        self,
        symbol: str,
        proposed_position: Decimal,
        current_price: Decimal,
        portfolio_value: Decimal,
    ) -> Tuple[bool, str, Decimal]:
        """
        Check if a proposed position meets risk criteria.

        Returns:
            Tuple[bool, str, Decimal]: (allowed?, reason, max_allowed_size)
        """
        self.total_checks += 1

        # Update state
        self.current_equity = portfolio_value
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        # 1. Check if trading is halted
        if self.trading_halted:
            self.total_rejections += 1
            return False, "Trading halted due to risk breach", Decimal("0")

        # 2. Check drawdown limit
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            if drawdown > self.max_drawdown_pct:
                self.trading_halted = True
                self.total_rejections += 1
                return False, f"Drawdown limit breached: {drawdown:.1%}", Decimal("0")

        # 3. Check position size limit
        max_position = portfolio_value * Decimal(str(self.max_position_pct))
        if abs(proposed_position) > max_position:
            self.total_rejections += 1
            return (
                False,
                f"Position size exceeds {self.max_position_pct:.0%} limit",
                max_position,
            )

        # 4. Check minimum trade size
        if abs(proposed_position) < self.min_trade_size:
            self.total_rejections += 1
            return (
                False,
                f"Below minimum trade size ${self.min_trade_size}",
                Decimal("0"),
            )

        # All checks passed
        return True, "Position approved", proposed_position

    def update_volatility(
        self, symbol: str, current_vol: float, avg_vol: float
    ) -> None:
        """Update volatility tracking and check circuit breakers."""
        vol_multiple = current_vol / max(avg_vol, 1e-9)  # Avoid div by 0
        self.vol_by_symbol[symbol] = vol_multiple

        if vol_multiple > self.vol_multiplier_limit:
            logger.warning(
                f"🚨 Circuit breaker: {symbol} volatility {vol_multiple:.1f}x normal"
            )
            self.trading_halted = True

    def get_stats(self) -> Dict:
        """Get risk manager statistics."""
        return {
            "Total checks": self.total_checks,
            "Total rejections": self.total_rejections,
            "Rejection rate": f"{(self.total_rejections/max(1,self.total_checks)):.1%}",
            "Trading halted": self.trading_halted,
            "Current drawdown": f"{((self.peak_equity - self.current_equity)/max(self.peak_equity,1)):.2%}",
            "Peak equity": f"${float(self.peak_equity):,.2f}",
            "Current equity": f"${float(self.current_equity):,.2f}",
        }

    # Compatibility helpers used by historical tests
    def check_position_limit(self, symbol: str, new_notional: float, portfolio) -> bool:
        """Return True if the additional position keeps exposure within limits."""

        portfolio_value = Decimal(str(getattr(portfolio, "total_value", 0) or 0))
        current_position = Decimal(
            str(getattr(portfolio.positions, "get", lambda *_: 0)(symbol, 0))
        )
        proposed = current_position + Decimal(str(new_notional))
        allowed, _, max_allowed = self.check_position_risk(
            symbol=symbol,
            proposed_position=proposed,
            current_price=Decimal("1"),
            portfolio_value=portfolio_value,
        )
        return allowed and abs(proposed) <= max_allowed

    def check_drawdown_limit(self, portfolio) -> bool:
        """Return False if the drawdown limit has been exceeded."""

        starting_value = Decimal(str(getattr(portfolio, "starting_value", 0) or 0))
        current_value = Decimal(str(getattr(portfolio, "total_value", 0) or 0))
        if starting_value <= 0:
            return True

        drawdown = (starting_value - current_value) / starting_value
        return drawdown < Decimal(str(self.max_drawdown_pct))

    def check_min_trade_size(self, notional: float) -> bool:
        """Return True if notional meets the minimum trade requirement."""

        return abs(notional) >= self.min_trade_size
