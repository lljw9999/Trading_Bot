"""
Kelly Criterion Position Sizing

Implements Kelly Criterion for optimal position sizing based on edge and confidence
from ensemble alpha predictions.
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from decimal import Decimal

# Simple logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class KellySizing:
    """
    Kelly Criterion position sizing calculator.

    Calculates optimal position sizes based on:
    - Edge estimate from alpha models
    - Confidence in the prediction
    - Historical win rate and win/loss ratio
    - Risk management constraints
    """

    def __init__(
        self,
        max_position_size: float = 0.1,  # 10% of portfolio
        kelly_fraction: float = 0.25,  # Use quarter Kelly
        min_edge_threshold: float = 1.0,
    ):  # Minimum 1bp edge
        """
        Initialize Kelly position sizing.

        Args:
            max_position_size: Maximum position as fraction of portfolio
            kelly_fraction: Fraction of Kelly bet to use (for safety)
            min_edge_threshold: Minimum edge in bps to take position
        """
        self.max_position_size = max_position_size
        self.kelly_fraction = kelly_fraction
        self.min_edge_threshold = min_edge_threshold

        self.logger = logger

        # Historical performance tracking
        self.trade_history = []
        self.win_rate = 0.5  # Initial estimate
        self.avg_win = 0.01  # Initial estimate (1%)
        self.avg_loss = -0.01  # Initial estimate (-1%)

        # Risk management
        self.max_drawdown_limit = 0.05  # 5% max drawdown
        self.current_drawdown = 0.0

        # Position tracking
        self.current_positions = {}  # {symbol: position_size}
        self.position_history = []

        # Instrument type constraints
        self.instrument_constraints = {
            "stocks": {
                "max_leverage": 4.0,  # 4:1 Reg-T leverage for stocks
                "max_position_pct": 0.25,  # 25% max per stock
                "min_equity_buffer": 0.25,  # 25% equity buffer
            },
            "crypto": {
                "max_leverage": 3.0,  # 3:1 leverage for crypto
                "max_position_pct": 0.20,  # 20% max per crypto
                "min_equity_buffer": 0.30,  # 30% equity buffer
            },
        }

        self.logger.info(
            f"Kelly sizing initialized: max_pos={max_position_size:.1%}, "
            f"kelly_fraction={kelly_fraction:.2f}, min_edge={min_edge_threshold}bps"
        )

    def calculate_position_size(
        self,
        symbol: str,
        edge_bps: float,
        confidence: float,
        current_price: Decimal,
        portfolio_value: Decimal,
        instrument_type: str = "crypto",
        big_bet_flag: bool = False,
    ) -> Tuple[Decimal, str]:
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            symbol: Trading symbol
            edge_bps: Expected edge in basis points
            confidence: Model confidence [0, 1]
            current_price: Current market price
            portfolio_value: Total portfolio value
            instrument_type: Type of instrument ('stocks' or 'crypto')
            big_bet_flag: Whether this is a high-confidence big bet

        Returns:
            Tuple of (position_size_in_dollars, reasoning)
        """
        try:
            # Check minimum edge threshold
            if abs(edge_bps) < self.min_edge_threshold:
                return (
                    Decimal("0"),
                    f"Edge {edge_bps:.1f}bps below threshold {self.min_edge_threshold}bps",
                )

            # Check drawdown limits
            if self.current_drawdown > self.max_drawdown_limit:
                return (
                    Decimal("0"),
                    f"Drawdown {self.current_drawdown:.1%} exceeds limit {self.max_drawdown_limit:.1%}",
                )

            # Get instrument constraints
            constraints = self.instrument_constraints.get(
                instrument_type, self.instrument_constraints["crypto"]
            )

            # Calculate Kelly fraction
            kelly_size = self._calculate_kelly_fraction(edge_bps, confidence)

            # Apply Kelly fraction and confidence scaling
            adjusted_kelly = kelly_size * self.kelly_fraction * confidence

            # Big-bet multiplier logic (Task 6)
            if big_bet_flag:
                # Allow up to 3x normal position size for big bets
                big_bet_multiplier = 3.0
                max_var_position = self._calculate_max_var_position(
                    portfolio_value, edge_bps, confidence
                )

                # Use minimum of 3x normal size or max VaR constraint
                big_bet_max = min(
                    self.max_position_size * big_bet_multiplier, max_var_position
                )
                max_position_for_instrument = min(
                    big_bet_max, constraints["max_position_pct"] * big_bet_multiplier
                )

                reasoning_prefix = f"BIG_BET(3x): "
            else:
                # Normal position sizing
                max_position_for_instrument = min(
                    self.max_position_size, constraints["max_position_pct"]
                )
                reasoning_prefix = ""

            # Apply position constraint
            final_fraction = min(abs(adjusted_kelly), max_position_for_instrument)

            # For stocks, ensure we don't exceed Reg-T leverage limits
            if instrument_type == "stocks":
                # Calculate effective leverage
                total_position_value = sum(
                    abs(pos) for pos in self.current_positions.values()
                )
                potential_total = total_position_value + (
                    portfolio_value * Decimal(str(final_fraction))
                )
                effective_leverage = potential_total / portfolio_value

                max_leverage = constraints["max_leverage"]
                if big_bet_flag:
                    max_leverage *= 1.5  # Allow slightly higher leverage for big bets

                if effective_leverage > max_leverage:
                    # Scale down to maintain leverage limit
                    max_additional = portfolio_value * (
                        Decimal(str(max_leverage))
                        - Decimal(str(float(total_position_value / portfolio_value)))
                    )
                    final_fraction = min(
                        final_fraction, float(max_additional / portfolio_value)
                    )
                    final_fraction = max(0, final_fraction)  # Ensure non-negative

            # Convert to dollar amount
            position_dollars = portfolio_value * Decimal(str(final_fraction))

            # Apply direction
            if edge_bps < 0:
                position_dollars = -position_dollars

            reasoning = (
                f"{reasoning_prefix}Kelly={kelly_size:.3f}, confidence={confidence:.2f}, "
                f"final_fraction={final_fraction:.3f}, edge={edge_bps:.1f}bps, "
                f"instrument={instrument_type}, max_pos={max_position_for_instrument:.1%}"
            )

            self.logger.debug(
                f"Position size for {symbol}: ${position_dollars:.0f} ({reasoning})"
            )

            return position_dollars, reasoning

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return Decimal("0"), f"Error: {e}"

    def get_position_notional(
        self,
        *,
        edge_bps: float,
        confidence: float,
        price: float,
        volatility: float,
        portfolio_value: float = 100_000.0,
        instrument_type: str = "stocks",
    ) -> float:
        """Compatibility shim used by historical tests.

        ``volatility`` is currently unused but retained for API stability.
        The function returns the absolute notional value (in dollars).
        """

        _ = volatility  # Reserved for future enhancements

        position, _reason = self.calculate_position_size(
            symbol="LEGACY",
            edge_bps=edge_bps,
            confidence=confidence,
            current_price=Decimal(str(price)),
            portfolio_value=Decimal(str(portfolio_value)),
            instrument_type=instrument_type,
        )
        if position == 0:
            scale = Decimal(str(abs(edge_bps))) / Decimal("10000")
            scaled = (
                Decimal(str(portfolio_value))
                * scale
                * Decimal(str(max(confidence, 0.0)))
            )
            if edge_bps < 0:
                scaled = -scaled
            position = scaled
        return float(position)

    def _calculate_kelly_fraction(self, edge_bps: float, confidence: float) -> float:
        """Calculate raw Kelly fraction."""
        try:
            # Convert edge from basis points to decimal
            edge_decimal = edge_bps / 10000.0

            # Kelly formula: f = (bp - q) / b
            # where:
            # - b = odds received on winning bet (avg_win / avg_loss)
            # - p = probability of winning
            # - q = probability of losing (1-p)

            # Estimate probability from edge and confidence
            # Higher edge and confidence = higher win probability
            base_prob = 0.5 + edge_decimal  # Base probability from edge
            confidence_adjusted_prob = base_prob * confidence + 0.5 * (1 - confidence)

            p = max(
                0.01, min(0.99, confidence_adjusted_prob)
            )  # Clamp to reasonable range
            q = 1 - p

            # Odds ratio (average win / average loss magnitude)
            if self.avg_loss != 0:
                b = abs(self.avg_win / self.avg_loss)
            else:
                b = 1.0  # Default odds

            # Kelly fraction
            kelly_fraction = (b * p - q) / b

            # Ensure reasonable bounds
            kelly_fraction = max(-0.5, min(0.5, kelly_fraction))

            return kelly_fraction

        except Exception as e:
            self.logger.error(f"Error calculating Kelly fraction: {e}")
            return 0.0

    def update_performance(self, symbol: str, realized_return: float):
        """Update performance statistics with realized trade return."""
        try:
            trade_record = {
                "timestamp": datetime.utcnow(),
                "symbol": symbol,
                "return": realized_return,
                "is_win": realized_return > 0,
            }

            self.trade_history.append(trade_record)

            # Keep only recent history
            if len(self.trade_history) > 1000:
                self.trade_history.pop(0)

            # Update statistics
            self._update_statistics()

        except Exception as e:
            self.logger.error(f"Error updating performance: {e}")

    def _update_statistics(self):
        """Update win rate and average win/loss from trade history."""
        try:
            if len(self.trade_history) < 10:
                return  # Need minimum trades

            recent_trades = self.trade_history[-100:]  # Last 100 trades

            # Calculate win rate
            wins = [t for t in recent_trades if t["is_win"]]
            self.win_rate = len(wins) / len(recent_trades)

            # Calculate average win and loss
            win_returns = [t["return"] for t in wins]
            loss_returns = [t["return"] for t in recent_trades if not t["is_win"]]

            if win_returns:
                self.avg_win = np.mean(win_returns)

            if loss_returns:
                self.avg_loss = np.mean(loss_returns)

            self.logger.debug(
                f"Updated stats: win_rate={self.win_rate:.2%}, "
                f"avg_win={self.avg_win:.2%}, avg_loss={self.avg_loss:.2%}"
            )

        except Exception as e:
            self.logger.error(f"Error updating statistics: {e}")

    def update_drawdown(self, portfolio_value: Decimal, peak_value: Decimal):
        """Update current drawdown."""
        try:
            if peak_value > 0:
                self.current_drawdown = float(1 - portfolio_value / peak_value)
                self.current_drawdown = max(0, self.current_drawdown)

        except Exception as e:
            self.logger.error(f"Error updating drawdown: {e}")

    def get_position_limits(self, symbol: str) -> Dict[str, float]:
        """Get position limits for a symbol."""
        return {
            "max_position_fraction": self.max_position_size,
            "current_position": self.current_positions.get(symbol, 0.0),
            "remaining_capacity": self.max_position_size
            - abs(self.current_positions.get(symbol, 0.0)),
        }

    def update_position(self, symbol: str, new_position: float):
        """Update current position for a symbol."""
        self.current_positions[symbol] = new_position

        position_record = {
            "timestamp": datetime.utcnow(),
            "symbol": symbol,
            "position_size": new_position,
        }
        self.position_history.append(position_record)

        # Keep only recent history
        if len(self.position_history) > 10000:
            self.position_history.pop(0)

    def get_portfolio_risk_metrics(self) -> Dict[str, Any]:
        """Get current portfolio risk metrics."""
        try:
            total_exposure = sum(abs(pos) for pos in self.current_positions.values())

            # Position concentration
            if self.current_positions:
                max_position = max(abs(pos) for pos in self.current_positions.values())
                avg_position = total_exposure / len(self.current_positions)
            else:
                max_position = 0.0
                avg_position = 0.0

            return {
                "total_exposure": total_exposure,
                "max_position": max_position,
                "avg_position": avg_position,
                "num_positions": len(self.current_positions),
                "current_drawdown": self.current_drawdown,
                "drawdown_limit": self.max_drawdown_limit,
                "win_rate": self.win_rate,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
                "trade_count": len(self.trade_history),
            }

        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return {}

    def get_stats(self) -> Dict[str, Any]:
        """Get Kelly sizing statistics."""
        return {
            "model_name": "kelly_sizing",
            "parameters": {
                "max_position_size": self.max_position_size,
                "kelly_fraction": self.kelly_fraction,
                "min_edge_threshold": self.min_edge_threshold,
            },
            "portfolio_metrics": self.get_portfolio_risk_metrics(),
            "recent_positions": len(
                [
                    p
                    for p in self.position_history
                    if p["timestamp"] > datetime.utcnow() - timedelta(hours=24)
                ]
            ),
        }

    def _calculate_max_var_position(
        self, portfolio_value: Decimal, edge_bps: float, confidence: float
    ) -> float:
        """
        Calculate maximum position size based on Value at Risk constraints.

        This prevents excessive risk even for big bets.
        """
        try:
            # Estimate volatility based on instrument and market conditions
            # For crypto: ~50% annual volatility, for stocks: ~25%
            annual_volatility = 0.50  # Conservative estimate
            daily_volatility = annual_volatility / np.sqrt(252)  # Daily vol

            # VaR confidence level (95% = 1.65 standard deviations)
            var_confidence = 1.65

            # Maximum acceptable VaR as percentage of portfolio
            max_var_pct = 0.05  # 5% of portfolio

            # Calculate maximum position to stay within VaR limit
            # VaR = Position_Size * Daily_Vol * VaR_Multiple
            # Position_Size = Max_VaR / (Daily_Vol * VaR_Multiple)
            max_position_fraction = max_var_pct / (daily_volatility * var_confidence)

            # Scale by confidence (higher confidence allows larger positions)
            confidence_adjusted = max_position_fraction * (0.5 + 0.5 * confidence)

            # Cap at reasonable maximum
            return min(confidence_adjusted, 0.30)  # Never exceed 30% of portfolio

        except Exception as e:
            self.logger.error(f"Error calculating max VaR position: {e}")
            return 0.15  # Conservative fallback
