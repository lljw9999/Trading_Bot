#!/usr/bin/env python3
"""
Portfolio Signals Module

Implements multi-asset portfolio signals using LightGBM Sharpe optimization.
Provides position sizing, risk management, and real-time signal generation
for quantitative trading strategies.

Key Features:
- Multi-asset position sizing
- Dynamic risk allocation
- Correlation-based diversification  
- Real-time signal generation
- Portfolio rebalancing logic
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

from .lgb_sharpe import SharpeOptimizer

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class PortfolioSignal:
    """Portfolio signal data structure."""

    symbol: str
    timestamp: datetime
    position_size: float
    confidence: float
    expected_sharpe: float
    risk_contribution: float
    signal_type: str  # 'entry', 'exit', 'rebalance'

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "position_size": self.position_size,
            "confidence": self.confidence,
            "expected_sharpe": self.expected_sharpe,
            "risk_contribution": self.risk_contribution,
            "signal_type": self.signal_type,
        }


class PortfolioSignalGenerator:
    """
    Multi-asset portfolio signal generator using LightGBM Sharpe optimization.

    Generates position sizing signals with risk management and diversification
    constraints for quantitative trading strategies.
    """

    def __init__(
        self,
        target_volatility: float = 0.15,  # 15% annual portfolio volatility
        max_concentration: float = 0.25,  # Max 25% in any single asset
        correlation_threshold: float = 0.7,  # Correlation limit for diversification
        rebalance_threshold: float = 0.05,  # 5% drift for rebalancing
        min_signal_strength: float = 0.6,  # Minimum confidence threshold
        lookback_days: int = 252,
    ):  # 1 year lookback for correlation
        """
        Initialize portfolio signal generator.

        Args:
            target_volatility: Target portfolio volatility (annual)
            max_concentration: Maximum weight in any single asset
            correlation_threshold: Maximum correlation for position scaling
            rebalance_threshold: Position drift threshold for rebalancing
            min_signal_strength: Minimum signal confidence
            lookback_days: Days for correlation/risk calculation
        """
        self.target_volatility = target_volatility
        self.max_concentration = max_concentration
        self.correlation_threshold = correlation_threshold
        self.rebalance_threshold = rebalance_threshold
        self.min_signal_strength = min_signal_strength
        self.lookback_days = lookback_days

        # Portfolio state
        self.current_positions: Dict[str, float] = {}
        self.last_rebalance: Optional[datetime] = None
        self.risk_budgets: Dict[str, float] = {}

        # Models per asset
        self.optimizers: Dict[str, SharpeOptimizer] = {}

        # Market data cache
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.volatility_estimates: Dict[str, float] = {}

        logger.info(f"PortfolioSignalGenerator initialized:")
        logger.info(f"  - Target volatility: {target_volatility:.1%}")
        logger.info(f"  - Max concentration: {max_concentration:.1%}")
        logger.info(f"  - Correlation threshold: {correlation_threshold:.2f}")

    def add_asset(self, symbol: str, optimizer: SharpeOptimizer) -> None:
        """Add an asset with its trained optimizer."""
        self.optimizers[symbol] = optimizer
        self.current_positions[symbol] = 0.0
        logger.info(f"Added asset {symbol} to portfolio")

    def update_market_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Update market data for an asset."""
        if symbol not in self.optimizers:
            raise ValueError(f"Asset {symbol} not in portfolio")

        # Store price data
        self.price_data[symbol] = data.copy()

        # Update volatility estimate
        returns = data["close"].pct_change().dropna()
        if len(returns) >= 20:
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            self.volatility_estimates[symbol] = volatility

        logger.debug(f"Updated market data for {symbol}")

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for all assets."""
        if len(self.price_data) < 2:
            return pd.DataFrame()

        # Align data and calculate returns
        returns_data = {}
        min_date = None

        for symbol, data in self.price_data.items():
            returns = data["close"].pct_change().dropna()
            if len(returns) > 0:
                returns_data[symbol] = returns
                if min_date is None or returns.index[0] > min_date:
                    min_date = returns.index[0]

        if len(returns_data) < 2:
            return pd.DataFrame()

        # Create aligned returns DataFrame
        aligned_returns = pd.DataFrame()
        for symbol, returns in returns_data.items():
            aligned_data = returns[returns.index >= min_date].tail(self.lookback_days)
            aligned_returns[symbol] = aligned_data

        # Calculate correlation
        correlation = aligned_returns.corr()
        self.correlation_matrix = correlation

        return correlation

    def calculate_risk_budgets(self) -> Dict[str, float]:
        """Calculate risk budgets for each asset based on diversification."""
        if not self.volatility_estimates:
            return {symbol: 1.0 / len(self.optimizers) for symbol in self.optimizers}

        # Equal risk contribution as starting point
        assets = list(self.optimizers.keys())
        n_assets = len(assets)

        if n_assets == 1:
            return {assets[0]: 1.0}

        # Calculate inverse volatility weights
        inv_vols = {}
        for symbol in assets:
            vol = self.volatility_estimates.get(symbol, 0.15)  # Default 15% volatility
            inv_vols[symbol] = 1.0 / vol

        # Normalize to sum to 1
        total_inv_vol = sum(inv_vols.values())
        risk_budgets = {
            symbol: inv_vol / total_inv_vol for symbol, inv_vol in inv_vols.items()
        }

        # Apply concentration limits
        for symbol in risk_budgets:
            risk_budgets[symbol] = min(risk_budgets[symbol], self.max_concentration)

        # Renormalize after concentration limits
        total_budget = sum(risk_budgets.values())
        if total_budget > 0:
            risk_budgets = {
                symbol: budget / total_budget for symbol, budget in risk_budgets.items()
            }

            # Apply concentration limits again after renormalization
            for symbol in risk_budgets:
                risk_budgets[symbol] = min(risk_budgets[symbol], self.max_concentration)

        self.risk_budgets = risk_budgets
        return risk_budgets

    def generate_raw_signals(
        self, timestamp: datetime
    ) -> Dict[str, Tuple[float, float]]:
        """
        Generate raw position signals for all assets.

        Args:
            timestamp: Current timestamp

        Returns:
            Dict mapping symbol to (position_size, confidence)
        """
        raw_signals = {}

        for symbol, optimizer in self.optimizers.items():
            if symbol not in self.price_data:
                continue

            try:
                # Get recent data for prediction
                data = self.price_data[symbol].tail(optimizer.lookback_window)

                if len(data) < 50:  # Minimum data requirement
                    continue

                # Generate position signal
                position_sizes = optimizer.predict_position_size(data)

                if len(position_sizes) > 0:
                    latest_position = position_sizes.iloc[-1]

                    # Calculate confidence based on consistency
                    recent_positions = position_sizes.tail(5)
                    position_std = recent_positions.std()
                    position_mean = recent_positions.mean()

                    # Higher consistency -> Higher confidence
                    if position_mean > 0:
                        confidence = max(
                            0.1, min(0.9, 1.0 - (position_std / position_mean))
                        )
                    else:
                        confidence = 0.5

                    raw_signals[symbol] = (float(latest_position), float(confidence))

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue

        return raw_signals

    def apply_portfolio_constraints(
        self, raw_signals: Dict[str, Tuple[float, float]], timestamp: datetime
    ) -> Dict[str, PortfolioSignal]:
        """
        Apply portfolio-level constraints to raw signals.

        Args:
            raw_signals: Raw position signals
            timestamp: Current timestamp

        Returns:
            Portfolio-adjusted signals
        """
        if not raw_signals:
            return {}

        # Calculate risk budgets and correlation
        risk_budgets = self.calculate_risk_budgets()
        correlation_matrix = self.calculate_correlation_matrix()

        # Filter signals by minimum strength
        filtered_signals = {
            symbol: (pos, conf)
            for symbol, (pos, conf) in raw_signals.items()
            if conf >= self.min_signal_strength
        }

        if not filtered_signals:
            return {}

        # Apply concentration limits and risk budgets
        portfolio_signals = {}
        total_position = 0.0

        for symbol, (raw_position, confidence) in filtered_signals.items():
            # Scale by risk budget
            risk_budget = risk_budgets.get(symbol, 0.0)

            # Apply correlation penalty
            correlation_penalty = self._calculate_correlation_penalty(
                symbol, correlation_matrix, filtered_signals
            )

            # Adjust position size
            adjusted_position = raw_position * risk_budget * correlation_penalty

            # Apply absolute limits
            adjusted_position = min(adjusted_position, self.max_concentration)

            # Calculate expected Sharpe (simplified)
            vol = self.volatility_estimates.get(symbol, 0.15)
            expected_return = adjusted_position * 0.08  # Assume 8% base return
            expected_sharpe = expected_return / vol if vol > 0 else 0.0

            # Determine signal type
            current_pos = self.current_positions.get(symbol, 0.0)
            position_change = abs(adjusted_position - current_pos)

            if current_pos == 0.0 and adjusted_position > 0.01:
                signal_type = "entry"
            elif adjusted_position < 0.01:
                signal_type = "exit"
            elif position_change > self.rebalance_threshold:
                signal_type = "rebalance"
            else:
                continue  # No significant change

            # Create portfolio signal
            signal = PortfolioSignal(
                symbol=symbol,
                timestamp=timestamp,
                position_size=adjusted_position,
                confidence=confidence,
                expected_sharpe=expected_sharpe,
                risk_contribution=risk_budget,
                signal_type=signal_type,
            )

            portfolio_signals[symbol] = signal
            total_position += adjusted_position

        # Scale down if total position exceeds limits
        if total_position > 1.0:
            scale_factor = 1.0 / total_position
            for signal in portfolio_signals.values():
                signal.position_size *= scale_factor

        return portfolio_signals

    def _calculate_correlation_penalty(
        self,
        target_symbol: str,
        correlation_matrix: pd.DataFrame,
        signals: Dict[str, Tuple[float, float]],
    ) -> float:
        """Calculate correlation penalty for position sizing."""
        if correlation_matrix.empty or target_symbol not in correlation_matrix.index:
            return 1.0

        penalties = []

        for other_symbol, (other_pos, _) in signals.items():
            if other_symbol == target_symbol or other_pos <= 0.01:
                continue

            if other_symbol in correlation_matrix.columns:
                correlation = correlation_matrix.loc[target_symbol, other_symbol]

                # Penalize high correlation
                if abs(correlation) > self.correlation_threshold:
                    penalty = 1.0 - (abs(correlation) - self.correlation_threshold) / (
                        1.0 - self.correlation_threshold
                    )
                    penalties.append(penalty)

        # Return minimum penalty (most restrictive)
        return min(penalties) if penalties else 1.0

    def generate_signals(self, timestamp: datetime) -> List[PortfolioSignal]:
        """
        Generate portfolio signals for the current timestamp.

        Args:
            timestamp: Current timestamp

        Returns:
            List of portfolio signals
        """
        # Generate raw signals
        raw_signals = self.generate_raw_signals(timestamp)

        if not raw_signals:
            logger.debug("No raw signals generated")
            return []

        # Apply portfolio constraints
        portfolio_signals = self.apply_portfolio_constraints(raw_signals, timestamp)

        if not portfolio_signals:
            logger.debug("No signals after portfolio constraints")
            return []

        # Update current positions
        for symbol, signal in portfolio_signals.items():
            if signal.signal_type in ["entry", "rebalance"]:
                self.current_positions[symbol] = signal.position_size
            elif signal.signal_type == "exit":
                self.current_positions[symbol] = 0.0

        self.last_rebalance = timestamp

        signals_list = list(portfolio_signals.values())

        logger.info(f"Generated {len(signals_list)} portfolio signals:")
        for signal in signals_list:
            logger.info(
                f"  {signal.symbol}: {signal.position_size:.2%} ({signal.signal_type})"
            )

        return signals_list

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status and metrics."""
        total_exposure = sum(self.current_positions.values())

        # Calculate portfolio risk
        portfolio_vol = 0.0
        if len(self.current_positions) > 1 and self.correlation_matrix is not None:
            weights = np.array(
                [
                    self.current_positions.get(symbol, 0.0)
                    for symbol in self.correlation_matrix.index
                ]
            )
            vols = np.array(
                [
                    self.volatility_estimates.get(symbol, 0.15)
                    for symbol in self.correlation_matrix.index
                ]
            )

            # Portfolio variance
            cov_matrix = self.correlation_matrix.values * np.outer(vols, vols)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_variance)

        return {
            "total_exposure": total_exposure,
            "portfolio_volatility": portfolio_vol,
            "target_volatility": self.target_volatility,
            "num_positions": len(
                [p for p in self.current_positions.values() if p > 0.01]
            ),
            "positions": self.current_positions.copy(),
            "risk_budgets": self.risk_budgets.copy(),
            "last_rebalance": self.last_rebalance,
            "volatility_estimates": self.volatility_estimates.copy(),
        }

    def should_rebalance(self, timestamp: datetime) -> bool:
        """Check if portfolio should be rebalanced."""
        if self.last_rebalance is None:
            return True

        # Time-based rebalancing (daily)
        time_diff = timestamp - self.last_rebalance
        if time_diff.total_seconds() >= 24 * 3600:  # 24 hours
            return True

        # Drift-based rebalancing
        # Only check drift if we have meaningful risk budgets and positions
        if not self.risk_budgets or not any(self.current_positions.values()):
            return False

        for symbol, current_pos in self.current_positions.items():
            if symbol in self.risk_budgets:
                target_pos = self.risk_budgets[symbol]
                drift = abs(current_pos - target_pos)
                # Only trigger if the drift is significant AND the position is meaningful
                if drift > self.rebalance_threshold and current_pos > 0.01:
                    return True

        return False


# Factory function for easy creation
def create_portfolio_generator(**kwargs) -> PortfolioSignalGenerator:
    """Create a PortfolioSignalGenerator with default or custom parameters."""
    return PortfolioSignalGenerator(**kwargs)
