"""
Advanced Risk Management System

Implements comprehensive risk management features:
- Value-at-Risk (VaR) calculation with multiple methods
- Stress testing with historical scenarios
- Dynamic risk limits based on market conditions
- Correlation and concentration monitoring
- Tail risk measures (CVaR, drawdown at risk)
- Independent risk oversight and kill-switch
- Real-time risk monitoring and alerting
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from decimal import Decimal

from .basic_risk_manager import BasicRiskManager
from .compliance_worm import EventType, write_event, get_audit_logger
from ..layer0_data_ingestion.schemas import FeatureSnapshot


class VaRMethod(Enum):
    """Value-at-Risk calculation methods."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


class RiskLevel(Enum):
    """Risk level classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class VaRResult:
    """VaR calculation result."""

    var_1d: float
    var_5d: float
    cvar_1d: float  # Conditional VaR
    cvar_5d: float
    confidence_level: float
    method: VaRMethod
    calculation_time: datetime


@dataclass
class StressTestResult:
    """Stress test result."""

    scenario_name: str
    portfolio_shock: float
    individual_shocks: Dict[str, float]
    estimated_pnl: float
    max_drawdown: float
    survival_probability: float
    recovery_time_days: Optional[int]


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""

    portfolio_value: float
    var_result: VaRResult
    stress_results: List[StressTestResult]
    correlation_matrix: np.ndarray
    concentration_risk: float
    leverage_ratio: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    risk_level: RiskLevel
    active_alerts: List[str]


class AdvancedRiskManager(BasicRiskManager):
    """
    Advanced risk management system with VaR, stress testing, and dynamic limits.

    Extends BasicRiskManager with sophisticated risk measures and monitoring.
    """

    def __init__(
        self,
        max_position_pct: float = 0.25,
        max_drawdown_pct: float = 0.15,  # Higher limit for aggressive strategy
        vol_multiplier_limit: float = 4.0,
        min_trade_size: float = 10.0,
        var_confidence_level: float = 0.95,
        var_horizon_days: int = 1,
        max_var_pct: float = 0.08,  # Max 8% VaR
        max_leverage: float = 5.0,
        correlation_threshold: float = 0.8,
        concentration_limit: float = 0.4,
    ):
        """
        Initialize advanced risk manager.

        Args:
            max_position_pct: Maximum position as % of portfolio
            max_drawdown_pct: Maximum drawdown threshold
            vol_multiplier_limit: Volatility circuit breaker multiplier
            min_trade_size: Minimum trade size in USD
            var_confidence_level: VaR confidence level (0.95 = 95%)
            var_horizon_days: VaR time horizon in days
            max_var_pct: Maximum VaR as % of portfolio
            max_leverage: Maximum leverage ratio
            correlation_threshold: High correlation alert threshold
            concentration_limit: Maximum concentration in any asset class
        """
        super().__init__(
            max_position_pct, max_drawdown_pct, vol_multiplier_limit, min_trade_size
        )

        # VaR parameters
        self.var_confidence_level = var_confidence_level
        self.var_horizon_days = var_horizon_days
        self.max_var_pct = max_var_pct

        # Advanced risk parameters
        self.max_leverage = max_leverage
        self.correlation_threshold = correlation_threshold
        self.concentration_limit = concentration_limit

        # Data storage
        self.return_history = deque(maxlen=252)  # 1 year of daily returns
        self.price_history = {}  # Per-symbol price history
        self.position_history = deque(maxlen=1000)  # Position history

        # Risk metrics tracking
        self.current_var = None
        self.stress_test_results = []
        self.correlation_matrix = None
        self.risk_alerts = []

        # Performance tracking
        self.pnl_history = deque(maxlen=252)
        self.drawdown_history = deque(maxlen=252)

        # Dynamic limits
        self.dynamic_limits = {
            "position_multiplier": 1.0,
            "var_multiplier": 1.0,
            "leverage_multiplier": 1.0,
        }

        # Independent oversight
        self.kill_switch_active = False
        self.last_risk_check = datetime.now()

        self.logger = logging.getLogger(f"advanced_risk_manager")
        self.logger.info(f"🛡️  Advanced Risk Manager initialized")

        # Initialize stress test scenarios
        self._initialize_stress_scenarios()

        # Exchange haircuts for different venues (as per Future_instruction.txt)
        self.exchange_haircuts = {
            "coinbase": 0.05,  # 5% haircut for Coinbase
            "binance": 0.08,  # 8% haircut for Binance
            "alpaca": 0.03,  # 3% haircut for Alpaca (equities)
            "default": 0.10,  # 10% default haircut for unknown venues
        }

    def _initialize_stress_scenarios(self):
        """Initialize historical stress test scenarios."""
        self.stress_scenarios = {
            "2020_covid_crash": {
                "name": "COVID-19 Crash (March 2020)",
                "equity_shock": -0.35,
                "crypto_shock": -0.50,
                "volatility_multiplier": 3.0,
                "correlation_increase": 0.3,
            },
            "2022_crypto_winter": {
                "name": "Crypto Winter (2022)",
                "equity_shock": -0.15,
                "crypto_shock": -0.70,
                "volatility_multiplier": 2.5,
                "correlation_increase": 0.2,
            },
            "flash_crash": {
                "name": "Flash Crash Scenario",
                "equity_shock": -0.10,
                "crypto_shock": -0.30,
                "volatility_multiplier": 5.0,
                "correlation_increase": 0.5,
                "duration_minutes": 30,
            },
            "liquidity_crisis": {
                "name": "Liquidity Crisis",
                "equity_shock": -0.25,
                "crypto_shock": -0.40,
                "volatility_multiplier": 4.0,
                "correlation_increase": 0.8,
                "bid_ask_spread_multiplier": 10.0,
            },
            "black_swan": {
                "name": "Black Swan Event",
                "equity_shock": -0.50,
                "crypto_shock": -0.80,
                "volatility_multiplier": 10.0,
                "correlation_increase": 0.9,
            },
        }

    def update_market_data(
        self,
        symbol: str,
        price: float,
        return_1d: float,
        volatility: float,
        feature_snapshot: FeatureSnapshot,
    ):
        """Update market data for risk calculations."""
        try:
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=252)

            self.price_history[symbol].append(
                {
                    "timestamp": datetime.now(),
                    "price": price,
                    "return": return_1d,
                    "volatility": volatility,
                }
            )

            # Update portfolio-level return history
            if return_1d is not None:
                self.return_history.append(return_1d)

            # Update volatility tracking
            if len(self.price_history[symbol]) > 1:
                recent_vol = np.std(
                    [p["return"] for p in list(self.price_history[symbol])[-20:]]
                )
                avg_vol = np.std([p["return"] for p in self.price_history[symbol]])
                self.update_volatility(symbol, recent_vol, avg_vol)

        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")

    def calculate_var(self, method: VaRMethod = VaRMethod.HISTORICAL) -> VaRResult:
        """
        Calculate Value-at-Risk using specified method.

        Args:
            method: VaR calculation method

        Returns:
            VaR calculation result
        """
        try:
            if len(self.return_history) < 30:
                self.logger.warning("Insufficient data for VaR calculation")
                return VaRResult(
                    var_1d=0.0,
                    var_5d=0.0,
                    cvar_1d=0.0,
                    cvar_5d=0.0,
                    confidence_level=self.var_confidence_level,
                    method=method,
                    calculation_time=datetime.now(),
                )

            returns = np.array(self.return_history)

            if method == VaRMethod.HISTORICAL:
                var_1d, cvar_1d = self._calculate_historical_var(returns, 1)
                var_5d, cvar_5d = self._calculate_historical_var(returns, 5)

            elif method == VaRMethod.PARAMETRIC:
                var_1d, cvar_1d = self._calculate_parametric_var(returns, 1)
                var_5d, cvar_5d = self._calculate_parametric_var(returns, 5)

            elif method == VaRMethod.MONTE_CARLO:
                var_1d, cvar_1d = self._calculate_monte_carlo_var(returns, 1)
                var_5d, cvar_5d = self._calculate_monte_carlo_var(returns, 5)

            result = VaRResult(
                var_1d=var_1d,
                var_5d=var_5d,
                cvar_1d=cvar_1d,
                cvar_5d=cvar_5d,
                confidence_level=self.var_confidence_level,
                method=method,
                calculation_time=datetime.now(),
            )

            self.current_var = result
            return result

        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return VaRResult(
                var_1d=0.0,
                var_5d=0.0,
                cvar_1d=0.0,
                cvar_5d=0.0,
                confidence_level=self.var_confidence_level,
                method=method,
                calculation_time=datetime.now(),
            )

    def _calculate_historical_var(
        self, returns: np.ndarray, horizon: int
    ) -> Tuple[float, float]:
        """Calculate historical VaR and CVaR."""
        # Scale returns to horizon
        scaled_returns = returns * np.sqrt(horizon)

        # Calculate VaR (percentile)
        var_percentile = (1 - self.var_confidence_level) * 100
        var = np.percentile(scaled_returns, var_percentile)

        # Calculate CVaR (expected shortfall)
        cvar = np.mean(scaled_returns[scaled_returns <= var])

        return abs(var), abs(cvar)

    def _calculate_parametric_var(
        self, returns: np.ndarray, horizon: int
    ) -> Tuple[float, float]:
        """Calculate parametric VaR assuming normal distribution."""
        # Scale returns to horizon
        scaled_returns = returns * np.sqrt(horizon)

        # Calculate parameters
        mean = np.mean(scaled_returns)
        std = np.std(scaled_returns)

        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(1 - self.var_confidence_level)
        var = mean + z_score * std

        # Calculate CVaR for normal distribution
        pdf_at_var = stats.norm.pdf(z_score)
        cvar = mean + std * pdf_at_var / (1 - self.var_confidence_level)

        return abs(var), abs(cvar)

    def _calculate_monte_carlo_var(
        self, returns: np.ndarray, horizon: int, n_simulations: int = 10000
    ) -> Tuple[float, float]:
        """Calculate Monte Carlo VaR."""
        # Estimate parameters
        mean = np.mean(returns)
        std = np.std(returns)

        # Generate random scenarios
        random_returns = np.random.normal(mean, std, n_simulations)
        scaled_returns = random_returns * np.sqrt(horizon)

        # Calculate VaR and CVaR
        var_percentile = (1 - self.var_confidence_level) * 100
        var = np.percentile(scaled_returns, var_percentile)
        cvar = np.mean(scaled_returns[scaled_returns <= var])

        return abs(var), abs(cvar)

    def run_stress_tests(
        self, current_positions: Dict[str, float], portfolio_value: float
    ) -> List[StressTestResult]:
        """
        Run comprehensive stress tests on current portfolio.

        Args:
            current_positions: Current positions by symbol
            portfolio_value: Current portfolio value

        Returns:
            List of stress test results
        """
        stress_results = []

        try:
            for scenario_name, scenario in self.stress_scenarios.items():
                result = self._run_single_stress_test(
                    scenario_name, scenario, current_positions, portfolio_value
                )
                stress_results.append(result)

            self.stress_test_results = stress_results
            return stress_results

        except Exception as e:
            self.logger.error(f"Error running stress tests: {e}")
            return []

    def _run_single_stress_test(
        self,
        scenario_name: str,
        scenario: Dict[str, Any],
        current_positions: Dict[str, float],
        portfolio_value: float,
    ) -> StressTestResult:
        """Run a single stress test scenario."""
        try:
            total_shock = 0.0
            individual_shocks = {}

            for symbol, position_value in current_positions.items():
                # Determine asset class
                if any(crypto in symbol.upper() for crypto in ["BTC", "ETH", "SOL"]):
                    shock = scenario["crypto_shock"]
                else:
                    shock = scenario["equity_shock"]

                individual_shocks[symbol] = shock
                total_shock += (position_value / portfolio_value) * shock

            # Calculate estimated PnL
            estimated_pnl = portfolio_value * total_shock

            # Estimate max drawdown (simplified)
            max_drawdown = abs(total_shock) * scenario.get("volatility_multiplier", 1.0)

            # Calculate survival probability (heuristic)
            survival_probability = max(0.0, 1.0 - (abs(total_shock) / 0.5))

            # Estimate recovery time
            recovery_time = None
            if abs(total_shock) > 0.1:
                recovery_time = int(abs(total_shock) * 365)  # Days

            return StressTestResult(
                scenario_name=scenario["name"],
                portfolio_shock=total_shock,
                individual_shocks=individual_shocks,
                estimated_pnl=estimated_pnl,
                max_drawdown=max_drawdown,
                survival_probability=survival_probability,
                recovery_time_days=recovery_time,
            )

        except Exception as e:
            self.logger.error(f"Error in stress test {scenario_name}: {e}")
            return StressTestResult(
                scenario_name=scenario_name,
                portfolio_shock=0.0,
                individual_shocks={},
                estimated_pnl=0.0,
                max_drawdown=0.0,
                survival_probability=1.0,
                recovery_time_days=None,
            )

    def calculate_correlation_matrix(self) -> Optional[np.ndarray]:
        """Calculate correlation matrix of assets."""
        try:
            if len(self.price_history) < 2:
                return None

            # Collect return data for all symbols
            return_data = {}
            min_length = float("inf")

            for symbol, history in self.price_history.items():
                if len(history) > 30:
                    returns = [
                        item["return"] for item in history if item["return"] is not None
                    ]
                    if len(returns) > 30:
                        return_data[symbol] = returns
                        min_length = min(min_length, len(returns))

            if len(return_data) < 2:
                return None

            # Align lengths and create matrix
            symbols = list(return_data.keys())
            return_matrix = np.zeros((min_length, len(symbols)))

            for i, symbol in enumerate(symbols):
                return_matrix[:, i] = return_data[symbol][-min_length:]

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(return_matrix.T)
            self.correlation_matrix = correlation_matrix

            return correlation_matrix

        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return None

    def check_correlation_risk(self, threshold: float = None) -> List[str]:
        """Check for high correlation risk."""
        if threshold is None:
            threshold = self.correlation_threshold

        alerts = []

        if self.correlation_matrix is None:
            return alerts

        try:
            symbols = list(self.price_history.keys())
            n_symbols = len(symbols)

            for i in range(n_symbols):
                for j in range(i + 1, n_symbols):
                    correlation = self.correlation_matrix[i, j]
                    if abs(correlation) > threshold:
                        alerts.append(
                            f"High correlation ({correlation:.3f}) between {symbols[i]} and {symbols[j]}"
                        )

            return alerts

        except Exception as e:
            self.logger.error(f"Error checking correlation risk: {e}")
            return []

    def calculate_concentration_risk(
        self, current_positions: Dict[str, float], portfolio_value: float
    ) -> float:
        """Calculate portfolio concentration risk."""
        try:
            if not current_positions or portfolio_value <= 0:
                return 0.0

            # Calculate Herfindahl-Hirschman Index
            position_weights = []
            for position_value in current_positions.values():
                weight = abs(position_value) / portfolio_value
                position_weights.append(weight)

            # HHI = sum of squared weights
            hhi = sum(w**2 for w in position_weights)

            # Normalize to 0-1 scale (1 = maximum concentration)
            n_positions = len(position_weights)
            if n_positions > 1:
                min_hhi = 1.0 / n_positions  # Perfectly diversified
                concentration_risk = (hhi - min_hhi) / (1.0 - min_hhi)
            else:
                concentration_risk = 1.0

            return max(0.0, min(1.0, concentration_risk))

        except Exception as e:
            self.logger.error(f"Error calculating concentration risk: {e}")
            return 0.0

    def update_dynamic_limits(
        self, market_regime: str = None, recent_performance: float = None
    ):
        """Update dynamic risk limits based on market conditions."""
        try:
            # Reset multipliers
            self.dynamic_limits = {
                "position_multiplier": 1.0,
                "var_multiplier": 1.0,
                "leverage_multiplier": 1.0,
            }

            # Adjust based on market regime
            if market_regime:
                if market_regime in ["high_volatility", "crisis"]:
                    # Tighten limits in high volatility
                    self.dynamic_limits["position_multiplier"] = 0.5
                    self.dynamic_limits["var_multiplier"] = 0.5
                    self.dynamic_limits["leverage_multiplier"] = 0.5
                elif market_regime in ["low_volatility", "trending"]:
                    # Relax limits in stable conditions
                    self.dynamic_limits["position_multiplier"] = 1.2
                    self.dynamic_limits["var_multiplier"] = 1.2
                    self.dynamic_limits["leverage_multiplier"] = 1.1

            # Adjust based on recent performance
            if recent_performance is not None:
                if recent_performance < -0.05:  # 5% loss
                    # Tighten after losses
                    multiplier = 0.7
                    for key in self.dynamic_limits:
                        self.dynamic_limits[key] *= multiplier
                elif recent_performance > 0.05:  # 5% gain
                    # Slightly relax after gains
                    multiplier = 1.1
                    for key in self.dynamic_limits:
                        self.dynamic_limits[key] *= multiplier

            self.logger.debug(f"Updated dynamic limits: {self.dynamic_limits}")

        except Exception as e:
            self.logger.error(f"Error updating dynamic limits: {e}")

    def check_advanced_risk(
        self,
        symbol: str,
        proposed_position: Decimal,
        current_price: Decimal,
        portfolio_value: Decimal,
        current_positions: Dict[str, float],
    ) -> Tuple[bool, str, Decimal]:
        """
        Comprehensive risk check including VaR, stress tests, and concentration.

        Args:
            symbol: Trading symbol
            proposed_position: Proposed position size
            current_price: Current market price
            portfolio_value: Current portfolio value
            current_positions: Current positions by symbol

        Returns:
            Tuple[bool, str, Decimal]: (allowed?, reason, max_allowed_size)
        """
        try:
            # First run basic risk checks
            basic_result = self.check_position_risk(
                symbol, proposed_position, current_price, portfolio_value
            )

            if not basic_result[0]:
                return basic_result

            # Kill switch check
            if self.kill_switch_active:
                return False, "Kill switch activated", Decimal("0")

            # Check VaR limits
            if self.current_var:
                portfolio_var = self.current_var.var_1d * float(portfolio_value)
                max_var = (
                    float(portfolio_value)
                    * self.max_var_pct
                    * self.dynamic_limits["var_multiplier"]
                )

                if portfolio_var > max_var:
                    return (
                        False,
                        f"Portfolio VaR {portfolio_var:.0f} exceeds limit {max_var:.0f}",
                        Decimal("0"),
                    )

            # Check leverage limits
            total_exposure = sum(abs(pos) for pos in current_positions.values())
            leverage_ratio = (
                total_exposure / float(portfolio_value) if portfolio_value > 0 else 0
            )
            max_leverage = (
                self.max_leverage * self.dynamic_limits["leverage_multiplier"]
            )

            if leverage_ratio > max_leverage:
                return (
                    False,
                    f"Leverage {leverage_ratio:.2f} exceeds limit {max_leverage:.2f}",
                    Decimal("0"),
                )

            # Check concentration risk
            concentration = self.calculate_concentration_risk(
                current_positions, float(portfolio_value)
            )
            if concentration > self.concentration_limit:
                return (
                    False,
                    f"Concentration risk {concentration:.2%} exceeds limit {self.concentration_limit:.2%}",
                    Decimal("0"),
                )

            # Check correlation risk
            correlation_alerts = self.check_correlation_risk()
            if len(correlation_alerts) > 3:  # Too many high correlations
                return (
                    False,
                    f"High correlation risk: {len(correlation_alerts)} alerts",
                    Decimal("0"),
                )

            # Adjust position size based on dynamic limits
            adjusted_position = proposed_position * Decimal(
                str(self.dynamic_limits["position_multiplier"])
            )

            return True, "Advanced risk checks passed", adjusted_position

        except Exception as e:
            self.logger.error(f"Error in advanced risk check: {e}")
            return False, f"Risk check error: {e}", Decimal("0")

    def independent_risk_oversight(
        self, current_positions: Dict[str, float], portfolio_value: float
    ) -> List[str]:
        """
        Independent risk oversight - parallel risk checking.

        Returns:
            List of risk alerts
        """
        alerts = []

        try:
            # Check position limits
            for symbol, position_value in current_positions.items():
                position_pct = abs(position_value) / portfolio_value
                if position_pct > self.max_position_pct:
                    alerts.append(
                        f"Position {symbol} exceeds limit: {position_pct:.2%}"
                    )

            # Check total leverage
            total_exposure = sum(abs(pos) for pos in current_positions.values())
            leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
            if leverage > self.max_leverage:
                alerts.append(f"Total leverage exceeds limit: {leverage:.2f}")

            # Check drawdown
            if self.peak_equity > 0:
                drawdown = (float(self.peak_equity) - portfolio_value) / float(
                    self.peak_equity
                )
                if drawdown > self.max_drawdown_pct:
                    alerts.append(f"Drawdown exceeds limit: {drawdown:.2%}")

            # Check VaR
            if self.current_var:
                portfolio_var = self.current_var.var_1d * portfolio_value
                var_pct = portfolio_var / portfolio_value
                if var_pct > self.max_var_pct:
                    alerts.append(f"VaR exceeds limit: {var_pct:.2%}")

            # Critical alert threshold
            if len(alerts) >= 3:
                self.kill_switch_active = True
                alerts.append(
                    "CRITICAL: Kill switch activated due to multiple risk breaches"
                )

            return alerts

        except Exception as e:
            self.logger.error(f"Error in independent risk oversight: {e}")
            return [f"Risk oversight error: {e}"]

    def get_comprehensive_risk_metrics(
        self, current_positions: Dict[str, float], portfolio_value: float
    ) -> RiskMetrics:
        """Get comprehensive risk metrics."""
        try:
            # Calculate all risk metrics
            var_result = self.calculate_var()
            stress_results = self.run_stress_tests(current_positions, portfolio_value)
            correlation_matrix = self.calculate_correlation_matrix()
            concentration_risk = self.calculate_concentration_risk(
                current_positions, portfolio_value
            )

            # Calculate leverage
            total_exposure = sum(abs(pos) for pos in current_positions.values())
            leverage_ratio = (
                total_exposure / portfolio_value if portfolio_value > 0 else 0
            )

            # Calculate performance metrics
            sharpe_ratio = self._calculate_sharpe_ratio()
            calmar_ratio = self._calculate_calmar_ratio()
            max_drawdown = self._calculate_max_drawdown()

            # Determine risk level
            risk_level = self._determine_risk_level(
                var_result, concentration_risk, leverage_ratio
            )

            # Get active alerts
            active_alerts = self.independent_risk_oversight(
                current_positions, portfolio_value
            )

            return RiskMetrics(
                portfolio_value=portfolio_value,
                var_result=var_result,
                stress_results=stress_results,
                correlation_matrix=correlation_matrix,
                concentration_risk=concentration_risk,
                leverage_ratio=leverage_ratio,
                sharpe_ratio=sharpe_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                risk_level=risk_level,
                active_alerts=active_alerts,
            )

        except Exception as e:
            self.logger.error(f"Error getting comprehensive risk metrics: {e}")
            return RiskMetrics(
                portfolio_value=portfolio_value,
                var_result=VaRResult(
                    0, 0, 0, 0, 0.95, VaRMethod.HISTORICAL, datetime.now()
                ),
                stress_results=[],
                correlation_matrix=None,
                concentration_risk=0.0,
                leverage_ratio=0.0,
                sharpe_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                risk_level=RiskLevel.LOW,
                active_alerts=[],
            )

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from PnL history."""
        try:
            if len(self.pnl_history) < 10:
                return 0.0

            returns = np.array(self.pnl_history)
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            if std_return == 0:
                return 0.0

            # Annualize (assuming daily returns)
            sharpe = (mean_return / std_return) * np.sqrt(252)
            return sharpe

        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (return/max drawdown)."""
        try:
            if len(self.pnl_history) < 10:
                return 0.0

            total_return = sum(self.pnl_history)
            max_dd = self._calculate_max_drawdown()

            if max_dd == 0:
                return 0.0

            # Annualize
            calmar = (total_return * 252) / max_dd
            return calmar

        except Exception as e:
            self.logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        try:
            if len(self.pnl_history) < 2:
                return 0.0

            cumulative = np.cumsum(self.pnl_history)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max

            return abs(np.min(drawdown))

        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    def _determine_risk_level(
        self, var_result: VaRResult, concentration_risk: float, leverage_ratio: float
    ) -> RiskLevel:
        """Determine overall risk level."""
        try:
            risk_score = 0

            # VaR component
            if var_result.var_1d > 0.1:  # 10% VaR
                risk_score += 3
            elif var_result.var_1d > 0.05:  # 5% VaR
                risk_score += 2
            elif var_result.var_1d > 0.02:  # 2% VaR
                risk_score += 1

            # Concentration component
            if concentration_risk > 0.7:
                risk_score += 3
            elif concentration_risk > 0.5:
                risk_score += 2
            elif concentration_risk > 0.3:
                risk_score += 1

            # Leverage component
            if leverage_ratio > 4.0:
                risk_score += 3
            elif leverage_ratio > 2.0:
                risk_score += 2
            elif leverage_ratio > 1.5:
                risk_score += 1

            # Classify risk level
            if risk_score >= 7:
                return RiskLevel.EXTREME
            elif risk_score >= 5:
                return RiskLevel.HIGH
            elif risk_score >= 3:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW

        except Exception as e:
            self.logger.error(f"Error determining risk level: {e}")
            return RiskLevel.MEDIUM

    def reset_kill_switch(self):
        """Reset kill switch - use with caution."""
        self.kill_switch_active = False
        self.logger.warning("Kill switch has been reset")

    def get_advanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive advanced risk statistics."""
        basic_stats = self.get_stats()

        advanced_stats = {
            **basic_stats,
            "VaR_1d": self.current_var.var_1d if self.current_var else None,
            "CVaR_1d": self.current_var.cvar_1d if self.current_var else None,
            "Stress_tests": len(self.stress_test_results),
            "Correlation_alerts": len(self.check_correlation_risk()),
            "Kill_switch_active": self.kill_switch_active,
            "Dynamic_limits": self.dynamic_limits,
            "Risk_history_length": len(self.return_history),
            "Max_leverage": self.max_leverage,
            "VaR_confidence": self.var_confidence_level,
            "Concentration_limit": self.concentration_limit,
        }

        return advanced_stats

    def calc_var(self, returns: np.ndarray, p: float = 0.99, horizon: int = 1) -> float:
        """
        Calculate Value-at-Risk (VaR) as specified in Future_instruction.txt.

        Args:
            returns: Array of portfolio returns
            p: Confidence level (0.99 = 99%)
            horizon: Time horizon in days

        Returns:
            VaR value as a positive number (loss)
        """
        try:
            if len(returns) < 30:
                return 0.0

            # Calculate VaR using historical method
            sorted_returns = np.sort(returns)
            var_index = int((1 - p) * len(sorted_returns))
            var_1d = -sorted_returns[var_index]  # Convert to positive loss

            # Scale to horizon
            var_horizon = var_1d * np.sqrt(horizon)

            return max(0.0, var_horizon)

        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return 0.0

    def calc_cvar(
        self, returns: np.ndarray, p: float = 0.99, horizon: int = 1
    ) -> float:
        """
        Calculate Conditional Value-at-Risk (CVaR) as specified in Future_instruction.txt.

        Args:
            returns: Array of portfolio returns
            p: Confidence level (0.99 = 99%)
            horizon: Time horizon in days

        Returns:
            CVaR value as a positive number (expected loss beyond VaR)
        """
        try:
            if len(returns) < 30:
                return 0.0

            # Calculate CVaR (expected shortfall beyond VaR)
            sorted_returns = np.sort(returns)
            var_index = int((1 - p) * len(sorted_returns))

            # CVaR is the average of returns worse than VaR
            cvar_1d = -np.mean(sorted_returns[:var_index]) if var_index > 0 else 0.0

            # Scale to horizon
            cvar_horizon = cvar_1d * np.sqrt(horizon)

            return max(0.0, cvar_horizon)

        except Exception as e:
            self.logger.error(f"Error calculating CVaR: {e}")
            return 0.0

    def calc_max_size_with_haircuts(
        self, capital: float, var: float, cvar: float, venue: str
    ) -> float:
        """
        Calculate maximum position size incorporating VaR, CVaR, and exchange haircuts.

        Args:
            capital: Available capital
            var: Value-at-Risk
            cvar: Conditional Value-at-Risk
            venue: Exchange venue for haircut calculation

        Returns:
            Maximum position size considering all risk factors
        """
        try:
            # Get haircut for venue
            haircut = self.exchange_haircuts.get(
                venue, self.exchange_haircuts["default"]
            )

            # Effective capital after haircut
            effective_capital = capital * (1 - haircut)

            # VaR-based limit (position size such that VaR doesn't exceed max_var_pct)
            # If VaR is 2% and max_var_pct is 8%, we can size 4x larger
            var_scaling = self.max_var_pct / max(var, 0.001)
            var_limit = effective_capital * min(var_scaling, self.max_position_pct)

            # CVaR-based limit (more conservative)
            cvar_scaling = (self.max_var_pct * 0.7) / max(cvar, 0.001)
            cvar_limit = effective_capital * min(cvar_scaling, self.max_position_pct)

            # Position limit (traditional limit with haircut applied)
            position_limit = effective_capital * self.max_position_pct

            # Return the most conservative limit
            max_size = min(var_limit, cvar_limit, position_limit)

            self.logger.debug(
                f"Position limits - VaR: ${var_limit:.0f}, CVaR: ${cvar_limit:.0f}, "
                f"Position: ${position_limit:.0f}, Venue: {venue} (haircut: {haircut:.1%})"
            )

            return max(0.0, max_size)

        except Exception as e:
            self.logger.error(f"Error calculating max size with haircuts: {e}")
            return 0.0

    async def log_risk_event(self, event_type: EventType, event_data: Dict[str, Any]):
        """Log risk management event to WORM audit trail."""
        try:
            metadata = {
                "risk_manager": "advanced",
                "timestamp": datetime.now().isoformat(),
                "module": "layer5_risk",
            }

            await write_event(event_type, event_data, metadata)

        except Exception as e:
            self.logger.error(f"Error logging risk event to WORM: {e}")

    async def check_position_async(
        self, symbol: str, size: float, price: float, venue: str = "default"
    ) -> Dict[str, Any]:
        """
        Async version of position check with WORM audit logging.

        Ensures each execution path calls compliance.write_event as specified
        in Future_instruction.txt.
        """
        try:
            # Log the risk check attempt
            await self.log_risk_event(
                EventType.POSITION_CHANGE,
                {
                    "action": "position_check_start",
                    "symbol": symbol,
                    "size": size,
                    "price": price,
                    "venue": venue,
                },
            )

            # Calculate current returns for VaR/CVaR
            returns = (
                np.array(list(self.return_history))
                if self.return_history
                else np.array([])
            )

            # Calculate VaR and CVaR
            var = self.calc_var(returns, p=0.99, horizon=1)
            cvar = self.calc_cvar(returns, p=0.99, horizon=1)

            # Calculate maximum size with haircuts
            portfolio_value = getattr(self, "current_positions", {}).get(
                "total_value", 100000.0
            )
            max_size = self.calc_max_size_with_haircuts(
                portfolio_value, var, cvar, venue
            )

            # Perform basic checks
            basic_result = self.check_position(symbol, size, price)

            # Enhanced risk assessment
            risk_score = 0
            risk_reasons = []

            # VaR-based check
            if var > 0 and abs(size * price) > max_size:
                risk_score += 3
                risk_reasons.append(
                    f"VaR limit exceeded (${abs(size * price):.0f} > ${max_size:.0f})"
                )

            # CVaR-based check
            if cvar > 0 and abs(size * price) > (max_size * 0.7):
                risk_score += 2
                risk_reasons.append(
                    f"CVaR limit warning (${abs(size * price):.0f} > ${max_size * 0.7:.0f})"
                )

            # Venue haircut warning
            haircut = self.exchange_haircuts.get(
                venue, self.exchange_haircuts["default"]
            )
            if haircut > 0.05:  # > 5% haircut
                risk_score += 1
                risk_reasons.append(f"High venue haircut ({haircut:.1%})")

            # Kill switch check
            if self.kill_switch_active:
                risk_score += 10
                risk_reasons.append("Kill switch is active")

            # Determine approval
            approved = (
                basic_result["approved"]
                and not self.kill_switch_active
                and abs(size * price) <= max_size
            )

            result = {
                "approved": approved,
                "risk_score": risk_score,
                "reasons": risk_reasons + basic_result.get("reasons", []),
                "max_size": max_size,
                "var_1d": var,
                "cvar_1d": cvar,
                "venue_haircut": haircut,
                "effective_capital": portfolio_value * (1 - haircut),
                "kill_switch_active": self.kill_switch_active,
            }

            # Log the risk check result
            await self.log_risk_event(
                EventType.POSITION_CHANGE,
                {
                    "action": "position_check_complete",
                    "symbol": symbol,
                    "size": size,
                    "price": price,
                    "venue": venue,
                    "approved": approved,
                    "risk_score": risk_score,
                    "max_size": max_size,
                    "var_1d": var,
                    "cvar_1d": cvar,
                },
            )

            # Log risk breach if position rejected
            if not approved:
                await self.log_risk_event(
                    EventType.RISK_BREACH,
                    {
                        "symbol": symbol,
                        "attempted_size": size,
                        "attempted_value": abs(size * price),
                        "max_allowed": max_size,
                        "risk_reasons": risk_reasons,
                        "venue": venue,
                        "haircut": haircut,
                    },
                )

            return result

        except Exception as e:
            self.logger.error(f"Error in async position check: {e}")

            # Log the error
            await self.log_risk_event(
                EventType.SYSTEM_START,
                {  # Using SYSTEM_START as generic error
                    "action": "position_check_error",
                    "error": str(e),
                    "symbol": symbol,
                    "size": size,
                    "price": price,
                },
            )

            return {
                "approved": False,
                "risk_score": 10,
                "reasons": [f"Risk check error: {e}"],
                "max_size": 0.0,
                "var_1d": 0.0,
                "cvar_1d": 0.0,
                "venue_haircut": 1.0,  # Assume worst case
                "effective_capital": 0.0,
                "kill_switch_active": True,  # Fail safe
            }
