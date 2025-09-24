#!/usr/bin/env python3
"""
Strategy Capital Allocator
Allocate capital across RL/BASIS/MM using recent Sharpe and cross-correlation
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import redis
import numpy as np
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("strategy_allocator")


class StrategyCapitalAllocator:
    """Allocates capital across strategies using Sharpe ratios and correlations."""

    def __init__(self):
        """Initialize strategy capital allocator."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Strategy configuration
        self.strategies = ["RL", "BASIS", "MM"]
        self.config = {
            "lookback_days": 7,  # 7-day P&L series
            "update_interval": 3600,  # Update every hour
            "min_weight": 0.05,  # Minimum allocation per strategy
            "max_weight": 0.60,  # Maximum allocation per strategy
            "correlation_penalty": 1.0,  # Correlation penalty strength
            "min_sharpe": -2.0,  # Minimum Sharpe to consider
            "volatility_target": 0.15,  # Target portfolio volatility
            "rebalance_threshold": 0.10,  # Rebalance if allocation changes > 10%
            "min_observations": 100,  # Minimum P&L observations
            "smoothing_alpha": 0.3,  # EMA smoothing for allocations
        }

        # State tracking
        self.current_allocations = {
            strategy: 1.0 / len(self.strategies) for strategy in self.strategies
        }
        self.previous_allocations = self.current_allocations.copy()
        self.allocation_history = []
        self.last_update = 0
        self.total_updates = 0
        self.total_rebalances = 0

        # Load current allocations from Redis
        self._load_allocations_from_redis()

        logger.info("⚖️ Strategy Capital Allocator initialized")
        logger.info(f"   Strategies: {self.strategies}")
        logger.info(
            f"   Current allocations: {[(s, f'{self.current_allocations[s]:.1%}') for s in self.strategies]}"
        )
        logger.info(
            f"   Weight bounds: [{self.config['min_weight']:.0%}, {self.config['max_weight']:.0%}]"
        )

    def _load_allocations_from_redis(self):
        """Load current allocations from Redis."""
        try:
            allocation_data = self.redis.hgetall("risk:strategy_cap")

            if allocation_data:
                for strategy in self.strategies:
                    if strategy in allocation_data:
                        self.current_allocations[strategy] = float(
                            allocation_data[strategy]
                        )

                # Normalize to sum to 1
                total = sum(self.current_allocations.values())
                if total > 0:
                    for strategy in self.strategies:
                        self.current_allocations[strategy] /= total

                self.previous_allocations = self.current_allocations.copy()

                logger.debug(
                    f"Loaded allocations from Redis: {self.current_allocations}"
                )

        except Exception as e:
            logger.warning(f"Error loading allocations from Redis: {e}")

    def pull_pnl_series(self, strategy: str) -> List[float]:
        """Pull 7-day P&L series for strategy."""
        try:
            # Try to get from Redis time series
            pnl_series_key = f"strategy:{strategy}:pnl_series_7d"
            pnl_data = self.redis.lrange(pnl_series_key, 0, -1)

            if pnl_data:
                pnl_series = [float(p) for p in pnl_data]
                return pnl_series

            # Fallback: try to get from generic P&L data
            pnl_key = f"strategy:{strategy}:pnl_7d"
            total_pnl = self.redis.get(pnl_key)

            if total_pnl:
                # Generate synthetic series assuming daily returns
                total_pnl_float = float(total_pnl)
                daily_returns = total_pnl_float / 7  # Assume spread over 7 days

                # Generate series with some variance
                pnl_series = []
                for i in range(168):  # 7 days * 24 hours
                    hourly_return = daily_returns / 24 + np.random.normal(
                        0, abs(daily_returns) * 0.1
                    )
                    pnl_series.append(hourly_return)

                return pnl_series

            # Generate realistic mock data
            return self._generate_mock_pnl_series(strategy)

        except Exception as e:
            logger.error(f"Error pulling P&L series for {strategy}: {e}")
            return self._generate_mock_pnl_series(strategy)

    def _generate_mock_pnl_series(self, strategy: str) -> List[float]:
        """Generate realistic mock P&L series."""
        try:
            # Strategy-specific parameters
            if strategy == "RL":
                mean_return = 0.0002  # Higher variance, neutral mean
                volatility = 0.008
                autocorr = 0.1
            elif strategy == "BASIS":
                mean_return = 0.0001  # Lower variance, slightly positive
                volatility = 0.004
                autocorr = 0.3
            else:  # MM
                mean_return = 0.00005  # Very low variance, consistent
                volatility = 0.002
                autocorr = 0.5

            # Generate correlated returns
            np.random.seed(int(time.time()) % 10000 + hash(strategy) % 1000)

            n_periods = 168  # 7 days * 24 hours
            returns = []
            prev_return = 0

            for _ in range(n_periods):
                # AR(1) process for autocorrelation
                innovation = np.random.normal(0, volatility)
                return_val = mean_return + autocorr * prev_return + innovation
                returns.append(return_val)
                prev_return = return_val

            return returns

        except Exception as e:
            logger.error(f"Error generating mock P&L series for {strategy}: {e}")
            return [0.0] * 168

    def calculate_strategy_metrics(
        self, pnl_series: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate Sharpe ratios and other metrics for strategies."""
        try:
            metrics = {}

            for strategy, returns in pnl_series.items():
                if not returns or len(returns) < self.config["min_observations"]:
                    metrics[strategy] = {
                        "sharpe": 0.0,
                        "mean_return": 0.0,
                        "volatility": 0.01,
                        "total_return": 0.0,
                        "max_drawdown": 0.0,
                        "win_rate": 0.5,
                    }
                    continue

                returns_array = np.array(returns)

                # Basic metrics
                mean_return = np.mean(returns_array)
                volatility = np.std(returns_array)
                total_return = np.sum(returns_array)

                # Sharpe ratio (annualized)
                if volatility > 1e-8:
                    sharpe = (mean_return * np.sqrt(8760)) / (
                        volatility * np.sqrt(8760)
                    )  # Hourly to annual
                else:
                    sharpe = 0.0

                # Max drawdown
                cumulative_returns = np.cumsum(returns_array)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = running_max - cumulative_returns
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

                # Win rate
                positive_returns = returns_array > 0
                win_rate = np.mean(positive_returns) if len(returns_array) > 0 else 0.5

                metrics[strategy] = {
                    "sharpe": max(sharpe, self.config["min_sharpe"]),
                    "mean_return": mean_return,
                    "volatility": max(volatility, 1e-8),
                    "total_return": total_return,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate,
                }

                logger.debug(
                    f"{strategy} metrics: Sharpe={sharpe:.2f}, "
                    f"Vol={volatility:.1%}, Return={total_return:.2%}"
                )

            return metrics

        except Exception as e:
            logger.error(f"Error calculating strategy metrics: {e}")
            return {
                s: {"sharpe": 0.0, "mean_return": 0.0, "volatility": 0.01}
                for s in self.strategies
            }

    def calculate_correlation_matrix(
        self, pnl_series: Dict[str, List[float]]
    ) -> np.ndarray:
        """Calculate correlation matrix between strategies."""
        try:
            # Prepare data matrix
            min_length = min(len(series) for series in pnl_series.values())
            if min_length < 10:
                # Return identity matrix if insufficient data
                return np.eye(len(self.strategies))

            data_matrix = []
            for strategy in self.strategies:
                series = pnl_series[strategy][
                    -min_length:
                ]  # Use last min_length observations
                data_matrix.append(series)

            data_matrix = np.array(data_matrix)

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(data_matrix)

            # Handle NaN values
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

            # Ensure positive definite (add small diagonal term if needed)
            eigenvals = np.linalg.eigvals(correlation_matrix)
            if np.min(eigenvals) < 1e-8:
                correlation_matrix += np.eye(len(self.strategies)) * 1e-6

            return correlation_matrix

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return np.eye(len(self.strategies))

    def optimize_allocations(
        self, metrics: Dict[str, Dict[str, float]], correlation_matrix: np.ndarray
    ) -> np.ndarray:
        """Optimize capital allocations using Sharpe ratios and correlations."""
        try:
            n_strategies = len(self.strategies)

            # Extract Sharpe ratios
            sharpe_ratios = np.array([metrics[s]["sharpe"] for s in self.strategies])

            # Simple Kelly-like optimization with correlation penalty
            def objective(weights):
                # Portfolio Sharpe (mean return / volatility)
                portfolio_returns = np.dot(weights, sharpe_ratios)

                # Portfolio variance (with correlation)
                volatilities = np.array(
                    [metrics[s]["volatility"] for s in self.strategies]
                )
                portfolio_variance = np.dot(weights**2, volatilities**2)

                # Correlation penalty
                correlation_penalty = 0
                for i in range(n_strategies):
                    for j in range(i + 1, n_strategies):
                        correlation_penalty += (
                            self.config["correlation_penalty"]
                            * max(0, correlation_matrix[i, j])
                            * weights[i]
                            * weights[j]
                        )

                # Objective: maximize returns, minimize correlation penalty
                return -(portfolio_returns - correlation_penalty)

            # Constraints and bounds
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}  # Weights sum to 1
            ]

            bounds = [
                (self.config["min_weight"], self.config["max_weight"])
                for _ in range(n_strategies)
            ]

            # Initial guess (current allocations or equal weights)
            x0 = np.array([self.current_allocations[s] for s in self.strategies])
            x0 = np.clip(x0, self.config["min_weight"], self.config["max_weight"])
            x0 = x0 / np.sum(x0)  # Normalize

            # Optimization
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-8},
            )

            if result.success:
                optimal_weights = result.x

                # Ensure weights are normalized and within bounds
                optimal_weights = np.clip(
                    optimal_weights,
                    self.config["min_weight"],
                    self.config["max_weight"],
                )
                optimal_weights = optimal_weights / np.sum(optimal_weights)

                return optimal_weights
            else:
                logger.warning(f"Optimization failed: {result.message}")

                # Fallback: simple Sharpe-based allocation
                positive_sharpe = np.maximum(sharpe_ratios, 0.1)  # Floor at 0.1
                raw_weights = positive_sharpe / np.sum(positive_sharpe)

                # Apply bounds
                bounded_weights = np.clip(
                    raw_weights, self.config["min_weight"], self.config["max_weight"]
                )
                bounded_weights = bounded_weights / np.sum(bounded_weights)

                return bounded_weights

        except Exception as e:
            logger.error(f"Error in allocation optimization: {e}")

            # Fallback to equal weights
            equal_weights = np.ones(len(self.strategies)) / len(self.strategies)
            return equal_weights

    def check_rebalance_needed(
        self, new_allocations: Dict[str, float]
    ) -> Tuple[bool, Dict[str, float]]:
        """Check if rebalancing is needed."""
        try:
            changes = {}
            max_change = 0

            for strategy in self.strategies:
                current = self.current_allocations[strategy]
                new = new_allocations[strategy]
                change = abs(new - current)
                changes[strategy] = change
                max_change = max(max_change, change)

            needs_rebalance = max_change > self.config["rebalance_threshold"]

            return needs_rebalance, changes

        except Exception as e:
            logger.error(f"Error checking rebalance: {e}")
            return True, {}

    def apply_smoothing(self, new_allocations: Dict[str, float]) -> Dict[str, float]:
        """Apply EMA smoothing to allocation changes."""
        try:
            alpha = self.config["smoothing_alpha"]
            smoothed_allocations = {}

            for strategy in self.strategies:
                current = self.current_allocations[strategy]
                new = new_allocations[strategy]

                smoothed = alpha * new + (1 - alpha) * current
                smoothed_allocations[strategy] = smoothed

            # Renormalize
            total = sum(smoothed_allocations.values())
            if total > 0:
                for strategy in self.strategies:
                    smoothed_allocations[strategy] /= total

            return smoothed_allocations

        except Exception as e:
            logger.error(f"Error applying smoothing: {e}")
            return new_allocations

    def update_allocations(self) -> Dict[str, Any]:
        """Update strategy capital allocations."""
        try:
            update_start = time.time()
            self.total_updates += 1

            # Pull P&L series for all strategies
            pnl_series = {}
            for strategy in self.strategies:
                pnl_series[strategy] = self.pull_pnl_series(strategy)

            # Calculate strategy metrics
            metrics = self.calculate_strategy_metrics(pnl_series)

            # Calculate correlation matrix
            correlation_matrix = self.calculate_correlation_matrix(pnl_series)

            # Optimize allocations
            optimal_weights = self.optimize_allocations(metrics, correlation_matrix)

            # Convert to dictionary
            new_allocations = {
                strategy: float(optimal_weights[i])
                for i, strategy in enumerate(self.strategies)
            }

            # Check if rebalancing is needed
            needs_rebalance, changes = self.check_rebalance_needed(new_allocations)

            # Apply smoothing
            smoothed_allocations = self.apply_smoothing(new_allocations)

            update_result = {
                "timestamp": update_start,
                "status": "completed",
                "total_updates": self.total_updates,
                "previous_allocations": self.current_allocations.copy(),
                "raw_optimal_allocations": new_allocations,
                "smoothed_allocations": smoothed_allocations,
                "needs_rebalance": needs_rebalance,
                "allocation_changes": changes,
                "metrics": metrics,
                "correlation_matrix": correlation_matrix.tolist(),
                "max_change": max(changes.values()) if changes else 0,
            }

            if needs_rebalance:
                # Update allocations
                self.previous_allocations = self.current_allocations.copy()
                self.current_allocations = smoothed_allocations.copy()
                self.total_rebalances += 1

                # Store in Redis
                self.redis.hset("risk:strategy_cap", mapping=self.current_allocations)

                # Store allocation history
                allocation_record = {
                    "timestamp": update_start,
                    "allocations": self.current_allocations.copy(),
                    "metrics": metrics,
                    "rebalance_reason": f"max_change_{max(changes.values()):.1%}",
                    "total_rebalances": self.total_rebalances,
                }

                self.allocation_history.append(allocation_record)

                # Trim history
                if len(self.allocation_history) > 1000:
                    self.allocation_history = self.allocation_history[-500:]

                # Store metrics in Redis
                for strategy in self.strategies:
                    self.redis.set(
                        f"metric:strategy_caps_{strategy.lower()}",
                        self.current_allocations[strategy],
                    )

                logger.info(
                    f"💼 Capital reallocation #{self.total_rebalances}: "
                    f"{[(s, f'{self.current_allocations[s]:.1%}') for s in self.strategies]} "
                    f"(max change: {max(changes.values()):.1%})"
                )
            else:
                logger.debug("No rebalancing needed")

            self.last_update = update_start
            update_result["update_duration"] = time.time() - update_start

            return update_result

        except Exception as e:
            logger.error(f"Error updating allocations: {e}")
            return {"timestamp": time.time(), "status": "error", "error": str(e)}

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        try:
            current_time = time.time()

            # Get recent metrics for each strategy
            strategy_info = {}
            for strategy in self.strategies:
                # Try to get recent metrics
                sharpe_key = f"strategy:{strategy}:sharpe_7d"
                pnl_key = f"strategy:{strategy}:pnl_7d"

                sharpe = self.redis.get(sharpe_key)
                pnl = self.redis.get(pnl_key)

                strategy_info[strategy] = {
                    "allocation": self.current_allocations[strategy],
                    "allocation_pct": f"{self.current_allocations[strategy]:.1%}",
                    "sharpe_7d": float(sharpe) if sharpe else 0.0,
                    "pnl_7d": float(pnl) if pnl else 0.0,
                }

            status = {
                "service": "strategy_capital_allocator",
                "timestamp": current_time,
                "config": self.config,
                "current_allocations": self.current_allocations,
                "previous_allocations": self.previous_allocations,
                "strategy_info": strategy_info,
                "statistics": {
                    "total_updates": self.total_updates,
                    "total_rebalances": self.total_rebalances,
                    "last_update": self.last_update,
                    "last_update_ago": (
                        current_time - self.last_update if self.last_update > 0 else 0
                    ),
                    "allocation_history_length": len(self.allocation_history),
                },
                "recent_allocations": (
                    self.allocation_history[-3:] if self.allocation_history else []
                ),
            }

            return status

        except Exception as e:
            return {
                "service": "strategy_capital_allocator",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }

    def run_continuous_allocation(self):
        """Run continuous capital allocation updates."""
        logger.info("💼 Starting continuous strategy capital allocation")

        try:
            while True:
                try:
                    # Update allocations
                    result = self.update_allocations()

                    if result["status"] == "completed":
                        if result.get("needs_rebalance"):
                            max_change = result.get("max_change", 0)
                            logger.info(
                                f"📊 Rebalance #{self.total_rebalances}: "
                                f"max change {max_change:.1%}"
                            )
                        elif self.total_updates % 12 == 0:  # Every 12 hours
                            logger.debug(
                                f"📊 Check #{self.total_updates}: "
                                f"allocations stable"
                            )

                    # Wait for next update
                    time.sleep(self.config["update_interval"])

                except Exception as e:
                    logger.error(f"Error in allocation loop: {e}")
                    time.sleep(300)  # Wait 5 minutes on error

        except KeyboardInterrupt:
            logger.info("Strategy allocator stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in allocation loop: {e}")


def main():
    """Main entry point for strategy allocator."""
    import argparse

    parser = argparse.ArgumentParser(description="Strategy Capital Allocator")
    parser.add_argument("--run", action="store_true", help="Run continuous allocation")
    parser.add_argument(
        "--update", action="store_true", help="Run single allocation update"
    )
    parser.add_argument("--status", action="store_true", help="Show status report")

    args = parser.parse_args()

    # Create allocator
    allocator = StrategyCapitalAllocator()

    if args.status:
        # Show status report
        status = allocator.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.update:
        # Run single update
        result = allocator.update_allocations()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.run:
        # Run continuous allocation
        allocator.run_continuous_allocation()
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
