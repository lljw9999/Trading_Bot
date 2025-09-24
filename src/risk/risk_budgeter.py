#!/usr/bin/env python3
"""
Cross-Asset Risk Budgeter
Allocates gross exposure by risk parity using EWMA covariance
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
import pandas as pd
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("risk_budgeter")


class CrossAssetRiskBudgeter:
    """Cross-asset risk budgeter using EWMA covariance."""

    def __init__(self):
        """Initialize risk budgeter."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Risk budgeting configuration
        self.config = {
            "symbols": ["BTC", "ETH", "SOL"],
            "lookback_periods": 480,  # 8 hours of 1-minute data
            "ewma_lambda": 0.94,  # EWMA decay factor
            "update_interval": 300,  # Update every 5 minutes
            "min_weight": 0.05,  # Minimum allocation per asset
            "max_weight": 0.60,  # Maximum allocation per asset
            "target_vol": 0.15,  # Target portfolio volatility (15%)
            "rebalance_threshold": 0.10,  # Rebalance if weight drift > 10%
        }

        # Risk budgeting methods
        self.budgeting_methods = {
            "equal_risk": self._equal_risk_contribution,
            "inverse_vol": self._inverse_volatility,
            "min_var": self._minimum_variance,
            "risk_parity": self._hierarchical_risk_parity,
        }

        self.current_method = "equal_risk"

        # State tracking
        self.covariance_matrix = None
        self.returns_history = pd.DataFrame()
        self.current_weights = {}
        self.weight_history = []
        self.last_update = 0
        self.total_updates = 0

        # Initialize with equal weights
        n_assets = len(self.config["symbols"])
        equal_weight = 1.0 / n_assets
        for symbol in self.config["symbols"]:
            self.current_weights[symbol] = equal_weight

        logger.info("⚖️ Cross-Asset Risk Budgeter initialized")
        logger.info(f"   Assets: {self.config['symbols']}")
        logger.info(f"   Method: {self.current_method}")
        logger.info(f"   EWMA λ: {self.config['ewma_lambda']}")
        logger.info(f"   Update interval: {self.config['update_interval']}s")

    def fetch_returns_data(self) -> pd.DataFrame:
        """Fetch returns data for all symbols."""
        try:
            returns_data = {}
            current_time = time.time()

            for symbol in self.config["symbols"]:
                # Try to get returns from Redis time series
                returns_key = f"returns:{symbol.lower()}:1m"

                # Get recent returns data
                try:
                    # Try to get from Redis list
                    returns_raw = self.redis.lrange(
                        returns_key, -self.config["lookback_periods"], -1
                    )
                    if returns_raw:
                        returns_list = [float(r) for r in returns_raw]
                        returns_data[symbol] = returns_list
                    else:
                        # Generate synthetic returns for demo
                        returns_data[symbol] = self._generate_synthetic_returns(symbol)

                except Exception as e:
                    logger.debug(f"Error fetching returns for {symbol}: {e}")
                    returns_data[symbol] = self._generate_synthetic_returns(symbol)

            # Create DataFrame
            max_length = max(len(returns) for returns in returns_data.values())

            # Pad shorter series with zeros at the beginning
            for symbol in returns_data:
                current_length = len(returns_data[symbol])
                if current_length < max_length:
                    padding = [0.0] * (max_length - current_length)
                    returns_data[symbol] = padding + returns_data[symbol]

            returns_df = pd.DataFrame(returns_data)

            # Add timestamps
            timestamps = pd.date_range(
                end=datetime.now(), periods=len(returns_df), freq="1min"
            )
            returns_df.index = timestamps

            logger.debug(f"Fetched returns data: {returns_df.shape}")
            return returns_df

        except Exception as e:
            logger.error(f"Error fetching returns data: {e}")
            return pd.DataFrame()

    def _generate_synthetic_returns(self, symbol: str) -> List[float]:
        """Generate realistic synthetic returns for demo."""
        try:
            np.random.seed(hash(symbol) % 2**32)  # Deterministic based on symbol

            # Asset-specific parameters
            params = {
                "BTC": {"vol": 0.60, "drift": 0.0001},  # 60% annual vol
                "ETH": {"vol": 0.75, "drift": 0.00005},  # 75% annual vol
                "SOL": {"vol": 1.20, "drift": -0.00002},  # 120% annual vol
            }

            asset_params = params.get(symbol, {"vol": 0.50, "drift": 0.0})

            # Scale to 1-minute returns
            vol_1m = asset_params["vol"] / np.sqrt(365 * 24 * 60)  # Annual to 1-minute
            drift_1m = asset_params["drift"] / (365 * 24 * 60)

            # Generate correlated returns using time-based seed
            time_seed = int(time.time()) % 1000
            np.random.seed(time_seed)

            returns = np.random.normal(
                drift_1m, vol_1m, self.config["lookback_periods"]
            )

            return returns.tolist()

        except Exception as e:
            logger.error(f"Error generating synthetic returns for {symbol}: {e}")
            return [0.0] * self.config["lookback_periods"]

    def calculate_ewma_covariance(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Calculate EWMA covariance matrix."""
        try:
            if returns_df.empty:
                # Return identity matrix scaled by default variance
                n_assets = len(self.config["symbols"])
                return np.eye(n_assets) * 0.0001  # 1% daily vol

            returns_array = returns_df.values
            n_obs, n_assets = returns_array.shape

            if n_obs < 2:
                return np.eye(n_assets) * 0.0001

            lambda_param = self.config["ewma_lambda"]

            # Initialize covariance matrix with sample covariance
            cov_matrix = np.cov(returns_array.T)

            # Apply EWMA weighting
            weights = np.array(
                [(1 - lambda_param) * (lambda_param**i) for i in range(n_obs)]
            )
            weights = weights / weights.sum()  # Normalize

            # Weighted covariance calculation
            mean_returns = np.average(returns_array, axis=0, weights=weights)

            weighted_cov = np.zeros((n_assets, n_assets))
            for i in range(n_obs):
                centered_returns = returns_array[i] - mean_returns
                weighted_cov += weights[i] * np.outer(
                    centered_returns, centered_returns
                )

            # Ensure positive definite
            eigenvals, eigenvecs = np.linalg.eigh(weighted_cov)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Floor eigenvalues
            weighted_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

            logger.debug(f"EWMA covariance calculated: {weighted_cov.shape}")
            return weighted_cov

        except Exception as e:
            logger.error(f"Error calculating EWMA covariance: {e}")
            n_assets = len(self.config["symbols"])
            return np.eye(n_assets) * 0.0001

    def _equal_risk_contribution(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Equal risk contribution portfolio."""
        try:
            n_assets = len(self.config["symbols"])

            def risk_budget_objective(weights, cov_matrix):
                """Objective function for equal risk contribution."""
                portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                marginal_risk = cov_matrix @ weights / portfolio_vol
                risk_contrib = weights * marginal_risk / portfolio_vol
                target_risk = 1.0 / n_assets
                return np.sum((risk_contrib - target_risk) ** 2)

            # Constraints and bounds
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
            bounds = [
                (self.config["min_weight"], self.config["max_weight"])
                for _ in range(n_assets)
            ]

            # Initial guess (equal weights)
            x0 = np.ones(n_assets) / n_assets

            # Optimization
            result = minimize(
                risk_budget_objective,
                x0,
                args=(cov_matrix,),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if result.success:
                return result.x
            else:
                logger.warning(
                    "Equal risk contribution optimization failed, using equal weights"
                )
                return np.ones(n_assets) / n_assets

        except Exception as e:
            logger.error(f"Error in equal risk contribution: {e}")
            n_assets = len(self.config["symbols"])
            return np.ones(n_assets) / n_assets

    def _inverse_volatility(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Inverse volatility weighting."""
        try:
            vols = np.sqrt(np.diag(cov_matrix))
            inv_vols = 1.0 / vols
            weights = inv_vols / inv_vols.sum()

            # Apply bounds
            weights = np.maximum(weights, self.config["min_weight"])
            weights = np.minimum(weights, self.config["max_weight"])
            weights = weights / weights.sum()  # Renormalize

            return weights

        except Exception as e:
            logger.error(f"Error in inverse volatility: {e}")
            n_assets = len(self.config["symbols"])
            return np.ones(n_assets) / n_assets

    def _minimum_variance(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Minimum variance portfolio."""
        try:
            n_assets = len(self.config["symbols"])

            def portfolio_variance(weights, cov_matrix):
                return weights.T @ cov_matrix @ weights

            # Constraints and bounds
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
            bounds = [
                (self.config["min_weight"], self.config["max_weight"])
                for _ in range(n_assets)
            ]

            # Initial guess
            x0 = np.ones(n_assets) / n_assets

            # Optimization
            result = minimize(
                portfolio_variance,
                x0,
                args=(cov_matrix,),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                return result.x
            else:
                logger.warning(
                    "Minimum variance optimization failed, using equal weights"
                )
                return np.ones(n_assets) / n_assets

        except Exception as e:
            logger.error(f"Error in minimum variance: {e}")
            n_assets = len(self.config["symbols"])
            return np.ones(n_assets) / n_assets

    def _hierarchical_risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Simplified hierarchical risk parity."""
        try:
            # For simplicity, use equal risk contribution as HRP proxy
            return self._equal_risk_contribution(cov_matrix)

        except Exception as e:
            logger.error(f"Error in hierarchical risk parity: {e}")
            n_assets = len(self.config["symbols"])
            return np.ones(n_assets) / n_assets

    def calculate_risk_budget(self, method: str = None) -> Dict[str, float]:
        """Calculate risk budget weights."""
        try:
            if method is None:
                method = self.current_method

            if method not in self.budgeting_methods:
                logger.warning(f"Unknown method {method}, using equal_risk")
                method = "equal_risk"

            # Get returns data and calculate covariance
            returns_df = self.fetch_returns_data()
            cov_matrix = self.calculate_ewma_covariance(returns_df)

            # Store for later use
            self.covariance_matrix = cov_matrix
            self.returns_history = returns_df

            # Calculate weights using selected method
            budgeting_func = self.budgeting_methods[method]
            weights_array = budgeting_func(cov_matrix)

            # Convert to dictionary
            risk_budget = {}
            for i, symbol in enumerate(self.config["symbols"]):
                risk_budget[symbol] = float(weights_array[i])

            # Validate and normalize weights
            total_weight = sum(risk_budget.values())
            if abs(total_weight - 1.0) > 1e-6:
                # Renormalize
                for symbol in risk_budget:
                    risk_budget[symbol] /= total_weight

            logger.info(f"📊 Risk budget calculated using {method}:")
            for symbol, weight in risk_budget.items():
                logger.info(f"   {symbol}: {weight:.1%}")

            return risk_budget

        except Exception as e:
            logger.error(f"Error calculating risk budget: {e}")
            # Fallback to equal weights
            n_assets = len(self.config["symbols"])
            equal_weight = 1.0 / n_assets
            return {symbol: equal_weight for symbol in self.config["symbols"]}

    def check_rebalance_needed(
        self, new_weights: Dict[str, float]
    ) -> Tuple[bool, Dict[str, float]]:
        """Check if rebalancing is needed."""
        try:
            weight_drifts = {}
            max_drift = 0

            for symbol in self.config["symbols"]:
                current_weight = self.current_weights.get(symbol, 0)
                new_weight = new_weights.get(symbol, 0)
                drift = abs(new_weight - current_weight)
                weight_drifts[symbol] = drift
                max_drift = max(max_drift, drift)

            rebalance_needed = max_drift > self.config["rebalance_threshold"]

            logger.debug(
                f"Max weight drift: {max_drift:.1%}, threshold: {self.config['rebalance_threshold']:.1%}"
            )

            return rebalance_needed, weight_drifts

        except Exception as e:
            logger.error(f"Error checking rebalance: {e}")
            return True, {}  # Default to rebalance on error

    def update_risk_budget(self) -> Dict[str, Any]:
        """Update risk budget and store in Redis."""
        try:
            update_start = time.time()
            self.total_updates += 1

            # Calculate new risk budget
            new_weights = self.calculate_risk_budget()

            # Check if rebalancing is needed
            rebalance_needed, weight_drifts = self.check_rebalance_needed(new_weights)

            update_result = {
                "timestamp": update_start,
                "method": self.current_method,
                "weights": new_weights,
                "previous_weights": self.current_weights.copy(),
                "weight_drifts": weight_drifts,
                "rebalance_needed": rebalance_needed,
                "total_updates": self.total_updates,
                "update_duration": 0,
            }

            if rebalance_needed:
                # Update weights
                self.current_weights = new_weights.copy()

                # Store in Redis
                self.redis.hset("risk:budget", mapping=new_weights)

                # Store budget history
                budget_record = {
                    "timestamp": update_start,
                    "method": self.current_method,
                    "weights": new_weights,
                    "total_updates": self.total_updates,
                }

                self.redis.lpush(
                    "risk:budget:history", json.dumps(budget_record, default=str)
                )
                self.redis.ltrim("risk:budget:history", 0, 99)  # Keep last 100

                # Track weight history
                self.weight_history.append(
                    {"timestamp": update_start, "weights": new_weights.copy()}
                )

                if len(self.weight_history) > 100:
                    self.weight_history = self.weight_history[-50:]

                logger.info(
                    f"⚖️ Risk budget updated: max drift {max(weight_drifts.values()):.1%}"
                )

                # Store portfolio metrics
                if self.covariance_matrix is not None:
                    portfolio_vol = self._calculate_portfolio_volatility(new_weights)
                    self.redis.set("risk:portfolio_vol", portfolio_vol)

                    # Calculate risk contributions
                    risk_contribs = self._calculate_risk_contributions(new_weights)
                    for symbol, contrib in risk_contribs.items():
                        self.redis.set(f"risk:contrib:{symbol.lower()}", contrib)

            else:
                logger.debug("No rebalancing needed")

            self.last_update = update_start
            update_result["update_duration"] = time.time() - update_start

            return update_result

        except Exception as e:
            logger.error(f"Error updating risk budget: {e}")
            return {"timestamp": time.time(), "status": "error", "error": str(e)}

    def _calculate_portfolio_volatility(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio volatility."""
        try:
            if self.covariance_matrix is None:
                return 0.0

            weights_array = np.array(
                [weights[symbol] for symbol in self.config["symbols"]]
            )
            portfolio_var = weights_array.T @ self.covariance_matrix @ weights_array
            portfolio_vol = np.sqrt(portfolio_var)

            # Annualize (from 1-minute to annual)
            annual_vol = portfolio_vol * np.sqrt(365 * 24 * 60)

            return float(annual_vol)

        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0

    def _calculate_risk_contributions(
        self, weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate risk contributions by asset."""
        try:
            if self.covariance_matrix is None:
                return {symbol: 0.0 for symbol in self.config["symbols"]}

            weights_array = np.array(
                [weights[symbol] for symbol in self.config["symbols"]]
            )
            portfolio_vol = np.sqrt(
                weights_array.T @ self.covariance_matrix @ weights_array
            )

            if portfolio_vol == 0:
                return {symbol: 0.0 for symbol in self.config["symbols"]}

            # Marginal risk contributions
            marginal_risk = self.covariance_matrix @ weights_array / portfolio_vol
            risk_contributions = weights_array * marginal_risk / portfolio_vol

            return {
                symbol: float(risk_contributions[i])
                for i, symbol in enumerate(self.config["symbols"])
            }

        except Exception as e:
            logger.error(f"Error calculating risk contributions: {e}")
            return {symbol: 0.0 for symbol in self.config["symbols"]}

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        try:
            # Get current portfolio metrics
            portfolio_vol = float(self.redis.get("risk:portfolio_vol") or 0)

            # Get risk contributions
            risk_contribs = {}
            for symbol in self.config["symbols"]:
                contrib = self.redis.get(f"risk:contrib:{symbol.lower()}")
                risk_contribs[symbol] = float(contrib) if contrib else 0.0

            # Calculate weight concentration (Herfindahl index)
            weights_squared = [w**2 for w in self.current_weights.values()]
            concentration = sum(weights_squared)

            status = {
                "service": "risk_budgeter",
                "timestamp": time.time(),
                "config": self.config,
                "current_method": self.current_method,
                "current_weights": self.current_weights,
                "portfolio_metrics": {
                    "portfolio_vol": portfolio_vol,
                    "target_vol": self.config["target_vol"],
                    "concentration": concentration,
                    "diversification_ratio": (
                        1.0 / concentration if concentration > 0 else 0
                    ),
                },
                "risk_contributions": risk_contribs,
                "statistics": {
                    "total_updates": self.total_updates,
                    "last_update": self.last_update,
                    "weight_history_length": len(self.weight_history),
                },
                "recent_weights": (
                    self.weight_history[-5:] if self.weight_history else []
                ),
            }

            # Add covariance matrix if available
            if self.covariance_matrix is not None:
                correlation_matrix = self.covariance_matrix.copy()
                # Convert to correlation
                vols = np.sqrt(np.diag(correlation_matrix))
                correlation_matrix = correlation_matrix / np.outer(vols, vols)

                status["correlation_matrix"] = {
                    f"{self.config['symbols'][i]}_{self.config['symbols'][j]}": float(
                        correlation_matrix[i, j]
                    )
                    for i in range(len(self.config["symbols"]))
                    for j in range(i + 1, len(self.config["symbols"]))
                }

            return status

        except Exception as e:
            return {
                "service": "risk_budgeter",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }

    def run_continuous_budgeting(self):
        """Run continuous risk budgeting."""
        logger.info("⚖️ Starting continuous risk budgeting")

        try:
            while True:
                try:
                    # Update risk budget
                    result = self.update_risk_budget()

                    if result.get("rebalance_needed"):
                        logger.info(f"🔄 Rebalanced: update #{self.total_updates}")
                    elif self.total_updates % 10 == 0:
                        logger.info(
                            f"📊 Check #{self.total_updates}: no rebalance needed"
                        )

                    # Wait for next update
                    time.sleep(self.config["update_interval"])

                except Exception as e:
                    logger.error(f"Error in risk budgeting loop: {e}")
                    time.sleep(60)  # Wait longer on error

        except KeyboardInterrupt:
            logger.info("Risk budgeting stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in risk budgeting loop: {e}")


def main():
    """Main entry point for risk budgeter."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Asset Risk Budgeter")
    parser.add_argument(
        "--run", action="store_true", help="Run continuous risk budgeting"
    )
    parser.add_argument(
        "--update", action="store_true", help="Run single budget update"
    )
    parser.add_argument("--status", action="store_true", help="Show status report")
    parser.add_argument(
        "--method",
        choices=["equal_risk", "inverse_vol", "min_var", "risk_parity"],
        help="Risk budgeting method",
    )

    args = parser.parse_args()

    # Create budgeter
    budgeter = CrossAssetRiskBudgeter()

    if args.method:
        budgeter.current_method = args.method
        logger.info(f"Using method: {args.method}")

    if args.status:
        # Show status report
        status = budgeter.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.update:
        # Run single update
        result = budgeter.update_risk_budget()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.run:
        # Run continuous budgeting
        budgeter.run_continuous_budgeting()
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
