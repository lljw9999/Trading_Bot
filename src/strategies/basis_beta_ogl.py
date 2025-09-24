#!/usr/bin/env python3
"""
Online GLS (Generalized Least Squares) Hedge Beta Calculator

Real-time hedge ratio estimation for spot-perp basis trading using online GLS
with Kalman filter fallback for stability during market stress.
"""

import os
import sys
import time
import asyncio
import logging
import json
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from scipy import linalg

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.utils.aredis import get_redis, get_batch_writer, set_metric

    ASYNC_REDIS_AVAILABLE = True
except ImportError:
    ASYNC_REDIS_AVAILABLE = False
    import redis

logger = logging.getLogger("basis_beta_ogl")


class OnlineGLSHedgeCalculator:
    """
    Online Generalized Least Squares hedge ratio calculator.

    Computes dynamic hedge ratios (beta) between spot and perpetual futures
    using rolling GLS estimation with heteroskedasticity-robust covariance.
    Falls back to Kalman filter during periods of instability.
    """

    def __init__(
        self, symbol: str = "BTC", window_size: int = 300, min_samples: int = 50
    ):
        """
        Initialize online GLS calculator.

        Args:
            symbol: Trading symbol (BTC, ETH, SOL)
            window_size: Rolling window size for GLS estimation
            min_samples: Minimum samples before publishing beta
        """
        self.symbol = symbol
        self.window_size = window_size
        self.min_samples = min_samples

        # Rolling data storage
        self.spot_prices = deque(maxlen=window_size)
        self.perp_prices = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.returns_spot = deque(maxlen=window_size - 1)
        self.returns_perp = deque(maxlen=window_size - 1)

        # GLS estimation state
        self.current_beta = 1.0
        self.beta_std = 0.1
        self.r_squared = 0.0
        self.gls_stable = True
        self.last_update = 0.0

        # Kalman filter fallback state
        self.kalman_beta = 1.0
        self.kalman_variance = 0.01
        self.process_noise = 1e-6
        self.measurement_noise = 1e-4

        # Performance and diagnostics
        self.stats = {
            "total_updates": 0,
            "gls_updates": 0,
            "kalman_updates": 0,
            "instability_events": 0,
            "avg_update_time_ms": 0.0,
            "last_beta_change": 0.0,
        }

        # Configuration
        self.config = {
            "gls_min_rsquared": 0.3,  # Min R² for GLS stability
            "gls_max_condition": 100.0,  # Max condition number for GLS
            "beta_change_threshold": 0.05,  # Max beta change per update
            "outlier_threshold": 3.0,  # Z-score threshold for outliers
            "update_frequency": 1.0,  # Seconds between updates
            "heteroskedasticity_window": 60,  # Window for variance estimation
        }

        logger.info(f"Initialized Online GLS hedge calculator for {symbol}")
        logger.info(f"  Window size: {window_size}, Min samples: {min_samples}")
        logger.info(
            f"  GLS stability: R²>{self.config['gls_min_rsquared']}, "
            f"cond<{self.config['gls_max_condition']}"
        )

    def add_price_tick(
        self, spot_price: float, perp_price: float, timestamp: float = None
    ) -> bool:
        """
        Add new price tick and update hedge ratio if needed.

        Args:
            spot_price: Current spot price
            perp_price: Current perpetual price
            timestamp: Timestamp (defaults to current time)

        Returns:
            True if beta was updated, False otherwise
        """
        try:
            if timestamp is None:
                timestamp = time.time()

            # Validate inputs
            if spot_price <= 0 or perp_price <= 0:
                logger.warning(f"Invalid prices: spot={spot_price}, perp={perp_price}")
                return False

            # Store prices
            self.spot_prices.append(spot_price)
            self.perp_prices.append(perp_price)
            self.timestamps.append(timestamp)

            # Calculate returns if we have enough data
            if len(self.spot_prices) >= 2:
                prev_spot = self.spot_prices[-2]
                prev_perp = self.perp_prices[-2]

                if prev_spot > 0 and prev_perp > 0:
                    ret_spot = np.log(spot_price / prev_spot)
                    ret_perp = np.log(perp_price / prev_perp)

                    # Filter outliers
                    if not self._is_outlier(ret_spot, ret_perp):
                        self.returns_spot.append(ret_spot)
                        self.returns_perp.append(ret_perp)

            # Update beta if we have enough samples and enough time has passed
            should_update = (
                len(self.returns_spot) >= self.min_samples
                and timestamp - self.last_update >= self.config["update_frequency"]
            )

            if should_update:
                return self._update_beta()

            return False

        except Exception as e:
            logger.error(f"Error adding price tick: {e}")
            return False

    def _is_outlier(self, ret_spot: float, ret_perp: float) -> bool:
        """Check if returns are outliers based on recent history."""
        try:
            if len(self.returns_spot) < 20:  # Need minimum history
                return False

            # Calculate z-scores
            recent_spot = list(self.returns_spot)[-50:]  # Last 50 returns
            recent_perp = list(self.returns_perp)[-50:]

            spot_mean = np.mean(recent_spot)
            spot_std = np.std(recent_spot)
            perp_mean = np.mean(recent_perp)
            perp_std = np.std(recent_perp)

            if spot_std > 0:
                z_spot = abs(ret_spot - spot_mean) / spot_std
            else:
                z_spot = 0

            if perp_std > 0:
                z_perp = abs(ret_perp - perp_mean) / perp_std
            else:
                z_perp = 0

            # Flag as outlier if either return is extreme
            is_outlier = (
                z_spot > self.config["outlier_threshold"]
                or z_perp > self.config["outlier_threshold"]
            )

            if is_outlier:
                logger.debug(
                    f"Outlier detected: spot_z={z_spot:.2f}, perp_z={z_perp:.2f}"
                )

            return is_outlier

        except Exception as e:
            logger.error(f"Error checking outliers: {e}")
            return False

    def _update_beta(self) -> bool:
        """Update hedge ratio using GLS with Kalman fallback."""
        try:
            start_time = time.time()

            # Try GLS first
            gls_success, beta_gls, std_gls, r_squared = self._estimate_gls_beta()

            if gls_success and self._validate_gls_result(beta_gls, std_gls, r_squared):
                # Use GLS result
                old_beta = self.current_beta
                self.current_beta = beta_gls
                self.beta_std = std_gls
                self.r_squared = r_squared
                self.gls_stable = True
                self.stats["gls_updates"] += 1

                logger.debug(
                    f"GLS update: β={beta_gls:.4f}±{std_gls:.4f}, R²={r_squared:.3f}"
                )

            else:
                # Fall back to Kalman filter
                self._estimate_kalman_beta()
                self.gls_stable = False
                self.stats["kalman_updates"] += 1
                self.stats["instability_events"] += 1

                logger.warning(f"GLS unstable, using Kalman: β={self.current_beta:.4f}")

            # Update statistics
            self.last_update = time.time()
            self.stats["total_updates"] += 1
            self.stats["last_beta_change"] = (
                abs(self.current_beta - old_beta) if "old_beta" in locals() else 0.0
            )

            update_duration = time.time() - start_time
            self.stats["avg_update_time_ms"] = (
                self.stats["avg_update_time_ms"] * (self.stats["total_updates"] - 1)
                + update_duration * 1000
            ) / self.stats["total_updates"]

            return True

        except Exception as e:
            logger.error(f"Error updating beta: {e}")
            return False

    def _estimate_gls_beta(self) -> Tuple[bool, float, float, float]:
        """
        Estimate hedge ratio using Generalized Least Squares.

        Returns:
            (success, beta, std_error, r_squared)
        """
        try:
            if len(self.returns_spot) < self.min_samples:
                return False, 1.0, 0.1, 0.0

            # Convert to numpy arrays
            y = np.array(self.returns_perp)  # Dependent variable (perp returns)
            X = np.column_stack(
                [np.ones(len(y)), np.array(self.returns_spot)]
            )  # [intercept, spot returns]

            if len(y) != len(X):
                return False, 1.0, 0.1, 0.0

            # Ordinary least squares first pass
            try:
                beta_ols = linalg.lstsq(X, y)[0]
                residuals = y - X @ beta_ols
            except linalg.LinAlgError:
                return False, 1.0, 0.1, 0.0

            # Estimate heteroskedastic variance (White's approach)
            weights = self._estimate_heteroskedastic_weights(residuals, X)

            # Weighted least squares (GLS)
            W = np.diag(weights)
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y

            # Check condition number
            try:
                cond_num = np.linalg.cond(XtWX)
                if cond_num > self.config["gls_max_condition"]:
                    logger.debug(f"GLS condition number too high: {cond_num:.1f}")
                    return False, 1.0, 0.1, 0.0
            except:
                return False, 1.0, 0.1, 0.0

            # Solve GLS
            try:
                beta_gls = linalg.solve(XtWX, XtWy)

                # Calculate standard errors
                residuals_gls = y - X @ beta_gls
                mse = np.sum(weights * residuals_gls**2) / (len(y) - 2)
                var_beta = mse * linalg.inv(XtWX)
                std_errors = np.sqrt(np.diag(var_beta))

                # Calculate R-squared
                y_mean = np.mean(y)
                tss = np.sum((y - y_mean) ** 2)
                rss = np.sum(residuals_gls**2)
                r_squared = 1 - (rss / tss) if tss > 0 else 0

                # Extract hedge ratio (slope coefficient)
                hedge_beta = beta_gls[1]  # Second coefficient is the slope
                hedge_std = std_errors[1]

                return True, hedge_beta, hedge_std, r_squared

            except linalg.LinAlgError as e:
                logger.debug(f"GLS solve failed: {e}")
                return False, 1.0, 0.1, 0.0

        except Exception as e:
            logger.error(f"Error in GLS estimation: {e}")
            return False, 1.0, 0.1, 0.0

    def _estimate_heteroskedastic_weights(
        self, residuals: np.ndarray, X: np.ndarray
    ) -> np.ndarray:
        """Estimate heteroskedastic weights using rolling variance."""
        try:
            n = len(residuals)
            window = min(self.config["heteroskedasticity_window"], n // 2)
            weights = np.ones(n)

            # Rolling variance estimation
            for i in range(n):
                start_idx = max(0, i - window // 2)
                end_idx = min(n, i + window // 2 + 1)
                local_residuals = residuals[start_idx:end_idx]

                if len(local_residuals) > 1:
                    local_var = np.var(local_residuals)
                    weights[i] = 1.0 / max(local_var, 1e-8)  # Prevent division by zero

            # Normalize weights
            weights = weights / np.mean(weights)
            return weights

        except Exception as e:
            logger.error(f"Error estimating heteroskedastic weights: {e}")
            return np.ones(len(residuals))

    def _validate_gls_result(
        self, beta: float, std_error: float, r_squared: float
    ) -> bool:
        """Validate GLS result for stability."""
        try:
            # Check R-squared threshold
            if r_squared < self.config["gls_min_rsquared"]:
                logger.debug(f"GLS R² too low: {r_squared:.3f}")
                return False

            # Check beta reasonableness (should be close to 1.0 for basis trading)
            if abs(beta) > 2.0 or abs(beta) < 0.1:
                logger.debug(f"GLS beta unreasonable: {beta:.4f}")
                return False

            # Check standard error (beta should be significant)
            if std_error > 0 and abs(beta / std_error) < 2.0:  # t-stat < 2
                logger.debug(
                    f"GLS beta not significant: t-stat={abs(beta/std_error):.2f}"
                )
                return False

            # Check for excessive change from previous estimate
            beta_change = abs(beta - self.current_beta)
            if beta_change > self.config["beta_change_threshold"]:
                logger.debug(f"GLS beta change too large: {beta_change:.4f}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating GLS result: {e}")
            return False

    def _estimate_kalman_beta(self):
        """Update hedge ratio using Kalman filter (fallback)."""
        try:
            if len(self.returns_spot) < 2:
                return

            # Get latest return pair
            y = self.returns_perp[-1]  # Perp return (observation)
            x = self.returns_spot[-1]  # Spot return (predictor)

            # Kalman filter update
            # Prediction step
            pred_beta = self.kalman_beta
            pred_variance = self.kalman_variance + self.process_noise

            # Update step
            if x != 0:
                innovation = y - pred_beta * x
                innovation_variance = pred_variance * x**2 + self.measurement_noise
                kalman_gain = (pred_variance * x) / innovation_variance

                # Update estimates
                self.kalman_beta = pred_beta + kalman_gain * innovation
                self.kalman_variance = pred_variance - kalman_gain * x * pred_variance
                self.current_beta = self.kalman_beta
                self.beta_std = np.sqrt(self.kalman_variance)

            logger.debug(
                f"Kalman update: β={self.kalman_beta:.4f}±{np.sqrt(self.kalman_variance):.4f}"
            )

        except Exception as e:
            logger.error(f"Error in Kalman estimation: {e}")

    def get_hedge_ratio(self) -> Tuple[float, float, Dict[str, Any]]:
        """
        Get current hedge ratio with metadata.

        Returns:
            (beta, std_error, metadata)
        """
        metadata = {
            "symbol": self.symbol,
            "last_update": self.last_update,
            "method": "gls" if self.gls_stable else "kalman",
            "r_squared": self.r_squared,
            "samples": len(self.returns_spot),
            "stable": self.gls_stable,
            "stats": self.stats.copy(),
        }

        return self.current_beta, self.beta_std, metadata

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics."""
        return {
            "symbol": self.symbol,
            "current_beta": self.current_beta,
            "beta_std": self.beta_std,
            "r_squared": self.r_squared,
            "gls_stable": self.gls_stable,
            "samples": len(self.returns_spot),
            "window_size": self.window_size,
            "last_update": self.last_update,
            "config": self.config.copy(),
            "stats": self.stats.copy(),
            "recent_prices": (
                {
                    "spot": list(self.spot_prices)[-5:],
                    "perp": list(self.perp_prices)[-5:],
                    "timestamps": list(self.timestamps)[-5:],
                }
                if len(self.spot_prices) > 0
                else {}
            ),
        }


class AsyncBasisBetaPublisher:
    """Async publisher for basis beta calculations."""

    def __init__(self, symbols: List[str] = None):
        """
        Initialize async publisher.

        Args:
            symbols: List of symbols to track (default: ["BTC", "ETH", "SOL"])
        """
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL"]

        self.symbols = symbols
        self.calculators = {}
        self.running = False

        # Initialize calculators
        for symbol in symbols:
            self.calculators[symbol] = OnlineGLSHedgeCalculator(symbol)

        logger.info(f"Initialized async beta publisher for {symbols}")

    async def start_publishing(self, update_interval: float = 1.0):
        """Start async publishing loop."""
        logger.info(f"Starting basis beta publisher with {update_interval}s interval")
        self.running = True

        try:
            while self.running:
                await asyncio.gather(
                    *[self._update_symbol_beta(symbol) for symbol in self.symbols],
                    return_exceptions=True,
                )

                await asyncio.sleep(update_interval)

        except Exception as e:
            logger.error(f"Error in publishing loop: {e}")
        finally:
            self.running = False

    async def _update_symbol_beta(self, symbol: str):
        """Update beta for single symbol."""
        try:
            calculator = self.calculators[symbol]

            # Get market data
            if ASYNC_REDIS_AVAILABLE:
                redis = await get_redis()
                spot_price = await redis.get(f"price:{symbol.lower()}:spot")
                perp_price = await redis.get(f"price:{symbol.lower()}:perp")
            else:
                redis = redis.Redis(decode_responses=True)
                spot_price = redis.get(f"price:{symbol.lower()}:spot")
                perp_price = redis.get(f"price:{symbol.lower()}:perp")

            if spot_price and perp_price:
                spot_px = float(spot_price)
                perp_px = float(perp_price)

                # Update calculator
                updated = calculator.add_price_tick(spot_px, perp_px)

                if updated:
                    beta, beta_std, metadata = calculator.get_hedge_ratio()

                    # Publish to Redis
                    beta_data = {
                        "beta": beta,
                        "beta_std": beta_std,
                        "timestamp": time.time(),
                        "method": metadata["method"],
                        "r_squared": metadata["r_squared"],
                        "stable": metadata["stable"],
                    }

                    if ASYNC_REDIS_AVAILABLE:
                        writer = await get_batch_writer()
                        await writer.set_batch(
                            f"basis:beta:{symbol}", json.dumps(beta_data)
                        )
                        await set_metric(f"basis_beta_{symbol.lower()}", beta)
                        await set_metric(
                            f"basis_beta_stable_{symbol.lower()}",
                            int(metadata["stable"]),
                        )
                    else:
                        redis.set(f"basis:beta:{symbol}", json.dumps(beta_data))
                        redis.set(f"metric:basis_beta_{symbol.lower()}", beta)

                    logger.debug(
                        f"Published beta for {symbol}: {beta:.4f} ({metadata['method']})"
                    )

        except Exception as e:
            logger.error(f"Error updating beta for {symbol}: {e}")

    def stop_publishing(self):
        """Stop publishing loop."""
        self.running = False
        logger.info("Stopped basis beta publisher")

    def get_calculator(self, symbol: str) -> Optional[OnlineGLSHedgeCalculator]:
        """Get calculator for symbol."""
        return self.calculators.get(symbol)


async def main():
    """Test the basis beta calculator."""
    import argparse

    parser = argparse.ArgumentParser(description="Basis Beta GLS Calculator")
    parser.add_argument("--symbol", default="BTC", help="Symbol to track")
    parser.add_argument(
        "--test", action="store_true", help="Run test with synthetic data"
    )
    parser.add_argument("--publish", action="store_true", help="Start publishing loop")

    args = parser.parse_args()

    if args.test:
        # Test with synthetic data
        calc = OnlineGLSHedgeCalculator(args.symbol)

        # Generate correlated prices
        np.random.seed(42)
        base_price = 50000
        for i in range(200):
            noise_spot = np.random.normal(0, 0.001)
            noise_perp = np.random.normal(0, 0.001)

            spot_price = base_price * (1 + noise_spot)
            perp_price = base_price * (
                1 + 0.98 * noise_spot + noise_perp
            )  # 0.98 hedge ratio

            calc.add_price_tick(spot_price, perp_price, time.time() + i)

            if i % 50 == 0:
                beta, std, meta = calc.get_hedge_ratio()
                print(
                    f"Step {i}: β={beta:.4f}±{std:.4f}, method={meta['method']}, R²={meta['r_squared']:.3f}"
                )

        # Final diagnostics
        diag = calc.get_diagnostics()
        print(f"\nFinal diagnostics:")
        print(json.dumps(diag, indent=2, default=str))

    elif args.publish:
        # Start publishing
        publisher = AsyncBasisBetaPublisher([args.symbol])
        try:
            await publisher.start_publishing()
        except KeyboardInterrupt:
            publisher.stop_publishing()

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
