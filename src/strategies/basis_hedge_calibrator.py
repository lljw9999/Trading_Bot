#!/usr/bin/env python3
"""
Basis Carry Hedge-Ratio & Entry Z-Score Calibrator
Kalman filter for dynamic hedge ratio + z-score based entry/exit signals
"""

import os
import sys
import json
import time
import math
import logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import redis
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("basis_calibrator")


class BasisHedgeCalibrator:
    """Calibrator for basis carry hedge ratios and z-score entries."""

    def __init__(self):
        """Initialize basis hedge calibrator."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Calibration configuration
        self.config = {
            "symbols": ["BTC", "ETH", "SOL"],
            "update_interval": 5.0,  # Update every 5 seconds
            "basis_window": 600,  # 10-minute rolling window for z-score
            "kalman_q": 1e-5,  # Process noise covariance
            "kalman_r": 1e-3,  # Measurement noise covariance
            "z_entry_threshold": -1.2,  # Entry when z < -1.2
            "z_exit_threshold": 0.3,  # Exit when |z| < 0.3
            "beta_drift_threshold": 0.001,  # 10 bps beta drift threshold
            "min_observations": 50,  # Minimum observations for z-score
        }

        # State tracking per symbol
        self.states = {}
        for symbol in self.config["symbols"]:
            self.states[symbol] = {
                "beta": 1.0,  # Current hedge ratio
                "beta_variance": 0.01,  # Kalman filter variance
                "basis_history": deque(maxlen=self.config["basis_window"]),
                "z_score": 0.0,
                "basis_mean": 0.0,
                "basis_std": 1.0,
                "last_beta": 1.0,
                "beta_drift_events": [],
                "total_updates": 0,
            }

        self.last_update = 0
        self.total_cycles = 0

        # Load initial states from Redis
        self._load_states_from_redis()

        logger.info("📈 Basis Hedge Calibrator initialized")
        logger.info(f"   Symbols: {self.config['symbols']}")
        logger.info(f"   Basis window: {self.config['basis_window']} periods")
        logger.info(
            f"   Z-score thresholds: entry<{self.config['z_entry_threshold']}, exit<{self.config['z_exit_threshold']}"
        )
        logger.info(
            f"   Initial betas: {[(s, self.states[s]['beta']) for s in self.config['symbols']]}"
        )

    def _load_states_from_redis(self):
        """Load calibrator states from Redis."""
        try:
            for symbol in self.config["symbols"]:
                calib_data = self.redis.hgetall(f"basis:calib:{symbol}")

                if calib_data:
                    state = self.states[symbol]

                    # Load beta and variance
                    if "beta" in calib_data:
                        state["beta"] = float(calib_data["beta"])
                        state["last_beta"] = state["beta"]

                    if "beta_variance" in calib_data:
                        state["beta_variance"] = float(calib_data["beta_variance"])

                    # Load z-score statistics
                    if "z" in calib_data:
                        state["z_score"] = float(calib_data["z"])

                    if "mu" in calib_data:
                        state["basis_mean"] = float(calib_data["mu"])

                    if "sd" in calib_data:
                        state["basis_std"] = float(calib_data["sd"])

                    logger.debug(
                        f"Loaded {symbol} state: β={state['beta']:.4f}, z={state['z_score']:.2f}"
                    )

        except Exception as e:
            logger.warning(f"Error loading states from Redis: {e}")

    def get_spot_perp_prices(
        self, symbol: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get current spot and perp prices for symbol."""
        try:
            # Try to get from Redis
            spot_key = f"price:{symbol.lower()}:spot"
            perp_key = f"price:{symbol.lower()}:perp"

            spot_price = self.redis.get(spot_key)
            perp_price = self.redis.get(perp_key)

            if spot_price and perp_price:
                return float(spot_price), float(perp_price)

            # Fallback: try generic price keys
            px_spot = self.redis.get("px:spot")
            px_perp = self.redis.get("px:perp")

            if px_spot and px_perp:
                return float(px_spot), float(px_perp)

            # Mock data for demo
            base_prices = {"BTC": 97600, "ETH": 3515, "SOL": 184}
            base_price = base_prices.get(symbol, 100)

            # Add some random walk and basis
            time_factor = time.time() % 3600  # Hourly cycle
            price_drift = math.sin(time_factor / 3600 * 2 * math.pi) * 0.001

            spot_price = base_price * (1 + price_drift + np.random.normal(0, 0.0005))

            # Perp with small basis
            basis_bps = np.random.normal(-2, 6)  # -2 ± 6 bps
            perp_price = spot_price * (1 + basis_bps / 10000)

            return spot_price, perp_price

        except Exception as e:
            logger.error(f"Error getting prices for {symbol}: {e}")
            return None, None

    def kalman_update_beta(
        self, symbol: str, spot_price: float, perp_price: float
    ) -> float:
        """Update hedge ratio using Kalman filter."""
        try:
            state = self.states[symbol]

            # Kalman filter parameters
            q = self.config["kalman_q"]  # Process noise
            r = self.config["kalman_r"]  # Measurement noise

            # Prediction step
            beta_pred = state["beta"]  # No motion model
            p_pred = state["beta_variance"] + q

            # Update step
            # Measurement: perp_price = beta * spot_price + noise
            # Innovation: y = perp_price - beta_pred * spot_price
            innovation = perp_price - beta_pred * spot_price

            # Innovation covariance: S = H * P * H' + R
            # H = spot_price (measurement jacobian)
            innovation_cov = (spot_price**2) * p_pred + r

            # Kalman gain: K = P * H' * S^(-1)
            kalman_gain = p_pred * spot_price / innovation_cov

            # State update
            new_beta = beta_pred + kalman_gain * innovation / spot_price
            new_variance = p_pred - kalman_gain * spot_price * p_pred

            # Store updated state
            state["beta"] = new_beta
            state["beta_variance"] = max(new_variance, 1e-6)  # Prevent numerical issues

            return new_beta

        except Exception as e:
            logger.error(f"Error in Kalman update for {symbol}: {e}")
            return self.states[symbol]["beta"]

    def simple_beta_update(
        self, symbol: str, spot_price: float, perp_price: float
    ) -> float:
        """Simple beta update (fallback method)."""
        try:
            state = self.states[symbol]

            # Simple exponential moving average
            alpha = 0.02  # Learning rate
            price_ratio = perp_price / spot_price if spot_price > 0 else 1.0

            new_beta = (1 - alpha) * state["beta"] + alpha * price_ratio
            state["beta"] = new_beta

            return new_beta

        except Exception as e:
            logger.error(f"Error in simple beta update for {symbol}: {e}")
            return self.states[symbol]["beta"]

    def calculate_basis_z_score(self, symbol: str, basis_bps: float) -> float:
        """Calculate z-score for basis."""
        try:
            state = self.states[symbol]

            # Add to history
            state["basis_history"].append(basis_bps)

            # Need minimum observations
            if len(state["basis_history"]) < self.config["min_observations"]:
                return 0.0

            # Calculate rolling statistics
            basis_values = list(state["basis_history"])
            basis_mean = np.mean(basis_values)
            basis_std = np.std(basis_values)

            # Prevent division by zero
            if basis_std < 1e-6:
                basis_std = 1.0

            # Calculate z-score
            z_score = (basis_bps - basis_mean) / basis_std

            # Update state
            state["basis_mean"] = basis_mean
            state["basis_std"] = basis_std
            state["z_score"] = z_score

            return z_score

        except Exception as e:
            logger.error(f"Error calculating z-score for {symbol}: {e}")
            return 0.0

    def check_beta_drift(self, symbol: str) -> Tuple[bool, float]:
        """Check if beta has drifted significantly."""
        try:
            state = self.states[symbol]
            current_beta = state["beta"]
            last_beta = state["last_beta"]

            # Calculate drift
            beta_drift = abs(current_beta - last_beta)
            drift_threshold = self.config["beta_drift_threshold"]

            is_drift = beta_drift > drift_threshold

            if is_drift:
                # Log drift event
                drift_event = {
                    "timestamp": time.time(),
                    "symbol": symbol,
                    "from_beta": last_beta,
                    "to_beta": current_beta,
                    "drift": beta_drift,
                    "drift_bps": beta_drift * 10000,
                }

                state["beta_drift_events"].append(drift_event)

                # Keep limited history
                if len(state["beta_drift_events"]) > 100:
                    state["beta_drift_events"] = state["beta_drift_events"][-50:]

                # Update last beta
                state["last_beta"] = current_beta

                logger.info(
                    f"📊 Beta drift detected for {symbol}: "
                    f"{last_beta:.4f} → {current_beta:.4f} "
                    f"({beta_drift*10000:.1f}bps)"
                )

            return is_drift, beta_drift

        except Exception as e:
            logger.error(f"Error checking beta drift for {symbol}: {e}")
            return False, 0.0

    def get_entry_exit_signals(self, symbol: str) -> Dict[str, Any]:
        """Get entry/exit signals based on z-score."""
        try:
            state = self.states[symbol]
            z_score = state["z_score"]

            signals = {
                "entry_signal": False,
                "exit_signal": False,
                "signal_strength": abs(z_score),
                "z_score": z_score,
            }

            # Entry signal: z < threshold (cheap carry)
            if z_score < self.config["z_entry_threshold"]:
                signals["entry_signal"] = True
                signals["entry_reason"] = (
                    f"z_score_{z_score:.2f}_below_{self.config['z_entry_threshold']}"
                )

            # Exit signal: |z| < threshold (mean reversion)
            if abs(z_score) < self.config["z_exit_threshold"]:
                signals["exit_signal"] = True
                signals["exit_reason"] = f"z_score_revert_{z_score:.2f}"

            return signals

        except Exception as e:
            logger.error(f"Error getting signals for {symbol}: {e}")
            return {"entry_signal": False, "exit_signal": False, "z_score": 0.0}

    def update_symbol_calibration(self, symbol: str) -> Dict[str, Any]:
        """Update calibration for a single symbol."""
        try:
            update_start = time.time()

            # Get current prices
            spot_price, perp_price = self.get_spot_perp_prices(symbol)

            if not spot_price or not perp_price:
                return {
                    "symbol": symbol,
                    "status": "no_data",
                    "timestamp": update_start,
                }

            # Calculate basis
            basis_bps = (perp_price - spot_price) / spot_price * 10000

            # Update hedge ratio using Kalman filter
            try:
                new_beta = self.kalman_update_beta(symbol, spot_price, perp_price)
            except Exception as e:
                logger.debug(
                    f"Kalman update failed for {symbol}, using simple update: {e}"
                )
                new_beta = self.simple_beta_update(symbol, spot_price, perp_price)

            # Calculate z-score
            z_score = self.calculate_basis_z_score(symbol, basis_bps)

            # Check beta drift
            has_drift, drift_amount = self.check_beta_drift(symbol)

            # Get entry/exit signals
            signals = self.get_entry_exit_signals(symbol)

            # Update state
            state = self.states[symbol]
            state["total_updates"] += 1

            # Store in Redis
            redis_data = {
                "beta": new_beta,
                "beta_variance": state["beta_variance"],
                "z": z_score,
                "mu": state["basis_mean"],
                "sd": state["basis_std"],
                "basis_bps": basis_bps,
                "spot_price": spot_price,
                "perp_price": perp_price,
                "last_update": update_start,
            }

            self.redis.hset(f"basis:calib:{symbol}", mapping=redis_data)

            # Store general basis calibration data
            self.redis.hset(
                "basis:calib",
                mapping={
                    f"{symbol}_beta": new_beta,
                    f"{symbol}_z": z_score,
                    f"{symbol}_mu": state["basis_mean"],
                    f"{symbol}_sd": state["basis_std"],
                },
            )

            update_duration = time.time() - update_start

            result = {
                "symbol": symbol,
                "status": "completed",
                "timestamp": update_start,
                "spot_price": spot_price,
                "perp_price": perp_price,
                "basis_bps": basis_bps,
                "beta": new_beta,
                "beta_variance": state["beta_variance"],
                "z_score": z_score,
                "basis_mean": state["basis_mean"],
                "basis_std": state["basis_std"],
                "beta_drift": has_drift,
                "drift_amount": drift_amount,
                "signals": signals,
                "observations": len(state["basis_history"]),
                "update_duration": update_duration,
            }

            return result

        except Exception as e:
            logger.error(f"Error updating calibration for {symbol}: {e}")
            return {
                "symbol": symbol,
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }

    def calibration_cycle(self) -> Dict[str, Any]:
        """Run one complete calibration cycle."""
        try:
            cycle_start = time.time()
            self.total_cycles += 1

            # Update calibration for each symbol
            symbol_results = {}
            total_drifts = 0
            active_signals = {"entry": [], "exit": []}

            for symbol in self.config["symbols"]:
                try:
                    result = self.update_symbol_calibration(symbol)
                    symbol_results[symbol] = result

                    if result.get("beta_drift"):
                        total_drifts += 1

                    # Collect signals
                    signals = result.get("signals", {})
                    if signals.get("entry_signal"):
                        active_signals["entry"].append(symbol)
                    if signals.get("exit_signal"):
                        active_signals["exit"].append(symbol)

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    symbol_results[symbol] = {"status": "error", "error": str(e)}

            # Update global metrics
            avg_beta = np.mean([self.states[s]["beta"] for s in self.config["symbols"]])
            avg_z_score = np.mean(
                [self.states[s]["z_score"] for s in self.config["symbols"]]
            )

            # Store global metrics in Redis
            global_metrics = {
                "basis_avg_beta": avg_beta,
                "basis_avg_z_score": avg_z_score,
                "basis_beta_drifts": total_drifts,
                "basis_entry_signals": len(active_signals["entry"]),
                "basis_exit_signals": len(active_signals["exit"]),
            }

            for metric, value in global_metrics.items():
                self.redis.set(f"metric:{metric}", value)

            self.last_update = cycle_start
            cycle_duration = time.time() - cycle_start

            # Log summary
            if total_drifts > 0 or active_signals["entry"] or active_signals["exit"]:
                logger.info(
                    f"📊 Basis Calibration #{self.total_cycles}: "
                    f"{total_drifts} drifts, {len(active_signals['entry'])} entry, {len(active_signals['exit'])} exit signals"
                )

            result = {
                "timestamp": cycle_start,
                "status": "completed",
                "total_cycles": self.total_cycles,
                "symbol_results": symbol_results,
                "total_drifts": total_drifts,
                "active_signals": active_signals,
                "global_metrics": global_metrics,
                "cycle_duration": cycle_duration,
            }

            return result

        except Exception as e:
            logger.error(f"Error in calibration cycle: {e}")
            return {"timestamp": time.time(), "status": "error", "error": str(e)}

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        try:
            current_time = time.time()

            # Collect per-symbol statistics
            symbol_stats = {}
            for symbol in self.config["symbols"]:
                state = self.states[symbol]

                symbol_stats[symbol] = {
                    "beta": state["beta"],
                    "beta_variance": state["beta_variance"],
                    "z_score": state["z_score"],
                    "basis_mean": state["basis_mean"],
                    "basis_std": state["basis_std"],
                    "observations": len(state["basis_history"]),
                    "total_updates": state["total_updates"],
                    "drift_events": len(state["beta_drift_events"]),
                    "recent_drifts": [
                        {
                            "timestamp": d["timestamp"],
                            "from_beta": d["from_beta"],
                            "to_beta": d["to_beta"],
                            "drift_bps": d["drift_bps"],
                        }
                        for d in state["beta_drift_events"][-3:]
                    ],
                }

            status = {
                "service": "basis_hedge_calibrator",
                "timestamp": current_time,
                "config": self.config,
                "total_cycles": self.total_cycles,
                "last_update": self.last_update,
                "last_update_ago": (
                    current_time - self.last_update if self.last_update > 0 else 0
                ),
                "symbols": symbol_stats,
                "summary": {
                    "avg_beta": np.mean([s["beta"] for s in symbol_stats.values()]),
                    "avg_z_score": np.mean(
                        [s["z_score"] for s in symbol_stats.values()]
                    ),
                    "total_observations": sum(
                        [s["observations"] for s in symbol_stats.values()]
                    ),
                    "total_drift_events": sum(
                        [s["drift_events"] for s in symbol_stats.values()]
                    ),
                },
            }

            return status

        except Exception as e:
            return {
                "service": "basis_hedge_calibrator",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }

    def run_continuous_calibration(self):
        """Run continuous basis hedge calibration."""
        logger.info("📈 Starting continuous basis hedge calibration")

        try:
            while True:
                try:
                    # Run calibration cycle
                    result = self.calibration_cycle()

                    if result["status"] == "completed":
                        total_signals = len(result["active_signals"]["entry"]) + len(
                            result["active_signals"]["exit"]
                        )

                        if result["total_drifts"] > 0 or total_signals > 0:
                            logger.debug(
                                f"📊 Cycle #{self.total_cycles}: "
                                f"{result['total_drifts']} drifts, {total_signals} signals"
                            )

                    # Sleep until next cycle
                    time.sleep(self.config["update_interval"])

                except Exception as e:
                    logger.error(f"Error in calibration loop: {e}")
                    time.sleep(30)  # Wait longer on error

        except KeyboardInterrupt:
            logger.info("Basis calibrator stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in calibration loop: {e}")


def main():
    """Main entry point for basis calibrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Basis Hedge Calibrator")
    parser.add_argument("--run", action="store_true", help="Run continuous calibration")
    parser.add_argument(
        "--cycle", action="store_true", help="Run single calibration cycle"
    )
    parser.add_argument("--status", action="store_true", help="Show status report")
    parser.add_argument(
        "--symbol", choices=["BTC", "ETH", "SOL"], help="Process specific symbol only"
    )

    args = parser.parse_args()

    # Create calibrator
    calibrator = BasisHedgeCalibrator()

    if args.symbol:
        calibrator.config["symbols"] = [args.symbol]

    if args.status:
        # Show status report
        status = calibrator.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.cycle:
        # Run single cycle
        result = calibrator.calibration_cycle()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.run:
        # Run continuous calibration
        calibrator.run_continuous_calibration()
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
