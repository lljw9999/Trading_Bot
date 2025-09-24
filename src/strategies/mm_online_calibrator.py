#!/usr/bin/env python3
"""
MM Online Calibrator
Markout-driven γ,k tuning for Avellaneda-Stoikov market maker
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
logger = logging.getLogger("mm_calibrator")


class MMOnlineCalibrator:
    """Online calibrator for MM parameters using markout analysis."""

    def __init__(self):
        """Initialize MM online calibrator."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Initial Avellaneda-Stoikov parameters
        self.gamma = 0.1  # Risk aversion
        self.k = 1.5  # Order book intensity

        # Calibration configuration
        self.config = {
            "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "markout_periods": [1, 5],  # 1s and 5s markouts
            "update_interval": 5.0,  # Update every 5 seconds
            "step_size_gamma": 0.02,  # Parameter step size
            "step_size_k": 0.05,  # Parameter step size
            "gamma_bounds": [0.05, 0.5],  # Gamma bounds
            "k_bounds": [1.0, 3.0],  # K bounds
            "markout_thresholds": {
                "mo1_negative": -1.0,  # 1s markout negative threshold (bps)
                "mo1_positive": 0.5,  # 1s markout positive threshold (bps)
                "mo5_negative": -1.5,  # 5s markout negative threshold (bps)
                "mo5_positive": 0.8,  # 5s markout positive threshold (bps)
            },
            "smoothing_alpha": 0.7,  # EMA smoothing for markouts
            "min_fills_for_update": 5,  # Minimum fills before update
            "toxic_flow_threshold": -3.0,  # Stop quoting if markout < -3bps
            "calibration_window": 300,  # 5-minute calibration window
        }

        # State tracking
        self.markout_history = {
            1: deque(maxlen=self.config["calibration_window"]),
            5: deque(maxlen=self.config["calibration_window"]),
        }
        self.fill_history = deque(maxlen=1000)
        self.parameter_history = []
        self.last_update = 0
        self.total_updates = 0
        self.total_calibrations = 0

        # Smoothed markout values
        self.smoothed_markouts = {1: 0.0, 5: 0.0}

        # Load initial parameters from Redis
        self._load_parameters_from_redis()

        # Metrics
        self.metrics = {
            "mm_calib_events_total": 0,
            "mm_gamma": self.gamma,
            "mm_k": self.k,
            "mm_markout_1s": 0.0,
            "mm_markout_5s": 0.0,
            "mm_toxic_flow_detected": 0.0,
        }

        logger.info("📊 MM Online Calibrator initialized")
        logger.info(f"   Initial γ: {self.gamma:.3f}")
        logger.info(f"   Initial k: {self.k:.3f}")
        logger.info(f"   Symbols: {self.config['symbols']}")
        logger.info(f"   Update interval: {self.config['update_interval']}s")

    def _load_parameters_from_redis(self):
        """Load current parameters from Redis."""
        try:
            gamma_redis = self.redis.get("mm:gamma")
            k_redis = self.redis.get("mm:k")

            if gamma_redis:
                self.gamma = float(gamma_redis)

            if k_redis:
                self.k = float(k_redis)

            logger.debug(
                f"Loaded parameters from Redis: γ={self.gamma:.3f}, k={self.k:.3f}"
            )

        except Exception as e:
            logger.warning(f"Error loading parameters from Redis: {e}")

    def get_recent_fills(self) -> List[Dict[str, Any]]:
        """Get recent MM fills from Redis."""
        try:
            fills = []

            # Get fills from Redis stream
            for symbol in self.config["symbols"]:
                fill_stream = f"mm:fills:{symbol.lower()}"
                recent_fills = self.redis.xrevrange(fill_stream, "+", "-", count=50)

                for stream_id, fill_data in recent_fills:
                    try:
                        fill = {
                            "stream_id": stream_id,
                            "symbol": fill_data.get("symbol", symbol),
                            "side": fill_data.get("side", ""),
                            "price": float(fill_data.get("price", 0)),
                            "size": float(fill_data.get("size", 0)),
                            "timestamp": float(fill_data.get("timestamp", time.time())),
                            "quote_id": fill_data.get("quote_id", ""),
                        }
                        fills.append(fill)
                    except Exception as e:
                        logger.debug(f"Error parsing fill data: {e}")

            # Sort by timestamp
            fills.sort(key=lambda x: x["timestamp"], reverse=True)

            # If no real fills, generate mock fills for demo
            if not fills and len(self.fill_history) == 0:
                fills = self._generate_mock_fills()

            return fills[:50]  # Return most recent 50 fills

        except Exception as e:
            logger.error(f"Error getting recent fills: {e}")
            return []

    def _generate_mock_fills(self) -> List[Dict[str, Any]]:
        """Generate mock fills for demonstration."""
        try:
            mock_fills = []
            current_time = time.time()

            # Generate fills for last few minutes
            for i in range(20):
                timestamp = current_time - (i * 15)  # Every 15 seconds
                symbol = np.random.choice(self.config["symbols"])
                side = np.random.choice(["bid", "ask"])

                # Mock prices
                base_prices = {"BTCUSDT": 97600, "ETHUSDT": 3515, "SOLUSDT": 184}
                base_price = base_prices.get(symbol, 100)
                price = base_price * (1 + np.random.uniform(-0.001, 0.001))

                fill = {
                    "stream_id": f"mock_{i}",
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "size": np.random.uniform(0.001, 0.1),
                    "timestamp": timestamp,
                    "quote_id": f"quote_{i}",
                }
                mock_fills.append(fill)

            return mock_fills

        except Exception as e:
            logger.error(f"Error generating mock fills: {e}")
            return []

    def get_mid_price(self, symbol: str, timestamp: float) -> Optional[float]:
        """Get mid price for symbol at timestamp."""
        try:
            # Try to get from Redis (current mid)
            mid_key = f"price:{symbol.lower()}:mid"
            mid_price = self.redis.get(mid_key)

            if mid_price:
                return float(mid_price)

            # Mock mid prices
            base_prices = {"BTCUSDT": 97600, "ETHUSDT": 3515, "SOLUSDT": 184}
            base_price = base_prices.get(symbol, 100)

            # Add some time-based drift
            time_drift = (timestamp - time.time()) * 0.0001  # Small drift
            return base_price * (1 + time_drift)

        except Exception as e:
            logger.debug(f"Error getting mid price for {symbol}: {e}")
            return None

    def calculate_fill_markout(
        self, fill: Dict[str, Any], markout_seconds: int
    ) -> Optional[float]:
        """Calculate markout for a fill."""
        try:
            fill_timestamp = fill["timestamp"]
            fill_price = fill["price"]
            fill_side = fill["side"]
            symbol = fill["symbol"]

            # Get mid price at fill time
            mid_at_fill = self.get_mid_price(symbol, fill_timestamp)
            if not mid_at_fill:
                return None

            # Get mid price after markout period
            markout_timestamp = fill_timestamp + markout_seconds
            mid_at_markout = self.get_mid_price(symbol, markout_timestamp)
            if not mid_at_markout:
                return None

            # Calculate markout in basis points
            if fill_side.lower() == "bid":  # We bought (filled our bid)
                # Markout = (mid_future - fill_price) / fill_price * 10000
                markout_bps = (mid_at_markout - fill_price) / fill_price * 10000
            else:  # We sold (filled our ask)
                # Markout = (fill_price - mid_future) / fill_price * 10000
                markout_bps = (fill_price - mid_at_markout) / fill_price * 10000

            return markout_bps

        except Exception as e:
            logger.debug(f"Error calculating markout: {e}")
            return None

    def update_markout_metrics(self):
        """Update markout metrics from recent fills."""
        try:
            # Get recent fills
            recent_fills = self.get_recent_fills()

            if not recent_fills:
                return

            # Update fill history
            for fill in recent_fills:
                if fill not in self.fill_history:
                    self.fill_history.append(fill)

            # Calculate markouts for recent fills
            current_time = time.time()
            markout_window = 300  # Look at fills from last 5 minutes

            relevant_fills = [
                f
                for f in recent_fills
                if current_time - f["timestamp"] < markout_window
            ]

            if len(relevant_fills) < self.config["min_fills_for_update"]:
                return

            # Calculate markouts for each period
            for period in self.config["markout_periods"]:
                markouts = []

                for fill in relevant_fills:
                    # Only calculate markout for fills old enough
                    if current_time - fill["timestamp"] >= period:
                        markout = self.calculate_fill_markout(fill, period)
                        if markout is not None:
                            markouts.append(markout)

                if markouts:
                    # Calculate average markout
                    avg_markout = np.mean(markouts)

                    # Update smoothed markout
                    alpha = self.config["smoothing_alpha"]
                    self.smoothed_markouts[period] = (
                        alpha * self.smoothed_markouts[period]
                        + (1 - alpha) * avg_markout
                    )

                    # Store in history
                    self.markout_history[period].append(avg_markout)

                    # Update Redis
                    self.redis.set(f"mm:markout_{period}s", avg_markout)

                    logger.debug(
                        f"Markout {period}s: {avg_markout:.2f}bps (smoothed: {self.smoothed_markouts[period]:.2f}bps)"
                    )

        except Exception as e:
            logger.error(f"Error updating markout metrics: {e}")

    def should_adjust_parameters(self) -> Tuple[bool, str, Dict[str, float]]:
        """Determine if parameters should be adjusted."""
        try:
            mo1 = self.smoothed_markouts[1]
            mo5 = self.smoothed_markouts[5]
            thresholds = self.config["markout_thresholds"]

            adjustments = {"gamma": 0.0, "k": 0.0}

            # Check for negative markouts (adverse selection)
            if mo1 < thresholds["mo1_negative"] or mo5 < thresholds["mo5_negative"]:
                # Increase risk aversion (higher gamma, wider spreads)
                adjustments["gamma"] = self.config["step_size_gamma"]
                adjustments["k"] = self.config["step_size_k"]
                reason = f"negative_markouts_mo1_{mo1:.1f}_mo5_{mo5:.1f}"
                return True, reason, adjustments

            # Check for positive markouts (profitable)
            elif mo1 > thresholds["mo1_positive"] and mo5 > thresholds["mo5_positive"]:
                # Decrease risk aversion (lower gamma, tighter spreads)
                adjustments["gamma"] = -self.config["step_size_gamma"]
                adjustments["k"] = -self.config["step_size_k"]
                reason = f"positive_markouts_mo1_{mo1:.1f}_mo5_{mo5:.1f}"
                return True, reason, adjustments

            # Check for toxic flow
            elif mo1 < self.config["toxic_flow_threshold"]:
                # Don't adjust parameters, but signal toxic flow
                reason = f"toxic_flow_mo1_{mo1:.1f}"
                return False, reason, adjustments

            return False, "no_adjustment_needed", adjustments

        except Exception as e:
            logger.error(f"Error checking parameter adjustment: {e}")
            return False, "error", {"gamma": 0.0, "k": 0.0}

    def apply_parameter_adjustments(
        self, adjustments: Dict[str, float], reason: str
    ) -> Dict[str, Any]:
        """Apply parameter adjustments with bounds checking."""
        try:
            adjustment_start = time.time()

            # Store previous values
            prev_gamma = self.gamma
            prev_k = self.k

            # Apply adjustments with bounds
            new_gamma = np.clip(
                self.gamma + adjustments["gamma"],
                self.config["gamma_bounds"][0],
                self.config["gamma_bounds"][1],
            )

            new_k = np.clip(
                self.k + adjustments["k"],
                self.config["k_bounds"][0],
                self.config["k_bounds"][1],
            )

            # Update parameters
            self.gamma = new_gamma
            self.k = new_k

            # Store in Redis
            self.redis.mset({"mm:gamma": self.gamma, "mm:k": self.k})

            # Update metrics
            self.metrics["mm_gamma"] = self.gamma
            self.metrics["mm_k"] = self.k
            self.metrics["mm_calib_events_total"] += 1
            self.total_calibrations += 1

            # Store calibration event
            calibration_record = {
                "timestamp": adjustment_start,
                "reason": reason,
                "previous": {"gamma": prev_gamma, "k": prev_k},
                "adjustments": adjustments,
                "new": {"gamma": self.gamma, "k": self.k},
                "markouts": self.smoothed_markouts.copy(),
                "bounded": {
                    "gamma": abs(new_gamma - (prev_gamma + adjustments["gamma"]))
                    > 1e-6,
                    "k": abs(new_k - (prev_k + adjustments["k"])) > 1e-6,
                },
            }

            self.parameter_history.append(calibration_record)

            # Trim history
            if len(self.parameter_history) > 1000:
                self.parameter_history = self.parameter_history[-500:]

            # Log calibration
            logger.info(
                f"🎛️ MM Parameter Calibration #{self.total_calibrations}: "
                f"γ {prev_gamma:.3f}→{self.gamma:.3f}, "
                f"k {prev_k:.3f}→{self.k:.3f} "
                f"({reason})"
            )

            return {
                "timestamp": adjustment_start,
                "status": "completed",
                "reason": reason,
                "previous_gamma": prev_gamma,
                "new_gamma": self.gamma,
                "previous_k": prev_k,
                "new_k": self.k,
                "markouts": self.smoothed_markouts.copy(),
                "bounded": calibration_record["bounded"],
            }

        except Exception as e:
            logger.error(f"Error applying parameter adjustments: {e}")
            return {"timestamp": time.time(), "status": "error", "error": str(e)}

    def check_toxic_flow(self) -> Tuple[bool, str]:
        """Check if toxic flow is detected."""
        try:
            mo1 = self.smoothed_markouts[1]
            toxic_threshold = self.config["toxic_flow_threshold"]

            if mo1 < toxic_threshold:
                return True, f"toxic_flow_mo1_{mo1:.1f}"

            return False, "no_toxic_flow"

        except Exception as e:
            logger.error(f"Error checking toxic flow: {e}")
            return False, "error"

    def calibration_cycle(self) -> Dict[str, Any]:
        """Run one calibration cycle."""
        try:
            cycle_start = time.time()
            self.total_updates += 1

            # Update markout metrics
            self.update_markout_metrics()

            # Update metrics in Redis
            self.metrics["mm_markout_1s"] = self.smoothed_markouts[1]
            self.metrics["mm_markout_5s"] = self.smoothed_markouts[5]

            # Check for toxic flow
            is_toxic, toxic_reason = self.check_toxic_flow()
            self.metrics["mm_toxic_flow_detected"] = 1.0 if is_toxic else 0.0

            if is_toxic:
                logger.warning(f"🚨 Toxic flow detected: {toxic_reason}")
                # Could set flag to stop quoting temporarily
                self.redis.set("mm:toxic_flow_flag", 1)
            else:
                self.redis.delete("mm:toxic_flow_flag")

            # Check if parameters should be adjusted
            should_adjust, reason, adjustments = self.should_adjust_parameters()

            adjustment_result = None
            if should_adjust:
                adjustment_result = self.apply_parameter_adjustments(
                    adjustments, reason
                )

            # Store all metrics in Redis
            for metric, value in self.metrics.items():
                self.redis.set(f"metric:{metric}", value)

            self.last_update = cycle_start
            cycle_duration = time.time() - cycle_start

            result = {
                "timestamp": cycle_start,
                "status": "completed",
                "total_updates": self.total_updates,
                "current_gamma": self.gamma,
                "current_k": self.k,
                "markouts": self.smoothed_markouts.copy(),
                "toxic_flow_detected": is_toxic,
                "parameter_adjusted": should_adjust,
                "adjustment_reason": reason if should_adjust else None,
                "adjustment_result": adjustment_result,
                "cycle_duration": cycle_duration,
            }

            return result

        except Exception as e:
            logger.error(f"Error in calibration cycle: {e}")
            return {"timestamp": time.time(), "status": "error", "error": str(e)}

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        try:
            # Calculate statistics
            if len(self.parameter_history) > 1:
                recent_calibrations = self.parameter_history[-10:]
                gamma_changes = [
                    abs(c["new"]["gamma"] - c["previous"]["gamma"])
                    for c in recent_calibrations
                ]
                k_changes = [
                    abs(c["new"]["k"] - c["previous"]["k"]) for c in recent_calibrations
                ]
                avg_gamma_change = np.mean(gamma_changes) if gamma_changes else 0
                avg_k_change = np.mean(k_changes) if k_changes else 0
            else:
                avg_gamma_change = 0
                avg_k_change = 0

            status = {
                "service": "mm_online_calibrator",
                "timestamp": time.time(),
                "config": self.config,
                "current_parameters": {"gamma": self.gamma, "k": self.k},
                "current_markouts": self.smoothed_markouts.copy(),
                "metrics": self.metrics.copy(),
                "statistics": {
                    "total_updates": self.total_updates,
                    "total_calibrations": self.total_calibrations,
                    "last_update": self.last_update,
                    "avg_gamma_change": avg_gamma_change,
                    "avg_k_change": avg_k_change,
                    "parameter_history_length": len(self.parameter_history),
                    "fill_history_length": len(self.fill_history),
                },
                "recent_calibrations": (
                    self.parameter_history[-3:] if self.parameter_history else []
                ),
                "toxic_flow_detected": self.metrics["mm_toxic_flow_detected"] > 0.5,
            }

            return status

        except Exception as e:
            return {
                "service": "mm_online_calibrator",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }

    def run_continuous_calibration(self):
        """Run continuous MM parameter calibration."""
        logger.info("🎛️ Starting continuous MM parameter calibration")

        try:
            while True:
                try:
                    # Run calibration cycle
                    result = self.calibration_cycle()

                    if result["status"] == "completed":
                        if result.get("parameter_adjusted"):
                            logger.debug(
                                f"📊 Cycle #{self.total_updates}: "
                                f"γ={self.gamma:.3f}, k={self.k:.3f} "
                                f"(MO1: {self.smoothed_markouts[1]:.1f}, MO5: {self.smoothed_markouts[5]:.1f})"
                            )
                        elif self.total_updates % 12 == 0:  # Every minute
                            logger.debug(
                                f"📊 Cycle #{self.total_updates}: "
                                f"γ={self.gamma:.3f}, k={self.k:.3f}, "
                                f"MO1: {self.smoothed_markouts[1]:.1f}bps"
                            )

                    # Sleep until next cycle
                    time.sleep(self.config["update_interval"])

                except Exception as e:
                    logger.error(f"Error in calibration loop: {e}")
                    time.sleep(30)  # Wait longer on error

        except KeyboardInterrupt:
            logger.info("MM calibrator stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in calibration loop: {e}")


def main():
    """Main entry point for MM calibrator."""
    import argparse

    parser = argparse.ArgumentParser(description="MM Online Calibrator")
    parser.add_argument("--run", action="store_true", help="Run continuous calibration")
    parser.add_argument(
        "--cycle", action="store_true", help="Run single calibration cycle"
    )
    parser.add_argument("--status", action="store_true", help="Show status report")

    args = parser.parse_args()

    # Create calibrator
    calibrator = MMOnlineCalibrator()

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
