#!/usr/bin/env python3
"""
Strategy Guardrail Daemon
Auto-halt any strategy (RL, BASIS, MM) if P&L or quality breaches thresholds
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("strategy_guard")


class StrategyGuardrailDaemon:
    """Per-strategy circuit breakers and guardrails."""

    def __init__(self):
        """Initialize strategy guardrail daemon."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.slack_webhook = os.getenv("SLACK_WEBHOOK", "")

        # Strategy configurations
        self.strategies = ["RL", "BASIS", "MM"]
        self.default_config = {
            "RL": {
                "dd60": -0.007,  # 60m drawdown threshold (-0.7%)
                "dd24h": -0.015,  # 24h drawdown threshold (-1.5%)
                "slip": 25.0,  # Slippage threshold (25 bps)
                "slip_duration": 15,  # Minutes to check slippage
                "enabled": True,
            },
            "BASIS": {
                "dd60": -0.007,  # 60m drawdown threshold (-0.7%)
                "dd24h": -0.012,  # 24h drawdown threshold (-1.2%)
                "enabled": True,
            },
            "MM": {
                "dd60": -0.005,  # 60m drawdown threshold (-0.5%)
                "dd24h": -0.010,  # 24h drawdown threshold (-1.0%)
                "slip": 20.0,  # Slippage threshold (20 bps)
                "slip_duration": 15,  # Minutes to check slippage
                "enabled": True,
            },
        }

        # Global thresholds
        self.global_config = {
            "max_trips_24h": 2,  # Max guard trips in 24h before disable
            "exec_timeout_threshold": 5,  # Max execution timeouts per hour
            "recon_breach_threshold": 2,  # Max recon breaches per hour
            "check_interval": 10,  # Check every 10 seconds
        }

        # State tracking
        self.guard_trips = {}  # strategy -> list of trip timestamps
        self.disabled_strategies = set()
        self.last_check = 0
        self.total_checks = 0
        self.total_trips = 0

        # Metrics
        self.metrics = {}
        for strategy in self.strategies:
            self.metrics[f"strategy_guard_trips_total_{strategy.lower()}"] = 0

        # Load configuration from Redis
        self._load_config_from_redis()

        logger.info("🛡️ Strategy Guardrail Daemon initialized")
        logger.info(f"   Strategies: {self.strategies}")
        logger.info(f"   Check interval: {self.global_config['check_interval']}s")
        logger.info(f"   Max trips 24h: {self.global_config['max_trips_24h']}")

    def _load_config_from_redis(self):
        """Load configuration from Redis parameter server."""
        try:
            for strategy in self.strategies:
                config_key = f"strategy_guard:config:{strategy}"
                config_data = self.redis.hgetall(config_key)

                if config_data:
                    # Update config with Redis values
                    for key, value in config_data.items():
                        try:
                            if key in ["enabled"]:
                                self.default_config[strategy][key] = bool(int(value))
                            else:
                                self.default_config[strategy][key] = float(value)
                        except (ValueError, KeyError):
                            logger.warning(
                                f"Invalid config value for {strategy}.{key}: {value}"
                            )

                logger.debug(f"Config for {strategy}: {self.default_config[strategy]}")

        except Exception as e:
            logger.warning(f"Error loading config from Redis: {e}")

    def get_metric_value(self, key: str, default: float = 0.0) -> float:
        """Get metric value from Redis."""
        try:
            value = self.redis.get(key)
            return float(value) if value else default
        except Exception as e:
            logger.debug(f"Error getting metric {key}: {e}")
            return default

    def get_strategy_pnl_60m(self, strategy: str) -> float:
        """Get 60-minute P&L for strategy."""
        try:
            # Try to get from Redis
            pnl_key = f"strategy:{strategy}:pnl_60m"
            pnl_60m = self.redis.get(pnl_key)

            if pnl_60m:
                return float(pnl_60m)

            # Fallback: calculate from recent P&L data
            pnl_series_key = f"strategy:{strategy}:pnl_series"
            pnl_data = self.redis.lrange(pnl_series_key, -60, -1)  # Last 60 minutes

            if pnl_data:
                pnl_values = [float(p) for p in pnl_data]
                return sum(pnl_values)

            # Mock data for demo
            if strategy == "RL":
                return np.random.normal(-0.002, 0.008)  # Slight negative drift
            elif strategy == "BASIS":
                return np.random.normal(0.001, 0.005)  # Slight positive
            else:  # MM
                return np.random.normal(0.0005, 0.003)  # Small positive

        except Exception as e:
            logger.error(f"Error getting 60m P&L for {strategy}: {e}")
            return 0.0

    def get_strategy_drawdown(self, strategy: str, period: str) -> float:
        """Get drawdown for strategy over period."""
        try:
            dd_key = f"strategy:{strategy}:dd_{period}"
            drawdown = self.redis.get(dd_key)

            if drawdown:
                return float(drawdown)

            # Calculate from P&L if available
            pnl_60m = self.get_strategy_pnl_60m(strategy)

            # Simple approximation: assume drawdown = max(0, -pnl)
            if period == "60m":
                return max(0, -pnl_60m)
            else:  # 24h
                # Scale up roughly
                return max(0, -pnl_60m * 2.5)

        except Exception as e:
            logger.error(f"Error getting drawdown for {strategy} {period}: {e}")
            return 0.0

    def get_strategy_slippage(self, strategy: str, duration_minutes: int) -> float:
        """Get average slippage for strategy over duration."""
        try:
            slip_key = f"strategy:{strategy}:slippage_bps_{duration_minutes}m"
            slippage = self.redis.get(slip_key)

            if slippage:
                return float(slippage)

            # Mock slippage data based on strategy
            if strategy == "RL":
                return np.random.uniform(5, 35)  # Higher variance for RL
            elif strategy == "MM":
                return np.random.uniform(2, 15)  # Lower for MM
            else:
                return np.random.uniform(3, 20)  # Medium for basis

        except Exception as e:
            logger.error(f"Error getting slippage for {strategy}: {e}")
            return 0.0

    def get_global_metrics(self) -> Dict[str, float]:
        """Get global execution and reconciliation metrics."""
        try:
            metrics = {}

            # Execution timeouts per hour
            exec_timeouts = self.get_metric_value("exec:timeouts_1h", 0)
            metrics["exec_timeouts"] = exec_timeouts

            # Reconciliation breaches per hour
            recon_breaches = self.get_metric_value("recon:breaches_1h", 0)
            metrics["recon_breaches"] = recon_breaches

            # System mode
            mode = self.redis.get("mode")
            metrics["halt_mode"] = 1.0 if mode == "halt" else 0.0

            return metrics

        except Exception as e:
            logger.error(f"Error getting global metrics: {e}")
            return {"exec_timeouts": 0, "recon_breaches": 0, "halt_mode": 0}

    def check_trip_history(self, strategy: str) -> Tuple[bool, int]:
        """Check if strategy has too many trips in 24h."""
        try:
            current_time = time.time()
            cutoff_time = current_time - 24 * 3600  # 24 hours ago

            # Get trip history for strategy
            if strategy not in self.guard_trips:
                self.guard_trips[strategy] = []

            # Filter trips to last 24 hours
            recent_trips = [t for t in self.guard_trips[strategy] if t > cutoff_time]
            self.guard_trips[strategy] = recent_trips

            trip_count = len(recent_trips)
            max_trips = self.global_config["max_trips_24h"]

            should_disable = trip_count >= max_trips

            return should_disable, trip_count

        except Exception as e:
            logger.error(f"Error checking trip history for {strategy}: {e}")
            return False, 0

    def trip_strategy(self, strategy: str, reason: str) -> Dict[str, Any]:
        """Trip/disable a strategy."""
        try:
            current_time = time.time()

            # Disable strategy in feature flags
            flag_key = f"STRAT_{strategy}"
            self.redis.hset("features:flags", flag_key, 0)

            # Add to disabled strategies
            self.disabled_strategies.add(strategy)

            # Log trip event in Redis stream
            trip_event = {
                "timestamp": current_time,
                "strategy": strategy,
                "reason": reason,
                "action": "disabled",
            }

            self.redis.xadd("strategy:guards", trip_event)

            # Update trip history
            if strategy not in self.guard_trips:
                self.guard_trips[strategy] = []

            self.guard_trips[strategy].append(current_time)

            # Update metrics
            metric_key = f"strategy_guard_trips_total_{strategy.lower()}"
            self.metrics[metric_key] = self.metrics.get(metric_key, 0) + 1
            self.redis.set(f"metric:{metric_key}", self.metrics[metric_key])

            self.total_trips += 1

            # Send Slack notification
            if self.slack_webhook:
                try:
                    message = (
                        f"⛔ *Strategy {strategy} Disabled*\n"
                        f"🔍 Reason: {reason.replace('_', ' ').title()}\n"
                        f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"🔢 Total trips today: {len(self.guard_trips[strategy])}"
                    )

                    payload = {
                        "text": message,
                        "username": "Strategy Guardrails",
                        "icon_emoji": ":shield:",
                    }

                    response = requests.post(
                        self.slack_webhook, json=payload, timeout=10
                    )
                    response.raise_for_status()

                except Exception as e:
                    logger.error(f"Error sending Slack notification: {e}")

            logger.critical(
                f"🛑 STRATEGY {strategy} DISABLED: {reason} "
                f"(trip #{len(self.guard_trips[strategy])} today)"
            )

            return {
                "strategy": strategy,
                "reason": reason,
                "timestamp": current_time,
                "action": "disabled",
                "trip_count_24h": len(self.guard_trips[strategy]),
            }

        except Exception as e:
            logger.error(f"Error tripping strategy {strategy}: {e}")
            return {
                "strategy": strategy,
                "reason": reason,
                "action": "error",
                "error": str(e),
            }

    def check_strategy_guardrails(self, strategy: str) -> List[Dict[str, Any]]:
        """Check all guardrails for a single strategy."""
        try:
            config = self.default_config.get(strategy, {})
            if not config.get("enabled", True):
                return []

            # Skip if already disabled
            if strategy in self.disabled_strategies:
                return []

            # Check if too many trips in 24h
            should_disable_history, trip_count = self.check_trip_history(strategy)
            if should_disable_history:
                return [self.trip_strategy(strategy, f"max_trips_24h_{trip_count}")]

            violations = []

            # Check 60-minute drawdown
            if "dd60" in config:
                dd_60m = self.get_strategy_drawdown(strategy, "60m")
                if dd_60m >= abs(config["dd60"]):  # Convert to positive for comparison
                    violations.append(self.trip_strategy(strategy, "drawdown_60m"))

            # Check 24-hour drawdown
            if "dd24h" in config:
                dd_24h = self.get_strategy_drawdown(strategy, "24h")
                if dd_24h >= abs(config["dd24h"]):
                    violations.append(self.trip_strategy(strategy, "drawdown_24h"))

            # Check slippage (for MM and RL)
            if "slip" in config:
                slip_duration = config.get("slip_duration", 15)
                slippage_bps = self.get_strategy_slippage(strategy, slip_duration)
                if slippage_bps > config["slip"]:
                    violations.append(self.trip_strategy(strategy, "slippage"))

            return violations

        except Exception as e:
            logger.error(f"Error checking guardrails for {strategy}: {e}")
            return []

    def check_global_guardrails(self) -> List[Dict[str, Any]]:
        """Check global system guardrails."""
        try:
            violations = []
            global_metrics = self.get_global_metrics()

            # Check execution timeouts
            exec_timeouts = global_metrics.get("exec_timeouts", 0)
            if exec_timeouts > self.global_config["exec_timeout_threshold"]:
                logger.warning(f"🚨 High execution timeouts: {exec_timeouts}/hour")
                # Could disable all strategies or specific actions here

            # Check reconciliation breaches
            recon_breaches = global_metrics.get("recon_breaches", 0)
            if recon_breaches > self.global_config["recon_breach_threshold"]:
                logger.warning(
                    f"🚨 High reconciliation breaches: {recon_breaches}/hour"
                )

            # Check if system is in halt mode
            if global_metrics.get("halt_mode", 0) > 0.5:
                logger.warning("🛑 System in halt mode")
                # All strategies should already be disabled by system halt

            return violations

        except Exception as e:
            logger.error(f"Error checking global guardrails: {e}")
            return []

    def run_guardrail_check(self) -> Dict[str, Any]:
        """Run complete guardrail check cycle."""
        try:
            check_start = time.time()
            self.total_checks += 1

            # Reload configuration from Redis periodically
            if self.total_checks % 60 == 0:  # Every 10 minutes
                self._load_config_from_redis()

            # Check each strategy
            all_violations = []
            strategy_results = {}

            for strategy in self.strategies:
                violations = self.check_strategy_guardrails(strategy)
                strategy_results[strategy] = {
                    "violations": len(violations),
                    "disabled": strategy in self.disabled_strategies,
                    "trip_count_24h": len(self.guard_trips.get(strategy, [])),
                }
                all_violations.extend(violations)

            # Check global guardrails
            global_violations = self.check_global_guardrails()
            all_violations.extend(global_violations)

            # Update last check time
            self.last_check = check_start

            check_duration = time.time() - check_start

            # Log summary if violations occurred
            if all_violations:
                logger.warning(
                    f"⚠️ Guardrail check #{self.total_checks}: "
                    f"{len(all_violations)} violations detected"
                )

            result = {
                "timestamp": check_start,
                "status": "completed",
                "total_checks": self.total_checks,
                "violations": len(all_violations),
                "strategy_results": strategy_results,
                "disabled_strategies": list(self.disabled_strategies),
                "violations_details": all_violations,
                "check_duration": check_duration,
            }

            return result

        except Exception as e:
            logger.error(f"Error in guardrail check: {e}")
            return {
                "timestamp": time.time(),
                "status": "error",
                "error": str(e),
                "total_checks": self.total_checks,
            }

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        try:
            current_time = time.time()

            # Calculate statistics
            active_strategies = [
                s for s in self.strategies if s not in self.disabled_strategies
            ]

            # Get recent trip history
            trip_summary = {}
            for strategy in self.strategies:
                trips_24h = len(self.guard_trips.get(strategy, []))
                trip_summary[strategy] = {
                    "trips_24h": trips_24h,
                    "disabled": strategy in self.disabled_strategies,
                    "config": self.default_config.get(strategy, {}),
                }

            status = {
                "service": "strategy_guardrail_daemon",
                "timestamp": current_time,
                "uptime_seconds": (
                    current_time
                    - (
                        self.last_check
                        - self.total_checks * self.global_config["check_interval"]
                    )
                    if self.last_check > 0
                    else 0
                ),
                "total_checks": self.total_checks,
                "total_trips": self.total_trips,
                "active_strategies": active_strategies,
                "disabled_strategies": list(self.disabled_strategies),
                "strategies_summary": trip_summary,
                "global_config": self.global_config,
                "metrics": self.metrics.copy(),
                "last_check": self.last_check,
                "last_check_ago": (
                    current_time - self.last_check if self.last_check > 0 else 0
                ),
            }

            return status

        except Exception as e:
            return {
                "service": "strategy_guardrail_daemon",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }

    def manual_reset_strategy(self, strategy: str) -> Dict[str, Any]:
        """Manually reset/re-enable a strategy."""
        try:
            if strategy not in self.strategies:
                return {
                    "strategy": strategy,
                    "status": "error",
                    "error": f"Unknown strategy: {strategy}",
                }

            # Re-enable strategy flag
            flag_key = f"STRAT_{strategy}"
            self.redis.hset("features:flags", flag_key, 1)

            # Remove from disabled set
            self.disabled_strategies.discard(strategy)

            # Clear trip history (optional)
            if strategy in self.guard_trips:
                self.guard_trips[strategy] = []

            # Log reset event
            reset_event = {
                "timestamp": time.time(),
                "strategy": strategy,
                "reason": "manual_reset",
                "action": "enabled",
            }

            self.redis.xadd("strategy:guards", reset_event)

            logger.info(f"✅ Strategy {strategy} manually reset and re-enabled")

            return {
                "strategy": strategy,
                "status": "success",
                "action": "enabled",
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error resetting strategy {strategy}: {e}")
            return {"strategy": strategy, "status": "error", "error": str(e)}

    def run_continuous_monitoring(self):
        """Run continuous guardrail monitoring."""
        logger.info("🛡️ Starting continuous strategy guardrail monitoring")

        try:
            while True:
                try:
                    # Run guardrail check
                    result = self.run_guardrail_check()

                    if result["status"] == "completed":
                        violations = result.get("violations", 0)
                        disabled_count = len(result.get("disabled_strategies", []))

                        if violations > 0 or disabled_count > 0:
                            logger.info(
                                f"🔍 Check #{self.total_checks}: "
                                f"{violations} violations, {disabled_count} disabled strategies"
                            )

                    # Sleep until next check
                    time.sleep(self.global_config["check_interval"])

                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(30)  # Wait longer on error

        except KeyboardInterrupt:
            logger.info("Strategy guardrail daemon stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {e}")


def main():
    """Main entry point for strategy guardrail daemon."""
    import argparse

    parser = argparse.ArgumentParser(description="Strategy Guardrail Daemon")
    parser.add_argument("--run", action="store_true", help="Run continuous monitoring")
    parser.add_argument(
        "--check", action="store_true", help="Run single guardrail check"
    )
    parser.add_argument("--status", action="store_true", help="Show status report")
    parser.add_argument(
        "--reset", choices=["RL", "BASIS", "MM"], help="Manually reset strategy"
    )

    args = parser.parse_args()

    # Create guardrail daemon
    guardian = StrategyGuardrailDaemon()

    if args.status:
        # Show status report
        status = guardian.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.reset:
        # Reset strategy
        result = guardian.manual_reset_strategy(args.reset)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.check:
        # Run single check
        result = guardian.run_guardrail_check()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.run:
        # Run continuous monitoring
        guardian.run_continuous_monitoring()
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
