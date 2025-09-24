#!/usr/bin/env python3
"""
Capital-Cap Enforcer
Smooth, rate-limited capital allocation to prevent sawtooth allocation changes
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
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import redis
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("capital_cap_enforcer")


class CapitalCapEnforcer:
    """Capital cap enforcement with smooth rate limiting."""

    def __init__(self):
        """Initialize capital cap enforcer."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.slack_webhook = os.getenv("SLACK_WEBHOOK", "")

        # Configuration
        self.config = {
            "rate_limit_pct_per_hour": 0.10,  # Maximum 10% change per hour
            "update_interval_minutes": 5,  # Update every 5 minutes
            "floor_allocation": 0.10,  # Minimum 10% when halted/drift
            "default_cap": 0.70,  # Default weekly cap (70%)
            "default_factor": 0.40,  # Default auto-ramp factor (40%)
            "smoothing_alpha": 0.85,  # EMA smoothing for changes
            "breach_threshold": 0.05,  # Log breach if change > 5%
            "history_length": 1440,  # Keep 24 hours of history (5min intervals)
        }

        # State tracking
        self.current_effective_cap = 0.40  # Start at 40%
        self.previous_effective_cap = 0.40
        self.target_cap = 0.40
        self.last_update = 0
        self.total_updates = 0
        self.cap_history = []
        self.breach_events = []

        # Initialize with Redis state if available
        self._load_state_from_redis()

        logger.info("🛡️ Capital Cap Enforcer initialized")
        logger.info(f"   Current effective cap: {self.current_effective_cap:.1%}")
        logger.info(f"   Rate limit: {self.config['rate_limit_pct_per_hour']:.0%}/hour")
        logger.info(f"   Update interval: {self.config['update_interval_minutes']}min")
        logger.info(f"   Floor allocation: {self.config['floor_allocation']:.0%}")

    def _load_state_from_redis(self):
        """Load current state from Redis."""
        try:
            # Get current effective cap
            effective_cap = self.redis.get("risk:capital_effective")
            if effective_cap:
                self.current_effective_cap = float(effective_cap)
                self.previous_effective_cap = self.current_effective_cap

            # Get last update timestamp
            last_update_ts = self.redis.get("capital_enforcer:last_update")
            if last_update_ts:
                self.last_update = float(last_update_ts)

            # Get total updates count
            update_count = self.redis.get("capital_enforcer:update_count")
            if update_count:
                self.total_updates = int(update_count)

            logger.debug(
                f"Loaded state from Redis: cap={self.current_effective_cap:.1%}, updates={self.total_updates}"
            )

        except Exception as e:
            logger.warning(f"Error loading state from Redis: {e}")

    def get_input_signals(self) -> Dict[str, float]:
        """Get input signals for capital cap calculation."""
        try:
            signals = {}

            # Get weekly cap from SLO gate
            cap_next_week = self.redis.get("risk:capital_cap_next_week")
            signals["cap_next_week"] = (
                float(cap_next_week) if cap_next_week else self.config["default_cap"]
            )

            # Get auto-ramp factor (hourly updated)
            capital_factor = self.redis.get("risk:capital_factor")
            signals["capital_factor"] = (
                float(capital_factor)
                if capital_factor
                else self.config["default_factor"]
            )

            # Check system state flags
            mode = self.redis.get("mode")
            signals["halt_mode"] = mode == "halt"

            drift_flag = self.redis.get("drift:flag")
            signals["drift_detected"] = bool(drift_flag) if drift_flag else False

            # Get additional risk metrics
            drawdown = self.redis.get("risk:max_drawdown")
            signals["max_drawdown"] = float(drawdown) if drawdown else 0.0

            sharpe = self.redis.get("metrics:sharpe_7d")
            signals["sharpe_7d"] = float(sharpe) if sharpe else 1.0

            logger.debug(f"Input signals: {signals}")
            return signals

        except Exception as e:
            logger.error(f"Error getting input signals: {e}")
            return {
                "cap_next_week": self.config["default_cap"],
                "capital_factor": self.config["default_factor"],
                "halt_mode": False,
                "drift_detected": False,
                "max_drawdown": 0.0,
                "sharpe_7d": 1.0,
            }

    def calculate_target_cap(self, signals: Dict[str, float]) -> Tuple[float, str]:
        """Calculate target capital allocation."""
        try:
            cap_next_week = signals["cap_next_week"]
            capital_factor = signals["capital_factor"]
            halt_mode = signals["halt_mode"]
            drift_detected = signals["drift_detected"]

            # Check if we should apply floor allocation
            if halt_mode or drift_detected:
                target = self.config["floor_allocation"]
                reason = "floor_allocation_" + ("halt" if halt_mode else "drift")
            else:
                # Normal operation: take minimum of weekly cap and auto-ramp factor
                target = min(cap_next_week, capital_factor)
                reason = "normal_operation"

            # Additional risk-based adjustments
            max_dd = signals.get("max_drawdown", 0.0)
            if max_dd > 0.05:  # > 5% drawdown
                adjustment = max(0.5, 1.0 - (max_dd - 0.05))  # Reduce allocation
                target *= adjustment
                reason += f"_dd_adjustment_{adjustment:.2f}"

            # Sharpe-based boost (if performing very well)
            sharpe = signals.get("sharpe_7d", 1.0)
            if sharpe > 2.0 and not halt_mode and target > 0.5:
                boost = min(1.2, 1.0 + (sharpe - 2.0) * 0.1)  # Max 20% boost
                target = min(1.0, target * boost)
                reason += f"_sharpe_boost_{boost:.2f}"

            # Ensure bounds
            target = max(0.05, min(1.0, target))  # Between 5% and 100%

            logger.debug(f"Target calculation: {target:.1%} (reason: {reason})")
            return target, reason

        except Exception as e:
            logger.error(f"Error calculating target cap: {e}")
            return self.config["floor_allocation"], "error_fallback"

    def calculate_rate_limited_step(
        self, current: float, target: float, dt_hours: float
    ) -> float:
        """Calculate rate-limited step towards target."""
        try:
            if dt_hours <= 0:
                return current

            # Maximum change allowed in this time period
            max_change_per_hour = self.config["rate_limit_pct_per_hour"]
            max_change = max_change_per_hour * dt_hours

            # Calculate desired change
            desired_change = target - current

            # Apply rate limiting
            if abs(desired_change) <= max_change:
                # Change is within limits
                new_value = target
            else:
                # Apply rate limit
                change_direction = 1 if desired_change > 0 else -1
                limited_change = change_direction * max_change
                new_value = current + limited_change

            # Ensure bounds
            new_value = max(0.05, min(1.0, new_value))

            logger.debug(
                f"Rate limiting: current={current:.1%}, target={target:.1%}, "
                f"dt={dt_hours:.2f}h, max_change={max_change:.1%}, new={new_value:.1%}"
            )

            return new_value

        except Exception as e:
            logger.error(f"Error calculating rate-limited step: {e}")
            return current

    def update_capital_cap(self) -> Dict[str, Any]:
        """Update effective capital cap with rate limiting."""
        try:
            update_start = time.time()
            self.total_updates += 1

            # Calculate time since last update
            dt_seconds = update_start - self.last_update if self.last_update > 0 else 0
            dt_hours = dt_seconds / 3600.0

            # Get input signals
            signals = self.get_input_signals()

            # Calculate target cap
            target_cap, target_reason = self.calculate_target_cap(signals)

            # Apply smoothing to target (EMA)
            alpha = self.config["smoothing_alpha"]
            if hasattr(self, "target_cap"):
                smoothed_target = alpha * self.target_cap + (1 - alpha) * target_cap
            else:
                smoothed_target = target_cap

            self.target_cap = smoothed_target

            # Calculate rate-limited effective cap
            new_effective_cap = self.calculate_rate_limited_step(
                self.current_effective_cap, smoothed_target, dt_hours
            )

            # Calculate change metrics
            cap_change = new_effective_cap - self.current_effective_cap
            cap_change_pct = (
                abs(cap_change / self.current_effective_cap)
                if self.current_effective_cap > 0
                else 0
            )

            # Update state
            self.previous_effective_cap = self.current_effective_cap
            self.current_effective_cap = new_effective_cap
            self.last_update = update_start

            # Store in Redis
            self.redis.set("risk:capital_effective", new_effective_cap)
            self.redis.set("capital_enforcer:last_update", update_start)
            self.redis.set("capital_enforcer:update_count", self.total_updates)

            # Store additional metrics
            self.redis.set("capital_enforcer:target", smoothed_target)
            self.redis.set(
                "capital_enforcer:change_rate",
                cap_change / dt_hours if dt_hours > 0 else 0,
            )

            # Track history
            history_record = {
                "timestamp": update_start,
                "previous": self.previous_effective_cap,
                "current": new_effective_cap,
                "target": smoothed_target,
                "target_reason": target_reason,
                "change": cap_change,
                "change_pct": cap_change_pct,
                "dt_hours": dt_hours,
                "signals": signals.copy(),
            }

            self.cap_history.append(history_record)

            # Trim history
            if len(self.cap_history) > self.config["history_length"]:
                self.cap_history = self.cap_history[
                    -self.config["history_length"] // 2 :
                ]

            # Check for breach events
            is_breach = cap_change_pct > self.config["breach_threshold"]

            if is_breach:
                breach_event = {
                    "timestamp": update_start,
                    "change_pct": cap_change_pct,
                    "from_cap": self.previous_effective_cap,
                    "to_cap": new_effective_cap,
                    "reason": target_reason,
                }
                self.breach_events.append(breach_event)

                if len(self.breach_events) > 100:
                    self.breach_events = self.breach_events[-50:]

                logger.warning(
                    f"⚠️ Capital cap breach: {self.previous_effective_cap:.1%} → {new_effective_cap:.1%} "
                    f"({cap_change_pct:.1%} change)"
                )

            # Log update
            if abs(cap_change) > 0.001:  # Log if change > 0.1%
                logger.info(
                    f"📊 Capital cap update #{self.total_updates}: "
                    f"{self.previous_effective_cap:.1%} → {new_effective_cap:.1%} "
                    f"(target: {smoothed_target:.1%}, reason: {target_reason})"
                )

            # Send notification for significant changes
            if is_breach or signals["halt_mode"] or signals["drift_detected"]:
                self.send_capital_notification(history_record)

            update_duration = time.time() - update_start

            result = {
                "timestamp": update_start,
                "status": "completed",
                "previous_cap": self.previous_effective_cap,
                "current_cap": new_effective_cap,
                "target_cap": smoothed_target,
                "target_reason": target_reason,
                "change": cap_change,
                "change_pct": cap_change_pct,
                "is_breach": is_breach,
                "rate_limited": abs(new_effective_cap - target_cap) > 1e-6,
                "signals": signals,
                "update_duration": update_duration,
                "total_updates": self.total_updates,
            }

            return result

        except Exception as e:
            logger.error(f"Error updating capital cap: {e}")
            return {
                "timestamp": time.time(),
                "status": "error",
                "error": str(e),
                "current_cap": self.current_effective_cap,
            }

    def send_capital_notification(self, update_record: Dict[str, Any]):
        """Send capital allocation change notification."""
        try:
            if not self.slack_webhook:
                return False

            current = update_record["current"]
            previous = update_record["previous"]
            target = update_record["target"]
            reason = update_record["target_reason"]
            change_pct = update_record["change_pct"]
            signals = update_record["signals"]

            # Determine notification urgency
            if signals.get("halt_mode"):
                icon = "🛑"
                urgency = "CRITICAL"
            elif signals.get("drift_detected"):
                icon = "⚠️"
                urgency = "WARNING"
            elif change_pct > 0.1:  # > 10% change
                icon = "📈" if current > previous else "📉"
                urgency = "INFO"
            else:
                icon = "💼"
                urgency = "DEBUG"

            message = (
                f"{icon} *Capital Allocation Update* [{urgency}]\n"
                f"🎯 Allocation: {previous:.1%} → {current:.1%} (target: {target:.1%})\n"
                f"📊 Change: {(current-previous):+.1%} ({change_pct:.1%})\n"
                f"🔍 Reason: {reason.replace('_', ' ').title()}\n\n"
                f"📋 *System State:*\n"
                f"• Weekly SLO Cap: {signals['cap_next_week']:.1%}\n"
                f"• Auto-ramp Factor: {signals['capital_factor']:.1%}\n"
                f"• Halt Mode: {'Yes' if signals['halt_mode'] else 'No'}\n"
                f"• Drift Detected: {'Yes' if signals['drift_detected'] else 'No'}\n"
                f"• Max Drawdown: {signals['max_drawdown']:.1%}\n"
                f"• 7d Sharpe: {signals['sharpe_7d']:.2f}"
            )

            payload = {
                "text": message,
                "username": "Capital Cap Enforcer",
                "icon_emoji": ":shield:",
            }

            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("📱 Sent capital allocation notification")
            return True

        except Exception as e:
            logger.error(f"Error sending capital notification: {e}")
            return False

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        try:
            current_signals = self.get_input_signals()

            # Calculate metrics
            if len(self.cap_history) > 1:
                recent_changes = [abs(h["change"]) for h in self.cap_history[-10:]]
                avg_change = np.mean(recent_changes)
                max_change = max(recent_changes)

                # Calculate velocity (change per hour)
                recent_history = [
                    h for h in self.cap_history if time.time() - h["timestamp"] < 3600
                ]
                if len(recent_history) >= 2:
                    velocity = sum(h["change"] for h in recent_history[-5:]) / len(
                        recent_history[-5:]
                    )
                else:
                    velocity = 0.0
            else:
                avg_change = 0.0
                max_change = 0.0
                velocity = 0.0

            status = {
                "service": "capital_cap_enforcer",
                "timestamp": time.time(),
                "config": self.config,
                "current_state": {
                    "effective_cap": self.current_effective_cap,
                    "target_cap": getattr(
                        self, "target_cap", self.current_effective_cap
                    ),
                    "previous_cap": self.previous_effective_cap,
                    "last_update": self.last_update,
                    "total_updates": self.total_updates,
                },
                "current_signals": current_signals,
                "metrics": {
                    "avg_change_10periods": avg_change,
                    "max_change_10periods": max_change,
                    "change_velocity_per_hour": velocity,
                    "total_breaches": len(self.breach_events),
                    "history_length": len(self.cap_history),
                },
                "recent_history": self.cap_history[-5:] if self.cap_history else [],
                "recent_breaches": (
                    self.breach_events[-3:] if self.breach_events else []
                ),
            }

            return status

        except Exception as e:
            return {
                "service": "capital_cap_enforcer",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }

    def run_continuous_enforcement(self):
        """Run continuous capital cap enforcement."""
        logger.info("🛡️ Starting continuous capital cap enforcement")

        try:
            while True:
                try:
                    # Update capital cap
                    result = self.update_capital_cap()

                    if result["status"] == "completed":
                        change_pct = result.get("change_pct", 0)
                        if change_pct > 0.001:  # Log if change > 0.1%
                            logger.debug(
                                f"💼 Update #{self.total_updates}: "
                                f"{result['current_cap']:.1%} cap "
                                f"({result['change']:+.1%} change)"
                            )

                    # Wait for next update
                    time.sleep(self.config["update_interval_minutes"] * 60)

                except Exception as e:
                    logger.error(f"Error in enforcement loop: {e}")
                    time.sleep(60)  # Wait longer on error

        except KeyboardInterrupt:
            logger.info("Capital cap enforcer stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in enforcement loop: {e}")


def main():
    """Main entry point for capital cap enforcer."""
    import argparse

    parser = argparse.ArgumentParser(description="Capital Cap Enforcer")
    parser.add_argument("--run", action="store_true", help="Run continuous enforcement")
    parser.add_argument("--update", action="store_true", help="Run single update")
    parser.add_argument("--status", action="store_true", help="Show status report")
    parser.add_argument(
        "--simulate", type=int, metavar="HOURS", help="Simulate enforcement for N hours"
    )

    args = parser.parse_args()

    # Create enforcer
    enforcer = CapitalCapEnforcer()

    if args.status:
        # Show status report
        status = enforcer.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.update:
        # Run single update
        result = enforcer.update_capital_cap()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.simulate:
        # Simulate enforcement
        print(f"Simulating {args.simulate} hours of capital cap enforcement...")

        for hour in range(args.simulate):
            for period in range(12):  # 12 periods per hour (5-min intervals)
                result = enforcer.update_capital_cap()
                if result["status"] == "completed":
                    print(
                        f"Hour {hour+1:2d}, Period {period+1:2d}: {result['current_cap']:.1%} "
                        f"(change: {result['change']:+.1%})"
                    )
                time.sleep(0.1)  # Small delay for simulation

        # Final status
        final_status = enforcer.get_status_report()
        print(f"\nFinal state: {final_status['current_state']['effective_cap']:.1%}")
        return

    if args.run:
        # Run continuous enforcement
        enforcer.run_continuous_enforcement()
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
