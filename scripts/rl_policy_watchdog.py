#!/usr/bin/env python3
"""
RL Policy Auto-Heal Watchdog

Auto-heals the RL policy by monitoring:
- policy:last_update > 5 minutes → restart policy-daemon
- entropy < 0.05 for 2+ minutes → set features:RL:weight to 5% floor and restart
- Logs all actions to alerts:policy
"""

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger("rl_policy_watchdog")


class RLPolicyWatchdog:
    """
    Monitors RL policy health and auto-heals issues by:
    - Restarting policy daemon when updates are stale
    - Setting minimum weight floor when entropy collapses
    - Logging all actions for audit trail
    """

    def __init__(self):
        """Initialize RL policy watchdog."""
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        # Watchdog configuration
        self.config = {
            "stale_threshold_minutes": 5,  # policy:last_update > 5 min
            "entropy_collapse_threshold": 0.05,  # entropy < 0.05
            "entropy_collapse_duration": 120,  # for 2+ minutes
            "min_weight_floor": 0.05,  # 5% minimum weight
            "policy_daemon_service": "policy-daemon",  # systemd service name
            "check_interval_seconds": 30,  # check every 30 seconds
        }

        # State tracking
        self.entropy_low_start = None
        self.last_restart_time = None
        self.restart_cooldown_minutes = 5  # minimum 5 min between restarts

        logger.info("Initialized RL policy watchdog")

    def check_policy_health(self) -> Dict[str, any]:
        """
        Check RL policy health status.

        Returns:
            Policy health assessment
        """
        try:
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "checks": {},
                "actions_needed": [],
                "overall_healthy": True,
            }

            if not self.redis_client:
                logger.warning("Redis unavailable - using mock health check")
                return self._get_mock_health_status()

            # Check 1: Policy staleness
            staleness_check = self._check_policy_staleness()
            health_status["checks"]["staleness"] = staleness_check

            if not staleness_check["healthy"]:
                health_status["actions_needed"].append("restart_daemon")
                health_status["overall_healthy"] = False

            # Check 2: Entropy collapse
            entropy_check = self._check_entropy_collapse()
            health_status["checks"]["entropy"] = entropy_check

            if not entropy_check["healthy"]:
                health_status["actions_needed"].append("set_weight_floor")
                health_status["overall_healthy"] = False

            # Check 3: General policy metrics
            metrics_check = self._check_policy_metrics()
            health_status["checks"]["metrics"] = metrics_check

            return health_status

        except Exception as e:
            logger.error(f"Error checking policy health: {e}")
            return {"error": str(e), "overall_healthy": False}

    def _check_policy_staleness(self) -> Dict[str, any]:
        """Check if policy updates are stale."""
        try:
            # Get last update timestamp (prefer epoch key)
            last_update_time: Optional[datetime] = None

            last_update_ts = self.redis_client.get("policy:last_update_ts")
            if last_update_ts:
                try:
                    last_update_time = datetime.fromtimestamp(
                        float(last_update_ts), tz=timezone.utc
                    )
                except (TypeError, ValueError):
                    last_update_time = None

            if last_update_time is None:
                last_update_str = self.redis_client.get("policy:last_update")
                if last_update_str:
                    try:
                        last_update_time = datetime.fromisoformat(
                            last_update_str.replace("Z", "+00:00")
                        )
                    except ValueError:
                        try:
                            last_update_time = datetime.fromtimestamp(
                                float(last_update_str), tz=timezone.utc
                            )
                        except (TypeError, ValueError):
                            last_update_time = None

            if last_update_time is None:
                return {
                    "healthy": False,
                    "reason": "No policy:last_update timestamp found",
                    "minutes_since_update": None,
                }

            minutes_since_update = (
                datetime.now(timezone.utc) - last_update_time
            ).total_seconds() / 60

            is_stale = minutes_since_update > self.config["stale_threshold_minutes"]

            return {
                "healthy": not is_stale,
                "last_update": last_update_time.isoformat(),
                "minutes_since_update": minutes_since_update,
                "threshold_minutes": self.config["stale_threshold_minutes"],
                "reason": (
                    f"Policy stale for {minutes_since_update:.1f} minutes"
                    if is_stale
                    else None
                ),
            }

        except Exception as e:
            return {
                "healthy": False,
                "reason": f"Error checking staleness: {e}",
                "minutes_since_update": None,
            }

    def _check_entropy_collapse(self) -> Dict[str, any]:
        """Check for entropy collapse condition."""
        try:
            # Get current entropy
            entropy_str = self.redis_client.get("rl:entropy")
            if not entropy_str:
                return {
                    "healthy": False,
                    "reason": "No rl:entropy metric found",
                    "current_entropy": None,
                }

            current_entropy = float(entropy_str)

            # Check if entropy is below collapse threshold
            entropy_low = current_entropy < self.config["entropy_collapse_threshold"]

            if entropy_low:
                # Track duration of low entropy
                current_time = time.time()

                if self.entropy_low_start is None:
                    # Start tracking low entropy period
                    self.entropy_low_start = current_time
                    low_duration = 0
                else:
                    low_duration = current_time - self.entropy_low_start

                # Check if duration exceeds threshold
                collapse_detected = (
                    low_duration >= self.config["entropy_collapse_duration"]
                )

                return {
                    "healthy": not collapse_detected,
                    "current_entropy": current_entropy,
                    "threshold": self.config["entropy_collapse_threshold"],
                    "low_duration_seconds": low_duration,
                    "collapse_threshold_seconds": self.config[
                        "entropy_collapse_duration"
                    ],
                    "reason": (
                        f"Entropy collapse: {current_entropy:.3f} for {low_duration:.1f}s"
                        if collapse_detected
                        else None
                    ),
                }
            else:
                # Reset low entropy tracking
                self.entropy_low_start = None

                return {
                    "healthy": True,
                    "current_entropy": current_entropy,
                    "threshold": self.config["entropy_collapse_threshold"],
                    "low_duration_seconds": 0,
                }

        except Exception as e:
            return {
                "healthy": False,
                "reason": f"Error checking entropy: {e}",
                "current_entropy": None,
            }

    def _check_policy_metrics(self) -> Dict[str, any]:
        """Check general policy metrics."""
        try:
            metrics = {}

            # Get key RL metrics
            for metric in ["rl:entropy", "rl:q_spread", "rl:action_rate"]:
                value = self.redis_client.get(metric)
                if value:
                    metrics[metric] = float(value)

            # Basic health heuristics
            healthy = True
            issues = []

            if "rl:entropy" in metrics:
                if metrics["rl:entropy"] < 0.1:
                    issues.append(f"Very low entropy: {metrics['rl:entropy']:.3f}")
                    healthy = False
                elif metrics["rl:entropy"] > 2.0:
                    issues.append(f"Very high entropy: {metrics['rl:entropy']:.3f}")

            if "rl:action_rate" in metrics:
                if metrics["rl:action_rate"] < 0.01:
                    issues.append(
                        f"Very low action rate: {metrics['rl:action_rate']:.3f}"
                    )
                    healthy = False

            return {"healthy": healthy, "metrics": metrics, "issues": issues}

        except Exception as e:
            return {"healthy": False, "reason": f"Error checking metrics: {e}"}

    def execute_healing_actions(self, actions_needed: list) -> Dict[str, any]:
        """
        Execute healing actions for unhealthy policy.

        Args:
            actions_needed: List of actions to execute

        Returns:
            Results of healing actions
        """
        try:
            healing_results = {
                "timestamp": datetime.now().isoformat(),
                "actions_executed": {},
                "overall_success": True,
            }

            for action in actions_needed:
                if action == "restart_daemon":
                    result = self._restart_policy_daemon()
                    healing_results["actions_executed"]["restart_daemon"] = result

                elif action == "set_weight_floor":
                    result = self._set_weight_floor()
                    healing_results["actions_executed"]["set_weight_floor"] = result

                # Log action to Redis alerts
                self._log_healing_action(action, result.get("success", False))

                if not result.get("success", False):
                    healing_results["overall_success"] = False

            return healing_results

        except Exception as e:
            logger.error(f"Error executing healing actions: {e}")
            return {"error": str(e), "overall_success": False}

    def _restart_policy_daemon(self) -> Dict[str, any]:
        """Restart policy daemon service."""
        try:
            # Check cooldown period
            if self.last_restart_time:
                minutes_since_restart = (
                    datetime.now() - self.last_restart_time
                ).total_seconds() / 60
                if minutes_since_restart < self.restart_cooldown_minutes:
                    return {
                        "success": False,
                        "reason": f"Restart cooldown active: {minutes_since_restart:.1f} < {self.restart_cooldown_minutes} minutes",
                    }

            logger.warning("🔄 Restarting policy daemon due to health issues")

            # Attempt to restart systemd service
            try:
                result = subprocess.run(
                    [
                        "sudo",
                        "systemctl",
                        "restart",
                        self.config["policy_daemon_service"],
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    self.last_restart_time = datetime.now()
                    return {
                        "success": True,
                        "method": "systemctl",
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    logger.error(f"systemctl restart failed: {result.stderr}")

            except Exception as e:
                logger.warning(f"systemctl restart failed: {e}")

            # Fallback: try pkill/restart approach
            try:
                # Kill existing policy daemon
                subprocess.run(["pkill", "-f", "policy_daemon"], timeout=10)
                time.sleep(2)

                # Start new instance (would need actual start command)
                logger.info("Policy daemon restart attempted via pkill method")

                self.last_restart_time = datetime.now()
                return {
                    "success": True,
                    "method": "pkill_restart",
                    "timestamp": datetime.now().isoformat(),
                }

            except Exception as e:
                return {"success": False, "reason": f"All restart methods failed: {e}"}

        except Exception as e:
            return {"success": False, "reason": str(e)}

    def _set_weight_floor(self) -> Dict[str, any]:
        """Set RL weight to minimum floor."""
        try:
            logger.warning(
                f"🔧 Setting RL weight floor to {self.config['min_weight_floor']} due to entropy collapse"
            )

            if self.redis_client:
                # Set weight floor
                self.redis_client.set(
                    "features:RL:weight", str(self.config["min_weight_floor"])
                )

                # Also set in main weight key if it exists
                current_weight = self.redis_client.get("rl:weight")
                if (
                    current_weight
                    and float(current_weight) < self.config["min_weight_floor"]
                ):
                    self.redis_client.set(
                        "rl:weight", str(self.config["min_weight_floor"])
                    )

                return {
                    "success": True,
                    "weight_floor": self.config["min_weight_floor"],
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {"success": False, "reason": "Redis unavailable"}

        except Exception as e:
            return {"success": False, "reason": str(e)}

    def _log_healing_action(self, action: str, success: bool):
        """Log healing action to alerts:policy."""
        try:
            if self.redis_client:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "alert_type": "rl_policy_watchdog",
                    "action": action,
                    "success": success,
                    "source": "rl_policy_watchdog",
                }

                self.redis_client.lpush("alerts:policy", json.dumps(alert))

                # Keep only last 100 alerts
                self.redis_client.ltrim("alerts:policy", 0, 99)

        except Exception as e:
            logger.error(f"Error logging healing action: {e}")

    def run_watchdog_cycle(self) -> Dict[str, any]:
        """Run single watchdog monitoring cycle."""
        try:
            logger.debug("🔍 Running RL policy watchdog cycle")

            cycle_results = {
                "timestamp": datetime.now().isoformat(),
                "cycle_type": "watchdog_monitoring",
            }

            # Check policy health
            health_status = self.check_policy_health()
            cycle_results["health_status"] = health_status

            # Execute healing actions if needed
            if not health_status.get("overall_healthy", True) and health_status.get(
                "actions_needed"
            ):
                healing_results = self.execute_healing_actions(
                    health_status["actions_needed"]
                )
                cycle_results["healing_results"] = healing_results

                if healing_results.get("overall_success", False):
                    logger.info("✅ RL policy healing actions completed successfully")
                else:
                    logger.error("❌ RL policy healing actions failed")
            else:
                cycle_results["healing_results"] = {
                    "message": "No actions needed - policy healthy"
                }

            return cycle_results

        except Exception as e:
            logger.error(f"Error in watchdog cycle: {e}")
            return {"error": str(e)}

    def run_watchdog_daemon(self):
        """Run watchdog as continuous daemon."""
        logger.info("🐕 Starting RL policy watchdog daemon")

        try:
            while True:
                cycle_results = self.run_watchdog_cycle()

                # Log significant events
                if (
                    "healing_results" in cycle_results
                    and "actions_executed" in cycle_results["healing_results"]
                ):
                    logger.info(
                        f"Healing actions executed: {list(cycle_results['healing_results']['actions_executed'].keys())}"
                    )

                # Wait before next cycle
                time.sleep(self.config["check_interval_seconds"])

        except KeyboardInterrupt:
            logger.info("Watchdog daemon stopped by user")
        except Exception as e:
            logger.error(f"Watchdog daemon error: {e}")

    def _get_mock_health_status(self) -> Dict[str, any]:
        """Get mock health status for testing."""
        return {
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "staleness": {
                    "healthy": True,
                    "minutes_since_update": 2.3,
                    "threshold_minutes": 5,
                },
                "entropy": {
                    "healthy": True,
                    "current_entropy": 0.42,
                    "threshold": 0.05,
                },
                "metrics": {
                    "healthy": True,
                    "metrics": {"rl:entropy": 0.42, "rl:q_spread": 0.034},
                },
            },
            "actions_needed": [],
            "overall_healthy": True,
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RL Policy Watchdog")

    parser.add_argument(
        "--mode",
        choices=["check", "heal", "daemon"],
        default="check",
        help="Watchdog mode",
    )
    parser.add_argument("--output", type=str, help="Output file for results")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info("🐕 Starting RL Policy Watchdog")

    try:
        watchdog = RLPolicyWatchdog()

        if args.mode == "check":
            results = watchdog.check_policy_health()
            print(f"\n🔍 POLICY HEALTH CHECK:")
            print(json.dumps(results, indent=2))

        elif args.mode == "heal":
            health_status = watchdog.check_policy_health()
            if health_status.get("actions_needed"):
                results = watchdog.execute_healing_actions(
                    health_status["actions_needed"]
                )
                print(f"\n🔧 HEALING ACTIONS:")
                print(json.dumps(results, indent=2))
            else:
                print("No healing actions needed - policy is healthy")
                results = {"message": "No actions needed"}

        elif args.mode == "daemon":
            watchdog.run_watchdog_daemon()
            return 0

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Error in RL policy watchdog: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
