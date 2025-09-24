#!/usr/bin/env python3
"""
Capital Ramp Policy

Implements capital ramping rules and model governance:
- Green window detection (4 consecutive 15-min windows)
- Ramp ladder: 10% → 15% → 20% → 30% with minimum trading day intervals
- Strategy caps: RL ≤ 40%, MM ≤ 35%, Basis ≤ 40% until proven
- Automatic reversion on stop conditions
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger("capital_ramp_policy")


class CapitalRampManager:
    """
    Manages capital ramping policy including green window detection,
    ramp ladder progression, strategy allocation limits, and safety reversions.
    """

    def __init__(self):
        """Initialize capital ramp manager."""
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        # Capital ramp configuration
        self.ramp_config = {
            "green_window_minutes": 15,
            "green_windows_required": 4,  # 4 consecutive windows
            "ramp_ladder": [10, 15, 20, 30],  # Percentage steps
            "min_days_between_ramps": 1,  # Minimum trading days between steps
            "strategy_caps": {
                "rl": 0.40,  # RL ≤ 40% until proven
                "market_maker": 0.35,  # MM ≤ 35%
                "basis_carry": 0.40,  # Basis ≤ 40%
            },
            "proven_days_threshold": 10,  # Days to prove strategy
            "green_window_criteria": {
                "min_sharpe": 0.0,  # Sharpe > 0
                "slippage_improving": True,  # Slippage must be improving
                "no_alerts": True,  # No alerts during window
            },
        }

        logger.info("Initialized capital ramp manager")

    def detect_green_windows(self) -> Dict[str, any]:
        """
        Detect consecutive green windows for capital ramping.

        Returns:
            Green window detection results
        """
        try:
            logger.info("🟢 Detecting green windows for capital ramp...")

            detection_results = {
                "timestamp": datetime.now().isoformat(),
                "window_minutes": self.ramp_config["green_window_minutes"],
                "windows_required": self.ramp_config["green_windows_required"],
                "windows": [],
            }

            # Analyze recent 15-minute windows
            current_time = datetime.now()
            for i in range(
                self.ramp_config["green_windows_required"] + 2
            ):  # Look back extra windows
                window_end = current_time - timedelta(
                    minutes=i * self.ramp_config["green_window_minutes"]
                )
                window_start = window_end - timedelta(
                    minutes=self.ramp_config["green_window_minutes"]
                )

                window_metrics = self._analyze_window_metrics(window_start, window_end)
                window_metrics["window_index"] = i
                window_metrics["start_time"] = window_start.isoformat()
                window_metrics["end_time"] = window_end.isoformat()

                detection_results["windows"].append(window_metrics)

            # Check for consecutive green windows
            consecutive_green = self._check_consecutive_green_windows(
                detection_results["windows"]
            )
            detection_results.update(
                {
                    "consecutive_green_count": consecutive_green["count"],
                    "ramp_eligible": consecutive_green["ramp_eligible"],
                    "next_ramp_level": (
                        self._get_next_ramp_level()
                        if consecutive_green["ramp_eligible"]
                        else None
                    ),
                }
            )

            return detection_results

        except Exception as e:
            logger.error(f"Error detecting green windows: {e}")
            return {"error": str(e)}

    def execute_capital_ramp(
        self, target_level: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Execute capital ramp to next level or specified level.

        Args:
            target_level: Specific ramp level (optional)

        Returns:
            Capital ramp execution results
        """
        try:
            logger.info(f"📈 Executing capital ramp...")

            ramp_results = {
                "timestamp": datetime.now().isoformat(),
                "procedure": "capital_ramp_execution",
                "status": "running",
            }

            # Get current capital level
            current_level = self._get_current_capital_level()
            ramp_results["current_level_percent"] = current_level

            # Determine target level
            if target_level is None:
                target_level = self._get_next_ramp_level()

            if target_level is None:
                ramp_results["status"] = "no_ramp_available"
                return ramp_results

            ramp_results["target_level_percent"] = target_level

            # Check ramp eligibility
            eligibility_check = self._check_ramp_eligibility(
                current_level, target_level
            )
            ramp_results["eligibility_check"] = eligibility_check

            if not eligibility_check["eligible"]:
                ramp_results["status"] = "not_eligible"
                return ramp_results

            # Execute ramp
            execution_result = self._execute_ramp_to_level(target_level)
            ramp_results["execution"] = execution_result

            # Update strategy allocations
            strategy_updates = self._update_strategy_allocations(target_level)
            ramp_results["strategy_updates"] = strategy_updates

            # Record ramp event
            self._record_ramp_event(current_level, target_level, ramp_results)

            ramp_results["status"] = (
                "completed" if execution_result["success"] else "failed"
            )

            if ramp_results["status"] == "completed":
                logger.info(
                    f"✅ Capital ramp executed: {current_level}% → {target_level}%"
                )
            else:
                logger.warning(
                    f"⚠️ Capital ramp failed: {execution_result.get('error', 'Unknown error')}"
                )

            return ramp_results

        except Exception as e:
            logger.error(f"Error executing capital ramp: {e}")
            return {"error": str(e), "status": "failed"}

    def revert_capital_ramp(self, reason: str) -> Dict[str, any]:
        """
        Revert capital to previous level due to stop condition.

        Args:
            reason: Reason for reversion

        Returns:
            Capital reversion results
        """
        try:
            logger.warning(f"📉 Reverting capital ramp due to: {reason}")

            reversion_results = {
                "timestamp": datetime.now().isoformat(),
                "procedure": "capital_ramp_reversion",
                "reason": reason,
                "status": "running",
            }

            # Get current and previous levels
            current_level = self._get_current_capital_level()
            previous_level = self._get_previous_capital_level()

            reversion_results.update(
                {
                    "current_level_percent": current_level,
                    "revert_to_level_percent": previous_level,
                }
            )

            if previous_level >= current_level:
                reversion_results["status"] = "no_reversion_needed"
                return reversion_results

            # Execute reversion
            reversion_execution = self._execute_ramp_to_level(previous_level)
            reversion_results["execution"] = reversion_execution

            # Update strategy allocations for lower level
            strategy_updates = self._update_strategy_allocations(previous_level)
            reversion_results["strategy_updates"] = strategy_updates

            # Record reversion event
            self._record_reversion_event(current_level, previous_level, reason)

            reversion_results["status"] = (
                "completed" if reversion_execution["success"] else "failed"
            )

            if reversion_results["status"] == "completed":
                logger.info(
                    f"✅ Capital reverted: {current_level}% → {previous_level}%"
                )
            else:
                logger.error(f"❌ Capital reversion failed")

            return reversion_results

        except Exception as e:
            logger.error(f"Error reverting capital ramp: {e}")
            return {"error": str(e), "status": "failed"}

    def manage_strategy_allocations(
        self, capital_level: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Manage strategy allocation limits based on capital level and performance.

        Args:
            capital_level: Capital level to manage (optional, uses current)

        Returns:
            Strategy allocation management results
        """
        try:
            logger.info("📊 Managing strategy allocations...")

            if capital_level is None:
                capital_level = self._get_current_capital_level()

            allocation_results = {
                "timestamp": datetime.now().isoformat(),
                "capital_level_percent": capital_level,
                "strategy_caps": self.ramp_config["strategy_caps"].copy(),
                "allocations": {},
            }

            # Check if strategies are "proven"
            proven_strategies = self._check_proven_strategies()
            allocation_results["proven_strategies"] = proven_strategies

            # Calculate effective allocations for each strategy
            for strategy_name, base_cap in self.ramp_config["strategy_caps"].items():
                strategy_info = {
                    "base_cap": base_cap,
                    "proven": proven_strategies.get(strategy_name, False),
                    "effective_cap": base_cap,
                }

                # Apply proven strategy bonus
                if strategy_info["proven"]:
                    # Proven strategies can use higher caps
                    strategy_info["effective_cap"] = min(
                        base_cap + 0.20, 0.60
                    )  # Max 60%
                    strategy_info["bonus_applied"] = 0.20

                # Calculate actual allocation
                current_allocation = self._get_current_strategy_allocation(
                    strategy_name
                )
                strategy_info["current_allocation"] = current_allocation

                # Check if allocation needs adjustment
                if current_allocation > strategy_info["effective_cap"]:
                    strategy_info["needs_reduction"] = True
                    strategy_info["target_allocation"] = strategy_info["effective_cap"]
                else:
                    strategy_info["needs_reduction"] = False
                    strategy_info["target_allocation"] = current_allocation

                allocation_results["allocations"][strategy_name] = strategy_info

            # Apply allocation adjustments if needed
            adjustments_made = self._apply_allocation_adjustments(
                allocation_results["allocations"]
            )
            allocation_results["adjustments_made"] = adjustments_made

            return allocation_results

        except Exception as e:
            logger.error(f"Error managing strategy allocations: {e}")
            return {"error": str(e)}

    def run_capital_ramp_governance(self) -> Dict[str, any]:
        """
        Run complete capital ramp governance cycle.

        Returns:
            Complete governance results
        """
        try:
            logger.info("🏛️ Running capital ramp governance...")

            governance_results = {
                "timestamp": datetime.now().isoformat(),
                "procedure": "capital_ramp_governance",
                "components": {},
            }

            # Step 1: Detect green windows
            green_windows = self.detect_green_windows()
            governance_results["components"]["green_windows"] = green_windows

            # Step 2: Check for ramp eligibility and execute if ready
            if green_windows.get("ramp_eligible", False):
                ramp_execution = self.execute_capital_ramp()
                governance_results["components"]["ramp_execution"] = ramp_execution
            else:
                governance_results["components"]["ramp_execution"] = {
                    "status": "not_eligible"
                }

            # Step 3: Manage strategy allocations
            allocation_management = self.manage_strategy_allocations()
            governance_results["components"][
                "allocation_management"
            ] = allocation_management

            # Step 4: Check for stop conditions (would trigger reversion)
            stop_conditions = self._check_stop_conditions()
            governance_results["components"]["stop_conditions"] = stop_conditions

            if stop_conditions.get("should_revert", False):
                reversion = self.revert_capital_ramp(
                    stop_conditions["reversion_reason"]
                )
                governance_results["components"]["reversion"] = reversion

            # Overall status
            governance_results["overall_status"] = self._assess_governance_status(
                governance_results["components"]
            )

            return governance_results

        except Exception as e:
            logger.error(f"Error in capital ramp governance: {e}")
            return {"error": str(e)}

    # Helper methods

    def _analyze_window_metrics(
        self, window_start: datetime, window_end: datetime
    ) -> Dict[str, any]:
        """Analyze metrics for a specific time window."""
        # Mock implementation - would read actual metrics from Redis/Prometheus
        import random

        # Simulate window metrics
        sharpe = random.uniform(-0.5, 1.5)
        slippage_bps = random.uniform(2, 8)
        alert_count = random.randint(0, 2)

        # Check green window criteria
        is_green = (
            sharpe >= self.ramp_config["green_window_criteria"]["min_sharpe"]
            and slippage_bps <= 5.0  # Assume improving if <= 5 bps
            and alert_count == 0
        )

        return {
            "sharpe": sharpe,
            "slippage_bps": slippage_bps,
            "alert_count": alert_count,
            "is_green": is_green,
            "criteria_met": {
                "sharpe_positive": sharpe
                >= self.ramp_config["green_window_criteria"]["min_sharpe"],
                "slippage_improving": slippage_bps <= 5.0,
                "no_alerts": alert_count == 0,
            },
        }

    def _check_consecutive_green_windows(
        self, windows: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Check for required consecutive green windows."""
        # Count consecutive green windows from most recent
        consecutive_count = 0
        for window in windows:
            if window.get("is_green", False):
                consecutive_count += 1
            else:
                break

        return {
            "count": consecutive_count,
            "ramp_eligible": consecutive_count
            >= self.ramp_config["green_windows_required"],
        }

    def _get_current_capital_level(self) -> int:
        """Get current capital level percentage."""
        if self.redis_client:
            try:
                return int(
                    float(self.redis_client.get("risk:capital_level_percent") or "10")
                )
            except:
                return 10
        return 10

    def _get_next_ramp_level(self) -> Optional[int]:
        """Get next ramp level in the ladder."""
        current_level = self._get_current_capital_level()

        for level in self.ramp_config["ramp_ladder"]:
            if level > current_level:
                return level

        return None  # Already at maximum

    def _get_previous_capital_level(self) -> int:
        """Get previous capital level for reversion."""
        current_level = self._get_current_capital_level()

        # Find previous level in ladder
        previous_level = 10  # Default to minimum
        for level in self.ramp_config["ramp_ladder"]:
            if level >= current_level:
                break
            previous_level = level

        return previous_level

    def _check_ramp_eligibility(
        self, current_level: int, target_level: int
    ) -> Dict[str, any]:
        """Check if capital ramp is eligible."""
        # Check minimum days between ramps
        if self.redis_client:
            last_ramp_timestamp = self.redis_client.get("risk:last_ramp_timestamp")
            if last_ramp_timestamp:
                days_since_ramp = (time.time() - float(last_ramp_timestamp)) / 86400
                min_days_met = (
                    days_since_ramp >= self.ramp_config["min_days_between_ramps"]
                )
            else:
                min_days_met = True
        else:
            min_days_met = True

        # Check if target level is valid
        valid_target = target_level in self.ramp_config["ramp_ladder"]

        # Check if moving to next level in sequence
        sequential_ramp = target_level == self._get_next_ramp_level()

        eligible = min_days_met and valid_target and sequential_ramp

        return {
            "eligible": eligible,
            "min_days_met": min_days_met,
            "valid_target": valid_target,
            "sequential_ramp": sequential_ramp,
            "current_level": current_level,
            "target_level": target_level,
        }

    def _execute_ramp_to_level(self, target_level: int) -> Dict[str, any]:
        """Execute capital ramp to specified level."""
        try:
            if self.redis_client:
                # Update capital level in Redis
                self.redis_client.set("risk:capital_level_percent", str(target_level))
                self.redis_client.set("risk:last_ramp_timestamp", str(time.time()))

                # Update effective capital
                base_capital = float(
                    self.redis_client.get("risk:capital_base_usd") or "100000"
                )
                effective_capital = base_capital * (target_level / 100.0)
                self.redis_client.set("risk:capital_effective", str(effective_capital))

            return {
                "success": True,
                "target_level": target_level,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _update_strategy_allocations(self, capital_level: int) -> Dict[str, any]:
        """Update strategy allocations for new capital level."""
        updates = {}

        for strategy_name, base_cap in self.ramp_config["strategy_caps"].items():
            # Calculate new allocation
            proven = self._check_proven_strategies().get(strategy_name, False)
            effective_cap = base_cap + (0.20 if proven else 0.0)
            effective_cap = min(effective_cap, 0.60)

            if self.redis_client:
                key = f"strategy:{strategy_name}:allocation_cap"
                self.redis_client.set(key, str(effective_cap))

            updates[strategy_name] = {"allocation_cap": effective_cap, "proven": proven}

        return updates

    def _check_proven_strategies(self) -> Dict[str, bool]:
        """Check which strategies have proven themselves over time."""
        proven = {}

        for strategy_name in self.ramp_config["strategy_caps"].keys():
            if self.redis_client:
                # Check days of good performance
                days_key = f"strategy:{strategy_name}:proven_days"
                proven_days = int(self.redis_client.get(days_key) or "0")
                proven[strategy_name] = (
                    proven_days >= self.ramp_config["proven_days_threshold"]
                )
            else:
                # Mock: RL is proven, others not yet
                proven[strategy_name] = strategy_name == "rl"

        return proven

    def _get_current_strategy_allocation(self, strategy_name: str) -> float:
        """Get current strategy allocation percentage."""
        if self.redis_client:
            key = f"strategy:{strategy_name}:current_allocation"
            return float(self.redis_client.get(key) or "0.0")
        return 0.0

    def _apply_allocation_adjustments(
        self, allocations: Dict[str, Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """Apply strategy allocation adjustments."""
        adjustments = []

        for strategy_name, allocation_info in allocations.items():
            if allocation_info.get("needs_reduction", False):
                adjustment = {
                    "strategy": strategy_name,
                    "action": "reduce",
                    "from_allocation": allocation_info["current_allocation"],
                    "to_allocation": allocation_info["target_allocation"],
                    "timestamp": datetime.now().isoformat(),
                }

                if self.redis_client:
                    key = f"strategy:{strategy_name}:current_allocation"
                    self.redis_client.set(
                        key, str(allocation_info["target_allocation"])
                    )

                adjustments.append(adjustment)

        return adjustments

    def _check_stop_conditions(self) -> Dict[str, any]:
        """Check for conditions that should trigger capital reversion."""
        # Mock stop condition checking
        # Real implementation would check actual system health

        return {
            "should_revert": False,
            "reversion_reason": None,
            "conditions_checked": [
                "recon_breach",
                "rl_entropy_collapse",
                "multiple_alerts",
                "drawdown_exceeded",
            ],
        }

    def _record_ramp_event(
        self, from_level: int, to_level: int, ramp_results: Dict[str, any]
    ):
        """Record capital ramp event for audit trail."""
        event = {
            "event_type": "capital_ramp",
            "from_level_percent": from_level,
            "to_level_percent": to_level,
            "timestamp": datetime.now().isoformat(),
            "success": ramp_results.get("status") == "completed",
        }

        if self.redis_client:
            self.redis_client.lpush("events:capital_ramp", json.dumps(event))

    def _record_reversion_event(self, from_level: int, to_level: int, reason: str):
        """Record capital reversion event."""
        event = {
            "event_type": "capital_reversion",
            "from_level_percent": from_level,
            "to_level_percent": to_level,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }

        if self.redis_client:
            self.redis_client.lpush("events:capital_ramp", json.dumps(event))

    def _assess_governance_status(self, components: Dict[str, any]) -> str:
        """Assess overall governance status."""
        # Check for any errors in components
        for component_name, component_data in components.items():
            if isinstance(component_data, dict) and component_data.get("error"):
                return "error"

        # Check if any reversions occurred
        if "reversion" in components:
            return "reverted"

        # Check if ramp was executed
        if components.get("ramp_execution", {}).get("status") == "completed":
            return "ramped"

        return "stable"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Capital Ramp Policy Manager")

    parser.add_argument(
        "--action",
        choices=["detect", "ramp", "revert", "allocations", "governance"],
        default="governance",
        help="Action to perform",
    )
    parser.add_argument(
        "--target-level", type=int, help="Target capital level for ramp"
    )
    parser.add_argument(
        "--revert-reason", type=str, help="Reason for capital reversion"
    )
    parser.add_argument("--output", type=str, help="Output file for results")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info("📈 Starting Capital Ramp Policy Manager")

    try:
        manager = CapitalRampManager()

        if args.action == "detect":
            results = manager.detect_green_windows()
        elif args.action == "ramp":
            results = manager.execute_capital_ramp(args.target_level)
        elif args.action == "revert":
            if not args.revert_reason:
                logger.error("--revert-reason required for revert action")
                return 1
            results = manager.revert_capital_ramp(args.revert_reason)
        elif args.action == "allocations":
            results = manager.manage_strategy_allocations()
        else:  # governance
            results = manager.run_capital_ramp_governance()

        print(f"\n📈 CAPITAL RAMP POLICY RESULTS:")
        print(json.dumps(results, indent=2))

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Error in capital ramp policy: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
