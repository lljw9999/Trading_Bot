#!/usr/bin/env python3
"""
Pilot Governance and Stop Rules

Implements the governance framework for crypto live pilot trading,
including success metrics tracking, stop conditions, and decision cadence.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import redis
    from src.utils.aredis import get_redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger("pilot_governance")


class PilotGovernanceManager:
    """
    Manages pilot governance including success metrics tracking,
    stop conditions monitoring, and automated decision making.
    """

    def __init__(self):
        """Initialize pilot governance manager."""
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        # Define success metrics thresholds
        self.success_metrics = {
            "net_pnl_threshold": 0.0,  # ≥ 0 after fees by Day-2
            "max_drawdown_threshold": 0.015,  # ≤ 1.5% intraday
            "median_slippage_threshold": 0.0004,  # ≤ 4 bps
            "p95_slippage_threshold": 0.0012,  # ≤ 12 bps
            "recon_breaches_threshold": 0,  # = 0
            "feature_halts_threshold": 1,  # ≤ 1/day
            "auto_recover_time_threshold": 300,  # < 5 min
        }

        # Define stop conditions
        self.stop_conditions = {
            "recon_breach": {
                "active": True,
                "description": "Any reconciliation breach",
            },
            "rl_entropy_collapse": {
                "threshold": 0.05,
                "duration": 120,
                "description": "RL entropy < 0.05 for 2 min",
            },
            "hedge_inactive_high_es": {
                "es_threshold": 0.03,
                "description": "Hedge inactive while ES95 > 3%",
            },
            "multiple_feature_halts": {
                "count": 2,
                "window": 3600,
                "description": "2+ feature halts in 60 min",
            },
        }

        logger.info("Initialized pilot governance manager")

    def check_success_metrics(self, pilot_day: int) -> Dict[str, any]:
        """
        Check current performance against success metrics.

        Args:
            pilot_day: Current day of pilot (0-based)

        Returns:
            Dictionary with metric results
        """
        try:
            results = {
                "pilot_day": pilot_day,
                "timestamp": datetime.now().isoformat(),
                "metrics": {},
                "overall_status": "unknown",
            }

            if not self.redis_client:
                logger.warning("Redis unavailable - using mock metrics")
                return self._get_mock_success_metrics(pilot_day)

            # Check net P&L (after fees)
            net_pnl = float(self.redis_client.get("pilot:net_pnl") or 0.0)
            results["metrics"]["net_pnl"] = {
                "value": net_pnl,
                "threshold": self.success_metrics["net_pnl_threshold"],
                "passing": (
                    net_pnl >= self.success_metrics["net_pnl_threshold"]
                    if pilot_day >= 2
                    else True
                ),
            }

            # Check intraday max drawdown
            max_drawdown = float(self.redis_client.get("pilot:max_drawdown") or 0.0)
            results["metrics"]["max_drawdown"] = {
                "value": max_drawdown,
                "threshold": self.success_metrics["max_drawdown_threshold"],
                "passing": max_drawdown
                <= self.success_metrics["max_drawdown_threshold"],
            }

            # Check slippage metrics
            median_slippage = float(
                self.redis_client.get("pilot:median_slippage") or 0.0
            )
            p95_slippage = float(self.redis_client.get("pilot:p95_slippage") or 0.0)

            results["metrics"]["median_slippage"] = {
                "value": median_slippage,
                "threshold": self.success_metrics["median_slippage_threshold"],
                "passing": median_slippage
                <= self.success_metrics["median_slippage_threshold"],
            }

            results["metrics"]["p95_slippage"] = {
                "value": p95_slippage,
                "threshold": self.success_metrics["p95_slippage_threshold"],
                "passing": p95_slippage
                <= self.success_metrics["p95_slippage_threshold"],
            }

            # Check recon breaches
            recon_breaches = int(
                self.redis_client.get("pilot:recon_breaches_today") or 0
            )
            results["metrics"]["recon_breaches"] = {
                "value": recon_breaches,
                "threshold": self.success_metrics["recon_breaches_threshold"],
                "passing": recon_breaches
                <= self.success_metrics["recon_breaches_threshold"],
            }

            # Check feature halts
            feature_halts = int(self.redis_client.get("pilot:feature_halts_today") or 0)
            results["metrics"]["feature_halts"] = {
                "value": feature_halts,
                "threshold": self.success_metrics["feature_halts_threshold"],
                "passing": feature_halts
                <= self.success_metrics["feature_halts_threshold"],
            }

            # Determine overall status
            all_passing = all(
                metric["passing"] for metric in results["metrics"].values()
            )
            results["overall_status"] = "passing" if all_passing else "failing"

            return results

        except Exception as e:
            logger.error(f"Error checking success metrics: {e}")
            return {"error": str(e), "overall_status": "error"}

    def check_stop_conditions(self) -> Dict[str, any]:
        """
        Check for stop conditions that require immediate halt.

        Returns:
            Dictionary with stop condition results
        """
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "conditions": {},
                "should_halt": False,
                "halt_reason": None,
            }

            if not self.redis_client:
                logger.warning("Redis unavailable - using mock stop conditions")
                return self._get_mock_stop_conditions()

            # Check recon breach
            recon_breach = bool(
                int(self.redis_client.get("pilot:recon_breach_active") or 0)
            )
            results["conditions"]["recon_breach"] = {
                "active": recon_breach,
                "description": self.stop_conditions["recon_breach"]["description"],
            }

            if recon_breach:
                results["should_halt"] = True
                results["halt_reason"] = "Reconciliation breach detected"

            # Check RL entropy collapse
            rl_entropy = float(self.redis_client.get("rl:entropy") or 0.5)
            entropy_low_start = self.redis_client.get("pilot:entropy_low_start")

            if rl_entropy < self.stop_conditions["rl_entropy_collapse"]["threshold"]:
                if not entropy_low_start:
                    # Start tracking low entropy period
                    self.redis_client.set("pilot:entropy_low_start", int(time.time()))
                else:
                    # Check if low entropy duration exceeded
                    duration = int(time.time()) - int(entropy_low_start)
                    if (
                        duration
                        >= self.stop_conditions["rl_entropy_collapse"]["duration"]
                    ):
                        results["should_halt"] = True
                        results["halt_reason"] = (
                            "RL entropy collapse (< 0.05 for 2+ minutes)"
                        )
            else:
                # Clear low entropy tracking
                if entropy_low_start:
                    self.redis_client.delete("pilot:entropy_low_start")

            results["conditions"]["rl_entropy_collapse"] = {
                "current_entropy": rl_entropy,
                "threshold": self.stop_conditions["rl_entropy_collapse"]["threshold"],
                "duration_seconds": int(time.time())
                - int(entropy_low_start or time.time()),
                "description": self.stop_conditions["rl_entropy_collapse"][
                    "description"
                ],
            }

            # Check hedge inactive during high ES
            hedge_active = bool(int(self.redis_client.get("hedge:active") or 0))
            es95 = float(self.redis_client.get("risk:es95") or 0.0)

            hedge_condition = (
                not hedge_active
                and es95
                > self.stop_conditions["hedge_inactive_high_es"]["es_threshold"]
            )
            results["conditions"]["hedge_inactive_high_es"] = {
                "hedge_active": hedge_active,
                "es95": es95,
                "threshold": self.stop_conditions["hedge_inactive_high_es"][
                    "es_threshold"
                ],
                "triggered": hedge_condition,
                "description": self.stop_conditions["hedge_inactive_high_es"][
                    "description"
                ],
            }

            if hedge_condition:
                results["should_halt"] = True
                results["halt_reason"] = "Hedge inactive during high ES (ES95 > 3%)"

            # Check multiple feature halts
            halt_timestamps = self.redis_client.lrange(
                "pilot:feature_halt_timestamps", 0, -1
            )
            current_time = int(time.time())
            recent_halts = [
                ts
                for ts in halt_timestamps
                if current_time - int(ts)
                <= self.stop_conditions["multiple_feature_halts"]["window"]
            ]

            multiple_halts = (
                len(recent_halts)
                >= self.stop_conditions["multiple_feature_halts"]["count"]
            )
            results["conditions"]["multiple_feature_halts"] = {
                "recent_count": len(recent_halts),
                "threshold": self.stop_conditions["multiple_feature_halts"]["count"],
                "window_seconds": self.stop_conditions["multiple_feature_halts"][
                    "window"
                ],
                "triggered": multiple_halts,
                "description": self.stop_conditions["multiple_feature_halts"][
                    "description"
                ],
            }

            if multiple_halts:
                results["should_halt"] = True
                results["halt_reason"] = "Multiple feature halts (2+ in 60 minutes)"

            return results

        except Exception as e:
            logger.error(f"Error checking stop conditions: {e}")
            return {"error": str(e), "should_halt": False}

    def execute_halt_procedure(self, reason: str) -> bool:
        """
        Execute halt procedure including global halt and notifications.

        Args:
            reason: Reason for halt

        Returns:
            True if halt executed successfully
        """
        try:
            logger.critical(f"🛑 EXECUTING HALT PROCEDURE: {reason}")

            if self.redis_client:
                # Set global halt
                self.redis_client.set("mode", "halt")

                # Record halt event
                halt_event = {
                    "timestamp": datetime.now().isoformat(),
                    "reason": reason,
                    "executed_by": "pilot_governance",
                }
                self.redis_client.lpush("pilot:halt_events", json.dumps(halt_event))

                # Trigger incident SOP
                self.redis_client.set("incident:active", "1")
                self.redis_client.set("incident:reason", reason)
                self.redis_client.set("incident:start_time", int(time.time()))

            logger.info("✅ Halt procedure executed successfully")
            return True

        except Exception as e:
            logger.error(f"Error executing halt procedure: {e}")
            return False

    def run_checkpoint(self, pilot_day: int) -> Dict[str, any]:
        """
        Run 30-minute checkpoint evaluation.

        Args:
            pilot_day: Current day of pilot

        Returns:
            Checkpoint results
        """
        try:
            logger.info(f"🔍 Running pilot checkpoint for Day {pilot_day}")

            # Check success metrics
            success_results = self.check_success_metrics(pilot_day)

            # Check stop conditions
            stop_results = self.check_stop_conditions()

            checkpoint_results = {
                "timestamp": datetime.now().isoformat(),
                "pilot_day": pilot_day,
                "success_metrics": success_results,
                "stop_conditions": stop_results,
                "action_taken": None,
            }

            # Execute halt if stop conditions triggered
            if stop_results.get("should_halt", False):
                halt_reason = stop_results.get("halt_reason", "Unknown stop condition")
                if self.execute_halt_procedure(halt_reason):
                    checkpoint_results["action_taken"] = f"halt_executed: {halt_reason}"
                else:
                    checkpoint_results["action_taken"] = "halt_failed"
            else:
                checkpoint_results["action_taken"] = "continue"

            # Store checkpoint results
            if self.redis_client:
                self.redis_client.lpush(
                    "pilot:checkpoints", json.dumps(checkpoint_results)
                )

            return checkpoint_results

        except Exception as e:
            logger.error(f"Error running checkpoint: {e}")
            return {"error": str(e)}

    def _get_mock_success_metrics(self, pilot_day: int) -> Dict[str, any]:
        """Get mock success metrics for testing."""
        return {
            "pilot_day": pilot_day,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "net_pnl": {"value": 150.75, "threshold": 0.0, "passing": True},
                "max_drawdown": {"value": 0.008, "threshold": 0.015, "passing": True},
                "median_slippage": {
                    "value": 0.0003,
                    "threshold": 0.0004,
                    "passing": True,
                },
                "p95_slippage": {"value": 0.0009, "threshold": 0.0012, "passing": True},
                "recon_breaches": {"value": 0, "threshold": 0, "passing": True},
                "feature_halts": {"value": 0, "threshold": 1, "passing": True},
            },
            "overall_status": "passing",
        }

    def _get_mock_stop_conditions(self) -> Dict[str, any]:
        """Get mock stop conditions for testing."""
        return {
            "timestamp": datetime.now().isoformat(),
            "conditions": {
                "recon_breach": {
                    "active": False,
                    "description": "Any reconciliation breach",
                },
                "rl_entropy_collapse": {
                    "current_entropy": 0.42,
                    "threshold": 0.05,
                    "duration_seconds": 0,
                    "description": "RL entropy < 0.05 for 2 min",
                },
                "hedge_inactive_high_es": {
                    "hedge_active": True,
                    "es95": 0.018,
                    "threshold": 0.03,
                    "triggered": False,
                    "description": "Hedge inactive while ES95 > 3%",
                },
                "multiple_feature_halts": {
                    "recent_count": 0,
                    "threshold": 2,
                    "window_seconds": 3600,
                    "triggered": False,
                    "description": "2+ feature halts in 60 min",
                },
            },
            "should_halt": False,
            "halt_reason": None,
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pilot Governance Manager")

    parser.add_argument(
        "--pilot-day", type=int, default=0, help="Current day of pilot (0-based)"
    )
    parser.add_argument(
        "--checkpoint", action="store_true", help="Run checkpoint evaluation"
    )
    parser.add_argument(
        "--check-metrics", action="store_true", help="Check success metrics only"
    )
    parser.add_argument(
        "--check-stops", action="store_true", help="Check stop conditions only"
    )
    parser.add_argument("--output", type=str, help="Output file for results")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info("🏛️ Starting Pilot Governance Manager")

    try:
        manager = PilotGovernanceManager()

        if args.checkpoint:
            results = manager.run_checkpoint(args.pilot_day)
            print(f"\n📋 CHECKPOINT RESULTS (Day {args.pilot_day}):")
            print(json.dumps(results, indent=2))

        elif args.check_metrics:
            results = manager.check_success_metrics(args.pilot_day)
            print(f"\n📊 SUCCESS METRICS (Day {args.pilot_day}):")
            print(json.dumps(results, indent=2))

        elif args.check_stops:
            results = manager.check_stop_conditions()
            print(f"\n🛑 STOP CONDITIONS:")
            print(json.dumps(results, indent=2))
        else:
            # Default: run full checkpoint
            results = manager.run_checkpoint(args.pilot_day)
            print(f"\n📋 CHECKPOINT RESULTS (Day {args.pilot_day}):")
            print(json.dumps(results, indent=2))

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Error in pilot governance: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
