#!/usr/bin/env python3
"""
Live Promotion Controller
Turn shadow winners into live features with auditable, automated promotion
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("live_promotion_controller")


class LivePromotionController:
    """Automated live feature promotion controller."""

    def __init__(self):
        """Initialize live promotion controller."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.slack_webhook = os.getenv("SLACK_WEBHOOK", "")

        # Feature flags to manage
        self.features = {
            "EXEC_RL_LIVE": {
                "shadow_key": "ab:last4:exec",
                "description": "RL Execution",
                "metrics": ["slippage_improvement", "fill_rate"],
            },
            "BANDIT_WEIGHTS": {
                "shadow_key": "ab:last4:bandit",
                "description": "Contextual Bandit",
                "metrics": ["sharpe_improvement"],
            },
            "LLM_SENTIMENT": {
                "shadow_key": "ab:last4:llm",
                "description": "LLM Sentiment",
                "metrics": ["llm_correlation", "sample_count"],
            },
        }

        # Promotion criteria
        self.promotion_criteria = {
            "consecutive_passes_required": 4,  # Must pass 4 consecutive 15-min windows
            "lookback_hours": 24,  # Look back 24 hours for metrics
            "min_evaluation_windows": 8,  # Need at least 8 evaluation windows
        }

        # Metric thresholds
        self.thresholds = {
            "slippage_improvement_bps": 5.0,  # Slippage improvement threshold
            "sharpe_improvement": 0.08,  # Sharpe improvement threshold
            "llm_correlation": 0.04,  # LLM correlation threshold
            "fill_rate_min": 0.93,  # Minimum fill rate
            "min_llm_samples": 300,  # Minimum LLM samples
        }

        logger.info("🚦 Live Promotion Controller initialized")
        logger.info(f"   Features managed: {list(self.features.keys())}")
        logger.info(
            f"   Consecutive passes required: {self.promotion_criteria['consecutive_passes_required']}"
        )
        logger.info(f"   Lookback period: {self.promotion_criteria['lookback_hours']}h")

    def get_metric(self, key: str, default: float = 0.0) -> float:
        """Get metric from Redis with default fallback."""
        try:
            value = self.redis.get(key)
            return float(value) if value is not None else default
        except Exception as e:
            logger.warning(f"Failed to get metric {key}: {e}")
            return default

    def get_feature_flag_status(self, flag: str) -> bool:
        """Get current feature flag status."""
        try:
            status = self.redis.hget("features:flags", flag)
            return bool(int(status)) if status is not None else False
        except Exception:
            return False

    def get_last_4_passes(self, feature: str) -> int:
        """Get number of consecutive passes for a feature."""
        try:
            shadow_key = self.features[feature]["shadow_key"]
            count = self.redis.get(shadow_key)
            return int(count) if count is not None else 0
        except Exception as e:
            logger.warning(f"Failed to get consecutive passes for {feature}: {e}")
            return 0

    def evaluate_24h_metrics(self, feature: str) -> dict:
        """Evaluate 24h performance metrics for a feature."""
        try:
            evaluation = {
                "feature": feature,
                "timestamp": time.time(),
                "pass": False,
                "reasons": [],
                "metrics": {},
            }

            if feature == "EXEC_RL_LIVE":
                # Evaluate RL execution metrics
                slip_live = self.get_metric("slippage:live:bps")
                slip_shadow = self.get_metric("slippage:shadow:bps")
                fill_live = self.get_metric("fills:live:rate", 1.0)
                fill_shadow = self.get_metric("fills:shadow:rate", 1.0)

                slippage_improvement = slip_live - slip_shadow

                evaluation["metrics"] = {
                    "slippage_live": slip_live,
                    "slippage_shadow": slip_shadow,
                    "slippage_improvement": slippage_improvement,
                    "fill_rate_live": fill_live,
                    "fill_rate_shadow": fill_shadow,
                }

                # Check thresholds
                meets_slippage = (
                    slippage_improvement >= self.thresholds["slippage_improvement_bps"]
                )
                meets_fill_rate = fill_shadow >= self.thresholds["fill_rate_min"]

                if not meets_slippage:
                    evaluation["reasons"].append(
                        f"Slippage improvement {slippage_improvement:.1f}bps < {self.thresholds['slippage_improvement_bps']:.1f}bps"
                    )

                if not meets_fill_rate:
                    evaluation["reasons"].append(
                        f"Fill rate {fill_shadow:.3f} < {self.thresholds['fill_rate_min']:.3f}"
                    )

                evaluation["pass"] = meets_slippage and meets_fill_rate

            elif feature == "BANDIT_WEIGHTS":
                # Evaluate bandit ensemble metrics
                sharpe_live = self.get_metric("sharpe:1h:live")
                sharpe_shadow = self.get_metric("sharpe:1h:shadow")

                sharpe_improvement = sharpe_shadow - sharpe_live

                evaluation["metrics"] = {
                    "sharpe_live": sharpe_live,
                    "sharpe_shadow": sharpe_shadow,
                    "sharpe_improvement": sharpe_improvement,
                }

                meets_sharpe = (
                    sharpe_improvement >= self.thresholds["sharpe_improvement"]
                )

                if not meets_sharpe:
                    evaluation["reasons"].append(
                        f"Sharpe improvement {sharpe_improvement:.3f} < {self.thresholds['sharpe_improvement']:.3f}"
                    )

                evaluation["pass"] = meets_sharpe

            elif feature == "LLM_SENTIMENT":
                # Evaluate LLM sentiment metrics
                llm_correlation = self.get_metric("llm:corr:futret5m")
                llm_samples = self.get_metric("llm:corr:samples")

                evaluation["metrics"] = {
                    "llm_correlation": llm_correlation,
                    "llm_samples": llm_samples,
                }

                meets_correlation = (
                    llm_correlation >= self.thresholds["llm_correlation"]
                )
                meets_samples = llm_samples >= self.thresholds["min_llm_samples"]

                if not meets_correlation:
                    evaluation["reasons"].append(
                        f"LLM correlation {llm_correlation:.4f} < {self.thresholds['llm_correlation']:.4f}"
                    )

                if not meets_samples:
                    evaluation["reasons"].append(
                        f"LLM samples {llm_samples:.0f} < {self.thresholds['min_llm_samples']}"
                    )

                evaluation["pass"] = meets_correlation and meets_samples

            if evaluation["pass"]:
                logger.debug(f"✅ {feature} passed 24h evaluation")
            else:
                logger.debug(
                    f"❌ {feature} failed 24h evaluation: {', '.join(evaluation['reasons'])}"
                )

            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating 24h metrics for {feature}: {e}")
            return {
                "feature": feature,
                "pass": False,
                "error": str(e),
                "timestamp": time.time(),
            }

    def promote_feature(self, feature: str, evaluation: dict) -> bool:
        """Promote a feature to live status."""
        try:
            # Set feature flag
            self.redis.hset("features:flags", feature, 1)

            # Publish promotion event
            promotion_event = {
                "ts": time.time(),
                "feature": feature,
                "decision": "promote_to_live",
                "evaluator": "live_promotion_controller",
                "consecutive_passes": self.get_last_4_passes(feature),
                "evaluation": evaluation,
                "previous_state": False,
                "new_state": True,
            }

            self.redis.publish("ab_gate:events", json.dumps(promotion_event))

            # Store promotion record
            promotion_record = {
                "feature": feature,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "consecutive_passes": self.get_last_4_passes(feature),
                "evaluation_summary": evaluation,
                "controller": "live_promotion_controller",
            }

            self.redis.lpush("promotions:live:history", json.dumps(promotion_record))
            self.redis.ltrim("promotions:live:history", 0, 99)  # Keep last 100

            # Reset consecutive pass counter
            shadow_key = self.features[feature]["shadow_key"]
            self.redis.delete(shadow_key)

            logger.info(f"✅ Promoted {feature} to LIVE status")

            # Send Slack notification
            if self.slack_webhook:
                feature_desc = self.features[feature]["description"]
                metrics_summary = self._format_metrics_for_slack(
                    evaluation.get("metrics", {})
                )

                slack_message = (
                    f"✅ LIVE PROMOTION: {feature_desc}\n"
                    f"Feature: {feature}=ON\n"
                    f"Consecutive passes: {promotion_event['consecutive_passes']}\n"
                    f"Metrics: {metrics_summary}"
                )

                self.send_slack_notification(slack_message)

            return True

        except Exception as e:
            logger.error(f"Failed to promote {feature}: {e}")
            return False

    def _format_metrics_for_slack(self, metrics: dict) -> str:
        """Format metrics for Slack notification."""
        try:
            if not metrics:
                return "N/A"

            formatted = []
            for key, value in metrics.items():
                if isinstance(value, float):
                    if "bps" in key or "slippage" in key:
                        formatted.append(f"{key}={value:.1f}bps")
                    elif "rate" in key or "correlation" in key:
                        formatted.append(f"{key}={value:.3f}")
                    else:
                        formatted.append(f"{key}={value:.2f}")
                else:
                    formatted.append(f"{key}={value}")

            return ", ".join(formatted)

        except Exception:
            return str(metrics)

    def send_slack_notification(self, message: str) -> bool:
        """Send notification to Slack."""
        try:
            if not self.slack_webhook:
                return False

            payload = {
                "text": message,
                "username": "Live Promotion Controller",
                "icon_emoji": ":rocket:",
            }

            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("📱 Sent promotion notification to Slack")
            return True

        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False

    def run_promotion_cycle(self) -> dict:
        """Run complete promotion evaluation cycle."""
        cycle_start = time.time()

        logger.info("🚦 Running live promotion evaluation cycle")

        results = {
            "timestamp": cycle_start,
            "evaluations": {},
            "promotions": [],
            "summary": {
                "total_features": len(self.features),
                "eligible_for_promotion": 0,
                "promoted": 0,
                "already_live": 0,
            },
        }

        try:
            for feature in self.features:
                try:
                    # Check if already live
                    if self.get_feature_flag_status(feature):
                        results["summary"]["already_live"] += 1
                        logger.debug(f"⏭️ {feature} already live, skipping")
                        continue

                    # Get consecutive passes
                    consecutive_passes = self.get_last_4_passes(feature)

                    # Evaluate 24h metrics
                    evaluation = self.evaluate_24h_metrics(feature)
                    results["evaluations"][feature] = evaluation

                    # Check promotion criteria
                    meets_consecutive = (
                        consecutive_passes
                        >= self.promotion_criteria["consecutive_passes_required"]
                    )
                    meets_current_eval = evaluation.get("pass", False)

                    if meets_consecutive and meets_current_eval:
                        results["summary"]["eligible_for_promotion"] += 1

                        # Promote to live
                        if self.promote_feature(feature, evaluation):
                            results["promotions"].append(
                                {
                                    "feature": feature,
                                    "consecutive_passes": consecutive_passes,
                                    "evaluation": evaluation,
                                }
                            )
                            results["summary"]["promoted"] += 1
                    else:
                        reasons = []
                        if not meets_consecutive:
                            reasons.append(
                                f"consecutive passes {consecutive_passes} < {self.promotion_criteria['consecutive_passes_required']}"
                            )
                        if not meets_current_eval:
                            reasons.append(
                                f"current evaluation failed: {evaluation.get('reasons', [])}"
                            )

                        logger.info(
                            f"⏳ {feature} not ready for promotion: {', '.join(reasons)}"
                        )

                except Exception as e:
                    logger.error(f"Error evaluating {feature}: {e}")
                    results["evaluations"][feature] = {
                        "feature": feature,
                        "pass": False,
                        "error": str(e),
                    }

            # Log summary
            cycle_duration = time.time() - cycle_start
            results["cycle_duration_seconds"] = cycle_duration

            promoted_count = results["summary"]["promoted"]
            eligible_count = results["summary"]["eligible_for_promotion"]

            if promoted_count > 0:
                promoted_features = [p["feature"] for p in results["promotions"]]
                logger.info(
                    f"🚀 Promoted {promoted_count} features to LIVE: {promoted_features}"
                )
            elif eligible_count > 0:
                logger.info(
                    f"⏳ {eligible_count} features eligible but promotion failed"
                )
            else:
                logger.info("⏸️ No features ready for live promotion")

            logger.info(f"📊 Cycle complete in {cycle_duration:.2f}s")

            return results

        except Exception as e:
            logger.error(f"Error in promotion cycle: {e}")
            results["error"] = str(e)
            results["cycle_duration_seconds"] = time.time() - cycle_start
            return results

    def update_consecutive_passes(self, feature: str, passed: bool):
        """Update consecutive pass counter for a feature (called by A/B gate)."""
        try:
            shadow_key = self.features.get(feature, {}).get("shadow_key")
            if not shadow_key:
                return

            if passed:
                # Increment consecutive passes
                current_count = int(self.redis.get(shadow_key) or 0)
                self.redis.set(shadow_key, current_count + 1)
                logger.debug(f"📈 {feature} consecutive passes: {current_count + 1}")
            else:
                # Reset consecutive passes
                self.redis.delete(shadow_key)
                logger.debug(f"🔄 {feature} consecutive passes reset")

        except Exception as e:
            logger.error(f"Error updating consecutive passes for {feature}: {e}")

    def get_status_report(self) -> dict:
        """Get comprehensive status report."""
        try:
            status = {
                "service": "live_promotion_controller",
                "timestamp": time.time(),
                "features": {},
                "promotion_criteria": self.promotion_criteria,
                "thresholds": self.thresholds,
            }

            for feature in self.features:
                status["features"][feature] = {
                    "description": self.features[feature]["description"],
                    "currently_live": self.get_feature_flag_status(feature),
                    "consecutive_passes": self.get_last_4_passes(feature),
                    "evaluation": self.evaluate_24h_metrics(feature),
                }

            return status

        except Exception as e:
            return {
                "service": "live_promotion_controller",
                "status": "error",
                "error": str(e),
            }


def main():
    """Main entry point for live promotion controller."""
    import argparse

    parser = argparse.ArgumentParser(description="Live Promotion Controller")
    parser.add_argument(
        "--run", action="store_true", help="Run promotion cycle (default mode)"
    )
    parser.add_argument("--status", action="store_true", help="Show status report")
    parser.add_argument(
        "--reset-passes",
        metavar="FEATURE",
        help="Reset consecutive passes for specific feature",
    )

    args = parser.parse_args()

    # Create controller
    controller = LivePromotionController()

    if args.status:
        # Show status report
        status = controller.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.reset_passes:
        # Reset consecutive passes
        feature = args.reset_passes.upper()
        if feature in controller.features:
            shadow_key = controller.features[feature]["shadow_key"]
            controller.redis.delete(shadow_key)
            logger.info(f"🔄 Reset consecutive passes for {feature}")
        else:
            logger.error(f"Unknown feature: {feature}")
            sys.exit(1)
        return

    # Run promotion cycle (default)
    results = controller.run_promotion_cycle()

    # Print results
    print(json.dumps(results, indent=2, default=str))

    # Exit with appropriate code
    if results.get("summary", {}).get("promoted", 0) > 0:
        sys.exit(0)  # Success - promoted features
    else:
        sys.exit(1)  # No promotions


if __name__ == "__main__":
    main()
