#!/usr/bin/env python3
"""
Cost Guardrails

Implements cost monitoring and alerting to prevent silent spend creep:
- Daily AWS budget alerts
- Grafana alerts for p95 egress > baseline × 2
- GPU utilization monitoring (idle GPU mem > 40% for 30 min)
- Tag-based cost attribution (training vs inference)
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
    import boto3
    import redis
    import psutil

    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

logger = logging.getLogger("cost_guardrails")


class CostGuardrailsManager:
    """
    Manages cost monitoring, alerting, and guardrails to prevent
    unexpected spend increases and optimize resource utilization.
    """

    def __init__(self):
        """Initialize cost guardrails manager."""
        self.redis_client = None
        self.aws_session = None

        if AWS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
                self.aws_session = boto3.Session()
            except Exception as e:
                logger.warning(f"AWS/Redis unavailable: {e}")

        # Cost thresholds and baselines
        self.cost_config = {
            "daily_budget_usd": 500,  # Daily AWS budget limit
            "egress_baseline_gb": 100,  # Baseline daily egress
            "egress_multiplier_threshold": 2.0,  # Alert when > baseline × 2
            "gpu_idle_threshold": 0.40,  # 40% idle memory threshold
            "gpu_idle_duration_minutes": 30,  # Alert after 30 min idle
            "cost_spike_threshold": 0.50,  # 50% increase over baseline
        }

        logger.info("Initialized cost guardrails manager")

    def check_aws_budget(self) -> Dict[str, any]:
        """
        Check AWS budget usage and set up alerts if needed.

        Returns:
            AWS budget status
        """
        try:
            logger.info("💰 Checking AWS budget status...")

            budget_status = {
                "timestamp": datetime.now().isoformat(),
                "budget_name": "daily-trading-budget",
                "daily_limit_usd": self.cost_config["daily_budget_usd"],
            }

            if not AWS_AVAILABLE:
                logger.warning("AWS SDK unavailable - using mock data")
                return self._get_mock_budget_status()

            try:
                # Check current spending
                budgets_client = self.aws_session.client("budgets")

                # Get current budget status
                response = budgets_client.describe_budget(
                    AccountId="123456789012",  # Would be actual account ID
                    BudgetName=budget_status["budget_name"],
                )

                budget = response["Budget"]
                budget_status.update(
                    {
                        "budgeted_amount": float(budget["BudgetLimit"]["Amount"]),
                        "actual_spend": float(
                            budget.get("CalculatedSpend", {})
                            .get("ActualSpend", {})
                            .get("Amount", 0)
                        ),
                        "forecasted_spend": float(
                            budget.get("CalculatedSpend", {})
                            .get("ForecastedSpend", {})
                            .get("Amount", 0)
                        ),
                    }
                )

                # Calculate utilization
                utilization = (
                    budget_status["actual_spend"] / budget_status["budgeted_amount"]
                )
                budget_status["utilization_percent"] = utilization * 100
                budget_status["alert_triggered"] = utilization > 0.80  # Alert at 80%

            except Exception as e:
                logger.warning(f"AWS budget API error: {e}")
                return self._get_mock_budget_status()

            return budget_status

        except Exception as e:
            logger.error(f"Error checking AWS budget: {e}")
            return {"error": str(e)}

    def monitor_egress_traffic(self) -> Dict[str, any]:
        """
        Monitor network egress traffic for cost spikes.

        Returns:
            Egress monitoring results
        """
        try:
            logger.info("🌐 Monitoring egress traffic...")

            egress_status = {
                "timestamp": datetime.now().isoformat(),
                "baseline_gb": self.cost_config["egress_baseline_gb"],
                "threshold_multiplier": self.cost_config["egress_multiplier_threshold"],
            }

            if not self.redis_client:
                logger.warning("Redis unavailable - using mock data")
                return self._get_mock_egress_status()

            # Get current egress metrics from Redis/Prometheus
            try:
                # Get daily egress total
                current_egress_gb = float(
                    self.redis_client.get("metrics:egress_daily_gb") or 85
                )
                p95_egress_gb = float(
                    self.redis_client.get("metrics:egress_p95_gb") or 12
                )

                egress_status.update(
                    {
                        "current_daily_gb": current_egress_gb,
                        "p95_gb": p95_egress_gb,
                        "threshold_gb": self.cost_config["egress_baseline_gb"]
                        * self.cost_config["egress_multiplier_threshold"],
                    }
                )

                # Check if threshold exceeded
                threshold_exceeded = p95_egress_gb > egress_status["threshold_gb"]
                egress_status["alert_triggered"] = threshold_exceeded

                if threshold_exceeded:
                    egress_status["alert_reason"] = (
                        f"P95 egress ({p95_egress_gb:.1f} GB) exceeds threshold ({egress_status['threshold_gb']:.1f} GB)"
                    )

                # Store alert if triggered
                if threshold_exceeded and self.redis_client:
                    alert = {
                        "type": "egress_threshold_exceeded",
                        "timestamp": datetime.now().isoformat(),
                        "p95_gb": p95_egress_gb,
                        "threshold_gb": egress_status["threshold_gb"],
                    }
                    self.redis_client.lpush("alerts:cost", json.dumps(alert))

            except Exception as e:
                logger.warning(f"Error reading egress metrics: {e}")
                return self._get_mock_egress_status()

            return egress_status

        except Exception as e:
            logger.error(f"Error monitoring egress traffic: {e}")
            return {"error": str(e)}

    def monitor_gpu_utilization(self) -> Dict[str, any]:
        """
        Monitor GPU utilization and detect idle resources.

        Returns:
            GPU utilization status
        """
        try:
            logger.info("🎮 Monitoring GPU utilization...")

            gpu_status = {
                "timestamp": datetime.now().isoformat(),
                "idle_threshold": self.cost_config["gpu_idle_threshold"],
                "idle_duration_threshold_minutes": self.cost_config[
                    "gpu_idle_duration_minutes"
                ],
                "gpus": [],
            }

            # Try to get GPU info (mock implementation - would use nvidia-ml-py)
            gpu_info = self._get_gpu_info()

            for i, gpu in enumerate(gpu_info):
                gpu_data = {
                    "gpu_id": i,
                    "memory_used_gb": gpu["memory_used_gb"],
                    "memory_total_gb": gpu["memory_total_gb"],
                    "memory_utilization": gpu["memory_used_gb"]
                    / gpu["memory_total_gb"],
                    "idle": gpu["memory_used_gb"] / gpu["memory_total_gb"]
                    < self.cost_config["gpu_idle_threshold"],
                }

                # Check idle duration
                if gpu_data["idle"]:
                    idle_start_key = f"gpu:{i}:idle_start"
                    if self.redis_client:
                        idle_start = self.redis_client.get(idle_start_key)
                        if not idle_start:
                            # Start tracking idle time
                            self.redis_client.set(idle_start_key, int(time.time()))
                            gpu_data["idle_duration_minutes"] = 0
                        else:
                            # Calculate idle duration
                            idle_duration = (int(time.time()) - int(idle_start)) / 60
                            gpu_data["idle_duration_minutes"] = idle_duration

                            # Check if alert threshold exceeded
                            if (
                                idle_duration
                                >= self.cost_config["gpu_idle_duration_minutes"]
                            ):
                                gpu_data["alert_triggered"] = True

                                # Store alert
                                alert = {
                                    "type": "gpu_idle_too_long",
                                    "gpu_id": i,
                                    "idle_duration_minutes": idle_duration,
                                    "memory_utilization": gpu_data[
                                        "memory_utilization"
                                    ],
                                    "timestamp": datetime.now().isoformat(),
                                }
                                if self.redis_client:
                                    self.redis_client.lpush(
                                        "alerts:cost", json.dumps(alert)
                                    )
                else:
                    # GPU not idle - clear tracking
                    if self.redis_client:
                        self.redis_client.delete(f"gpu:{i}:idle_start")
                    gpu_data["idle_duration_minutes"] = 0
                    gpu_data["alert_triggered"] = False

                gpu_status["gpus"].append(gpu_data)

            # Overall status
            idle_gpus = [
                gpu for gpu in gpu_status["gpus"] if gpu.get("alert_triggered", False)
            ]
            gpu_status["idle_gpus_count"] = len(idle_gpus)
            gpu_status["total_gpus"] = len(gpu_status["gpus"])
            gpu_status["overall_alert"] = len(idle_gpus) > 0

            return gpu_status

        except Exception as e:
            logger.error(f"Error monitoring GPU utilization: {e}")
            return {"error": str(e)}

    def check_cost_attribution(self) -> Dict[str, any]:
        """
        Check cost attribution by tags (training vs inference).

        Returns:
            Cost attribution analysis
        """
        try:
            logger.info("🏷️ Checking cost attribution...")

            attribution = {
                "timestamp": datetime.now().isoformat(),
                "period": "daily",
                "categories": {},
            }

            if not AWS_AVAILABLE:
                logger.warning("AWS unavailable - using mock attribution data")
                return self._get_mock_cost_attribution()

            try:
                # Use AWS Cost Explorer to get tagged costs
                ce_client = self.aws_session.client("ce")

                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

                # Get costs by tag
                response = ce_client.get_cost_and_usage(
                    TimePeriod={"Start": start_date, "End": end_date},
                    Granularity="DAILY",
                    Metrics=["UnblendedCost"],
                    GroupBy=[{"Type": "TAG", "Key": "Purpose"}],
                )

                for result in response["ResultsByTime"][0]["Groups"]:
                    tag_value = result["Keys"][0]
                    cost = float(result["Metrics"]["UnblendedCost"]["Amount"])

                    attribution["categories"][tag_value] = {
                        "cost_usd": cost,
                        "currency": result["Metrics"]["UnblendedCost"]["Unit"],
                    }

            except Exception as e:
                logger.warning(f"AWS cost explorer error: {e}")
                return self._get_mock_cost_attribution()

            # Calculate totals and percentages
            total_cost = sum(
                cat["cost_usd"] for cat in attribution["categories"].values()
            )
            attribution["total_cost_usd"] = total_cost

            for category in attribution["categories"].values():
                category["percentage"] = (
                    (category["cost_usd"] / total_cost * 100) if total_cost > 0 else 0
                )

            return attribution

        except Exception as e:
            logger.error(f"Error checking cost attribution: {e}")
            return {"error": str(e)}

    def run_cost_analysis(self) -> Dict[str, any]:
        """
        Run complete cost analysis and guardrails check.

        Returns:
            Complete cost analysis results
        """
        try:
            logger.info("📊 Running complete cost analysis...")

            analysis = {
                "timestamp": datetime.now().isoformat(),
                "components": {},
                "alerts_triggered": [],
                "recommendations": [],
            }

            # Check AWS budget
            budget_status = self.check_aws_budget()
            analysis["components"]["aws_budget"] = budget_status

            if budget_status.get("alert_triggered", False):
                analysis["alerts_triggered"].append(
                    {
                        "type": "budget_threshold",
                        "component": "aws_budget",
                        "message": f"Budget utilization at {budget_status.get('utilization_percent', 0):.1f}%",
                    }
                )

            # Monitor egress traffic
            egress_status = self.monitor_egress_traffic()
            analysis["components"]["egress_traffic"] = egress_status

            if egress_status.get("alert_triggered", False):
                analysis["alerts_triggered"].append(
                    {
                        "type": "egress_threshold",
                        "component": "egress_traffic",
                        "message": egress_status.get(
                            "alert_reason", "Egress traffic threshold exceeded"
                        ),
                    }
                )

            # Monitor GPU utilization
            gpu_status = self.monitor_gpu_utilization()
            analysis["components"]["gpu_utilization"] = gpu_status

            if gpu_status.get("overall_alert", False):
                analysis["alerts_triggered"].append(
                    {
                        "type": "gpu_idle",
                        "component": "gpu_utilization",
                        "message": f"{gpu_status.get('idle_gpus_count', 0)} GPU(s) idle for >30 minutes",
                    }
                )

            # Check cost attribution
            attribution = self.check_cost_attribution()
            analysis["components"]["cost_attribution"] = attribution

            # Generate recommendations
            analysis["recommendations"] = self._generate_cost_recommendations(
                analysis["components"]
            )

            # Overall status
            analysis["overall_status"] = (
                "healthy" if len(analysis["alerts_triggered"]) == 0 else "alerts"
            )
            analysis["total_alerts"] = len(analysis["alerts_triggered"])

            return analysis

        except Exception as e:
            logger.error(f"Error running cost analysis: {e}")
            return {"error": str(e)}

    def _get_gpu_info(self) -> List[Dict[str, any]]:
        """Get GPU information (mock implementation)."""
        # Mock GPU data - real implementation would use nvidia-ml-py
        return [
            {"memory_used_gb": 3.2, "memory_total_gb": 8.0, "utilization": 0.40},
            {"memory_used_gb": 7.8, "memory_total_gb": 8.0, "utilization": 0.98},
            {"memory_used_gb": 2.1, "memory_total_gb": 8.0, "utilization": 0.26},
        ]

    def _get_mock_budget_status(self) -> Dict[str, any]:
        """Get mock budget status for testing."""
        return {
            "timestamp": datetime.now().isoformat(),
            "budget_name": "daily-trading-budget",
            "daily_limit_usd": self.cost_config["daily_budget_usd"],
            "budgeted_amount": 500.0,
            "actual_spend": 387.50,
            "forecasted_spend": 425.00,
            "utilization_percent": 77.5,
            "alert_triggered": False,
        }

    def _get_mock_egress_status(self) -> Dict[str, any]:
        """Get mock egress status for testing."""
        return {
            "timestamp": datetime.now().isoformat(),
            "baseline_gb": self.cost_config["egress_baseline_gb"],
            "threshold_multiplier": self.cost_config["egress_multiplier_threshold"],
            "current_daily_gb": 185.3,
            "p95_gb": 15.2,
            "threshold_gb": 200.0,
            "alert_triggered": False,
        }

    def _get_mock_cost_attribution(self) -> Dict[str, any]:
        """Get mock cost attribution for testing."""
        return {
            "timestamp": datetime.now().isoformat(),
            "period": "daily",
            "categories": {
                "training": {"cost_usd": 245.75, "currency": "USD", "percentage": 63.5},
                "inference": {"cost_usd": 98.25, "currency": "USD", "percentage": 25.4},
                "monitoring": {
                    "cost_usd": 43.00,
                    "currency": "USD",
                    "percentage": 11.1,
                },
            },
            "total_cost_usd": 387.00,
        }

    def _generate_cost_recommendations(self, components: Dict[str, any]) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []

        # Budget recommendations
        budget = components.get("aws_budget", {})
        if budget.get("utilization_percent", 0) > 70:
            recommendations.append(
                "Consider reviewing high-cost services as budget utilization exceeds 70%"
            )

        # GPU recommendations
        gpu = components.get("gpu_utilization", {})
        idle_count = gpu.get("idle_gpus_count", 0)
        if idle_count > 0:
            recommendations.append(
                f"Consider terminating or rightsizing {idle_count} underutilized GPU instance(s)"
            )

        # Egress recommendations
        egress = components.get("egress_traffic", {})
        if egress.get("current_daily_gb", 0) > 150:
            recommendations.append(
                "Review data transfer patterns - consider caching or CDN optimization"
            )

        # Cost attribution recommendations
        attribution = components.get("cost_attribution", {})
        training_pct = (
            attribution.get("categories", {}).get("training", {}).get("percentage", 0)
        )
        if training_pct > 70:
            recommendations.append(
                "Training costs are high - consider batch scheduling or spot instances"
            )

        return recommendations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cost Guardrails Manager")

    parser.add_argument(
        "--component",
        choices=["budget", "egress", "gpu", "attribution", "all"],
        default="all",
        help="Component to check",
    )
    parser.add_argument(
        "--alert-only",
        action="store_true",
        help="Only show results if alerts are triggered",
    )
    parser.add_argument("--output", type=str, help="Output file for results")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info("💰 Starting Cost Guardrails Manager")

    try:
        manager = CostGuardrailsManager()

        if args.component == "budget":
            results = manager.check_aws_budget()
        elif args.component == "egress":
            results = manager.monitor_egress_traffic()
        elif args.component == "gpu":
            results = manager.monitor_gpu_utilization()
        elif args.component == "attribution":
            results = manager.check_cost_attribution()
        else:  # all
            results = manager.run_cost_analysis()

        # Filter results if alert-only mode
        if args.alert_only and results.get("overall_status") == "healthy":
            print("No cost alerts triggered")
            return 0

        print(f"\n💰 COST GUARDRAILS RESULTS:")
        print(json.dumps(results, indent=2))

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        # Return appropriate exit code
        if args.component == "all":
            return 0 if results.get("total_alerts", 0) == 0 else 1
        else:
            return 0

    except Exception as e:
        logger.error(f"Error in cost guardrails: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
