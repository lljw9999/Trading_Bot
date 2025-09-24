#!/usr/bin/env python3
"""
SLO Monitoring and Data Quality SLAs

Implements Service Level Objectives monitoring and data quality SLAs:
- Uptime: 99.9% for price ingestion and order routing
- Freshness: feature staleness < 2× sampling interval for 99.5% of the day
- Alert hygiene: < 2 false positives/week across critical alerts
- MTTR: median < 10 min for P1 incidents
- Data quality: NaN/Inf rate < 0.01% of feature vectors
"""

import argparse
import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import math

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import redis
    import pandas as pd

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger("slo_monitoring")


class SLOMonitoringManager:
    """
    Monitors Service Level Objectives and data quality SLAs,
    tracks compliance, and generates SLO reports.
    """

    def __init__(self):
        """Initialize SLO monitoring manager."""
        self.redis_client = None
        if DEPS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        # SLO definitions
        self.slo_config = {
            "uptime_slos": {
                "price_ingestion": {
                    "target": 0.999,  # 99.9%
                    "measurement_window": "rolling_30_days",
                    "service_name": "price_ingestion",
                },
                "order_routing": {
                    "target": 0.999,  # 99.9%
                    "measurement_window": "rolling_30_days",
                    "service_name": "order_routing",
                },
            },
            "freshness_slos": {
                "feature_staleness": {
                    "target": 0.995,  # 99.5% of the day
                    "threshold_multiplier": 2.0,  # < 2× sampling interval
                    "sampling_intervals": {
                        "market_data": 1.0,  # 1 second
                        "alpha_signals": 5.0,  # 5 seconds
                        "risk_metrics": 10.0,  # 10 seconds
                    },
                }
            },
            "alert_hygiene_slos": {
                "false_positive_rate": {
                    "target": 2,  # < 2 false positives/week
                    "measurement_window": "weekly",
                    "critical_alerts_only": True,
                }
            },
            "incident_response_slos": {
                "mttr_p1": {
                    "target_minutes": 10,  # median < 10 min
                    "measurement_window": "rolling_90_days",
                    "incident_priority": "P1",
                }
            },
            "data_quality_slas": {
                "nan_inf_rate": {
                    "target": 0.0001,  # < 0.01%
                    "measurement_scope": "feature_vectors",
                    "measurement_window": "daily",
                },
                "drift_detection": {
                    "kl_divergence_threshold": 0.1,
                    "auto_tuned_threshold": "median_3iqr",
                    "measurement_window": "daily",
                },
            },
        }

        logger.info("Initialized SLO monitoring manager")

    def measure_uptime_slos(self, measurement_period_hours: int = 24) -> Dict[str, any]:
        """
        Measure uptime SLOs for critical services.

        Args:
            measurement_period_hours: Period to measure (hours)

        Returns:
            Uptime SLO measurements
        """
        try:
            logger.info("📈 Measuring uptime SLOs...")

            uptime_results = {
                "timestamp": datetime.now().isoformat(),
                "measurement_period_hours": measurement_period_hours,
                "services": {},
            }

            for service_name, slo_config in self.slo_config["uptime_slos"].items():
                service_results = {
                    "service_name": service_name,
                    "target_uptime": slo_config["target"],
                    "measurement_window": slo_config["measurement_window"],
                }

                # Get service uptime data
                uptime_data = self._get_service_uptime_data(
                    service_name, measurement_period_hours
                )
                service_results.update(uptime_data)

                # Calculate SLO compliance
                actual_uptime = uptime_data["uptime_percentage"]
                slo_met = actual_uptime >= slo_config["target"]

                service_results.update(
                    {
                        "slo_met": slo_met,
                        "slo_margin": actual_uptime - slo_config["target"],
                        "error_budget_consumed": max(
                            0, (1 - actual_uptime) / (1 - slo_config["target"])
                        ),
                    }
                )

                uptime_results["services"][service_name] = service_results

            # Overall uptime SLO status
            all_slos_met = all(
                service["slo_met"] for service in uptime_results["services"].values()
            )
            uptime_results["overall_slo_status"] = "met" if all_slos_met else "breached"

            return uptime_results

        except Exception as e:
            logger.error(f"Error measuring uptime SLOs: {e}")
            return {"error": str(e)}

    def measure_freshness_slos(self) -> Dict[str, any]:
        """
        Measure data freshness SLOs.

        Returns:
            Freshness SLO measurements
        """
        try:
            logger.info("🕐 Measuring freshness SLOs...")

            freshness_results = {
                "timestamp": datetime.now().isoformat(),
                "measurement_period": "24_hours",
                "feature_types": {},
            }

            slo_config = self.slo_config["freshness_slos"]["feature_staleness"]

            for feature_type, sampling_interval in slo_config[
                "sampling_intervals"
            ].items():
                feature_results = {
                    "feature_type": feature_type,
                    "sampling_interval_seconds": sampling_interval,
                    "staleness_threshold_seconds": sampling_interval
                    * slo_config["threshold_multiplier"],
                    "target_compliance": slo_config["target"],
                }

                # Get freshness measurements
                freshness_data = self._measure_feature_freshness(
                    feature_type, sampling_interval
                )
                feature_results.update(freshness_data)

                # Calculate SLO compliance
                compliance_rate = freshness_data["compliance_percentage"]
                slo_met = compliance_rate >= slo_config["target"]

                feature_results.update(
                    {
                        "slo_met": slo_met,
                        "slo_margin": compliance_rate - slo_config["target"],
                    }
                )

                freshness_results["feature_types"][feature_type] = feature_results

            # Overall freshness SLO status
            all_slos_met = all(
                feature["slo_met"]
                for feature in freshness_results["feature_types"].values()
            )
            freshness_results["overall_slo_status"] = (
                "met" if all_slos_met else "breached"
            )

            return freshness_results

        except Exception as e:
            logger.error(f"Error measuring freshness SLOs: {e}")
            return {"error": str(e)}

    def measure_alert_hygiene_slos(
        self, measurement_period_days: int = 7
    ) -> Dict[str, any]:
        """
        Measure alert hygiene SLOs (false positive rate).

        Args:
            measurement_period_days: Period to measure (days)

        Returns:
            Alert hygiene SLO measurements
        """
        try:
            logger.info("🚨 Measuring alert hygiene SLOs...")

            hygiene_results = {
                "timestamp": datetime.now().isoformat(),
                "measurement_period_days": measurement_period_days,
                "alert_analysis": {},
            }

            slo_config = self.slo_config["alert_hygiene_slos"]["false_positive_rate"]

            # Get alert data
            alert_data = self._get_alert_history_data(measurement_period_days)

            # Analyze false positives
            critical_alerts = [
                alert for alert in alert_data if alert.get("severity") == "critical"
            ]
            false_positives = [
                alert
                for alert in critical_alerts
                if alert.get("resolution") == "false_positive"
            ]

            hygiene_results["alert_analysis"] = {
                "total_alerts": len(alert_data),
                "critical_alerts": len(critical_alerts),
                "false_positives": len(false_positives),
                "false_positive_rate": len(false_positives)
                / max(len(critical_alerts), 1),
                "target_max_false_positives": slo_config["target"],
            }

            # Calculate SLO compliance
            weekly_false_positives = len(false_positives) * (
                7 / measurement_period_days
            )
            slo_met = weekly_false_positives <= slo_config["target"]

            hygiene_results.update(
                {
                    "weekly_false_positives": weekly_false_positives,
                    "slo_met": slo_met,
                    "slo_margin": slo_config["target"] - weekly_false_positives,
                }
            )

            return hygiene_results

        except Exception as e:
            logger.error(f"Error measuring alert hygiene SLOs: {e}")
            return {"error": str(e)}

    def measure_incident_response_slos(
        self, measurement_period_days: int = 90
    ) -> Dict[str, any]:
        """
        Measure incident response SLOs (MTTR).

        Args:
            measurement_period_days: Period to measure (days)

        Returns:
            Incident response SLO measurements
        """
        try:
            logger.info("⚠️ Measuring incident response SLOs...")

            incident_results = {
                "timestamp": datetime.now().isoformat(),
                "measurement_period_days": measurement_period_days,
                "incident_analysis": {},
            }

            slo_config = self.slo_config["incident_response_slos"]["mttr_p1"]

            # Get P1 incident data
            p1_incidents = self._get_p1_incident_data(measurement_period_days)

            if not p1_incidents:
                incident_results.update(
                    {"no_p1_incidents": True, "slo_met": True, "median_mttr_minutes": 0}
                )
                return incident_results

            # Calculate MTTR statistics
            mttr_minutes = [
                incident["resolution_time_minutes"] for incident in p1_incidents
            ]

            incident_results["incident_analysis"] = {
                "total_p1_incidents": len(p1_incidents),
                "mttr_statistics": {
                    "median_minutes": statistics.median(mttr_minutes),
                    "mean_minutes": statistics.mean(mttr_minutes),
                    "p95_minutes": self._percentile(mttr_minutes, 95),
                    "min_minutes": min(mttr_minutes),
                    "max_minutes": max(mttr_minutes),
                },
            }

            # Calculate SLO compliance
            median_mttr = statistics.median(mttr_minutes)
            slo_met = median_mttr <= slo_config["target_minutes"]

            incident_results.update(
                {
                    "median_mttr_minutes": median_mttr,
                    "target_mttr_minutes": slo_config["target_minutes"],
                    "slo_met": slo_met,
                    "slo_margin_minutes": slo_config["target_minutes"] - median_mttr,
                }
            )

            return incident_results

        except Exception as e:
            logger.error(f"Error measuring incident response SLOs: {e}")
            return {"error": str(e)}

    def measure_data_quality_slas(self) -> Dict[str, any]:
        """
        Measure data quality SLAs.

        Returns:
            Data quality SLA measurements
        """
        try:
            logger.info("📊 Measuring data quality SLAs...")

            quality_results = {
                "timestamp": datetime.now().isoformat(),
                "measurement_period": "24_hours",
                "quality_metrics": {},
            }

            # Measure NaN/Inf rate
            nan_inf_sla = self.slo_config["data_quality_slas"]["nan_inf_rate"]
            nan_inf_data = self._measure_nan_inf_rate()

            quality_results["quality_metrics"]["nan_inf_rate"] = {
                "measurement": nan_inf_data,
                "target": nan_inf_sla["target"],
                "sla_met": nan_inf_data["rate"] <= nan_inf_sla["target"],
                "margin": nan_inf_sla["target"] - nan_inf_data["rate"],
            }

            # Measure drift detection
            drift_sla = self.slo_config["data_quality_slas"]["drift_detection"]
            drift_data = self._measure_data_drift()

            quality_results["quality_metrics"]["data_drift"] = {
                "measurement": drift_data,
                "threshold": drift_sla["kl_divergence_threshold"],
                "auto_tuned_threshold": drift_data.get("auto_tuned_threshold"),
                "sla_met": drift_data["kl_divergence"]
                <= drift_sla["kl_divergence_threshold"],
            }

            # Overall data quality status
            all_slas_met = all(
                metric["sla_met"]
                for metric in quality_results["quality_metrics"].values()
            )
            quality_results["overall_sla_status"] = (
                "met" if all_slas_met else "breached"
            )

            return quality_results

        except Exception as e:
            logger.error(f"Error measuring data quality SLAs: {e}")
            return {"error": str(e)}

    def generate_slo_report(self, report_period_days: int = 30) -> Dict[str, any]:
        """
        Generate comprehensive SLO compliance report.

        Args:
            report_period_days: Period to report on (days)

        Returns:
            Complete SLO compliance report
        """
        try:
            logger.info(f"📋 Generating SLO report for {report_period_days} days...")

            report = {
                "timestamp": datetime.now().isoformat(),
                "report_period_days": report_period_days,
                "slo_measurements": {},
                "summary": {},
            }

            # Measure all SLO categories
            uptime_slos = self.measure_uptime_slos(report_period_days * 24)
            freshness_slos = self.measure_freshness_slos()
            alert_hygiene_slos = self.measure_alert_hygiene_slos(report_period_days)
            incident_response_slos = self.measure_incident_response_slos(
                report_period_days
            )
            data_quality_slas = self.measure_data_quality_slas()

            report["slo_measurements"] = {
                "uptime": uptime_slos,
                "freshness": freshness_slos,
                "alert_hygiene": alert_hygiene_slos,
                "incident_response": incident_response_slos,
                "data_quality": data_quality_slas,
            }

            # Generate summary
            report["summary"] = self._generate_slo_summary(report["slo_measurements"])

            # Generate recommendations
            report["recommendations"] = self._generate_slo_recommendations(
                report["slo_measurements"]
            )

            return report

        except Exception as e:
            logger.error(f"Error generating SLO report: {e}")
            return {"error": str(e)}

    def publish_slo_metrics(self, slo_data: Dict[str, any]) -> Dict[str, any]:
        """
        Publish SLO metrics to Grafana/Prometheus.

        Args:
            slo_data: SLO measurement data

        Returns:
            Publication results
        """
        try:
            logger.info("📊 Publishing SLO metrics...")

            publication_results = {
                "timestamp": datetime.now().isoformat(),
                "metrics_published": [],
                "publication_errors": [],
            }

            # Extract metrics for publication
            metrics_to_publish = self._extract_slo_metrics(slo_data)

            for metric_name, metric_value in metrics_to_publish.items():
                try:
                    # Publish to Redis (mock Prometheus gateway)
                    if self.redis_client:
                        metric_key = f"slo_metrics:{metric_name}"
                        metric_data = {
                            "value": metric_value,
                            "timestamp": time.time(),
                            "labels": {"slo_type": metric_name.split("_")[0]},
                        }
                        self.redis_client.set(metric_key, json.dumps(metric_data))

                    publication_results["metrics_published"].append(
                        {
                            "metric_name": metric_name,
                            "value": metric_value,
                            "status": "published",
                        }
                    )

                except Exception as e:
                    publication_results["publication_errors"].append(
                        {"metric_name": metric_name, "error": str(e)}
                    )

            publication_results["total_published"] = len(
                publication_results["metrics_published"]
            )
            publication_results["total_errors"] = len(
                publication_results["publication_errors"]
            )
            publication_results["publication_success_rate"] = publication_results[
                "total_published"
            ] / max(
                publication_results["total_published"]
                + publication_results["total_errors"],
                1,
            )

            return publication_results

        except Exception as e:
            logger.error(f"Error publishing SLO metrics: {e}")
            return {"error": str(e)}

    # Helper methods (mock implementations)

    def _get_service_uptime_data(
        self, service_name: str, period_hours: int
    ) -> Dict[str, any]:
        """Get service uptime data (mock implementation)."""
        import random

        # Mock uptime calculation
        total_minutes = period_hours * 60
        downtime_minutes = random.randint(
            0, int(total_minutes * 0.002)
        )  # Usually very low downtime
        uptime_percentage = (total_minutes - downtime_minutes) / total_minutes

        return {
            "total_minutes": total_minutes,
            "downtime_minutes": downtime_minutes,
            "uptime_minutes": total_minutes - downtime_minutes,
            "uptime_percentage": uptime_percentage,
            "downtime_incidents": random.randint(0, 3),
        }

    def _measure_feature_freshness(
        self, feature_type: str, sampling_interval: float
    ) -> Dict[str, any]:
        """Measure feature freshness (mock implementation)."""
        import random

        # Mock freshness measurement
        total_samples = 86400 // int(sampling_interval)  # Samples in 24h
        stale_samples = random.randint(
            0, int(total_samples * 0.008)
        )  # Usually very low
        compliance_percentage = (total_samples - stale_samples) / total_samples

        return {
            "total_samples": total_samples,
            "fresh_samples": total_samples - stale_samples,
            "stale_samples": stale_samples,
            "compliance_percentage": compliance_percentage,
            "max_staleness_seconds": random.uniform(
                sampling_interval, sampling_interval * 3
            ),
            "p95_staleness_seconds": random.uniform(
                sampling_interval * 0.5, sampling_interval * 1.5
            ),
        }

    def _get_alert_history_data(self, period_days: int) -> List[Dict[str, any]]:
        """Get alert history data (mock implementation)."""
        import random

        alerts = []
        num_alerts = random.randint(10, 50)

        for i in range(num_alerts):
            alert = {
                "alert_id": f"alert_{i:03d}",
                "severity": random.choice(["critical", "warning", "info"]),
                "timestamp": (
                    datetime.now() - timedelta(days=random.uniform(0, period_days))
                ).isoformat(),
                "resolution": random.choices(
                    ["resolved", "false_positive", "ongoing"], weights=[0.8, 0.15, 0.05]
                )[0],
            }
            alerts.append(alert)

        return alerts

    def _get_p1_incident_data(self, period_days: int) -> List[Dict[str, any]]:
        """Get P1 incident data (mock implementation)."""
        import random

        # Mock P1 incidents (typically rare)
        num_incidents = random.randint(0, 5)
        incidents = []

        for i in range(num_incidents):
            incident = {
                "incident_id": f"P1_{i:03d}",
                "created_timestamp": (
                    datetime.now() - timedelta(days=random.uniform(0, period_days))
                ).isoformat(),
                "resolution_time_minutes": random.randint(
                    3, 45
                ),  # Usually under target
                "severity": "P1",
            }
            incidents.append(incident)

        return incidents

    def _measure_nan_inf_rate(self) -> Dict[str, any]:
        """Measure NaN/Inf rate in feature vectors (mock implementation)."""
        import random

        total_features = random.randint(50000, 200000)
        nan_inf_count = random.randint(0, int(total_features * 0.0002))  # Very low rate

        return {
            "total_feature_values": total_features,
            "nan_inf_count": nan_inf_count,
            "rate": nan_inf_count / total_features,
            "feature_types_affected": random.randint(0, 3),
        }

    def _measure_data_drift(self) -> Dict[str, any]:
        """Measure data drift (mock implementation)."""
        import random

        kl_divergence = random.uniform(0.01, 0.15)  # Usually low drift

        return {
            "kl_divergence": kl_divergence,
            "auto_tuned_threshold": 0.08,  # median + 3*IQR
            "baseline_updated": datetime.now().isoformat(),
            "features_drifted": random.randint(0, 2),
        }

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _generate_slo_summary(self, slo_measurements: Dict[str, any]) -> Dict[str, any]:
        """Generate SLO compliance summary."""
        summary = {
            "overall_slo_compliance": True,
            "slo_categories": {},
            "breached_slos": [],
            "compliance_score": 0.0,
        }

        total_slos = 0
        met_slos = 0

        for category, measurements in slo_measurements.items():
            if category == "uptime":
                category_status = measurements.get("overall_slo_status") == "met"
                category_count = len(measurements.get("services", {}))
                category_met = sum(
                    1
                    for s in measurements.get("services", {}).values()
                    if s.get("slo_met", False)
                )
            elif category == "freshness":
                category_status = measurements.get("overall_slo_status") == "met"
                category_count = len(measurements.get("feature_types", {}))
                category_met = sum(
                    1
                    for f in measurements.get("feature_types", {}).values()
                    if f.get("slo_met", False)
                )
            elif category == "data_quality":
                category_status = measurements.get("overall_sla_status") == "met"
                category_count = len(measurements.get("quality_metrics", {}))
                category_met = sum(
                    1
                    for q in measurements.get("quality_metrics", {}).values()
                    if q.get("sla_met", False)
                )
            else:
                category_status = measurements.get("slo_met", False)
                category_count = 1
                category_met = 1 if category_status else 0

            summary["slo_categories"][category] = {
                "status": "met" if category_status else "breached",
                "slos_met": category_met,
                "total_slos": category_count,
            }

            total_slos += category_count
            met_slos += category_met

            if not category_status:
                summary["overall_slo_compliance"] = False
                summary["breached_slos"].append(category)

        summary["compliance_score"] = met_slos / max(total_slos, 1)
        summary["total_slos"] = total_slos
        summary["slos_met"] = met_slos

        return summary

    def _generate_slo_recommendations(
        self, slo_measurements: Dict[str, any]
    ) -> List[str]:
        """Generate SLO improvement recommendations."""
        recommendations = []

        # Check uptime SLOs
        uptime_data = slo_measurements.get("uptime", {})
        for service_name, service_data in uptime_data.get("services", {}).items():
            if not service_data.get("slo_met", True):
                recommendations.append(
                    f"Investigate downtime causes for {service_name} service"
                )

        # Check freshness SLOs
        freshness_data = slo_measurements.get("freshness", {})
        for feature_type, feature_data in freshness_data.get(
            "feature_types", {}
        ).items():
            if not feature_data.get("slo_met", True):
                recommendations.append(
                    f"Optimize {feature_type} data pipeline to reduce staleness"
                )

        # Check alert hygiene
        alert_data = slo_measurements.get("alert_hygiene", {})
        if not alert_data.get("slo_met", True):
            recommendations.append(
                "Review and tune alert thresholds to reduce false positives"
            )

        # Check incident response
        incident_data = slo_measurements.get("incident_response", {})
        if not incident_data.get("slo_met", True):
            recommendations.append(
                "Improve incident response procedures to reduce MTTR"
            )

        # Check data quality
        quality_data = slo_measurements.get("data_quality", {})
        for metric_name, metric_data in quality_data.get("quality_metrics", {}).items():
            if not metric_data.get("sla_met", True):
                recommendations.append(f"Address data quality issues in {metric_name}")

        return recommendations

    def _extract_slo_metrics(self, slo_data: Dict[str, any]) -> Dict[str, float]:
        """Extract metrics for publication to monitoring systems."""
        metrics = {}

        # Extract uptime metrics
        uptime_data = slo_data.get("uptime", {})
        for service_name, service_data in uptime_data.get("services", {}).items():
            metrics[f"uptime_{service_name}_percentage"] = service_data.get(
                "uptime_percentage", 0
            )
            metrics[f"uptime_{service_name}_slo_met"] = (
                1 if service_data.get("slo_met", False) else 0
            )

        # Extract freshness metrics
        freshness_data = slo_data.get("freshness", {})
        for feature_type, feature_data in freshness_data.get(
            "feature_types", {}
        ).items():
            metrics[f"freshness_{feature_type}_compliance"] = feature_data.get(
                "compliance_percentage", 0
            )
            metrics[f"freshness_{feature_type}_slo_met"] = (
                1 if feature_data.get("slo_met", False) else 0
            )

        # Extract other metrics
        if "alert_hygiene" in slo_data:
            metrics["alert_hygiene_false_positive_rate"] = slo_data[
                "alert_hygiene"
            ].get("false_positive_rate", 0)
            metrics["alert_hygiene_slo_met"] = (
                1 if slo_data["alert_hygiene"].get("slo_met", False) else 0
            )

        if "incident_response" in slo_data:
            metrics["incident_response_mttr_minutes"] = slo_data[
                "incident_response"
            ].get("median_mttr_minutes", 0)
            metrics["incident_response_slo_met"] = (
                1 if slo_data["incident_response"].get("slo_met", False) else 0
            )

        return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SLO Monitoring Manager")

    parser.add_argument(
        "--category",
        choices=["uptime", "freshness", "alerts", "incidents", "quality", "all"],
        default="all",
        help="SLO category to measure",
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate comprehensive SLO report"
    )
    parser.add_argument(
        "--publish", action="store_true", help="Publish metrics to monitoring systems"
    )
    parser.add_argument(
        "--period-days", type=int, default=30, help="Measurement/report period in days"
    )
    parser.add_argument("--output", type=str, help="Output file for results")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info("📊 Starting SLO Monitoring Manager")

    try:
        manager = SLOMonitoringManager()

        if args.report:
            results = manager.generate_slo_report(args.period_days)
        elif args.category == "uptime":
            results = manager.measure_uptime_slos(args.period_days * 24)
        elif args.category == "freshness":
            results = manager.measure_freshness_slos()
        elif args.category == "alerts":
            results = manager.measure_alert_hygiene_slos(args.period_days)
        elif args.category == "incidents":
            results = manager.measure_incident_response_slos(args.period_days)
        elif args.category == "quality":
            results = manager.measure_data_quality_slas()
        else:  # all
            results = manager.generate_slo_report(args.period_days)

        # Publish metrics if requested
        if args.publish:
            publish_results = manager.publish_slo_metrics(results)
            results["publication"] = publish_results

        print(f"\n📊 SLO MONITORING RESULTS:")
        print(json.dumps(results, indent=2))

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Error in SLO monitoring: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
