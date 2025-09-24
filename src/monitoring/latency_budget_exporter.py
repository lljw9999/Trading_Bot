#!/usr/bin/env python3
"""
Latency Budget Exporter: Real-time Execution Latency Monitoring
Export latency metrics to Prometheus/Redis for alerting and dashboards.
"""
import os
import sys
import json
import time
import datetime
import threading
import numpy as np
import redis
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class LatencyBudget:
    """Latency budget configuration by component."""

    # Component latency budgets (milliseconds)
    signal_generation_ms: float = 5.0  # Alpha models
    feature_engineering_ms: float = 3.0  # Feature computation
    ensemble_ms: float = 2.0  # Meta-learner
    position_sizing_ms: float = 1.0  # Kelly sizing
    risk_checks_ms: float = 2.0  # Risk manager
    order_routing_ms: float = 5.0  # Smart routing
    venue_latency_ms: float = 50.0  # Exchange round-trip

    # Total budget
    total_budget_ms: float = 75.0  # End-to-end SLA

    # Alert thresholds (as percentage of budget)
    warning_threshold_pct: float = 0.8  # 80% of budget
    critical_threshold_pct: float = 0.95  # 95% of budget


class LatencyBudgetExporter:
    """Export latency metrics for monitoring and alerting."""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.budget = LatencyBudget()
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

        # Redis connection for metrics
        try:
            self.redis_client = redis.Redis(
                host=redis_host, port=redis_port, decode_responses=True
            )
            self.redis_client.ping()  # Test connection
        except Exception as e:
            print(f"⚠️ Redis not available: {e}")
            self.redis_client = None

        # Metrics storage
        self.metrics_buffer = {}
        self.alert_history = []

    def record_component_latency(
        self, component: str, latency_ms: float, metadata: Dict[str, Any] = None
    ) -> None:
        """Record latency for a specific component."""
        timestamp = time.time()

        metric_key = f"latency:{component}"

        # Store in buffer for local analysis
        if metric_key not in self.metrics_buffer:
            self.metrics_buffer[metric_key] = []

        self.metrics_buffer[metric_key].append(
            {
                "timestamp": timestamp,
                "latency_ms": latency_ms,
                "metadata": metadata or {},
            }
        )

        # Keep only recent metrics (last 1000 points)
        if len(self.metrics_buffer[metric_key]) > 1000:
            self.metrics_buffer[metric_key] = self.metrics_buffer[metric_key][-1000:]

        # Export to Redis
        if self.redis_client:
            try:
                # Set current latency
                self.redis_client.set(f"{metric_key}:current", latency_ms, ex=300)

                # Add to time series (keep last 100 points)
                self.redis_client.lpush(
                    f"{metric_key}:history",
                    json.dumps({"ts": timestamp, "val": latency_ms}),
                )
                self.redis_client.ltrim(f"{metric_key}:history", 0, 99)

                # Update statistics
                self._update_component_stats(component, latency_ms)

            except Exception as e:
                print(f"⚠️ Redis export error: {e}")

        # Check for budget violations
        self._check_budget_violation(component, latency_ms, metadata)

    def record_end_to_end_latency(
        self,
        total_latency_ms: float,
        component_breakdown: Dict[str, float] = None,
        order_id: str = None,
    ) -> None:
        """Record end-to-end execution latency."""

        # Record total latency
        self.record_component_latency(
            "total_e2e",
            total_latency_ms,
            {"order_id": order_id, "breakdown": component_breakdown},
        )

        # Record individual components if provided
        if component_breakdown:
            for component, latency in component_breakdown.items():
                self.record_component_latency(
                    component,
                    latency,
                    {"order_id": order_id, "total_e2e_ms": total_latency_ms},
                )

    def _update_component_stats(self, component: str, latency_ms: float) -> None:
        """Update component statistics in Redis."""
        if not self.redis_client:
            return

        stats_key = f"latency:{component}:stats"

        # Get or initialize stats
        try:
            stats_json = self.redis_client.get(stats_key)
            if stats_json:
                stats = json.loads(stats_json)
            else:
                stats = {
                    "count": 0,
                    "sum": 0.0,
                    "sum_squares": 0.0,
                    "min": float("inf"),
                    "max": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                }

            # Update running statistics
            stats["count"] += 1
            stats["sum"] += latency_ms
            stats["sum_squares"] += latency_ms * latency_ms
            stats["min"] = min(stats["min"], latency_ms)
            stats["max"] = max(stats["max"], latency_ms)

            # Calculate percentiles from recent history
            history_key = f"latency:{component}:history"
            recent_latencies = []
            history = self.redis_client.lrange(history_key, 0, -1)

            for entry in history:
                try:
                    data = json.loads(entry)
                    recent_latencies.append(data["val"])
                except:
                    continue

            if recent_latencies:
                stats["p95"] = float(np.percentile(recent_latencies, 95))
                stats["p99"] = float(np.percentile(recent_latencies, 99))

            # Store updated stats
            self.redis_client.set(stats_key, json.dumps(stats), ex=3600)

        except Exception as e:
            print(f"⚠️ Stats update error: {e}")

    def _check_budget_violation(
        self, component: str, latency_ms: float, metadata: Dict[str, Any] = None
    ) -> None:
        """Check for latency budget violations and generate alerts."""

        # Get budget for component
        budget_ms = getattr(self.budget, f"{component}_ms", None)
        if budget_ms is None:
            # Use total budget for unknown components
            budget_ms = self.budget.total_budget_ms

        # Calculate threshold violations
        warning_threshold = budget_ms * self.budget.warning_threshold_pct
        critical_threshold = budget_ms * self.budget.critical_threshold_pct

        violation_type = None
        if latency_ms >= critical_threshold:
            violation_type = "CRITICAL"
        elif latency_ms >= warning_threshold:
            violation_type = "WARNING"

        if violation_type:
            alert = {
                "timestamp": datetime.datetime.now().isoformat(),
                "component": component,
                "violation_type": violation_type,
                "latency_ms": latency_ms,
                "budget_ms": budget_ms,
                "threshold_ms": (
                    critical_threshold
                    if violation_type == "CRITICAL"
                    else warning_threshold
                ),
                "budget_utilization_pct": (latency_ms / budget_ms) * 100,
                "metadata": metadata or {},
            }

            self.alert_history.append(alert)

            # Keep only recent alerts
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]

            # Export alert to Redis
            if self.redis_client:
                try:
                    alert_key = f"alerts:latency:{component}"
                    self.redis_client.lpush(alert_key, json.dumps(alert))
                    self.redis_client.ltrim(alert_key, 0, 49)  # Keep last 50 alerts
                    self.redis_client.expire(alert_key, 7200)  # 2 hour TTL

                    # Set current alert status
                    status_key = f"latency:{component}:alert_status"
                    self.redis_client.set(status_key, violation_type, ex=300)

                except Exception as e:
                    print(f"⚠️ Alert export error: {e}")

            print(
                f"🚨 {violation_type}: {component} latency {latency_ms:.1f}ms exceeds {budget_ms:.1f}ms budget"
            )

    def export_budget_status(self) -> Dict[str, Any]:
        """Export current budget status and utilization."""
        status = {
            "timestamp": datetime.datetime.now().isoformat(),
            "budget_config": asdict(self.budget),
            "component_status": {},
            "alerts_last_hour": 0,
            "overall_status": "OK",
        }

        # Analyze recent metrics for each component
        current_time = time.time()
        recent_window = 300  # 5 minutes

        for metric_key, measurements in self.metrics_buffer.items():
            component = metric_key.split(":")[1]

            # Get recent measurements
            recent_measurements = [
                m
                for m in measurements
                if current_time - m["timestamp"] <= recent_window
            ]

            if recent_measurements:
                latencies = [m["latency_ms"] for m in recent_measurements]

                component_status = {
                    "count_5m": len(recent_measurements),
                    "avg_latency_ms": float(np.mean(latencies)),
                    "p95_latency_ms": float(np.percentile(latencies, 95)),
                    "max_latency_ms": float(np.max(latencies)),
                    "budget_ms": getattr(
                        self.budget, f"{component}_ms", self.budget.total_budget_ms
                    ),
                    "budget_utilization_pct": 0,
                }

                budget_ms = component_status["budget_ms"]
                component_status["budget_utilization_pct"] = (
                    component_status["p95_latency_ms"] / budget_ms
                ) * 100

                # Determine component status
                if component_status["budget_utilization_pct"] >= 95:
                    component_status["status"] = "CRITICAL"
                    status["overall_status"] = "CRITICAL"
                elif component_status["budget_utilization_pct"] >= 80:
                    component_status["status"] = "WARNING"
                    if status["overall_status"] == "OK":
                        status["overall_status"] = "WARNING"
                else:
                    component_status["status"] = "OK"

                status["component_status"][component] = component_status

        # Count recent alerts
        hour_ago = current_time - 3600
        recent_alerts = [
            a
            for a in self.alert_history
            if datetime.datetime.fromisoformat(a["timestamp"]).timestamp() >= hour_ago
        ]
        status["alerts_last_hour"] = len(recent_alerts)

        # Export to Redis
        if self.redis_client:
            try:
                self.redis_client.set(
                    "latency:budget_status", json.dumps(status), ex=300
                )
            except Exception as e:
                print(f"⚠️ Budget status export error: {e}")

        return status

    def generate_synthetic_metrics(self, duration_seconds: int = 300) -> None:
        """Generate synthetic latency metrics for testing."""
        print(f"📊 Generating synthetic latency metrics for {duration_seconds}s...")

        components = [
            "signal_generation",
            "feature_engineering",
            "ensemble",
            "position_sizing",
            "risk_checks",
            "order_routing",
            "venue_latency",
        ]

        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            # Simulate processing pipeline
            total_latency = 0
            component_breakdown = {}

            for component in components:
                # Get base budget for component
                base_budget = getattr(self.budget, f"{component}_ms", 10.0)

                # Add realistic variations
                if component == "venue_latency":
                    # Venue latency is more variable
                    latency = np.random.lognormal(np.log(base_budget * 0.6), 0.8)
                elif component == "signal_generation":
                    # Signal generation can spike
                    if np.random.random() < 0.1:  # 10% chance of spike
                        latency = base_budget * np.random.uniform(2, 5)
                    else:
                        latency = np.random.normal(base_budget * 0.7, base_budget * 0.2)
                else:
                    # Other components are more stable
                    latency = np.random.normal(base_budget * 0.8, base_budget * 0.15)

                latency = max(0.1, latency)  # Ensure positive
                component_breakdown[component] = latency
                total_latency += latency

                # Record component latency
                self.record_component_latency(component, latency)

            # Record end-to-end latency
            self.record_end_to_end_latency(
                total_latency, component_breakdown, f"test_order_{int(time.time())}"
            )

            # Random delay between measurements
            time.sleep(np.random.uniform(0.1, 2.0))

        print(
            f"✅ Generated {len(components)} component metrics over {duration_seconds}s"
        )

    def create_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Create data for latency monitoring dashboard."""

        budget_status = self.export_budget_status()

        # Create dashboard-friendly format
        dashboard_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_status": budget_status["overall_status"],
            "total_budget_ms": self.budget.total_budget_ms,
            "components": [],
        }

        for component, status in budget_status["component_status"].items():
            dashboard_data["components"].append(
                {
                    "name": component,
                    "current_p95_ms": status["p95_latency_ms"],
                    "budget_ms": status["budget_ms"],
                    "utilization_pct": status["budget_utilization_pct"],
                    "status": status["status"],
                    "count_5m": status["count_5m"],
                }
            )

        # Sort by utilization
        dashboard_data["components"].sort(
            key=lambda x: x["utilization_pct"], reverse=True
        )

        # Recent alerts
        recent_alerts = self.alert_history[-10:] if self.alert_history else []
        dashboard_data["recent_alerts"] = recent_alerts

        return dashboard_data


def test_latency_budget_exporter():
    """Test latency budget exporter with synthetic data."""

    print("🕐 Testing Latency Budget Exporter")
    print("=" * 40)

    exporter = LatencyBudgetExporter()

    print(f"\n📋 Budget Configuration:")
    print(f"  Signal generation: {exporter.budget.signal_generation_ms}ms")
    print(f"  Feature engineering: {exporter.budget.feature_engineering_ms}ms")
    print(f"  Ensemble: {exporter.budget.ensemble_ms}ms")
    print(f"  Position sizing: {exporter.budget.position_sizing_ms}ms")
    print(f"  Risk checks: {exporter.budget.risk_checks_ms}ms")
    print(f"  Order routing: {exporter.budget.order_routing_ms}ms")
    print(f"  Venue latency: {exporter.budget.venue_latency_ms}ms")
    print(f"  Total budget: {exporter.budget.total_budget_ms}ms")

    # Generate synthetic metrics
    print(f"\n📊 Generating synthetic metrics...")
    exporter.generate_synthetic_metrics(30)  # 30 seconds of data

    # Export budget status
    print(f"\n📈 Exporting budget status...")
    status = exporter.export_budget_status()

    print(f"\n🚦 Budget Status Summary:")
    print(f"  Overall status: {status['overall_status']}")
    print(f"  Alerts last hour: {status['alerts_last_hour']}")

    print(f"\n📊 Component Status:")
    for component, comp_status in status["component_status"].items():
        utilization = comp_status["budget_utilization_pct"]
        status_emoji = (
            "✅"
            if comp_status["status"] == "OK"
            else "⚠️" if comp_status["status"] == "WARNING" else "🚨"
        )

        print(
            f"    {status_emoji} {component}: {utilization:.1f}% ({comp_status['p95_latency_ms']:.1f}/{comp_status['budget_ms']:.1f}ms)"
        )

    # Create dashboard data
    dashboard_data = exporter.create_monitoring_dashboard_data()

    print(f"\n📱 Dashboard Data:")
    print(f"  Status: {dashboard_data['overall_status']}")
    print(f"  Components tracked: {len(dashboard_data['components'])}")
    print(f"  Recent alerts: {len(dashboard_data['recent_alerts'])}")

    print(f"\n✅ Latency budget exporter test complete!")

    return exporter, status


if __name__ == "__main__":
    test_latency_budget_exporter()
