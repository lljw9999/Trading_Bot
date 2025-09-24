#!/usr/bin/env python3
"""
Model Drift Detection for SAC-DiF RL Trading Bot
Uses Evidently AI to detect when the RL model performance degrades
"""

import pandas as pd
import redis
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# Uncomment when you install evidently
# pip install evidently
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import *

    EVIDENTLY_AVAILABLE = True
except ImportError:
    print("⚠️ Evidently not installed. Install with: pip install evidently")
    EVIDENTLY_AVAILABLE = False


@dataclass
class DriftAlert:
    """Data drift alert structure"""

    timestamp: datetime
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    drift_type: str  # DATA, TARGET, PERFORMANCE
    affected_features: List[str]
    drift_score: float
    recommended_action: str
    details: Dict


class RLModelDriftDetector:
    """Detect drift in RL model performance and data distribution"""

    def __init__(self, redis_client=None):
        self.redis_client = redis_client or redis.Redis(
            host="localhost", port=6379, decode_responses=True
        )
        self.baseline_window_days = 7  # Use 7 days as baseline
        self.comparison_window_hours = 24  # Compare last 24 hours

        # Feature columns for drift detection
        self.feature_columns = [
            "entropy",
            "q_spread",
            "price_change",
            "volume",
            "rsi",
            "macd",
            "bb_position",
            "sentiment_score",
        ]

        # Performance metrics to track
        self.performance_metrics = [
            "sharpe_ratio",
            "win_rate",
            "avg_return",
            "max_drawdown",
            "profit_factor",
            "avg_trade_duration",
        ]

    def collect_rl_data(self, hours_back: int = 24) -> pd.DataFrame:
        """Collect RL policy data for drift analysis"""
        try:
            # Get policy actions from Redis stream
            cutoff_time = int(
                (datetime.now() - timedelta(hours=hours_back)).timestamp() * 1000
            )

            policy_actions = self.redis_client.xrevrange("policy:actions", count=1000)

            data_rows = []
            for stream_id, fields in policy_actions:
                timestamp_ms = int(stream_id.split("-")[0])
                if timestamp_ms >= cutoff_time:

                    # Extract features
                    entropy = float(fields.get("entropy", 0))
                    q_spread = float(fields.get("q_spread", 0))

                    # Simulate additional features (in production, get from actual data)
                    price_change = np.random.normal(0, 0.02)  # Mock price change
                    volume = np.random.lognormal(10, 0.5)  # Mock volume
                    rsi = np.random.uniform(20, 80)  # Mock RSI
                    macd = np.random.normal(0, 0.1)  # Mock MACD
                    bb_position = np.random.uniform(0, 1)  # Mock Bollinger position
                    sentiment_score = np.random.uniform(-1, 1)  # Mock sentiment

                    data_rows.append(
                        {
                            "timestamp": datetime.fromtimestamp(timestamp_ms / 1000),
                            "entropy": entropy,
                            "q_spread": q_spread,
                            "price_change": price_change,
                            "volume": volume,
                            "rsi": rsi,
                            "macd": macd,
                            "bb_position": bb_position,
                            "sentiment_score": sentiment_score,
                        }
                    )

            return pd.DataFrame(data_rows)

        except Exception as e:
            print(f"❌ Error collecting RL data: {e}")
            return pd.DataFrame()

    def collect_performance_data(self, hours_back: int = 24) -> pd.DataFrame:
        """Collect trading performance metrics"""
        try:
            # In production, this would come from trading results
            # For demo, simulate performance metrics

            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)

            # Generate hourly performance data
            performance_data = []
            current_time = start_time

            while current_time <= end_time:
                # Simulate degrading performance over time for demo
                age_hours = (end_time - current_time).total_seconds() / 3600
                performance_decay = max(0.5, 1 - age_hours / 168)  # Decay over 7 days

                perf_row = {
                    "timestamp": current_time,
                    "sharpe_ratio": np.random.normal(1.5 * performance_decay, 0.3),
                    "win_rate": np.random.normal(0.6 * performance_decay, 0.1),
                    "avg_return": np.random.normal(0.002 * performance_decay, 0.001),
                    "max_drawdown": np.random.normal(0.05 / performance_decay, 0.01),
                    "profit_factor": np.random.normal(1.3 * performance_decay, 0.2),
                    "avg_trade_duration": np.random.normal(300, 50),  # seconds
                }

                performance_data.append(perf_row)
                current_time += timedelta(hours=1)

            return pd.DataFrame(performance_data)

        except Exception as e:
            print(f"❌ Error collecting performance data: {e}")
            return pd.DataFrame()

    def detect_data_drift(
        self, baseline_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> Dict:
        """Detect drift in input features"""
        if not EVIDENTLY_AVAILABLE:
            return self._simulate_drift_detection(baseline_data, current_data)

        try:
            # Column mapping for Evidently
            column_mapping = ColumnMapping()
            column_mapping.numerical_features = self.feature_columns

            # Create drift report
            data_drift_report = Report(
                metrics=[
                    DatasetDriftMetric(),
                    DatasetMissingValuesMetric(),
                    DataDriftTable(),
                ]
            )

            data_drift_report.run(
                reference_data=baseline_data[self.feature_columns],
                current_data=current_data[self.feature_columns],
                column_mapping=column_mapping,
            )

            # Extract results
            results = data_drift_report.as_dict()

            drift_detected = results["metrics"][0]["result"]["dataset_drift"]
            drift_score = results["metrics"][0]["result"]["drift_share"]

            drifted_features = []
            if len(results["metrics"]) > 2:  # DataDriftTable available
                for feature_result in results["metrics"][2]["result"][
                    "drift_by_columns"
                ].values():
                    if feature_result["drift_detected"]:
                        drifted_features.append(feature_result["column_name"])

            return {
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "drifted_features": drifted_features,
                "total_features": len(self.feature_columns),
                "drift_share": drift_score,
            }

        except Exception as e:
            print(f"⚠️ Evidently drift detection error: {e}")
            return self._simulate_drift_detection(baseline_data, current_data)

    def _simulate_drift_detection(
        self, baseline_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> Dict:
        """Simulate drift detection when Evidently is not available"""
        drift_detected = False
        drifted_features = []
        drift_score = 0.0

        if len(baseline_data) > 0 and len(current_data) > 0:
            # Simple statistical drift detection
            for feature in self.feature_columns:
                if feature in baseline_data.columns and feature in current_data.columns:
                    baseline_mean = baseline_data[feature].mean()
                    current_mean = current_data[feature].mean()
                    baseline_std = baseline_data[feature].std()

                    if baseline_std > 0:
                        # Z-score based drift detection
                        z_score = abs(current_mean - baseline_mean) / baseline_std
                        if z_score > 2:  # Significant change
                            drift_detected = True
                            drifted_features.append(feature)
                            drift_score += z_score

            drift_score = min(1.0, drift_score / len(self.feature_columns))

        return {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "drifted_features": drifted_features,
            "total_features": len(self.feature_columns),
            "drift_share": drift_score,
        }

    def detect_performance_drift(
        self, baseline_perf: pd.DataFrame, current_perf: pd.DataFrame
    ) -> Dict:
        """Detect drift in model performance metrics"""
        try:
            performance_drift = {
                "drift_detected": False,
                "degraded_metrics": [],
                "performance_change": {},
            }

            for metric in self.performance_metrics:
                if metric in baseline_perf.columns and metric in current_perf.columns:
                    baseline_mean = baseline_perf[metric].mean()
                    current_mean = current_perf[metric].mean()

                    # Calculate percentage change
                    if baseline_mean != 0:
                        pct_change = (current_mean - baseline_mean) / abs(baseline_mean)
                        performance_drift["performance_change"][metric] = pct_change

                        # Detect significant degradation (>20% drop for positive metrics)
                        if metric in [
                            "sharpe_ratio",
                            "win_rate",
                            "avg_return",
                            "profit_factor",
                        ]:
                            if pct_change < -0.2:  # 20% degradation
                                performance_drift["drift_detected"] = True
                                performance_drift["degraded_metrics"].append(metric)
                        elif metric in ["max_drawdown"]:  # Negative metric
                            if pct_change > 0.2:  # 20% increase in drawdown
                                performance_drift["drift_detected"] = True
                                performance_drift["degraded_metrics"].append(metric)

            return performance_drift

        except Exception as e:
            print(f"❌ Performance drift detection error: {e}")
            return {
                "drift_detected": False,
                "degraded_metrics": [],
                "performance_change": {},
            }

    def generate_drift_alerts(
        self, data_drift: Dict, perf_drift: Dict
    ) -> List[DriftAlert]:
        """Generate actionable drift alerts"""
        alerts = []

        # Data drift alerts
        if data_drift["drift_detected"]:
            severity = (
                "CRITICAL"
                if data_drift["drift_score"] > 0.7
                else (
                    "HIGH"
                    if data_drift["drift_score"] > 0.4
                    else "MEDIUM" if data_drift["drift_score"] > 0.2 else "LOW"
                )
            )

            recommended_action = {
                "CRITICAL": "Immediate model retraining required",
                "HIGH": "Schedule model retraining within 24 hours",
                "MEDIUM": "Monitor closely, consider retraining",
                "LOW": "Continue monitoring",
            }[severity]

            alerts.append(
                DriftAlert(
                    timestamp=datetime.now(),
                    severity=severity,
                    drift_type="DATA",
                    affected_features=data_drift["drifted_features"],
                    drift_score=data_drift["drift_score"],
                    recommended_action=recommended_action,
                    details=data_drift,
                )
            )

        # Performance drift alerts
        if perf_drift["drift_detected"]:
            severity = (
                "CRITICAL"
                if len(perf_drift["degraded_metrics"]) >= 3
                else "HIGH" if len(perf_drift["degraded_metrics"]) >= 2 else "MEDIUM"
            )

            recommended_action = {
                "CRITICAL": "Stop trading and retrain model immediately",
                "HIGH": "Reduce position sizes and retrain model",
                "MEDIUM": "Monitor performance closely",
            }[severity]

            alerts.append(
                DriftAlert(
                    timestamp=datetime.now(),
                    severity=severity,
                    drift_type="PERFORMANCE",
                    affected_features=perf_drift["degraded_metrics"],
                    drift_score=len(perf_drift["degraded_metrics"])
                    / len(self.performance_metrics),
                    recommended_action=recommended_action,
                    details=perf_drift,
                )
            )

        return alerts

    def store_drift_results(self, alerts: List[DriftAlert]):
        """Store drift detection results in Redis"""
        try:
            # Store current alerts
            alert_data = [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity,
                    "drift_type": alert.drift_type,
                    "affected_features": alert.affected_features,
                    "drift_score": alert.drift_score,
                    "recommended_action": alert.recommended_action,
                    "details": alert.details,
                }
                for alert in alerts
            ]

            self.redis_client.set(
                "model:drift_alerts", json.dumps(alert_data), ex=86400
            )

            # Add to time series
            timestamp = int(time.time() * 1000)

            drift_summary = {
                "timestamp": timestamp,
                "alert_count": len(alerts),
                "max_severity": max([a.severity for a in alerts] + ["NONE"]),
                "drift_types": list(set(a.drift_type for a in alerts)),
            }

            self.redis_client.xadd("model:drift_history", drift_summary, maxlen=1000)

            print(f"✅ Stored {len(alerts)} drift alerts")

        except Exception as e:
            print(f"❌ Error storing drift results: {e}")

    def run_drift_monitoring(self):
        """Execute complete drift monitoring cycle"""
        print("🔍 Starting Model Drift Detection...")

        try:
            # Collect baseline data (last 7 days, excluding recent 24 hours)
            baseline_data = self.collect_rl_data(hours_back=168)  # 7 days
            baseline_data = baseline_data[
                baseline_data["timestamp"] < (datetime.now() - timedelta(hours=24))
            ]

            baseline_perf = self.collect_performance_data(hours_back=168)
            baseline_perf = baseline_perf[
                baseline_perf["timestamp"] < (datetime.now() - timedelta(hours=24))
            ]

            # Collect current data (last 24 hours)
            current_data = self.collect_rl_data(hours_back=24)
            current_perf = self.collect_performance_data(hours_back=24)

            print(
                f"📊 Data collected: {len(baseline_data)} baseline, {len(current_data)} current samples"
            )

            if len(baseline_data) < 100 or len(current_data) < 10:
                print("⚠️ Insufficient data for drift detection")
                return

            # Detect drifts
            data_drift = self.detect_data_drift(baseline_data, current_data)
            perf_drift = self.detect_performance_drift(baseline_perf, current_perf)

            print(
                f"🔍 Data drift detected: {data_drift['drift_detected']} (score: {data_drift['drift_score']:.3f})"
            )
            print(f"🔍 Performance drift detected: {perf_drift['drift_detected']}")

            # Generate alerts
            alerts = self.generate_drift_alerts(data_drift, perf_drift)

            # Store results
            self.store_drift_results(alerts)

            # Log alerts
            for alert in alerts:
                print(
                    f"🚨 {alert.severity} {alert.drift_type} DRIFT: {alert.recommended_action}"
                )
                if alert.affected_features:
                    print(f"   Affected: {', '.join(alert.affected_features)}")

            if not alerts:
                print("✅ No significant drift detected - model is healthy")

        except Exception as e:
            print(f"❌ Drift monitoring error: {e}")


def main():
    """Run drift monitoring demo"""
    detector = RLModelDriftDetector()
    detector.run_drift_monitoring()


if __name__ == "__main__":
    main()
