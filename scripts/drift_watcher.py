#!/usr/bin/env python3
"""
Model Drift Watcher using Evidently
Detects data drift and alerts via Slack
"""

import pickle, json, os, datetime
import pandas as pd
import redis
import requests
import numpy as np
from pathlib import Path
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

# Try to import evidently, fall back to simple drift detection if unavailable
try:
    import evidently
    from evidently.report import Report
    from evidently.metrics import DataDriftTable

    EVIDENTLY_AVAILABLE = True
except ImportError:
    print("⚠️ Evidently not available, using simple drift detection")
    EVIDENTLY_AVAILABLE = False

R = redis.Redis(host="localhost", port=6379, decode_responses=True)


def create_baseline_data():
    """Create baseline data if it doesn't exist"""
    baseline_path = "data/baseline_state_sample.parquet"

    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)

    if not os.path.exists(baseline_path):
        print("📊 Creating baseline state sample...")

        # Generate synthetic baseline data
        np.random.seed(42)  # Reproducible baseline
        n_samples = 10000

        baseline_data = pd.DataFrame(
            {
                "price": 50000 + np.cumsum(np.random.randn(n_samples) * 100),
                "volume": np.random.lognormal(8, 1, n_samples),
                "rsi": np.random.uniform(20, 80, n_samples),
                "macd": np.random.normal(0, 0.1, n_samples),
                "sentiment": np.random.uniform(-1, 1, n_samples),
                "entropy": np.random.uniform(0.5, 2.0, n_samples),
                "q_spread": np.random.uniform(20, 100, n_samples),
                "whale_high_impact": np.random.binomial(
                    1, 0.05, n_samples
                ),  # 5% chance
            }
        )

        baseline_data.to_parquet(baseline_path)
        print(f"✅ Created baseline with {n_samples} samples")

    return pd.read_parquet(baseline_path)


def sample_live_data(n=10000):
    """Sample live state data from Redis streams"""
    try:
        # Try to get from policy actions stream (our main data source)
        policy_data = R.xrevrange("policy:actions", count=n)

        if not policy_data:
            print("⚠️ No live policy data available, generating mock data")
            return generate_mock_live_data(n)

        rows = []
        for stream_id, fields in policy_data:
            # Extract timestamp from stream ID
            timestamp_ms = int(stream_id.split("-")[0])

            # Create state row from Redis fields
            row = {
                "price": 50000 + np.random.randn() * 1000,  # Mock price
                "volume": np.random.lognormal(8, 1),  # Mock volume
                "rsi": np.random.uniform(20, 80),  # Mock RSI
                "macd": np.random.normal(0, 0.1),  # Mock MACD
                "sentiment": np.random.uniform(-1, 1),  # Mock sentiment
                "entropy": float(fields.get("entropy", 1.0)),
                "q_spread": float(fields.get("q_spread", 50.0)),
                "whale_high_impact": 0,  # Would get from whale stream
            }
            rows.append(row)

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"⚠️ Error sampling live data: {e}")
        return generate_mock_live_data(n)


def generate_mock_live_data(n=10000):
    """Generate mock live data for testing"""
    # Simulate some drift by shifting distributions
    drift_factor = 0.1  # 10% drift

    return pd.DataFrame(
        {
            "price": 50000 + np.cumsum(np.random.randn(n) * 100 * (1 + drift_factor)),
            "volume": np.random.lognormal(8 * (1 + drift_factor), 1, n),
            "rsi": np.random.uniform(20, 80, n)
            + np.random.randn(n) * drift_factor * 10,
            "macd": np.random.normal(0, 0.1 * (1 + drift_factor), n),
            "sentiment": np.random.uniform(-1, 1, n)
            + np.random.randn(n) * drift_factor,
            "entropy": np.random.uniform(
                0.5 * (1 - drift_factor), 2.0 * (1 + drift_factor), n
            ),
            "q_spread": np.random.uniform(20, 100, n)
            + np.random.randn(n) * drift_factor * 5,
            "whale_high_impact": np.random.binomial(1, 0.05 * (1 + drift_factor), n),
        }
    )


def detect_drift_evidently(baseline_data, live_data):
    """Use Evidently to detect drift"""
    if not EVIDENTLY_AVAILABLE:
        return detect_drift_simple(baseline_data, live_data)

    try:
        # Create drift report
        report = Report(metrics=[DataDriftTable()])
        report.run(reference_data=baseline_data, current_data=live_data)

        # Extract results
        result = report.as_dict()
        drift_detected = result["metrics"][0]["result"]["dataset_drift"]

        return {
            "drift_detected": drift_detected,
            "method": "evidently",
            "details": result["metrics"][0]["result"],
        }

    except Exception as e:
        print(f"⚠️ Evidently drift detection failed: {e}")
        return detect_drift_simple(baseline_data, live_data)


def calculate_kl_divergence(baseline_data, current_data):
    """Calculate KL divergence between baseline and current distributions."""
    try:
        # Standardize data
        scaler = StandardScaler()
        baseline_scaled = scaler.fit_transform(
            baseline_data.select_dtypes(include=[np.number])
        )
        current_scaled = scaler.transform(
            current_data.select_dtypes(include=[np.number])
        )

        # Create histograms for KL divergence calculation
        kl_divs = []

        for i in range(baseline_scaled.shape[1]):
            # Create probability distributions from histograms
            baseline_col = baseline_scaled[:, i]
            current_col = current_scaled[:, i]

            # Use same bins for both distributions
            bins = np.histogram_bin_edges(baseline_col, bins=50)
            p, _ = np.histogram(baseline_col, bins=bins, density=True)
            q, _ = np.histogram(current_col, bins=bins, density=True)

            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            p = p + epsilon
            q = q + epsilon

            # Normalize to ensure they sum to 1
            p = p / p.sum()
            q = q / q.sum()

            # Calculate KL divergence
            kl_div = entropy(p, q)
            kl_divs.append(kl_div)

        return np.mean(kl_divs)

    except Exception as e:
        print(f"⚠️ Error calculating KL divergence: {e}")
        return 0.0


def update_baseline_history(current_data):
    """Keep 7-day rolling baseline as specified in task brief."""
    try:
        baseline_history_path = "data/baseline_history.parquet"

        # Create data directory if needed
        Path("data").mkdir(exist_ok=True)

        # Load existing history or create new
        if os.path.exists(baseline_history_path):
            history_df = pd.read_parquet(baseline_history_path)
        else:
            history_df = pd.DataFrame()

        # Add current data with timestamp
        current_with_ts = current_data.copy()
        current_with_ts["baseline_date"] = datetime.date.today()

        # Append to history
        history_df = pd.concat([history_df, current_with_ts], ignore_index=True)

        # Keep only last 7 days
        cutoff_date = datetime.date.today() - datetime.timedelta(days=7)
        history_df = history_df[history_df["baseline_date"] > cutoff_date]

        # Save updated history
        history_df.to_parquet(baseline_history_path)

        print(
            f"📁 Updated baseline history: {len(history_df)} samples across {history_df['baseline_date'].nunique()} days"
        )

        return history_df

    except Exception as e:
        print(f"⚠️ Error updating baseline history: {e}")
        return current_data


def calculate_adaptive_threshold():
    """Auto-adjust drift thresholds based on historical KL divergence."""
    try:
        # Get historical KL divergence values from Redis zset
        kl_history = R.zrevrange("drift_hist", 0, -1, withscores=True)

        if len(kl_history) < 3:
            return 0.5  # Default threshold

        # Extract KL values (scores)
        kl_values = [float(score) for _, score in kl_history[-7:]]  # Last 7 days

        if len(kl_values) < 3:
            return 0.5

        # Calculate median and IQR
        kl_median = np.median(kl_values)
        q75 = np.percentile(kl_values, 75)
        q25 = np.percentile(kl_values, 25)
        iqr = q75 - q25

        # Set new alert_threshold = median(last 7) + 3×IQR as specified
        new_threshold = kl_median + 3 * iqr

        print(f"📊 Adaptive threshold calculation:")
        print(f"   Median KL: {kl_median:.4f}")
        print(f"   IQR: {iqr:.4f}")
        print(f"   New threshold: {new_threshold:.4f}")

        return max(0.1, min(2.0, new_threshold))  # Clamp between reasonable bounds

    except Exception as e:
        print(f"⚠️ Error calculating adaptive threshold: {e}")
        return 0.5  # Safe default


def detect_drift_simple(baseline_data, live_data):
    """Enhanced drift detection with adaptive thresholds and KL divergence."""
    try:
        # Calculate KL divergence
        kl_div = calculate_kl_divergence(baseline_data, live_data)

        # Store KL divergence in Redis zset with timestamp
        current_timestamp = int(datetime.datetime.now().timestamp())
        R.zadd("drift_hist", {current_timestamp: kl_div})

        # Keep only last 30 days of history
        cutoff_time = current_timestamp - (30 * 24 * 3600)
        R.zremrangebyscore("drift_hist", 0, cutoff_time)

        # Get adaptive threshold
        adaptive_threshold = calculate_adaptive_threshold()

        # Check if drift detected using adaptive threshold
        drift_detected = kl_div > adaptive_threshold

        # Update baseline history
        update_baseline_history(live_data)

        # Traditional feature-level drift detection
        drift_features = []
        for column in baseline_data.columns:
            if column in live_data.columns:
                baseline_mean = baseline_data[column].mean()
                live_mean = live_data[column].mean()
                baseline_std = baseline_data[column].std()

                if baseline_std > 0:
                    z_score = abs(live_mean - baseline_mean) / baseline_std
                    if z_score > 2:  # 2 standard deviations
                        drift_features.append(column)

        # Send Slack notification about threshold change if significant
        threshold_changed = abs(adaptive_threshold - 0.5) > 0.1  # Default was 0.5
        if threshold_changed:
            threshold_message = f"📊 New drift alert threshold set to {adaptive_threshold:.4f} (KL divergence method)"
            send_slack_alert(threshold_message)
            print(f"🔄 {threshold_message}")

        return {
            "drift_detected": drift_detected,
            "method": "adaptive_kl",
            "kl_divergence": kl_div,
            "adaptive_threshold": adaptive_threshold,
            "drifted_features": drift_features,
            "drift_count": len(drift_features),
            "threshold_updated": threshold_changed,
        }

    except Exception as e:
        print(f"⚠️ Error in enhanced drift detection: {e}")
        # Fallback to basic detection
        drift_features = []

        for column in baseline_data.columns:
            if column in live_data.columns:
                baseline_mean = baseline_data[column].mean()
                live_mean = live_data[column].mean()
                baseline_std = baseline_data[column].std()

                if baseline_std > 0:
                    z_score = abs(live_mean - baseline_mean) / baseline_std
                    if z_score > 2:  # 2 standard deviations
                        drift_features.append(column)

        return {
            "drift_detected": len(drift_features) > 0,
            "method": "simple_fallback",
            "drifted_features": drift_features,
            "drift_count": len(drift_features),
        }


def send_slack_alert(message):
    """Send alert to Slack if webhook configured"""
    slack_webhook = os.getenv("SLACK_WEBHOOK")

    if not slack_webhook:
        print(f"📢 ALERT (no Slack webhook): {message}")
        return

    try:
        payload = {"text": message}
        response = requests.post(slack_webhook, json=payload, timeout=10)

        if response.status_code == 200:
            print(f"✅ Slack alert sent: {message}")
        else:
            print(f"❌ Slack alert failed ({response.status_code}): {message}")

    except Exception as e:
        print(f"❌ Slack alert error: {e}")


def main():
    """Main drift detection routine"""
    print(f"🔍 Running drift detection - {datetime.datetime.now()}")

    try:
        # Load or create baseline data
        baseline_data = create_baseline_data()
        print(f"📊 Loaded baseline: {len(baseline_data)} samples")

        # Sample live data
        live_data = sample_live_data(n=10000)
        print(f"📈 Sampled live data: {len(live_data)} samples")

        # Detect drift
        drift_result = detect_drift_evidently(baseline_data, live_data)

        print(f"🔍 Drift detection method: {drift_result['method']}")
        print(f"🎯 Drift detected: {drift_result['drift_detected']}")

        if drift_result["drift_detected"]:
            # Compose alert message
            alert_message = f":warning: Data-drift detected on {datetime.date.today()}!"

            if "drifted_features" in drift_result:
                alert_message += (
                    f" Affected features: {', '.join(drift_result['drifted_features'])}"
                )

            if "drift_count" in drift_result:
                alert_message += f" ({drift_result['drift_count']} features)"

            alert_message += f" Method: {drift_result['method']}"

            # Send alert
            send_slack_alert(alert_message)

            # Store drift result in Redis
            drift_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "drift_detected": True,
                "method": drift_result["method"],
                "details": drift_result,
            }
            R.set(
                "model:last_drift_check", json.dumps(drift_record), ex=86400 * 7
            )  # Keep for 1 week

            print("🚨 DRIFT ALERT SENT")

        else:
            print("✅ No significant drift detected")

            # Store healthy result
            healthy_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "drift_detected": False,
                "method": drift_result["method"],
            }
            R.set("model:last_drift_check", json.dumps(healthy_record), ex=86400 * 7)

    except Exception as e:
        error_message = f"❌ Drift detection failed: {e}"
        print(error_message)
        send_slack_alert(f":x: Drift watcher error: {e}")


if __name__ == "__main__":
    main()
