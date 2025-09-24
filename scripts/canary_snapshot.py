#!/usr/bin/env python3
"""
Canary Watch Dashboard Snapshotter
Captures key metrics every 5 minutes during canary deployment
"""
import os
import sys
import json
import time
import datetime
import pathlib
import argparse
import requests
from datetime import timezone


def get_redis_metrics():
    """Get metrics from Redis if available."""
    try:
        import redis

        r = redis.Redis(decode_responses=True)

        metrics = {}

        # Try to get key metrics from Redis
        redis_keys = [
            "policy:allowed_influence_pct",
            "gpu:mem_frac",
            "alerts:policy",
            "ops:go_live",
        ]

        for key in redis_keys:
            try:
                value = r.get(key)
                metrics[key] = value
            except Exception:
                metrics[key] = None

        # Get list length for alerts
        try:
            alert_count = r.llen("alerts:policy")
            metrics["alert_count"] = alert_count
        except Exception:
            metrics["alert_count"] = 0

        return metrics

    except Exception as e:
        return {"error": f"Redis unavailable: {e}"}


def get_exporter_metrics():
    """Get metrics from Prometheus exporter if available."""
    try:
        response = requests.get("http://localhost:9100/metrics", timeout=5)
        if response.status_code != 200:
            return {"error": "Exporter not available"}

        metrics = {}
        for line in response.text.split("\n"):
            if line.startswith("rl_policy_"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        metric_name = parts[0]
                        metric_value = float(parts[1])
                        metrics[metric_name] = metric_value
                    except ValueError:
                        continue

        return metrics

    except Exception as e:
        return {"error": f"Exporter unavailable: {e}"}


def get_influence_controller_status():
    """Get status from influence controller."""
    try:
        sys.path.insert(0, "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")
        from src.rl.influence_controller import InfluenceController

        ic = InfluenceController()
        status = ic.get_status()
        return status

    except Exception as e:
        return {"error": f"Influence controller unavailable: {e}"}


def get_system_health():
    """Get basic system health indicators."""
    try:
        import psutil

        health = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "load_avg": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None,
        }

        return health

    except Exception as e:
        return {"error": f"System health unavailable: {e}"}


def capture_snapshot():
    """Capture complete metrics snapshot."""
    timestamp = datetime.datetime.now(timezone.utc)

    print(f"📸 Capturing snapshot at {timestamp.isoformat()}")

    snapshot = {
        "timestamp": timestamp.isoformat(),
        "capture_time_unix": timestamp.timestamp(),
        "snapshot_version": "1.0",
    }

    # Gather metrics from all sources
    print("  📊 Getting Redis metrics...")
    snapshot["redis_metrics"] = get_redis_metrics()

    print("  📈 Getting exporter metrics...")
    snapshot["exporter_metrics"] = get_exporter_metrics()

    print("  🎛️ Getting influence controller status...")
    snapshot["influence_status"] = get_influence_controller_status()

    print("  🖥️ Getting system health...")
    snapshot["system_health"] = get_system_health()

    # Derived summary metrics
    summary = {
        "current_influence_pct": 0,
        "entropy": None,
        "q_spread": None,
        "heartbeat_age_seconds": None,
        "alert_count": 0,
        "system_healthy": True,
    }

    # Extract key metrics for summary
    if (
        isinstance(snapshot["influence_status"], dict)
        and "percentage" in snapshot["influence_status"]
    ):
        summary["current_influence_pct"] = snapshot["influence_status"]["percentage"]

    if isinstance(snapshot["exporter_metrics"], dict):
        summary["entropy"] = snapshot["exporter_metrics"].get("rl_policy_entropy")
        summary["q_spread"] = snapshot["exporter_metrics"].get("rl_policy_q_spread")
        summary["heartbeat_age_seconds"] = snapshot["exporter_metrics"].get(
            "rl_policy_heartbeat_age_seconds"
        )

    if isinstance(snapshot["redis_metrics"], dict):
        summary["alert_count"] = snapshot["redis_metrics"].get("alert_count", 0)

    # System health check
    if (
        isinstance(snapshot["system_health"], dict)
        and "error" not in snapshot["system_health"]
    ):
        cpu_ok = snapshot["system_health"].get("cpu_percent", 0) < 90
        mem_ok = snapshot["system_health"].get("memory_percent", 0) < 90
        disk_ok = snapshot["system_health"].get("disk_percent", 0) < 90
        summary["system_healthy"] = cpu_ok and mem_ok and disk_ok

    snapshot["summary"] = summary

    print(f"  ✅ Snapshot captured: {summary['current_influence_pct']}% influence")
    if summary["entropy"]:
        print(f"      Entropy: {summary['entropy']:.2f}")
    if summary["alert_count"] > 0:
        print(f"      ⚠️ {summary['alert_count']} alerts")

    return snapshot


def save_snapshot(snapshot, output_file):
    """Save snapshot to JSONL file."""
    # Ensure output directory exists
    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Append to JSONL file
    with open(output_file, "a") as f:
        f.write(json.dumps(snapshot, separators=(",", ":")) + "\n")

    print(f"  💾 Snapshot saved to {output_file}")


def analyze_snapshots(snapshot_file):
    """Analyze existing snapshots for trends."""
    if not os.path.exists(snapshot_file):
        return {"error": "No snapshots file found"}

    snapshots = []
    try:
        with open(snapshot_file, "r") as f:
            for line in f:
                if line.strip():
                    snapshots.append(json.loads(line))
    except Exception as e:
        return {"error": f"Failed to read snapshots: {e}"}

    if len(snapshots) < 2:
        return {"message": "Not enough snapshots for trend analysis"}

    # Analyze trends
    analysis = {
        "total_snapshots": len(snapshots),
        "time_span_minutes": 0,
        "influence_trend": "stable",
        "entropy_trend": "stable",
        "alert_trend": "stable",
    }

    if len(snapshots) >= 2:
        first_time = datetime.datetime.fromisoformat(
            snapshots[0]["timestamp"]
        ).timestamp()
        last_time = datetime.datetime.fromisoformat(
            snapshots[-1]["timestamp"]
        ).timestamp()
        analysis["time_span_minutes"] = int((last_time - first_time) / 60)

        # Influence trend
        influences = [
            s["summary"]["current_influence_pct"] for s in snapshots if "summary" in s
        ]
        if len(influences) >= 2:
            if influences[-1] > influences[0]:
                analysis["influence_trend"] = "increasing"
            elif influences[-1] < influences[0]:
                analysis["influence_trend"] = "decreasing"

        # Entropy trend (if available)
        entropies = [
            s["summary"]["entropy"]
            for s in snapshots
            if "summary" in s and s["summary"]["entropy"] is not None
        ]
        if len(entropies) >= 2:
            if entropies[-1] > entropies[0]:
                analysis["entropy_trend"] = "increasing"
            elif entropies[-1] < entropies[0]:
                analysis["entropy_trend"] = "decreasing"

        # Alert trend
        alerts = [s["summary"]["alert_count"] for s in snapshots if "summary" in s]
        if len(alerts) >= 2:
            if alerts[-1] > alerts[0]:
                analysis["alert_trend"] = "increasing"
            elif alerts[-1] < alerts[0]:
                analysis["alert_trend"] = "decreasing"

    return analysis


def main():
    """Main canary snapshot function."""
    parser = argparse.ArgumentParser(
        description="Canary deployment metrics snapshotter"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="artifacts/golive/current/snapshots.jsonl",
        help="Output JSONL file for snapshots",
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=5, help="Snapshot interval in minutes"
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=0,
        help="Total duration in minutes (0 = single snapshot)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze existing snapshots instead of capturing new ones",
    )
    args = parser.parse_args()

    if args.analyze:
        print("📈 Analyzing existing snapshots...")
        analysis = analyze_snapshots(args.output)
        print(json.dumps(analysis, indent=2))
        return 0

    print("📸 Canary Dashboard Snapshotter")
    print("=" * 40)
    print(f"Output: {args.output}")
    print(f"Interval: {args.interval} minutes")
    if args.duration > 0:
        print(f"Duration: {args.duration} minutes")
    else:
        print("Mode: Single snapshot")
    print("=" * 40)

    if args.duration > 0:
        # Continuous monitoring mode
        start_time = time.time()
        snapshot_count = 0

        while time.time() - start_time < args.duration * 60:
            snapshot = capture_snapshot()
            save_snapshot(snapshot, args.output)
            snapshot_count += 1

            elapsed = int((time.time() - start_time) / 60)
            remaining = args.duration - elapsed

            if remaining > 0:
                print(f"  ⏱️ {elapsed}m elapsed, {remaining}m remaining...")
                print(f"  😴 Sleeping {args.interval} minutes until next snapshot...")
                time.sleep(args.interval * 60)
            else:
                break

        print(f"✅ Continuous monitoring complete: {snapshot_count} snapshots captured")

        # Final analysis
        analysis = analyze_snapshots(args.output)
        print("\n📈 Final Analysis:")
        print(json.dumps(analysis, indent=2))

    else:
        # Single snapshot mode
        snapshot = capture_snapshot()
        save_snapshot(snapshot, args.output)
        print("✅ Single snapshot captured")

    return 0


if __name__ == "__main__":
    sys.exit(main())
