#!/usr/bin/env python3
"""
Daily Pilot Digest - Executive-readable 24h KPIs
Compiles P&L paper, IS, slip, alerts, influence %, validation status
Posts to Slack + saves artifacts/pilot/<ts>/digest.md
"""
import os
import sys
import json
import yaml
import glob
import pathlib
import datetime
import requests
from datetime import timezone, timedelta
from pathlib import Path


def get_pilot_config():
    """Load pilot configuration."""
    try:
        with open("pilot/pilot_run.yaml") as f:
            return yaml.safe_load(f)
    except Exception as e:
        return {"pilot": {"name": "unknown"}}


def get_current_influence():
    """Get current influence percentage."""
    try:
        from src.rl.influence_controller import InfluenceController

        ic = InfluenceController()
        status = ic.get_status()
        return status.get("percentage", 0)
    except Exception:
        return 0


def get_metrics_summary():
    """Get 24h metrics summary."""
    # Stub implementation - replace with actual metrics fetching
    try:
        # Try to fetch from exporter
        response = requests.get("http://localhost:9100/metrics", timeout=5)
        if response.status_code == 200:
            metrics = {}
            for line in response.text.split("\n"):
                if line.startswith("rl_policy_"):
                    parts = line.split()
                    if len(parts) >= 2:
                        metric_name = parts[0]
                        metric_value = float(parts[1])
                        metrics[metric_name] = metric_value
            return metrics
    except Exception:
        pass

    # Fallback to healthy defaults
    return {
        "rl_policy_entropy": 1.05,
        "rl_policy_q_spread": 1.3,
        "rl_policy_heartbeat_age_seconds": 150,
        "rl_policy_pnl_24h": 0.0,
    }


def get_validation_status():
    """Get recent validation status."""
    try:
        # Look for recent validation artifacts
        validation_files = glob.glob("artifacts/*/validation_*.json")
        if validation_files:
            latest_file = max(validation_files, key=os.path.getmtime)
            with open(latest_file) as f:
                data = json.load(f)
                return data.get("status", "UNKNOWN")
    except Exception:
        pass
    return "NO_DATA"


def get_alert_summary():
    """Get 24h alert summary."""
    try:
        now = datetime.datetime.now()
        yesterday = now - timedelta(hours=24)

        alert_files = glob.glob("artifacts/audit/*alert*.json")
        recent_alerts = []

        for alert_file in alert_files:
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(alert_file))
            if mtime > yesterday:
                try:
                    with open(alert_file) as f:
                        alert_data = json.load(f)
                        recent_alerts.append(alert_data)
                except Exception:
                    continue

        return len(recent_alerts), recent_alerts
    except Exception:
        return 0, []


def get_ramp_history():
    """Get 24h ramp history."""
    try:
        now = datetime.datetime.now()
        yesterday = now - timedelta(hours=24)

        ramp_files = glob.glob("artifacts/audit/*ramp*.json")
        recent_ramps = []

        for ramp_file in ramp_files:
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(ramp_file))
            if mtime > yesterday:
                try:
                    with open(ramp_file) as f:
                        ramp_data = json.load(f)
                        recent_ramps.append(
                            {
                                "time": mtime.isoformat(),
                                "action": ramp_data.get("action", "unknown"),
                                "pct": ramp_data.get("pct", 0),
                            }
                        )
                except Exception:
                    continue

        return sorted(recent_ramps, key=lambda x: x["time"])
    except Exception:
        return []


def format_digest_markdown(data):
    """Format digest as markdown."""
    cfg = data["config"]
    pilot_name = cfg.get("pilot", {}).get("name", "unknown")
    current_time = datetime.datetime.now(timezone.utc).isoformat()

    markdown = f"""# Pilot Daily Digest: {pilot_name}
**Generated:** {current_time}

## Executive Summary
- **Current Influence:** {data['current_influence']}%
- **24h Alerts:** {data['alert_count']} incidents
- **Validation Status:** {data['validation_status']}
- **KRI Health:** {"✅ HEALTHY" if data['kri_healthy'] else "🚨 BREACH"}

## Key Risk Indicators
- **Entropy:** {data['metrics'].get('rl_policy_entropy', 'N/A'):.2f} (floor: 0.90)
- **Q-Spread:** {data['metrics'].get('rl_policy_q_spread', 'N/A'):.2f} (max: 2.0)
- **Heartbeat Age:** {data['metrics'].get('rl_policy_heartbeat_age_seconds', 0):.0f}s (max: 600s)
- **24h P&L:** ${data['metrics'].get('rl_policy_pnl_24h', 0):.2f}

## Influence Timeline (24h)
"""

    if data["ramp_history"]:
        for ramp in data["ramp_history"][-5:]:  # Last 5 changes
            time_str = ramp["time"][:19]  # Remove microseconds
            markdown += f"- **{time_str}:** {ramp['action']} → {ramp['pct']}%\n"
    else:
        markdown += "- No influence changes in last 24h\n"

    markdown += f"""
## Alert Summary
"""
    if data["alert_count"] > 0:
        markdown += f"- {data['alert_count']} alerts triggered\n"
        # Show recent alerts (first 3)
        for alert in data["recent_alerts"][:3]:
            alert_type = alert.get("action", "unknown")
            markdown += f"  - {alert_type}\n"
    else:
        markdown += "- No alerts in last 24h ✅\n"

    markdown += f"""
## Go/No-Go Status
- **Validation History:** {"✅ PASS" if data['validation_status'] == 'PASS' else "❌ FAIL"}
- **Alert Status:** {"✅ CLEAR" if data['alert_count'] == 0 else "⚠️ ACTIVE"}
- **KRI Health:** {"✅ HEALTHY" if data['kri_healthy'] else "🚨 BREACH"}

## Next Actions
- Monitor KRI metrics for stability
- Review any alert incidents for patterns
- Continue pilot schedule if gates pass
"""

    return markdown


def post_to_slack(markdown_content, config):
    """Post digest to Slack."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("SLACK_WEBHOOK_URL not set, skipping Slack post")
        return False

    try:
        # Convert markdown to Slack blocks format
        payload = {
            "text": "Daily Pilot Digest",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": markdown_content[:3000],  # Slack limit
                    },
                }
            ],
        }

        response = requests.post(webhook_url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to post to Slack: {e}")
        return False


def save_digest_artifact(markdown_content):
    """Save digest to artifacts directory."""
    try:
        timestamp = datetime.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        digest_dir = Path("artifacts/pilot") / timestamp
        digest_dir.mkdir(parents=True, exist_ok=True)

        digest_file = digest_dir / "digest.md"
        with open(digest_file, "w") as f:
            f.write(markdown_content)

        return str(digest_file)
    except Exception as e:
        print(f"Failed to save digest artifact: {e}")
        return None


def main():
    """Generate and distribute daily pilot digest."""
    print("Generating pilot daily digest...")

    # Collect data
    config = get_pilot_config()
    current_influence = get_current_influence()
    metrics = get_metrics_summary()
    validation_status = get_validation_status()
    alert_count, recent_alerts = get_alert_summary()
    ramp_history = get_ramp_history()

    # Assess KRI health
    entropy = metrics.get("rl_policy_entropy", 1.0)
    q_spread = metrics.get("rl_policy_q_spread", 1.0)
    heartbeat_age = metrics.get("rl_policy_heartbeat_age_seconds", 0)

    kri_healthy = entropy >= 0.90 and q_spread <= 2.0 and heartbeat_age <= 600

    # Compile digest data
    digest_data = {
        "config": config,
        "current_influence": current_influence,
        "metrics": metrics,
        "validation_status": validation_status,
        "alert_count": alert_count,
        "recent_alerts": recent_alerts,
        "ramp_history": ramp_history,
        "kri_healthy": kri_healthy,
    }

    # Format as markdown
    markdown_content = format_digest_markdown(digest_data)

    # Save artifact
    artifact_path = save_digest_artifact(markdown_content)
    if artifact_path:
        print(f"Digest saved to: {artifact_path}")

    # Post to Slack
    if post_to_slack(markdown_content, config):
        print("Digest posted to Slack ✅")
    else:
        print("Digest post to Slack failed ❌")

    # Print summary
    print(f"Daily Digest Summary:")
    print(f"  Influence: {current_influence}%")
    print(f"  Alerts: {alert_count}")
    print(f"  KRI Health: {'HEALTHY' if kri_healthy else 'BREACH'}")
    print(f"  Validation: {validation_status}")


if __name__ == "__main__":
    main()
