#!/usr/bin/env python3
"""
Model Card Builder
Generates human-readable model card from YAML configuration and artifacts
"""
import os
import sys
import yaml
import json
import glob
import argparse
import datetime
from pathlib import Path


def load_model_card_yaml(yaml_file):
    """Load model card YAML configuration."""
    try:
        with open(yaml_file, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML: {e}")
        sys.exit(1)


def get_latest_postmortem():
    """Get latest postmortem data if available."""
    try:
        postmortem_files = glob.glob("artifacts/pilot/*/postmortem.json")
        if postmortem_files:
            latest_file = max(postmortem_files, key=os.path.getmtime)
            with open(latest_file, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def get_latest_ab_results():
    """Get latest A/B test results if available."""
    try:
        ab_files = glob.glob("artifacts/ab/*/ab_results.json")
        if ab_files:
            latest_file = max(ab_files, key=os.path.getmtime)
            with open(latest_file, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def get_validation_summary():
    """Get validation summary from recent artifacts."""
    try:
        validation_files = glob.glob("artifacts/*/validation_*.json")
        if validation_files:
            recent_validations = []
            for file in validation_files[-5:]:  # Last 5
                with open(file, "r") as f:
                    data = json.load(f)
                    recent_validations.append(data.get("status", "UNKNOWN"))

            pass_count = recent_validations.count("PASS")
            total_count = len(recent_validations)
            return {
                "pass_rate": f"{pass_count}/{total_count} ({100*pass_count/total_count:.1f}%)",
                "recent_status": (
                    recent_validations[-1] if recent_validations else "NO_DATA"
                ),
            }
    except Exception:
        pass
    return {"pass_rate": "N/A", "recent_status": "NO_DATA"}


def format_risk_section(risk_config):
    """Format the risk management section."""
    risk_md = f"""## Risk Management

### Risk Limits
- **Entropy Floor:** {risk_config.get('entropy_floor', 'N/A')}
- **Q-Spread Guard:** {risk_config.get('qspread_guard', 'N/A')}
- **Daily Drawdown Max:** {risk_config.get('drawdown_day_max', 'N/A')}
- **Max Position Size:** {risk_config.get('max_position_size', 'N/A')}

### Kill Switch Triggers
"""
    for trigger in risk_config.get("kill_switch_triggers", []):
        risk_md += f"- {trigger}\n"

    risk_md += f"""
### Influence Limits
- **Pilot Maximum:** {risk_config.get('influence_limits', {}).get('pilot_max', 'N/A')}
- **Production Maximum:** {risk_config.get('influence_limits', {}).get('production_max', 'N/A')}
- **Ramp Schedule:** {risk_config.get('influence_limits', {}).get('ramp_schedule', 'N/A')}
"""
    return risk_md


def format_governance_section(governance_config):
    """Format the governance and controls section."""
    gov_md = f"""## Governance & Controls

### Audit Trail
"""
    for path in governance_config.get("worm_audit_paths", []):
        gov_md += f"- `{path}`\n"

    gov_md += f"""
### Operational Controls
"""
    for control in governance_config.get("controls", []):
        gov_md += f"- {control}\n"

    gov_md += f"""
### Approval Chain
"""
    for approver in governance_config.get("approval_chain", []):
        gov_md += f"- {approver}\n"

    gov_md += f"""
### Monitoring Infrastructure
#### Prometheus Metrics
"""
    for metric in governance_config.get("monitoring", {}).get("prometheus_metrics", []):
        gov_md += f"- `{metric}`\n"

    gov_md += f"""
#### Alert Rules
"""
    for alert in governance_config.get("monitoring", {}).get("alert_rules", []):
        gov_md += f"- {alert}\n"

    return gov_md


def generate_model_card_markdown(
    card_config, postmortem_data=None, ab_results=None, validation_summary=None
):
    """Generate comprehensive model card in markdown format."""

    # Update evaluation section with real data if available
    if ab_results:
        card_config["evaluation"]["ab_test"] = {
            "delta_pnl": ab_results.get("delta_pnl_mean", "N/A"),
            "ci95": ab_results.get("ci95", ["N/A", "N/A"]),
            "verdict": ab_results.get("verdict", "PENDING"),
            "sample_size": ab_results.get("n", "N/A"),
        }

    if validation_summary:
        card_config["evaluation"]["offline_gate"]["pass_rate"] = validation_summary[
            "pass_rate"
        ]

    markdown = f"""# Model Card: {card_config['model']}

**Version:** {card_config['version']}  
**Date:** {card_config['date']}  
**Owner:** {card_config['owner']}

## Overview

{card_config.get('description', 'Reinforcement Learning policy for algorithmic trading.')}

## Model Details

### Architecture
- **Algorithm:** {card_config['training']['algo']}
- **Environment:** {card_config['training']['env']}
- **Training Data Window:** {card_config['training']['data_window']}
- **Training Duration:** {card_config['training'].get('training_duration', 'N/A')}
- **Compute Resources:** {card_config['training'].get('compute_resources', 'N/A')}

### Hyperparameters
"""

    hyperparams = card_config["training"].get("hyperparameters", {})
    for param, value in hyperparams.items():
        markdown += f"- **{param.replace('_', ' ').title()}:** {value}\n"

    markdown += f"""
## Performance Evaluation

### Offline Validation
- **Pass Rate:** {card_config['evaluation']['offline_gate']['pass_rate']}
- **Mean Entropy:** {card_config['evaluation']['offline_gate']['entropy']}
- **Mean Return:** {card_config['evaluation']['offline_gate']['return_mean']}
- **Sharpe Ratio:** {card_config['evaluation']['offline_gate'].get('sharpe_ratio', 'N/A')}
- **Max Drawdown:** {card_config['evaluation']['offline_gate'].get('max_drawdown', 'N/A')}

### A/B Test Results
- **PnL Delta:** {card_config['evaluation']['ab_test']['delta_pnl']}
- **95% Confidence Interval:** {card_config['evaluation']['ab_test']['ci95']}
- **Verdict:** **{card_config['evaluation']['ab_test']['verdict']}**
- **Sample Size:** {card_config['evaluation']['ab_test'].get('sample_size', 'N/A')}
"""

    # Add postmortem summary if available
    if postmortem_data:
        summary = postmortem_data.get("summary", {})
        markdown += f"""
### Pilot Summary
- **Max Influence Reached:** {summary.get('max_influence', 0)}%
- **System Uptime:** {summary.get('uptime_pct', 0):.1f}%
- **Total Alerts:** {summary.get('total_alerts', 0)}
- **Emergency Rollbacks:** {summary.get('rollbacks', 0)}
"""

    # Add risk management section
    markdown += format_risk_section(card_config["risk"])

    # Add explainability section
    expl_config = card_config.get("explainability", {})
    markdown += f"""
## Explainability & Interpretability

### Input Features
"""
    for feature in expl_config.get("features", []):
        markdown += f"- {feature}\n"

    markdown += f"""
### Attribution Methods
"""
    for method in expl_config.get("attribution_methods", []):
        markdown += f"- {method}\n"

    markdown += f"""
### Tools
"""
    for tool in expl_config.get("tools", []):
        markdown += f"- {tool}\n"

    # Add governance section
    markdown += format_governance_section(card_config["governance"])

    # Add compliance section
    compliance_config = card_config.get("compliance", {})
    markdown += f"""
## Compliance

- **Model Risk Tier:** {compliance_config.get('model_risk_tier', 'N/A')}
- **Regulatory Framework:** {compliance_config.get('regulatory_framework', 'N/A')}
- **Validation Frequency:** {compliance_config.get('validation_frequency', 'N/A')}
- **Documentation Status:** {compliance_config.get('documentation_status', 'N/A')}
- **Audit Trail:** {compliance_config.get('audit_trail', 'N/A')}

## Deployment

### Infrastructure
- **Environment:** {card_config.get('deployment', {}).get('infrastructure', {}).get('environment', 'N/A')}
- **Scaling:** {card_config.get('deployment', {}).get('infrastructure', {}).get('scaling', 'N/A')}
- **Persistence:** {card_config.get('deployment', {}).get('infrastructure', {}).get('persistence', 'N/A')}

### Rollout Strategy
- **Blue/Green:** {card_config.get('deployment', {}).get('rollout_strategy', {}).get('blue_green', False)}
- **Canary Percentage:** {card_config.get('deployment', {}).get('rollout_strategy', {}).get('canary_pct', 'N/A')}
- **Monitoring Period:** {card_config.get('deployment', {}).get('rollout_strategy', {}).get('monitoring_period', 'N/A')}

### Rollback Criteria
"""

    rollback_criteria = card_config.get("deployment", {}).get("rollback_criteria", [])
    for criteria in rollback_criteria:
        markdown += f"- {criteria}\n"

    # Add metadata section
    metadata = card_config.get("metadata", {})
    markdown += f"""
## Metadata

- **Created By:** {metadata.get('created_by', 'N/A')}
- **Reviewed By:** {metadata.get('reviewed_by', 'N/A')}
- **Approved By:** {metadata.get('approved_by', 'N/A')}
- **Last Updated:** {metadata.get('last_updated', 'N/A')}
- **Next Review:** {metadata.get('next_review', 'N/A')}

### Version History
"""

    for version in metadata.get("version_history", []):
        markdown += f"- {version}\n"

    markdown += f"""
---
*This model card was generated automatically on {datetime.datetime.now(datetime.timezone.utc).isoformat()}*
"""

    return markdown


def main():
    """Main model card builder function."""
    parser = argparse.ArgumentParser(
        description="Generate model card from YAML configuration"
    )
    parser.add_argument("--yaml", required=True, help="Model card YAML file")
    parser.add_argument("--out", required=True, help="Output markdown file")
    parser.add_argument(
        "--include-artifacts", action="store_true", help="Include latest artifacts data"
    )
    args = parser.parse_args()

    print(f"Loading model card configuration from: {args.yaml}")
    card_config = load_model_card_yaml(args.yaml)

    postmortem_data = None
    ab_results = None
    validation_summary = None

    if args.include_artifacts:
        print("Loading latest artifacts...")
        postmortem_data = get_latest_postmortem()
        ab_results = get_latest_ab_results()
        validation_summary = get_validation_summary()

        if postmortem_data:
            print("✅ Found postmortem data")
        if ab_results:
            print("✅ Found A/B test results")
        if validation_summary:
            print("✅ Found validation summary")

    print("Generating model card markdown...")
    markdown_content = generate_model_card_markdown(
        card_config, postmortem_data, ab_results, validation_summary
    )

    # Ensure output directory exists
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing model card to: {args.out}")
    with open(args.out, "w") as f:
        f.write(markdown_content)

    print("✅ Model card generated successfully")


if __name__ == "__main__":
    main()
