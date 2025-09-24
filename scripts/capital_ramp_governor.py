#!/usr/bin/env python3
"""
Capital Ramp Governor
Portfolio-aware influence allocation with TCA gates and risk controls
"""
import os
import sys
import json
import yaml
import datetime
import argparse
import pathlib
import subprocess
from datetime import timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, Any


def load_config(config_path: str) -> dict:
    """Load portfolio pilot configuration."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load config from {config_path}: {e}")
        sys.exit(1)


def write_audit(action: str, payload: dict):
    """Write WORM audit record."""
    ts = datetime.datetime.now(timezone.utc).isoformat().replace(":", "_")
    pathlib.Path("artifacts/audit").mkdir(parents=True, exist_ok=True)
    audit_file = f"artifacts/audit/{ts}_capital_ramp_governor.json"

    audit_record = {
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        "action": action,
        "payload": payload,
        "operator": "capital_ramp_governor",
        "session_id": f"governor_{ts}",
    }

    with open(audit_file, "w") as f:
        json.dump(audit_record, f, indent=2)

    print(f"[AUDIT] {audit_file}")
    return audit_file


def check_go_live_flag():
    """Check GO_LIVE environment variable."""
    if os.getenv("GO_LIVE", "0") != "1":
        print("❌ GO_LIVE flag not set - governor will not apply changes")
        return False
    return True


def check_recent_alerts(config: dict) -> bool:
    """Check for recent page-severity alerts."""
    governance = config.get("governance", {})
    no_pages_hours = governance.get("no_pages_hours", 48)

    # Check for recent alert audit files
    cutoff = datetime.datetime.now(timezone.utc) - timedelta(hours=no_pages_hours)

    alert_files = Path("artifacts/audit").glob("*alert*.json")
    recent_alerts = 0

    for alert_file in alert_files:
        try:
            mtime = datetime.datetime.fromtimestamp(
                alert_file.stat().st_mtime, tz=timezone.utc
            )
            if mtime > cutoff:
                recent_alerts += 1
        except Exception:
            continue

    if recent_alerts > 0:
        print(
            f"❌ {recent_alerts} recent alerts found within {no_pages_hours}h - blocking ramp"
        )
        return False

    print(f"✅ No recent alerts found within {no_pages_hours}h")
    return True


def get_risk_metrics() -> dict:
    """Get risk metrics (stub implementation)."""
    # In production, this would integrate with risk systems
    return {
        "SOL-USD": {
            "var_95_usd": 2500,
            "sharpe_ratio": 1.2,
            "max_drawdown_24h": 1.1,
            "volatility": 0.45,
        },
        "BTC-USD": {
            "var_95_usd": 3200,
            "sharpe_ratio": 0.9,
            "max_drawdown_24h": 1.8,
            "volatility": 0.38,
        },
        "ETH-USD": {
            "var_95_usd": 2800,
            "sharpe_ratio": 1.0,
            "max_drawdown_24h": 1.5,
            "volatility": 0.42,
        },
        "NVDA": {
            "var_95_usd": 1800,
            "sharpe_ratio": 1.4,
            "max_drawdown_24h": 0.8,
            "volatility": 0.28,
        },
    }


def get_tca_metrics() -> dict:
    """Get TCA metrics (stub implementation)."""
    # In production, this would integrate with execution analytics
    return {
        "SOL-USD": {
            "is_bps": 28,
            "slip_bps_p95": 32,
            "fill_ratio": 0.94,
            "cancel_ratio": 0.12,
        },
        "BTC-USD": {
            "is_bps": 22,
            "slip_bps_p95": 26,
            "fill_ratio": 0.96,
            "cancel_ratio": 0.08,
        },
        "ETH-USD": {
            "is_bps": 31,
            "slip_bps_p95": 35,
            "fill_ratio": 0.91,
            "cancel_ratio": 0.15,
        },
        "NVDA": {
            "is_bps": 18,
            "slip_bps_p95": 22,
            "fill_ratio": 0.97,
            "cancel_ratio": 0.05,
        },
    }


def get_slo_status() -> dict:
    """Get SLO status (stub implementation)."""
    return {
        "entropy_current": 1.15,
        "qspread_ratio_24h": 1.6,
        "heartbeat_age_seconds": 45,
        "overall_healthy": True,
    }


def check_tca_gates(config: dict, tca_metrics: dict) -> dict:
    """Check TCA gates for each asset."""
    tca_gates = config.get("tca_gates", {})
    results = {}

    for asset, metrics in tca_metrics.items():
        asset_result = {"asset": asset, "pass": True, "reasons": []}

        # Check implementation shortfall
        is_max = tca_gates.get("is_bps_max", 50)
        if metrics.get("is_bps", 0) > is_max:
            asset_result["pass"] = False
            asset_result["reasons"].append(f"IS {metrics['is_bps']} > {is_max} bps")

        # Check slippage
        slip_max = tca_gates.get("slip_bps_p95_max", 50)
        if metrics.get("slip_bps_p95", 0) > slip_max:
            asset_result["pass"] = False
            asset_result["reasons"].append(
                f"Slippage {metrics['slip_bps_p95']} > {slip_max} bps"
            )

        # Check fill ratio
        fill_min = tca_gates.get("fill_ratio_min", 0.80)
        if metrics.get("fill_ratio", 1.0) < fill_min:
            asset_result["pass"] = False
            asset_result["reasons"].append(
                f"Fill ratio {metrics['fill_ratio']:.2f} < {fill_min}"
            )

        # Check cancel ratio
        cancel_max = tca_gates.get("cancel_ratio_max", 0.50)
        if metrics.get("cancel_ratio", 0.0) > cancel_max:
            asset_result["pass"] = False
            asset_result["reasons"].append(
                f"Cancel ratio {metrics['cancel_ratio']:.2f} > {cancel_max}"
            )

        if asset_result["pass"]:
            asset_result["reasons"] = ["All TCA gates passed"]

        results[asset] = asset_result

    return results


def compute_budgeted_weights(
    config: dict, risk_metrics: dict, tca_results: dict
) -> dict:
    """Compute budgeted weights based on risk and performance."""
    assets = config.get("pilot", {}).get("assets", [])
    portfolio_limits = config.get("portfolio_limits", {})

    weights = {}
    total_budget = 0

    for asset_config in assets:
        asset = asset_config["symbol"]
        max_influence = asset_config["max_influence_pct"]

        # Start with max allowed
        budgeted_weight = max_influence

        # Adjust based on TCA performance
        if asset in tca_results and not tca_results[asset]["pass"]:
            print(f"  ⚠️ {asset}: TCA gates failed - setting to 0%")
            budgeted_weight = 0
        else:
            # Risk-adjusted allocation (better metrics = higher allocation)
            if asset in risk_metrics:
                risk = risk_metrics[asset]
                sharpe = risk.get("sharpe_ratio", 1.0)
                drawdown = risk.get("max_drawdown_24h", 1.0)

                # Simple risk adjustment: higher Sharpe, lower drawdown = higher weight
                risk_multiplier = min(1.0, (sharpe / 1.0) * (1.0 / max(drawdown, 0.5)))
                budgeted_weight = int(budgeted_weight * risk_multiplier)

        weights[asset] = budgeted_weight
        total_budget += budgeted_weight

    # Enforce portfolio-level cap
    max_total = portfolio_limits.get("max_total_influence_pct", 100)
    if total_budget > max_total:
        print(f"  📉 Portfolio cap enforcement: {total_budget}% → {max_total}%")
        scale_factor = max_total / total_budget
        weights = {
            asset: int(weight * scale_factor) for asset, weight in weights.items()
        }
        total_budget = sum(weights.values())

    return weights, total_budget


def enforce_risk_limits(weights: dict, config: dict, risk_metrics: dict) -> dict:
    """Enforce portfolio risk limits."""
    portfolio_limits = config.get("portfolio_limits", {})
    max_var = portfolio_limits.get("max_var_95_usd", float("inf"))

    # Calculate portfolio VaR (simplified)
    total_var = 0
    for asset, weight in weights.items():
        if asset in risk_metrics and weight > 0:
            asset_var = risk_metrics[asset].get("var_95_usd", 0)
            # Scale VaR by influence weight
            total_var += asset_var * (weight / 100.0)

    if total_var > max_var:
        print(f"  📊 VaR budget enforcement: {total_var:,.0f} > {max_var:,.0f} USD")
        # Scale down all weights proportionally
        scale_factor = max_var / total_var
        weights = {
            asset: int(weight * scale_factor) for asset, weight in weights.items()
        }
        print(f"     Scaled weights by {scale_factor:.2f}")

    return weights


def generate_proposal(config: dict) -> dict:
    """Generate influence allocation proposal."""
    print("📊 Generating capital allocation proposal...")

    # Get current metrics
    print("  📈 Fetching risk metrics...")
    risk_metrics = get_risk_metrics()

    print("  🎯 Fetching TCA metrics...")
    tca_metrics = get_tca_metrics()

    print("  🚦 Checking TCA gates...")
    tca_results = check_tca_gates(config, tca_metrics)

    # Show TCA gate results
    for asset, result in tca_results.items():
        status = "✅" if result["pass"] else "❌"
        print(f"    {status} {asset}: {', '.join(result['reasons'])}")

    print("  ⚖️ Computing budgeted weights...")
    weights, total_budget = compute_budgeted_weights(config, risk_metrics, tca_results)

    print("  🛡️ Enforcing risk limits...")
    weights = enforce_risk_limits(weights, config, risk_metrics)

    # Final totals
    final_total = sum(weights.values())

    proposal = {
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        "total_influence_pct": final_total,
        "asset_weights": weights,
        "risk_metrics": risk_metrics,
        "tca_metrics": tca_metrics,
        "tca_results": tca_results,
        "governance_checks": {
            "go_live_flag": check_go_live_flag(),
            "recent_alerts": check_recent_alerts(config),
            "slo_status": get_slo_status(),
        },
    }

    return proposal


def save_proposal(proposal: dict, dry_run: bool = False) -> str:
    """Save proposal to artifacts."""
    timestamp = datetime.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    output_dir = Path("artifacts/governor") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "dry_run" if dry_run else "live"
    proposal_file = output_dir / f"proposal_{mode}.json"

    with open(proposal_file, "w") as f:
        json.dump(proposal, f, indent=2)

    print(f"  💾 Proposal saved: {proposal_file}")
    return str(proposal_file)


def validate_proposal(proposal: dict, config: dict) -> tuple:
    """Validate proposal against all gates."""
    errors = []

    governance = proposal["governance_checks"]

    # Check GO_LIVE flag
    if not governance.get("go_live_flag", False):
        errors.append("GO_LIVE flag not set")

    # Check alerts
    if not governance.get("recent_alerts", False):
        errors.append("Recent alerts detected")

    # Check SLO status
    slo = governance.get("slo_status", {})
    if not slo.get("overall_healthy", False):
        errors.append("SLO health check failed")

    # Check entropy
    entropy = slo.get("entropy_current", 0)
    entropy_floor = config.get("governance", {}).get("entropy_floor", 0.9)
    if entropy < entropy_floor:
        errors.append(f"Entropy {entropy:.2f} < {entropy_floor}")

    # Check total portfolio limit
    total_pct = proposal.get("total_influence_pct", 0)
    max_total = config.get("portfolio_limits", {}).get("max_total_influence_pct", 100)
    if total_pct > max_total:
        errors.append(f"Total influence {total_pct}% > {max_total}%")

    return len(errors) == 0, errors


def apply_proposal(proposal: dict) -> bool:
    """Apply proposal using influence controller."""
    print("🚀 Applying capital allocation proposal...")

    try:
        sys.path.insert(0, "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")
        from src.rl.influence_controller import InfluenceController

        ic = InfluenceController()

        for asset, pct in proposal["asset_weights"].items():
            if pct > 0:
                print(f"  📈 Setting {asset} to {pct}%")
                ic.set_weight_asset(asset, pct, f"Capital ramp governor allocation")
            else:
                print(f"  📉 Setting {asset} to 0% (TCA gate failed or risk limit)")
                ic.set_weight_asset(asset, 0, f"TCA gate failed or risk limit")

        print("✅ Proposal application complete")
        return True

    except Exception as e:
        print(f"❌ Failed to apply proposal: {e}")
        return False


def main():
    """Main capital ramp governor function."""
    parser = argparse.ArgumentParser(description="Capital Ramp Governor")
    parser.add_argument(
        "-c",
        "--config",
        default="pilot/portfolio_pilot.yaml",
        help="Portfolio configuration file",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate proposal but don't apply"
    )
    parser.add_argument(
        "--force", action="store_true", help="Skip governance checks (dangerous)"
    )
    args = parser.parse_args()

    print("🏛️ Capital Ramp Governor")
    print("=" * 50)
    print(f"Config: {args.config}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE APPLICATION'}")
    print("=" * 50)

    # Load configuration
    config = load_config(args.config)

    # Generate proposal
    try:
        proposal = generate_proposal(config)

        # Save proposal
        proposal_file = save_proposal(proposal, args.dry_run)

        # Validate proposal
        valid, errors = validate_proposal(proposal, config)

        print("\n📋 Proposal Summary:")
        print(f"  Total Portfolio Influence: {proposal['total_influence_pct']}%")
        print("  Asset Allocations:")

        for asset, pct in proposal["asset_weights"].items():
            print(f"    • {asset}: {pct}%")

        print(f"\n🚦 Validation: {'✅ PASS' if valid else '❌ FAIL'}")

        if not valid:
            print("  Blocking issues:")
            for error in errors:
                print(f"    • {error}")

            write_audit(
                "proposal_blocked",
                {
                    "proposal_file": proposal_file,
                    "errors": errors,
                    "total_influence": proposal["total_influence_pct"],
                },
            )

            if not args.force:
                print("\n🛑 Proposal blocked - use --force to override")
                return 1
            else:
                print("\n⚠️ Force mode - applying despite validation failures")

        # Apply proposal (if not dry run)
        if not args.dry_run:
            if apply_proposal(proposal):
                write_audit(
                    "proposal_applied",
                    {
                        "proposal_file": proposal_file,
                        "asset_weights": proposal["asset_weights"],
                        "total_influence": proposal["total_influence_pct"],
                    },
                )
                print("\n✅ Capital ramp governor completed successfully")
                return 0
            else:
                write_audit(
                    "proposal_application_failed",
                    {"proposal_file": proposal_file, "error": "Application failed"},
                )
                return 1
        else:
            write_audit(
                "proposal_dry_run",
                {
                    "proposal_file": proposal_file,
                    "asset_weights": proposal["asset_weights"],
                    "total_influence": proposal["total_influence_pct"],
                },
            )
            print("\n🧪 Dry run completed - no changes applied")
            return 0

    except Exception as e:
        print(f"\n❌ Governor failed: {e}")
        write_audit("governor_error", {"error": str(e)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
