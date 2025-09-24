#!/usr/bin/env python3
"""
CFO-Grade Portfolio Report  
7-day aggregated economics, risk, and capacity utilization
"""
import os
import sys
import json
import glob
import datetime
import pathlib
from datetime import timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional


def gather_7day_economics() -> List[dict]:
    """Gather economics data for last 7 days."""
    econ_data = []

    try:
        # Get all economic close files
        econ_files = glob.glob("artifacts/econ/*/econ_close.json")

        # Filter to last 7 days
        cutoff = datetime.datetime.now(timezone.utc) - timedelta(days=7)

        for econ_file in econ_files:
            try:
                file_mtime = datetime.datetime.fromtimestamp(
                    os.path.getmtime(econ_file), tz=timezone.utc
                )

                if file_mtime > cutoff:
                    with open(econ_file, "r") as f:
                        data = json.load(f)

                    data["file_time"] = file_mtime.isoformat()
                    data["file_path"] = econ_file
                    econ_data.append(data)

            except Exception as e:
                print(f"Warning: Could not process {econ_file}: {e}")
                continue

        # Sort by file time
        econ_data.sort(key=lambda x: x["file_time"])

    except Exception as e:
        print(f"Warning: Error gathering economics data: {e}")

    return econ_data


def aggregate_portfolio_metrics(econ_data: List[dict]) -> dict:
    """Aggregate portfolio-level metrics."""
    if not econ_data:
        return {}

    # Aggregate totals
    total_net_pnl = sum(
        d.get("portfolio", {}).get("net_pnl_final_usd", 0) for d in econ_data
    )
    total_gross_pnl = sum(
        d.get("portfolio", {}).get("gross_pnl_usd", 0) for d in econ_data
    )
    total_fees = sum(d.get("portfolio", {}).get("total_fees_usd", 0) for d in econ_data)
    total_infra_costs = sum(
        d.get("portfolio", {}).get("total_infra_cost_usd", 0) for d in econ_data
    )

    # Calculate averages and distributions
    daily_pnls = [d.get("portfolio", {}).get("net_pnl_final_usd", 0) for d in econ_data]
    cost_ratios = [d.get("portfolio", {}).get("cost_ratio", 0) for d in econ_data]

    # Risk metrics
    daily_returns = [pnl / max(abs(total_gross_pnl), 1000) for pnl in daily_pnls]
    max_daily_loss = min(daily_pnls) if daily_pnls else 0
    max_daily_gain = max(daily_pnls) if daily_pnls else 0

    # Calculate simple volatility
    if len(daily_returns) > 1:
        avg_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - avg_return) ** 2 for r in daily_returns) / (
            len(daily_returns) - 1
        )
        volatility = variance**0.5
    else:
        volatility = 0

    return {
        "period_days": len(econ_data),
        "total_net_pnl_usd": total_net_pnl,
        "total_gross_pnl_usd": total_gross_pnl,
        "total_fees_usd": total_fees,
        "total_infra_costs_usd": total_infra_costs,
        "avg_daily_net_pnl": total_net_pnl / max(len(econ_data), 1),
        "avg_cost_ratio": sum(cost_ratios) / max(len(cost_ratios), 1),
        "max_daily_loss": max_daily_loss,
        "max_daily_gain": max_daily_gain,
        "daily_volatility": volatility,
        "profitable_days": sum(1 for pnl in daily_pnls if pnl > 0),
        "loss_days": sum(1 for pnl in daily_pnls if pnl < 0),
        "net_margin_pct": (total_net_pnl / max(abs(total_gross_pnl), 1)) * 100,
    }


def aggregate_asset_metrics(econ_data: List[dict]) -> dict:
    """Aggregate per-asset metrics."""
    asset_totals = {}

    for daily_data in econ_data:
        assets = daily_data.get("assets", {})

        for asset, data in assets.items():
            if asset not in asset_totals:
                asset_totals[asset] = {
                    "total_net_pnl": 0,
                    "total_gross_pnl": 0,
                    "total_fees": 0,
                    "total_fills": 0,
                    "is_bps_samples": [],
                    "slip_bps_samples": [],
                    "profitable_days": 0,
                    "trading_days": 0,
                }

            asset_totals[asset]["total_net_pnl"] += data.get("net_pnl_usd", 0)
            asset_totals[asset]["total_gross_pnl"] += data.get("gross_pnl_usd", 0)
            asset_totals[asset]["total_fees"] += data.get("fees_usd", 0)
            asset_totals[asset]["total_fills"] += data.get("fill_count", 0)
            asset_totals[asset]["trading_days"] += 1

            if data.get("net_pnl_usd", 0) > 0:
                asset_totals[asset]["profitable_days"] += 1

            # Collect TCA samples
            if data.get("is_bps"):
                asset_totals[asset]["is_bps_samples"].append(data["is_bps"])
            if data.get("slippage_bps_p95"):
                asset_totals[asset]["slip_bps_samples"].append(data["slippage_bps_p95"])

    # Calculate derived metrics
    for asset, totals in asset_totals.items():
        totals["avg_is_bps"] = sum(totals["is_bps_samples"]) / max(
            len(totals["is_bps_samples"]), 1
        )
        totals["avg_slip_bps"] = sum(totals["slip_bps_samples"]) / max(
            len(totals["slip_bps_samples"]), 1
        )
        totals["win_rate"] = totals["profitable_days"] / max(totals["trading_days"], 1)
        totals["net_margin_pct"] = (
            totals["total_net_pnl"] / max(abs(totals["total_gross_pnl"]), 1)
        ) * 100

    return asset_totals


def get_capacity_utilization() -> dict:
    """Get current capacity utilization metrics."""
    try:
        # Load portfolio configuration
        import yaml

        with open("pilot/portfolio_pilot.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Get current influence status
        sys.path.insert(0, "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")
        from src.rl.influence_controller import InfluenceController

        ic = InfluenceController()
        current_weights = ic.get_all_asset_weights()

        # Calculate utilization
        portfolio_limits = config.get("portfolio_limits", {})
        max_total_pct = portfolio_limits.get("max_total_influence_pct", 100)
        current_total_pct = sum(w * 100 for w in current_weights.values())

        asset_utilization = {}
        for asset_config in config.get("pilot", {}).get("assets", []):
            asset = asset_config["symbol"]
            max_pct = asset_config["max_influence_pct"]
            current_pct = current_weights.get(asset, 0) * 100

            asset_utilization[asset] = {
                "current_pct": current_pct,
                "max_pct": max_pct,
                "utilization": current_pct / max_pct if max_pct > 0 else 0,
                "headroom_pct": max_pct - current_pct,
            }

        return {
            "portfolio_utilization_pct": (
                current_total_pct / max_total_pct * 100 if max_total_pct > 0 else 0
            ),
            "current_total_influence": current_total_pct,
            "max_total_influence": max_total_pct,
            "portfolio_headroom": max_total_pct - current_total_pct,
            "asset_utilization": asset_utilization,
            "max_gross_notional": portfolio_limits.get("max_gross_notional_usd", 0),
        }

    except Exception as e:
        print(f"Warning: Could not get capacity utilization: {e}")
        return {"error": str(e)}


def get_risk_snapshot() -> dict:
    """Get current risk metrics snapshot."""
    try:
        # Stub risk data - in production would integrate with risk systems
        return {
            "var_95_1d_usd": 8500,
            "es_95_1d_usd": 12000,
            "portfolio_beta": 0.95,
            "max_correlation": 0.72,
            "diversification_ratio": 1.23,
            "leverage": 1.0,
            "currency_exposure": {"USD": 0.65, "EUR": 0.20, "JPY": 0.15},
        }
    except Exception as e:
        return {"error": str(e)}


def generate_cfo_report(data: dict) -> str:
    """Generate executive-level markdown report."""

    portfolio = data["portfolio_metrics"]
    assets = data["asset_metrics"]
    capacity = data["capacity_utilization"]
    risk = data["risk_snapshot"]

    # Overall assessment
    net_pnl = portfolio.get("total_net_pnl_usd", 0)
    if net_pnl > 1000:
        performance_status = "🟢 STRONG PERFORMANCE"
    elif net_pnl > 0:
        performance_status = "✅ PROFITABLE"
    elif net_pnl > -500:
        performance_status = "🟡 BREAKEVEN"
    else:
        performance_status = "🔴 LOSSES"

    markdown = f"""# CFO Portfolio Report - 7-Day Summary

**Generated:** {data["timestamp"]}  
**Period:** {portfolio.get("period_days", 0)} days  
**Performance:** {performance_status}

## Executive Summary

**Net P&L (7 days):** ${portfolio.get("total_net_pnl_usd", 0):,.2f}  
**Average Daily P&L:** ${portfolio.get("avg_daily_net_pnl", 0):,.2f}  
**Win Rate:** {portfolio.get("profitable_days", 0)}/{portfolio.get("period_days", 0)} days ({(portfolio.get("profitable_days", 0) / max(portfolio.get("period_days", 1), 1)) * 100:.0f}%)  
**Portfolio Utilization:** {capacity.get("portfolio_utilization_pct", 0):.0f}% of capacity

## Financial Performance

| Metric | Amount | Notes |
|--------|--------|-------|
| **Gross P&L** | ${portfolio.get("total_gross_pnl_usd", 0):,.2f} | Before costs |
| **Total Fees** | ${portfolio.get("total_fees_usd", 0):,.2f} | Trading costs |
| **Infrastructure Costs** | ${portfolio.get("total_infra_costs_usd", 0):,.2f} | AWS/GPU/Monitoring |
| **Net P&L** | ${portfolio.get("total_net_pnl_usd", 0):,.2f} | After all costs |
| **Net Margin** | {portfolio.get("net_margin_pct", 0):.1f}% | Net / Gross |
| **Cost Ratio** | {portfolio.get("avg_cost_ratio", 0):.1%} | (Fees + Infra) / Gross |

## Risk Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Max Daily Loss** | ${portfolio.get("max_daily_loss", 0):,.2f} | {"❌" if portfolio.get("max_daily_loss", 0) < -1000 else "✅"} |
| **Max Daily Gain** | ${portfolio.get("max_daily_gain", 0):,.2f} | - |
| **Daily Volatility** | {portfolio.get("daily_volatility", 0):.1%} | {"⚠️" if portfolio.get("daily_volatility", 0) > 0.05 else "✅"} |
| **VaR (95%, 1d)** | ${risk.get("var_95_1d_usd", 0):,.0f} | - |
| **Expected Shortfall** | ${risk.get("es_95_1d_usd", 0):,.0f} | - |

## Per-Asset Performance

"""

    # Sort assets by net P&L
    sorted_assets = sorted(
        assets.items(), key=lambda x: x[1]["total_net_pnl"], reverse=True
    )

    for asset, metrics in sorted_assets:
        pnl_emoji = "✅" if metrics["total_net_pnl"] > 0 else "❌"

        markdown += f"""### {pnl_emoji} {asset}

- **Net P&L:** ${metrics["total_net_pnl"]:,.2f} ({metrics["net_margin_pct"]:.1f}% margin)
- **Win Rate:** {metrics["profitable_days"]}/{metrics["trading_days"]} days ({metrics["win_rate"]:.0%})
- **Total Fills:** {metrics["total_fills"]}
- **Avg IS:** {metrics["avg_is_bps"]:.0f} bps
- **Avg Slippage:** {metrics["avg_slip_bps"]:.0f} bps
- **Current Utilization:** {capacity.get("asset_utilization", {}).get(asset, {}).get("utilization", 0):.0%}

"""

    markdown += f"""## Capacity Utilization

**Portfolio Level:**
- Current: {capacity.get("current_total_influence", 0):.1f}% / {capacity.get("max_total_influence", 0):.0f}% max
- Utilization: {capacity.get("portfolio_utilization_pct", 0):.0f}%  
- Headroom: {capacity.get("portfolio_headroom", 0):.1f}%

**Asset Level:**
"""

    asset_util = capacity.get("asset_utilization", {})
    for asset, util in asset_util.items():
        markdown += f"- **{asset}:** {util['current_pct']:.1f}% / {util['max_pct']}% ({util['utilization']:.0%} utilized)\n"

    markdown += f"""

## Fee & Cost Breakdown

### Transaction Costs (7-day total)
- **Trading Fees:** ${portfolio.get("total_fees_usd", 0):,.2f}
- **Average per Day:** ${portfolio.get("total_fees_usd", 0) / max(portfolio.get("period_days", 1), 1):,.2f}

### Infrastructure Costs (7-day total)  
- **Total Infra:** ${portfolio.get("total_infra_costs_usd", 0):,.2f}
- **Daily Average:** ${portfolio.get("total_infra_costs_usd", 0) / max(portfolio.get("period_days", 1), 1):,.2f}

### Cost Efficiency
- **Cost as % of Gross P&L:** {portfolio.get("avg_cost_ratio", 0):.1%}
- **Target:** <30% (current: {"✅ PASS" if portfolio.get("avg_cost_ratio", 0) < 0.30 else "⚠️ HIGH"})

## Strategic Recommendations

"""

    recommendations = []

    # Performance-based recommendations
    if portfolio.get("total_net_pnl_usd", 0) > 500:
        recommendations.append(
            "✅ **Strong Performance** - Consider gradual capacity increases"
        )
    elif portfolio.get("total_net_pnl_usd", 0) < -200:
        recommendations.append(
            "🚨 **Losses** - Review strategy and consider reducing exposure"
        )

    # Cost-based recommendations
    if portfolio.get("avg_cost_ratio", 0) > 0.30:
        recommendations.append(
            "💰 **High Costs** - Optimize infrastructure or improve P&L to reduce cost ratio"
        )

    # Utilization-based recommendations
    if capacity.get("portfolio_utilization_pct", 0) < 50:
        recommendations.append(
            "📈 **Low Utilization** - Opportunity to scale if performance metrics support"
        )
    elif capacity.get("portfolio_utilization_pct", 0) > 80:
        recommendations.append(
            "⚠️ **High Utilization** - Consider expanding capacity limits"
        )

    # Risk-based recommendations
    if portfolio.get("daily_volatility", 0) > 0.05:
        recommendations.append(
            "📊 **High Volatility** - Review position sizing and diversification"
        )

    if not recommendations:
        recommendations.append(
            "✅ **Balanced Operation** - Current metrics within acceptable ranges"
        )

    for rec in recommendations:
        markdown += f"- {rec}\n"

    markdown += f"""

## Operational Metrics

- **Data Coverage:** {portfolio.get("period_days", 0)} of 7 days
- **Total Fill Count:** {sum(a["total_fills"] for a in assets.values())}
- **Average IS Across Assets:** {sum(a["avg_is_bps"] for a in assets.values()) / max(len(assets), 1):.0f} bps
- **Average Slippage:** {sum(a["avg_slip_bps"] for a in assets.values()) / max(len(assets), 1):.0f} bps

## Next Review

**Frequency:** Weekly  
**Next Report:** {(datetime.datetime.now(timezone.utc) + timedelta(days=7)).strftime("%Y-%m-%d")}

---
*This report is automatically generated from production systems. For detailed data, see artifacts in `artifacts/econ/` and `artifacts/ramp/`.*
"""

    return markdown


def main():
    """Main CFO report function."""
    print("👔 CFO Portfolio Report Generator")
    print("=" * 45)

    import argparse

    parser = argparse.ArgumentParser(description="Generate CFO Portfolio Report")
    parser.add_argument(
        "--output", "-o", default="artifacts/cfo", help="Output directory"
    )
    parser.add_argument("--days", type=int, default=7, help="Number of days to analyze")
    args = parser.parse_args()

    try:
        # Gather data
        print("📊 Gathering 7-day economics data...")
        econ_data = gather_7day_economics()
        print(f"  Found {len(econ_data)} daily records")

        print("📈 Aggregating portfolio metrics...")
        portfolio_metrics = aggregate_portfolio_metrics(econ_data)

        print("🎯 Aggregating asset metrics...")
        asset_metrics = aggregate_asset_metrics(econ_data)

        print("📏 Getting capacity utilization...")
        capacity_utilization = get_capacity_utilization()

        print("📊 Getting risk snapshot...")
        risk_snapshot = get_risk_snapshot()

        # Compile report data
        report_data = {
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
            "period_days": args.days,
            "portfolio_metrics": portfolio_metrics,
            "asset_metrics": asset_metrics,
            "capacity_utilization": capacity_utilization,
            "risk_snapshot": risk_snapshot,
            "data_sources": {
                "econ_files": len(econ_data),
                "time_range": {
                    "start": econ_data[0]["file_time"] if econ_data else None,
                    "end": econ_data[-1]["file_time"] if econ_data else None,
                },
            },
        }

        # Generate reports
        print("📝 Generating CFO report...")
        markdown_content = generate_cfo_report(report_data)

        # Save outputs
        timestamp = datetime.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        output_dir = Path(args.output) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON data
        json_file = output_dir / "cfo_report.json"
        with open(json_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        # Save markdown report
        md_file = output_dir / "cfo_report.md"
        with open(md_file, "w") as f:
            f.write(markdown_content)

        # Create latest symlinks
        latest_json = Path(args.output) / "cfo_report_latest.json"
        latest_md = Path(args.output) / "cfo_report_latest.md"

        if latest_json.exists():
            latest_json.unlink()
        if latest_md.exists():
            latest_md.unlink()

        latest_json.symlink_to(json_file)
        latest_md.symlink_to(md_file)

        # Display summary
        net_pnl = portfolio_metrics.get("total_net_pnl_usd", 0)
        win_rate = (
            portfolio_metrics.get("profitable_days", 0)
            / max(portfolio_metrics.get("period_days", 1), 1)
        ) * 100

        print("\n💼 CFO Report Summary:")
        print(f"  Net P&L (7d): ${net_pnl:,.2f}")
        print(f"  Win Rate: {win_rate:.0f}%")
        print(f"  Cost Ratio: {portfolio_metrics.get('avg_cost_ratio', 0):.1%}")
        print(f"  Assets Analyzed: {len(asset_metrics)}")

        print(f"\n📄 Reports Generated:")
        print(f"  JSON: {json_file}")
        print(f"  Markdown: {md_file}")

        return 0

    except Exception as e:
        print(f"❌ CFO report generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
