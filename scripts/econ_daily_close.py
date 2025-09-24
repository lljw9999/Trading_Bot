#!/usr/bin/env python3
"""
Economic Daily Close
Compute net P&L after all costs (fees, slippage, infra) with per-asset breakdown
"""
import os
import sys
import json
import glob
import datetime
import argparse
import pathlib
from datetime import timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any


def load_fee_engine_data() -> dict:
    """Load fee engine data (stub implementation)."""
    # In production, this would integrate with actual fee/fill data
    return {
        "SOL-USD": {
            "fills": [
                {
                    "timestamp": "2025-08-14T06:00:00Z",
                    "side": "buy",
                    "qty": 100,
                    "price": 145.50,
                    "venue": "coinbase",
                    "fee_usd": 14.55,
                },
                {
                    "timestamp": "2025-08-14T10:30:00Z",
                    "side": "sell",
                    "qty": 50,
                    "price": 147.20,
                    "venue": "coinbase",
                    "fee_usd": 7.36,
                },
            ],
            "gross_pnl_usd": 85.00,
            "total_fees_usd": 21.91,
            "is_bps": 28,
            "slippage_bps_p95": 32,
        },
        "BTC-USD": {
            "fills": [
                {
                    "timestamp": "2025-08-14T08:15:00Z",
                    "side": "buy",
                    "qty": 0.02,
                    "price": 58500.00,
                    "venue": "binance",
                    "fee_usd": 11.70,
                }
            ],
            "gross_pnl_usd": 45.00,
            "total_fees_usd": 11.70,
            "is_bps": 22,
            "slippage_bps_p95": 26,
        },
        "ETH-USD": {
            "fills": [
                {
                    "timestamp": "2025-08-14T09:45:00Z",
                    "side": "sell",
                    "qty": 0.5,
                    "price": 2420.00,
                    "venue": "coinbase",
                    "fee_usd": 12.10,
                }
            ],
            "gross_pnl_usd": 28.00,
            "total_fees_usd": 12.10,
            "is_bps": 31,
            "slippage_bps_p95": 35,
        },
        "NVDA": {
            "fills": [],
            "gross_pnl_usd": 0.00,
            "total_fees_usd": 0.00,
            "is_bps": 18,
            "slippage_bps_p95": 22,
        },
    }


def load_infra_costs(cost_file: Optional[str] = None) -> dict:
    """Load infrastructure costs."""
    if cost_file and os.path.exists(cost_file):
        try:
            with open(cost_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cost file {cost_file}: {e}")

    # Stub infrastructure costs
    return {
        "daily_cost_usd": 95.50,
        "breakdown": {
            "aws_compute": 42.30,
            "aws_storage": 8.20,
            "gpu_instances": 35.00,
            "network_data": 6.50,
            "monitoring": 3.50,
        },
        "cost_per_hour": 3.98,
    }


def compute_net_economics(fee_data: dict, infra_costs: dict, target_date: str) -> dict:
    """Compute net economics with all cost deductions."""

    # Portfolio totals
    total_gross_pnl = sum(asset["gross_pnl_usd"] for asset in fee_data.values())
    total_fees = sum(asset["total_fees_usd"] for asset in fee_data.values())
    total_infra_cost = infra_costs.get("daily_cost_usd", 0)

    # Net P&L calculation
    net_pnl_before_infra = total_gross_pnl - total_fees
    net_pnl_final = net_pnl_before_infra - total_infra_cost

    # Per-asset breakdown
    asset_breakdown = {}
    for asset, data in fee_data.items():
        # Allocate infra costs proportionally by gross PnL
        if total_gross_pnl > 0:
            infra_allocation = (
                data["gross_pnl_usd"] / total_gross_pnl
            ) * total_infra_cost
        else:
            infra_allocation = total_infra_cost / len(fee_data)  # Equal split if no PnL

        asset_net = data["gross_pnl_usd"] - data["total_fees_usd"] - infra_allocation

        asset_breakdown[asset] = {
            "gross_pnl_usd": data["gross_pnl_usd"],
            "fees_usd": data["total_fees_usd"],
            "infra_cost_allocated_usd": infra_allocation,
            "net_pnl_usd": asset_net,
            "fill_count": len(data.get("fills", [])),
            "is_bps": data.get("is_bps", 0),
            "slippage_bps_p95": data.get("slippage_bps_p95", 0),
            "net_margin_pct": (asset_net / max(abs(data["gross_pnl_usd"]), 1)) * 100,
        }

    # Portfolio summary
    economics = {
        "date": target_date,
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        "portfolio": {
            "gross_pnl_usd": total_gross_pnl,
            "total_fees_usd": total_fees,
            "total_infra_cost_usd": total_infra_cost,
            "net_pnl_before_infra_usd": net_pnl_before_infra,
            "net_pnl_final_usd": net_pnl_final,
            "cost_ratio": (total_fees + total_infra_cost)
            / max(abs(total_gross_pnl), 1),
            "net_margin_pct": (net_pnl_final / max(abs(total_gross_pnl), 1)) * 100,
        },
        "assets": asset_breakdown,
        "infra_costs": infra_costs,
        "summary": {
            "profitable_assets": sum(
                1 for a in asset_breakdown.values() if a["net_pnl_usd"] > 0
            ),
            "total_assets": len(asset_breakdown),
            "avg_is_bps": sum(a["is_bps"] for a in asset_breakdown.values())
            / len(asset_breakdown),
            "avg_slippage_bps": sum(
                a["slippage_bps_p95"] for a in asset_breakdown.values()
            )
            / len(asset_breakdown),
        },
    }

    return economics


def generate_readable_report(economics: dict) -> str:
    """Generate human-readable markdown report."""

    date = economics["date"]
    portfolio = economics["portfolio"]
    assets = economics["assets"]
    summary = economics["summary"]

    # Status determination
    net_pnl = portfolio["net_pnl_final_usd"]
    status = (
        "🟢 PROFITABLE"
        if net_pnl > 0
        else "🔴 LOSS" if net_pnl < -50 else "🟡 BREAKEVEN"
    )

    markdown = f"""# Daily Economic Close - {date}

**Status:** {status}  
**Net P&L After All Costs:** ${net_pnl:,.2f}  
**Generated:** {economics["timestamp"]}

## Portfolio Summary

| Metric | Value |
|--------|-------|
| **Gross P&L** | ${portfolio["gross_pnl_usd"]:,.2f} |
| **Total Fees** | ${portfolio["total_fees_usd"]:,.2f} |
| **Infrastructure Costs** | ${portfolio["total_infra_cost_usd"]:,.2f} |
| **Net P&L (Pre-Infra)** | ${portfolio["net_pnl_before_infra_usd"]:,.2f} |
| **Net P&L (Final)** | ${portfolio["net_pnl_final_usd"]:,.2f} |
| **Cost Ratio** | {portfolio["cost_ratio"]:.1%} |
| **Net Margin** | {portfolio["net_margin_pct"]:.1f}% |

## Per-Asset Breakdown

"""

    for asset, data in assets.items():
        pnl_emoji = (
            "✅"
            if data["net_pnl_usd"] > 0
            else "❌" if data["net_pnl_usd"] < -10 else "⚖️"
        )
        markdown += f"""### {pnl_emoji} {asset}

- **Net P&L:** ${data["net_pnl_usd"]:,.2f}
- **Gross P&L:** ${data["gross_pnl_usd"]:,.2f}  
- **Fees:** ${data["fees_usd"]:,.2f}
- **Infra Allocation:** ${data["infra_cost_allocated_usd"]:,.2f}
- **Fills:** {data["fill_count"]}
- **IS:** {data["is_bps"]} bps
- **Slippage P95:** {data["slippage_bps_p95"]} bps
- **Net Margin:** {data["net_margin_pct"]:.1f}%

"""

    markdown += f"""## Infrastructure Costs

"""
    infra = economics["infra_costs"]
    for component, cost in infra.get("breakdown", {}).items():
        markdown += f"- **{component.replace('_', ' ').title()}:** ${cost:,.2f}\n"

    markdown += f"""
**Total Daily:** ${infra.get("daily_cost_usd", 0):,.2f} (${infra.get("cost_per_hour", 0):.2f}/hour)

## Key Metrics

- **Profitable Assets:** {summary["profitable_assets"]}/{summary["total_assets"]}
- **Average IS:** {summary["avg_is_bps"]:.0f} bps  
- **Average Slippage:** {summary["avg_slippage_bps"]:.0f} bps

## Economic Assessment

"""

    if net_pnl > 100:
        markdown += (
            "🎯 **STRONG PERFORMANCE** - Well above breakeven with healthy margins"
        )
    elif net_pnl > 0:
        markdown += "✅ **POSITIVE** - Generating profit after all costs"
    elif net_pnl > -50:
        markdown += "⚠️ **BREAKEVEN** - Close to cost neutrality, monitor closely"
    else:
        markdown += "🚨 **LOSSES** - Net negative after costs, review strategy"

    markdown += f"""

---
*Report generated by Economic Daily Close at {economics["timestamp"]}*
"""

    return markdown


def save_economics(economics: dict, output_dir: str) -> tuple:
    """Save economics data and report."""

    # Create output directory
    timestamp = datetime.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    econ_dir = Path(output_dir) / timestamp
    econ_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON data
    json_file = econ_dir / "econ_close.json"
    with open(json_file, "w") as f:
        json.dump(economics, f, indent=2)

    # Save markdown report
    markdown_content = generate_readable_report(economics)
    md_file = econ_dir / "econ_close.md"
    with open(md_file, "w") as f:
        f.write(markdown_content)

    # Create latest symlinks
    latest_json = Path(output_dir) / "econ_close_latest.json"
    latest_md = Path(output_dir) / "econ_close_latest.md"

    if latest_json.exists():
        latest_json.unlink()
    if latest_md.exists():
        latest_md.unlink()

    latest_json.symlink_to(
        json_file.name if json_file.parent == latest_json.parent else json_file
    )
    latest_md.symlink_to(
        md_file.name if md_file.parent == latest_md.parent else md_file
    )

    return str(json_file), str(md_file)


def main():
    """Main economic daily close function."""
    parser = argparse.ArgumentParser(description="Economic Daily Close")
    parser.add_argument(
        "--date",
        default="UTC-1",
        help="Target date (UTC-1 for yesterday, or YYYY-MM-DD)",
    )
    parser.add_argument("--out", default="artifacts/econ", help="Output directory")
    parser.add_argument("--cost-file", help="Infrastructure cost file (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Parse target date
    if args.date == "UTC-1":
        target_date = (
            datetime.datetime.now(timezone.utc) - timedelta(days=1)
        ).strftime("%Y-%m-%d")
    else:
        target_date = args.date

    print("💰 Economic Daily Close")
    print("=" * 40)
    print(f"Target Date: {target_date}")
    print(f"Output: {args.out}")
    if args.cost_file:
        print(f"Cost File: {args.cost_file}")
    print("=" * 40)

    try:
        # Load data
        print("📊 Loading fee engine data...")
        fee_data = load_fee_engine_data()

        print("🏗️ Loading infrastructure costs...")
        infra_costs = load_infra_costs(args.cost_file)

        # Compute economics
        print("🧮 Computing net economics...")
        economics = compute_net_economics(fee_data, infra_costs, target_date)

        # Save results
        print("💾 Saving economics data...")
        json_file, md_file = save_economics(economics, args.out)

        # Display summary
        portfolio = economics["portfolio"]
        net_pnl = portfolio["net_pnl_final_usd"]

        print("\n📋 Economic Summary:")
        print(f"  Net P&L (Final): ${net_pnl:,.2f}")
        print(f"  Cost Ratio: {portfolio['cost_ratio']:.1%}")
        print(f"  Net Margin: {portfolio['net_margin_pct']:.1f}%")
        print(
            f"  Profitable Assets: {economics['summary']['profitable_assets']}/{economics['summary']['total_assets']}"
        )

        print(f"\n📄 Reports:")
        print(f"  JSON: {json_file}")
        print(f"  Markdown: {md_file}")

        if args.verbose:
            print(f"\n📊 Per-Asset Details:")
            for asset, data in economics["assets"].items():
                print(
                    f"  {asset}: ${data['net_pnl_usd']:,.2f} net ({data['fill_count']} fills)"
                )

        return 0

    except Exception as e:
        print(f"❌ Economic daily close failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
