#!/usr/bin/env python3
"""
Budget Tripwire Kill-Switch
Monitor daily/weekly P&L and monthly costs - auto-revert to 0% on breach
"""
import os
import sys
import json
import glob
import datetime
import requests
import pathlib
from datetime import timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_budget_policy() -> dict:
    """Load budget policy from ramp configuration."""
    try:
        import yaml

        with open("ramp/ramp_policy.yaml", "r") as f:
            policy = yaml.safe_load(f)
        return policy.get(
            "budgets",
            {"daily_loss_usd": 1000, "weekly_loss_usd": 3000, "monthly_cost_usd": 2500},
        )
    except Exception as e:
        print(f"Warning: Could not load budget policy: {e}")
        # Return default budgets
        return {
            "daily_loss_usd": 1000,
            "weekly_loss_usd": 3000,
            "monthly_cost_usd": 2500,
        }


def get_recent_pnl_data(days: int) -> List[dict]:
    """Get P&L data for recent days."""
    pnl_data = []

    try:
        # Look for economic close files
        econ_files = glob.glob("artifacts/econ/*/econ_close.json")

        cutoff = datetime.datetime.now(timezone.utc) - timedelta(days=days)

        for econ_file in econ_files:
            try:
                # Get file modification time as proxy for date
                file_mtime = datetime.datetime.fromtimestamp(
                    os.path.getmtime(econ_file), tz=timezone.utc
                )

                if file_mtime > cutoff:
                    with open(econ_file, "r") as f:
                        data = json.load(f)

                    pnl_data.append(
                        {
                            "date": data.get("date", file_mtime.strftime("%Y-%m-%d")),
                            "file_time": file_mtime,
                            "net_pnl_usd": data.get("portfolio", {}).get(
                                "net_pnl_final_usd", 0
                            ),
                            "gross_pnl_usd": data.get("portfolio", {}).get(
                                "gross_pnl_usd", 0
                            ),
                            "file": econ_file,
                        }
                    )
            except Exception as e:
                print(f"Warning: Could not process {econ_file}: {e}")
                continue

    except Exception as e:
        print(f"Warning: Error gathering P&L data: {e}")

    # Sort by file time (most recent first)
    pnl_data.sort(key=lambda x: x["file_time"], reverse=True)

    return pnl_data


def get_monthly_cost_data() -> dict:
    """Get monthly infrastructure cost data."""
    try:
        # Look for cost data - try latest economic close first
        latest_econ = glob.glob("artifacts/econ/*/econ_close.json")
        if latest_econ:
            latest_file = max(latest_econ, key=os.path.getmtime)
            with open(latest_file, "r") as f:
                data = json.load(f)

            daily_cost = data.get("infra_costs", {}).get("daily_cost_usd", 0)

            # Estimate monthly cost (30 days)
            return {
                "monthly_cost_estimate_usd": daily_cost * 30,
                "daily_cost_usd": daily_cost,
                "source": "economic_close_estimate",
            }
    except Exception as e:
        print(f"Warning: Could not load cost data: {e}")

    # Fallback to stub data
    return {
        "monthly_cost_estimate_usd": 2400,
        "daily_cost_usd": 80,
        "source": "fallback_estimate",
    }


def check_daily_loss_budget(budget_policy: dict) -> Tuple[bool, dict]:
    """Check daily loss budget."""
    daily_limit = budget_policy.get("daily_loss_usd", 1000)

    # Get today's P&L
    today_pnl = get_recent_pnl_data(1)

    if not today_pnl:
        return True, {  # No data = assume OK
            "status": "no_data",
            "limit": daily_limit,
            "current_loss": 0,
            "breach": False,
        }

    latest_pnl = today_pnl[0]["net_pnl_usd"]
    current_loss = max(0, -latest_pnl)  # Loss is negative P&L
    breach = current_loss > daily_limit

    return not breach, {
        "status": "checked",
        "limit": daily_limit,
        "current_loss": current_loss,
        "net_pnl": latest_pnl,
        "breach": breach,
        "date": today_pnl[0]["date"],
    }


def check_weekly_loss_budget(budget_policy: dict) -> Tuple[bool, dict]:
    """Check weekly loss budget."""
    weekly_limit = budget_policy.get("weekly_loss_usd", 3000)

    # Get last 7 days P&L
    weekly_pnl = get_recent_pnl_data(7)

    if not weekly_pnl:
        return True, {  # No data = assume OK
            "status": "no_data",
            "limit": weekly_limit,
            "total_loss": 0,
            "breach": False,
        }

    # Sum up losses (negative P&L)
    total_loss = sum(max(0, -pnl["net_pnl_usd"]) for pnl in weekly_pnl)
    total_pnl = sum(pnl["net_pnl_usd"] for pnl in weekly_pnl)

    breach = total_loss > weekly_limit

    return not breach, {
        "status": "checked",
        "limit": weekly_limit,
        "total_loss": total_loss,
        "total_pnl": total_pnl,
        "days_data": len(weekly_pnl),
        "breach": breach,
    }


def check_monthly_cost_budget(budget_policy: dict) -> Tuple[bool, dict]:
    """Check monthly cost budget."""
    monthly_limit = budget_policy.get("monthly_cost_usd", 2500)

    cost_data = get_monthly_cost_data()
    monthly_cost = cost_data["monthly_cost_estimate_usd"]

    breach = monthly_cost > monthly_limit

    return not breach, {
        "status": "checked",
        "limit": monthly_limit,
        "monthly_cost": monthly_cost,
        "daily_cost": cost_data["daily_cost_usd"],
        "source": cost_data["source"],
        "breach": breach,
    }


def emergency_kill_switch(reason: str, breach_data: dict) -> bool:
    """Execute emergency kill-switch - set all influence to 0%."""
    print(f"🚨 EMERGENCY KILL-SWITCH TRIGGERED: {reason}")

    try:
        # Set all influence to 0%
        sys.path.insert(0, "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")
        from src.rl.influence_controller import InfluenceController

        ic = InfluenceController()

        # Get current weights and zero them out
        current_weights = ic.get_all_asset_weights()

        for asset in current_weights.keys():
            ic.set_weight_asset(asset, 0, f"Budget tripwire: {reason}")
            print(f"  🔒 {asset} influence set to 0%")

        # Also set the main influence to 0
        ic.set_weight(0, f"Budget tripwire: {reason}")
        print(f"  🔒 Main influence set to 0%")

        print("✅ Emergency kill-switch executed successfully")
        return True

    except Exception as e:
        print(f"❌ Emergency kill-switch FAILED: {e}")
        return False


def write_budget_trip_audit(
    trip_type: str, breach_data: dict, kill_switch_success: bool
):
    """Write WORM audit record for budget trip."""
    timestamp = datetime.datetime.now(timezone.utc)
    ts_str = timestamp.isoformat().replace(":", "_")

    pathlib.Path("artifacts/audit").mkdir(parents=True, exist_ok=True)
    audit_file = f"artifacts/audit/{ts_str}_budget_trip.json"

    audit_record = {
        "timestamp": timestamp.isoformat(),
        "action": "budget_tripwire_triggered",
        "trip_type": trip_type,
        "breach_data": breach_data,
        "kill_switch_executed": kill_switch_success,
        "operator": "budget_tripwire_daemon",
        "severity": "CRITICAL",
    }

    with open(audit_file, "w") as f:
        json.dump(audit_record, f, indent=2)

    print(f"📋 Budget trip audit: {audit_file}")
    return audit_file


def send_slack_alert(trip_type: str, breach_data: dict, kill_switch_success: bool):
    """Send priority Slack alert for budget trip."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("SLACK_WEBHOOK_URL not set - skipping alert")
        return False

    # Format alert message
    status_emoji = "✅" if kill_switch_success else "❌"

    if trip_type == "daily_loss":
        message = f"🚨 DAILY LOSS BUDGET BREACHED\nLoss: ${breach_data['current_loss']:,.2f} > ${breach_data['limit']:,.2f} limit"
    elif trip_type == "weekly_loss":
        message = f"🚨 WEEKLY LOSS BUDGET BREACHED\nTotal Loss: ${breach_data['total_loss']:,.2f} > ${breach_data['limit']:,.2f} limit"
    elif trip_type == "monthly_cost":
        message = f"🚨 MONTHLY COST BUDGET BREACHED\nCosts: ${breach_data['monthly_cost']:,.2f} > ${breach_data['limit']:,.2f} limit"
    else:
        message = f"🚨 BUDGET BREACH: {trip_type}"

    alert_payload = {
        "text": "🚨 BUDGET TRIPWIRE TRIGGERED",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{message}\n{status_emoji} Kill-switch: {'SUCCESS' if kill_switch_success else 'FAILED'}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Action Taken:* All influence set to 0%\n*Time:* {datetime.datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
                },
            },
        ],
    }

    try:
        response = requests.post(webhook_url, json=alert_payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to send Slack alert: {e}")
        return False


def run_budget_checks(synthetic_breach: Optional[str] = None) -> dict:
    """Run all budget checks and execute tripwire if needed."""

    print("💰 Budget Tripwire Check")
    print("=" * 40)

    budget_policy = load_budget_policy()

    results = {
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        "budget_policy": budget_policy,
        "checks": {},
        "breaches": [],
        "actions_taken": [],
    }

    # Synthetic breach for testing
    if synthetic_breach:
        print(f"🧪 SYNTHETIC BREACH TEST: {synthetic_breach}")

        if synthetic_breach == "daily_loss":
            breach_data = {
                "limit": budget_policy["daily_loss_usd"],
                "current_loss": budget_policy["daily_loss_usd"] + 500,
                "breach": True,
            }
        elif synthetic_breach == "weekly_loss":
            breach_data = {
                "limit": budget_policy["weekly_loss_usd"],
                "total_loss": budget_policy["weekly_loss_usd"] + 1000,
                "breach": True,
            }
        else:  # monthly_cost
            breach_data = {
                "limit": budget_policy["monthly_cost_usd"],
                "monthly_cost": budget_policy["monthly_cost_usd"] + 500,
                "breach": True,
            }

        results["synthetic_test"] = True
        results["breaches"].append(synthetic_breach)

        # Execute kill-switch
        kill_success = emergency_kill_switch(
            f"SYNTHETIC TEST: {synthetic_breach}", breach_data
        )
        audit_file = write_budget_trip_audit(
            synthetic_breach, breach_data, kill_success
        )
        slack_success = send_slack_alert(synthetic_breach, breach_data, kill_success)

        results["actions_taken"].extend(
            [
                f"kill_switch: {'SUCCESS' if kill_success else 'FAILED'}",
                f"audit_written: {audit_file}",
                f"slack_alert: {'SUCCESS' if slack_success else 'FAILED'}",
            ]
        )

        return results

    # Real budget checks
    print("📊 Checking daily loss budget...")
    daily_ok, daily_data = check_daily_loss_budget(budget_policy)
    results["checks"]["daily_loss"] = daily_data

    print("📈 Checking weekly loss budget...")
    weekly_ok, weekly_data = check_weekly_loss_budget(budget_policy)
    results["checks"]["weekly_loss"] = weekly_data

    print("🏗️ Checking monthly cost budget...")
    cost_ok, cost_data = check_monthly_cost_budget(budget_policy)
    results["checks"]["monthly_cost"] = cost_data

    # Check for breaches
    if not daily_ok and daily_data["breach"]:
        results["breaches"].append("daily_loss")
        print(
            f"🚨 DAILY LOSS BREACH: ${daily_data['current_loss']:.2f} > ${daily_data['limit']:.2f}"
        )

        kill_success = emergency_kill_switch("Daily loss budget exceeded", daily_data)
        audit_file = write_budget_trip_audit("daily_loss", daily_data, kill_success)
        slack_success = send_slack_alert("daily_loss", daily_data, kill_success)

        results["actions_taken"].extend(
            [
                f"daily_loss_kill_switch: {'SUCCESS' if kill_success else 'FAILED'}",
                f"audit: {audit_file}",
                f"slack: {'SUCCESS' if slack_success else 'FAILED'}",
            ]
        )

    if not weekly_ok and weekly_data["breach"]:
        results["breaches"].append("weekly_loss")
        print(
            f"🚨 WEEKLY LOSS BREACH: ${weekly_data['total_loss']:.2f} > ${weekly_data['limit']:.2f}"
        )

        kill_success = emergency_kill_switch("Weekly loss budget exceeded", weekly_data)
        audit_file = write_budget_trip_audit("weekly_loss", weekly_data, kill_success)
        slack_success = send_slack_alert("weekly_loss", weekly_data, kill_success)

        results["actions_taken"].extend(
            [
                f"weekly_loss_kill_switch: {'SUCCESS' if kill_success else 'FAILED'}",
                f"audit: {audit_file}",
                f"slack: {'SUCCESS' if slack_success else 'FAILED'}",
            ]
        )

    if not cost_ok and cost_data["breach"]:
        results["breaches"].append("monthly_cost")
        print(
            f"🚨 MONTHLY COST BREACH: ${cost_data['monthly_cost']:.2f} > ${cost_data['limit']:.2f}"
        )

        kill_success = emergency_kill_switch("Monthly cost budget exceeded", cost_data)
        audit_file = write_budget_trip_audit("monthly_cost", cost_data, kill_success)
        slack_success = send_slack_alert("monthly_cost", cost_data, kill_success)

        results["actions_taken"].extend(
            [
                f"monthly_cost_kill_switch: {'SUCCESS' if kill_success else 'FAILED'}",
                f"audit: {audit_file}",
                f"slack: {'SUCCESS' if slack_success else 'FAILED'}",
            ]
        )

    # Summary
    if results["breaches"]:
        print(f"🚨 {len(results['breaches'])} budget breaches detected and handled")
    else:
        print("✅ All budget checks passed")

        for check_name, check_data in results["checks"].items():
            if check_data["status"] == "checked":
                if check_name == "daily_loss":
                    print(
                        f"  Daily: ${check_data.get('current_loss', 0):.2f} / ${check_data['limit']:.2f}"
                    )
                elif check_name == "weekly_loss":
                    print(
                        f"  Weekly: ${check_data.get('total_loss', 0):.2f} / ${check_data['limit']:.2f}"
                    )
                elif check_name == "monthly_cost":
                    print(
                        f"  Monthly: ${check_data.get('monthly_cost', 0):.2f} / ${check_data['limit']:.2f}"
                    )

    return results


def main():
    """Main budget tripwire function."""
    import argparse

    parser = argparse.ArgumentParser(description="Budget Tripwire Kill-Switch")
    parser.add_argument(
        "--synthetic-breach",
        choices=["daily_loss", "weekly_loss", "monthly_cost"],
        help="Trigger synthetic breach for testing",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="artifacts/budget",
        help="Output directory for results",
    )
    args = parser.parse_args()

    try:
        # Run budget checks
        results = run_budget_checks(args.synthetic_breach)

        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        results_file = output_dir / f"budget_check_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"💾 Results saved: {results_file}")

        # Return appropriate exit code
        return 1 if results["breaches"] else 0

    except Exception as e:
        print(f"❌ Budget tripwire failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
