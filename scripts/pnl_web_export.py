#!/usr/bin/env python3
"""
Export P&L data to JSON for web dashboard
"""
import os
import json
import time
from live_pnl_tracker import LivePnLTracker


def export_pnl_to_json():
    """Export current P&L to JSON file for frontend"""
    try:
        tracker = LivePnLTracker()
        pnl_data = tracker.calculate_pnl()

        if pnl_data:
            # Export to web-accessible location
            web_data = {
                "timestamp": time.time(),
                "status": "live",
                "portfolio": pnl_data["portfolio"],
                "positions": {
                    "BTC": {
                        "symbol": "BTC",
                        "amount": pnl_data["holdings"]["BTC"]["amount"],
                        "value": pnl_data["holdings"]["BTC"]["value"],
                        "cost": pnl_data["holdings"]["BTC"]["cost"],
                        "pnl": pnl_data["holdings"]["BTC"]["pnl"],
                        "return_pct": pnl_data["holdings"]["BTC"]["return_pct"],
                        "current_price": pnl_data["prices"]["BTC"],
                        "avg_price": float(tracker.trades["BTC"]["avg_cost"]),
                    },
                    "ETH": {
                        "symbol": "ETH",
                        "amount": pnl_data["holdings"]["ETH"]["amount"],
                        "value": pnl_data["holdings"]["ETH"]["value"],
                        "cost": pnl_data["holdings"]["ETH"]["cost"],
                        "pnl": pnl_data["holdings"]["ETH"]["pnl"],
                        "return_pct": pnl_data["holdings"]["ETH"]["return_pct"],
                        "current_price": pnl_data["prices"]["ETH"],
                        "avg_price": float(tracker.trades["ETH"]["avg_cost"]),
                    },
                    "USDT": {
                        "symbol": "USDT",
                        "amount": pnl_data["holdings"]["USDT"]["amount"],
                        "value": pnl_data["holdings"]["USDT"]["value"],
                    },
                },
                "trades_executed": 4,
                "last_trade": "ETH FILLED",
                "strategy": "50/50 BTC/ETH",
            }

            # Write to frontend directory
            output_file = "../frontend/live_pnl_data.json"
            with open(output_file, "w") as f:
                json.dump(web_data, f, indent=2)

            print(f"✅ P&L data exported to {output_file}")
            return True

    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False


if __name__ == "__main__":
    export_pnl_to_json()
