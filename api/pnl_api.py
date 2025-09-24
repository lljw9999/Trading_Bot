#!/usr/bin/env python3
"""
P&L API Server for Live Trading Dashboard
Serves real-time portfolio data to the frontend
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys
import json
import time

# Add project root to path
sys.path.append("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

from scripts.live_pnl_tracker import LivePnLTracker

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global tracker instance
pnl_tracker = None


def init_tracker():
    """Initialize the P&L tracker"""
    global pnl_tracker
    try:
        pnl_tracker = LivePnLTracker()
        print("✅ P&L Tracker initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize P&L Tracker: {e}")
        return False


@app.route("/api/health")
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": time.time(),
            "tracker_initialized": pnl_tracker is not None,
        }
    )


@app.route("/api/pnl")
def get_pnl():
    """Get current P&L data"""
    global pnl_tracker

    if not pnl_tracker:
        return jsonify({"error": "P&L tracker not initialized"}), 500

    try:
        pnl_data = pnl_tracker.calculate_pnl()
        if pnl_data:
            return jsonify(pnl_data)
        else:
            return jsonify({"error": "Failed to fetch P&L data"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/portfolio")
def get_portfolio():
    """Get portfolio summary for dashboard"""
    global pnl_tracker

    if not pnl_tracker:
        return jsonify({"error": "P&L tracker not initialized"}), 500

    try:
        pnl_data = pnl_tracker.calculate_pnl()
        if not pnl_data:
            return jsonify({"error": "Failed to fetch portfolio data"}), 500

        # Format data for frontend dashboard
        portfolio = pnl_data["portfolio"]
        holdings = pnl_data["holdings"]
        prices = pnl_data["prices"]

        dashboard_data = {
            "status": "live",
            "timestamp": pnl_data["timestamp"],
            "portfolio": {
                "total_value": portfolio["total_value"],
                "total_pnl": portfolio["total_pnl"],
                "total_return_pct": portfolio["total_return_pct"],
                "initial_investment": portfolio["initial_investment"],
                "deployed_capital": portfolio["deployed_capital"],
            },
            "positions": {
                "BTC": {
                    "symbol": "BTC",
                    "amount": holdings["BTC"]["amount"],
                    "value": holdings["BTC"]["value"],
                    "cost": holdings["BTC"]["cost"],
                    "pnl": holdings["BTC"]["pnl"],
                    "return_pct": holdings["BTC"]["return_pct"],
                    "current_price": prices["BTC"],
                    "avg_price": pnl_tracker.trades["BTC"]["avg_cost"],
                },
                "ETH": {
                    "symbol": "ETH",
                    "amount": holdings["ETH"]["amount"],
                    "value": holdings["ETH"]["value"],
                    "cost": holdings["ETH"]["cost"],
                    "pnl": holdings["ETH"]["pnl"],
                    "return_pct": holdings["ETH"]["return_pct"],
                    "current_price": prices["ETH"],
                    "avg_price": pnl_tracker.trades["ETH"]["avg_cost"],
                },
                "USDT": {
                    "symbol": "USDT",
                    "amount": holdings["USDT"]["amount"],
                    "value": holdings["USDT"]["value"],
                },
            },
            "price_changes_24h": pnl_data["price_changes_24h"],
            "trades_executed": 4,  # Your successful trades
            "last_trade": "ETH FILLED",
            "strategy": "50/50 BTC/ETH",
        }

        return jsonify(dashboard_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/balances")
def get_balances():
    """Get current Binance balances"""
    global pnl_tracker

    if not pnl_tracker:
        return jsonify({"error": "P&L tracker not initialized"}), 500

    try:
        balances = pnl_tracker.get_current_balances()
        return jsonify(balances)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Starting P&L API Server...")

    if init_tracker():
        print("🌐 Server running on http://localhost:5000")
        print("📊 Endpoints:")
        print("  - GET /api/health")
        print("  - GET /api/pnl")
        print("  - GET /api/portfolio")
        print("  - GET /api/balances")

        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        print("❌ Failed to start server - P&L tracker initialization failed")
        sys.exit(1)
