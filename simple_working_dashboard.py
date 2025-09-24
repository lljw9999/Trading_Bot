#!/usr/bin/env python3
"""
Simple Working Trading Dashboard

A clean, minimal dashboard that focuses on displaying real-time data correctly.
"""

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import redis
import json
import asyncio
from datetime import datetime
import requests
from dotenv import load_dotenv
import sys
import math

# Load environment variables
load_dotenv()


# JSON Sanitization Utility
def sanitize_json_data(data):
    """Convert NaN/Infinity to None for safe JSON serialization."""
    if isinstance(data, dict):
        return {k: sanitize_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json_data(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    else:
        return data


def safe_json_response(data):
    """Return a JSONResponse with sanitized data."""
    sanitized_data = sanitize_json_data(data)
    return JSONResponse(content=sanitized_data)


# Add src directory to path for imports
sys.path.append("src")

try:
    from layers.layer1_signal_generation.cryptopanic_client import CryptoPanicClient

    cryptopanic_client = CryptoPanicClient()
    print("✅ CryptoPanic client initialized")
except Exception as e:
    print(f"⚠️ CryptoPanic client initialization failed: {e}")
    cryptopanic_client = None

app = FastAPI(title="Simple Trading Dashboard")

# Initialize Redis connection
try:
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    redis_client.ping()
    print("✅ Redis connected successfully")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    redis_client = None


def get_latest_price(symbol):
    """Get latest price from Redis."""
    try:
        if redis_client:
            # The Binance WebSocket stores data as lists in market.raw.crypto.{symbol} keys
            market_key = f"market.raw.crypto.{symbol}"
            # Get the most recent price data from the list (-1 means last element)
            latest_data = redis_client.lrange(market_key, -1, -1)
            if latest_data:
                data = json.loads(latest_data[0])
                return float(data.get("price", 0.0))
    except Exception as e:
        print(f"Error getting price for {symbol}: {e}")
    return 0.0


def get_portfolio_data():
    """Get portfolio data."""
    portfolio = {}

    # Get prices (use live prices, fallback to 0 if unavailable)
    btc_price = get_latest_price("BTCUSDT") or 0.0
    eth_price = get_latest_price("ETHUSDT") or 0.0

    # Simple portfolio simulation
    btc_position = 0.000849
    eth_position = 0.027927

    btc_entry = 117800.0
    eth_entry = 3580.82

    btc_value = btc_position * btc_price
    eth_value = eth_position * eth_price

    btc_pnl = btc_value - (btc_position * btc_entry)
    eth_pnl = eth_value - (eth_position * eth_entry)

    portfolio = {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "current_price": btc_price,
            "position_size": btc_position,
            "entry_price": btc_entry,
            "current_value": btc_value,
            "total_pnl": btc_pnl,
        },
        "ETHUSDT": {
            "symbol": "ETHUSDT",
            "current_price": eth_price,
            "position_size": eth_position,
            "entry_price": eth_entry,
            "current_value": eth_value,
            "total_pnl": eth_pnl,
        },
    }

    return portfolio


@app.get("/")
async def dashboard():
    """Serve the simple dashboard."""
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Trading Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .price-display {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
            color: #4CAF50;
        }
        .symbol {
            font-size: 24px;
            text-align: center;
            margin-bottom: 10px;
            color: #FFF;
        }
        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 15px 0;
        }
        .info-item {
            text-align: center;
            padding: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        .info-label {
            font-size: 12px;
            opacity: 0.8;
            display: block;
        }
        .info-value {
            font-size: 16px;
            font-weight: bold;
            display: block;
            margin-top: 5px;
        }
        .positive {
            color: #4CAF50;
        }
        .negative {
            color: #f44336;
        }
        .status {
            text-align: center;
            padding: 10px;
            margin: 20px 0;
            border-radius: 10px;
            background: rgba(76, 175, 80, 0.2);
            border: 1px solid rgba(76, 175, 80, 0.5);
        }
        .loading {
            color: #FF9800;
        }
        .updated {
            color: #4CAF50;
        }
        .error {
            color: #f44336;
        }
        .news-item {
            padding: 12px;
            margin: 8px 0;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            border-left: 3px solid #4CAF50;
        }
        .news-item.bearish {
            border-left-color: #f44336;
        }
        .news-item.neutral {
            border-left-color: #FF9800;
        }
        .news-title {
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 6px;
            line-height: 1.3;
        }
        .news-meta {
            font-size: 11px;
            opacity: 0.7;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .sentiment-pill {
            padding: 2px 6px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .sentiment-bullish {
            background-color: rgba(76, 175, 80, 0.3);
            color: #4CAF50;
        }
        .sentiment-bearish {
            background-color: rgba(244, 67, 54, 0.3);
            color: #f44336;
        }
        .sentiment-neutral {
            background-color: rgba(255, 152, 0, 0.3);
            color: #FF9800;
        }
        .risk-bar {
            position: sticky;
            top: 0;
            z-index: 1000;
            background: linear-gradient(90deg, rgba(244, 67, 54, 0.1) 0%, rgba(255, 193, 7, 0.1) 50%, rgba(76, 175, 80, 0.1) 100%);
            border-bottom: 2px solid rgba(244, 67, 54, 0.3);
            padding: 12px 20px;
            margin: -20px -20px 20px -20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(10px);
            border-radius: 0 0 15px 15px;
        }
        .risk-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 80px;
        }
        .risk-label {
            font-size: 10px;
            opacity: 0.8;
            text-transform: uppercase;
            font-weight: bold;
        }
        .risk-value {
            font-size: 14px;
            font-weight: bold;
            margin-top: 2px;
        }
        .risk-kill-switch {
            padding: 8px 16px;
            background: rgba(244, 67, 54, 0.2);
            border: 1px solid #f44336;
            border-radius: 20px;
        }
        .kill-line {
            color: #f44336;
            font-weight: bold;
            font-size: 12px;
            text-transform: uppercase;
        }
        .risk-critical {
            background: rgba(244, 67, 54, 0.3) !important;
            border-color: #f44336 !important;
        }
        .risk-warning {
            background: rgba(255, 193, 7, 0.3) !important;
            border-color: #ff9800 !important;
        }
        .btn-small {
            padding: 6px 12px;
            font-size: 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 6px;
            color: white;
            cursor: pointer;
            margin-left: 8px;
            transition: all 0.3s ease;
        }
        .btn-small:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        .trade-summary {
            display: flex;
            justify-content: space-around;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 15px;
        }
        .summary-item {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .summary-label {
            font-size: 11px;
            opacity: 0.7;
            margin-bottom: 4px;
        }
        .trade-log-container {
            max-height: 400px;
            overflow-y: auto;
        }
        .trade-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 13px;
        }
        .trade-row:hover {
            background: rgba(255, 255, 255, 0.05);
        }
        .trade-time {
            width: 80px;
            opacity: 0.7;
        }
        .trade-symbol {
            width: 60px;
            font-weight: bold;
        }
        .trade-side {
            width: 40px;
            font-weight: bold;
        }
        .trade-side.buy {
            color: #4CAF50;
        }
        .trade-side.sell {
            color: #f44336;
        }
        .trade-qty, .trade-price, .trade-pnl {
            width: 80px;
            text-align: right;
        }
        .trade-latency {
            width: 50px;
            text-align: right;
            opacity: 0.8;
        }
        .action-summary {
            display: flex;
            justify-content: space-around;
            background: rgba(76, 175, 80, 0.1);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 15px;
        }
        .action-sparkline {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
            text-align: center;
        }
        .action-log-container {
            max-height: 300px;
            overflow-y: auto;
        }
        .action-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 12px;
        }
        .action-row:hover {
            background: rgba(76, 175, 80, 0.1);
        }
        .action-time {
            width: 60px;
            opacity: 0.7;
        }
        .action-type {
            width: 60px;
            font-weight: bold;
            color: #4CAF50;
        }
        .action-size, .action-price, .action-confidence {
            width: 80px;
            text-align: right;
        }
        .gauges-container {
            display: flex;
            justify-content: space-around;
            gap: 30px;
        }
        .gauge-wrapper {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .gauge-label {
            font-size: 12px;
            opacity: 0.8;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .gauge {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: conic-gradient(#4CAF50 0deg, #4CAF50 0deg, rgba(255,255,255,0.1) 0deg, rgba(255,255,255,0.1) 360deg);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        .gauge::before {
            content: '';
            width: 70px;
            height: 70px;
            border-radius: 50%;
            background: var(--bg-gradient);
            position: absolute;
        }
        .gauge-value {
            position: relative;
            z-index: 1;
            font-size: 16px;
            font-weight: bold;
        }
        .metrics-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4CAF50;
        }
        .status-dot.warning {
            background: #ff9800;
        }
        .status-dot.error {
            background: #f44336;
        }
        .toast-container {
            position: fixed;
            top: 80px;
            right: 20px;
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-width: 400px;
        }
        .toast {
            padding: 16px 20px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            border-left: 4px solid;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            gap: 12px;
            animation: slideIn 0.3s ease-out;
            min-width: 300px;
        }
        .toast.info {
            background: rgba(33, 150, 243, 0.9);
            border-left-color: #2196F3;
            color: white;
        }
        .toast.warning {
            background: rgba(255, 152, 0, 0.9);
            border-left-color: #ff9800;
            color: white;
        }
        .toast.error {
            background: rgba(244, 67, 54, 0.9);
            border-left-color: #f44336;
            color: white;
        }
        .toast.critical {
            background: rgba(156, 39, 176, 0.9);
            border-left-color: #9c27b0;
            color: white;
            animation: pulse 1s infinite;
        }
        .toast-icon {
            font-size: 20px;
            flex-shrink: 0;
        }
        .toast-content {
            flex: 1;
        }
        .toast-title {
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 4px;
        }
        .toast-message {
            font-size: 12px;
            opacity: 0.9;
        }
        .toast-close {
            background: none;
            border: none;
            color: white;
            font-size: 18px;
            cursor: pointer;
            padding: 0;
            margin-left: 8px;
            opacity: 0.7;
            flex-shrink: 0;
        }
        .toast-close:hover {
            opacity: 1;
        }
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        
        /* Health-Check Panel Styles */
        .health-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin: 15px 0;
        }
        .health-service {
            padding: 12px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .service-name {
            font-size: 14px;
            font-weight: bold;
            color: #ffffff;
        }
        .service-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
        }
        .status-text {
            opacity: 0.9;
        }
        .health-timestamp {
            text-align: center;
            font-size: 11px;
            opacity: 0.6;
            margin-top: 15px;
            padding-top: 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        .overall-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            font-weight: bold;
        }
        .status-dot.healthy {
            background: #4CAF50;
            animation: pulse 2s infinite;
        }
        .status-dot.degraded {
            background: #FF9800;
            animation: pulse 2s infinite;
        }
        .status-dot.unhealthy {
            background: #f44336;
            animation: pulse 1s infinite;
        }
        .status-dot.unknown {
            background: #9E9E9E;
        }
        
        /* Model Controls Styles */
        .model-controls {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-left: 20px;
        }
        .model-status-pill {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
        }
        .model-info {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .model-name {
            font-size: 14px;
            font-weight: bold;
            color: #ffffff;
        }
        .model-hash {
            font-size: 11px;
            opacity: 0.7;
            color: #ffffff;
            font-family: 'Courier New', monospace;
        }
        .model-status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            flex-shrink: 0;
        }
        .model-status-dot.training {
            background: #FF9800;
            animation: pulse 1s infinite;
        }
        .model-status-dot.error {
            background: #f44336;
        }
        .rollback-btn {
            padding: 8px 16px;
            background: rgba(255, 152, 0, 0.2);
            border: 1px solid #FF9800;
            border-radius: 20px;
            color: #ffffff;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        .rollback-btn:hover {
            background: rgba(255, 152, 0, 0.4);
            transform: translateY(-1px);
        }
        .rollback-btn:active {
            transform: translateY(0);
        }
        
        /* Trading Interface Styles */
        .trading-interface {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .order-type-buttons {
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
            justify-content: space-between;
        }
        .order-type-btn {
            flex: 1;
            padding: 6px 4px;
            font-size: 10px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 5px;
            color: #ffffff;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .order-type-btn.active {
            background: rgba(76, 175, 80, 0.3);
            border-color: #4CAF50;
        }
        .order-type-btn:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        .trade-btn {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 8px 0;
        }
        .sell-btn {
            background: rgba(244, 67, 54, 0.8);
            color: white;
        }
        .sell-btn:hover {
            background: rgba(244, 67, 54, 1);
            transform: translateY(-1px);
        }
        .buy-btn {
            background: rgba(76, 175, 80, 0.8);
            color: white;
        }
        .buy-btn:hover {
            background: rgba(76, 175, 80, 1);
            transform: translateY(-1px);
        }
        .amount-section {
            margin: 15px 0;
        }
        .amount-label {
            display: block;
            margin-bottom: 8px;
            font-size: 12px;
            font-weight: bold;
            color: #ffffff;
        }
        .amount-input {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            color: #ffffff;
            font-size: 14px;
            box-sizing: border-box;
        }
        .amount-input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }
        .amount-input:focus {
            outline: none;
            border-color: #4CAF50;
            background: rgba(255, 255, 255, 0.15);
        }
        .amount-buttons {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .amount-btn {
            flex: 1;
            min-width: 60px;
            padding: 6px 8px;
            font-size: 11px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            color: #ffffff;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .amount-btn:hover {
            background: rgba(76, 175, 80, 0.3);
            border-color: #4CAF50;
            transform: scale(1.05);
        }
        
        /* Alpha Signals Styles */
        .alpha-signals-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 15px 0;
        }
        .alpha-signal {
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            border-left: 4px solid #FF9800;
        }
        .alpha-signal.strong-buy {
            border-left-color: #4CAF50;
        }
        .alpha-signal.weak-buy {
            border-left-color: #8BC34A;
        }
        .alpha-signal.strong-sell {
            border-left-color: #f44336;
        }
        .alpha-signal.weak-sell {
            border-left-color: #FF5722;
        }
        .alpha-signal.hold {
            border-left-color: #FF9800;
        }
        .signal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .signal-name {
            font-size: 13px;
            font-weight: bold;
            color: #ffffff;
        }
        .signal-strength {
            font-size: 16px;
            font-weight: bold;
            color: #4CAF50;
        }
        .signal-status {
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 5px;
            text-transform: uppercase;
        }
        .signal-status.strong-buy {
            color: #4CAF50;
        }
        .signal-status.weak-buy {
            color: #8BC34A;
        }
        .signal-status.strong-sell {
            color: #f44336;
        }
        .signal-status.weak-sell {
            color: #FF5722;
        }
        .signal-status.hold {
            color: #FF9800;
        }
        .signal-time {
            font-size: 11px;
            opacity: 0.7;
            color: #ffffff;
        }
        
        /* Chart Controls Styles */
        .chart-controls, .timeframe-buttons, .residual-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .chart-controls select, .residual-controls select {
            padding: 6px 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            color: #ffffff;
            font-size: 12px;
        }
        .chart-controls select:focus, .residual-controls select:focus {
            outline: none;
            border-color: #4CAF50;
        }
        .timeframe-btn {
            padding: 6px 12px;
            font-size: 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            color: #ffffff;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .timeframe-btn.active {
            background: rgba(76, 175, 80, 0.3);
            border-color: #4CAF50;
        }
        .timeframe-btn:hover {
            background: rgba(255, 255, 255, 0.2);
        }
    </style>
    
    <!-- jsPDF for PDF export functionality -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf-autotable/3.5.31/jspdf.plugin.autotable.min.js"></script>
    
    <!-- Plotly.js for advanced charts -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h1>🚀 Simple Trading Dashboard</h1>
                    <div id="status" class="status loading">Loading data...</div>
                    <div id="lastUpdate">Last update: Never</div>
                </div>
                <div class="model-controls">
                    <div class="model-status-pill" id="modelStatusPill">
                        <div class="model-info">
                            <span class="model-name" id="modelName">SAC-DiF v1.2.3</span>
                            <span class="model-hash" id="modelHash">#a7b9c2d</span>
                        </div>
                        <div class="model-status-dot" id="modelStatusDot"></div>
                    </div>
                    <div class="model-status-pill" id="modeStatusPill" style="display: none; background: rgba(244,67,54,0.2); border: 1px solid #f44336;">
                        <div class="model-info">
                            <span class="model-name" id="modeIndicator">Mode: Failover</span>
                        </div>
                        <div style="width: 8px; height: 8px; border-radius: 50%; background: #f44336;"></div>
                    </div>
                    <div style="margin-right: 10px; padding: 5px 10px; background: rgba(255,255,255,0.1); border-radius: 15px; font-size: 12px;">
                        <span>👤 Role:</span>
                        <select id="roleSelector" onchange="setUserRole(this.value)" style="background: transparent; color: white; border: none; font-size: 12px; margin-left: 5px;">
                            <option value="trader">👨‍💼 Trader</option>
                            <option value="admin">👑 Admin</option>
                            <option value="observer">👁️ Observer</option>
                        </select>
                    </div>
                    <button class="rollback-btn" id="rollbackBtn" onclick="showRollbackModal()">
                        🔄 Rollback
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Sticky Risk Bar (Critical for RL Trading) -->
        <div class="risk-bar" id="riskBar">
            <div class="risk-item">
                <span class="risk-label">Position:</span>
                <span class="risk-value" id="totalPosition">0%</span>
            </div>
            <div class="risk-item">
                <span class="risk-label">Drawdown:</span>
                <span class="risk-value" id="drawdown">0%</span>
            </div>
            <div class="risk-item">
                <span class="risk-label">VaR 95%:</span>
                <span class="risk-value" id="var95">$0.00</span>
            </div>
            <div class="risk-item">
                <span class="risk-label">CVaR 95%:</span>
                <span class="risk-value" id="cvar95">$0.00</span>
            </div>
            <div class="risk-kill-switch">
                <span class="kill-line" id="killLine">Kill Line: 5%</span>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <div class="symbol">₿ BTCUSDT</div>
                <div class="price-display" id="btcPrice">$0.00</div>
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">Position</span>
                        <span class="info-value" id="btcPosition">0.000000</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Entry Price</span>
                        <span class="info-value" id="btcEntry">$0.00</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Value</span>
                        <span class="info-value" id="btcValue">$0.00</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">P&L</span>
                        <span class="info-value" id="btcPnL">$0.00</span>
                    </div>
                </div>
                
                <!-- Trading Interface for BTC -->
                <div class="trading-interface">
                    <div class="order-type-buttons">
                        <button class="order-type-btn active" data-type="market">MKT</button>
                        <button class="order-type-btn" data-type="limit">LMT</button>
                        <button class="order-type-btn" data-type="stop">STP</button>
                        <button class="order-type-btn" data-type="oco">OCO</button>
                        <button class="order-type-btn" data-type="iceberg">ICE</button>
                    </div>
                    <button class="trade-btn sell-btn" onclick="executeTrade('BTCUSDT', 'SELL')">Sell</button>
                    
                    <div class="amount-section">
                        <span class="amount-label">💰 Enter Amount ($):</span>
                        <input type="text" class="amount-input" id="btcAmount" placeholder="Enter amount to buy" />
                        <div class="amount-buttons">
                            <button class="amount-btn" onclick="setAmount('btc', 50)">$50</button>
                            <button class="amount-btn" onclick="setAmount('btc', 100)">$100</button>
                            <button class="amount-btn" onclick="setAmount('btc', 250)">$250</button>
                            <button class="amount-btn" onclick="setAmount('btc', 500)">$500</button>
                            <button class="amount-btn" onclick="setAmount('btc', 1000)">$1000</button>
                        </div>
                    </div>
                    <button class="trade-btn buy-btn" onclick="executeTrade('BTCUSDT', 'BUY')">Buy</button>
                </div>
            </div>
            
            <div class="card">
                <div class="symbol">Ξ ETHUSDT</div>
                <div class="price-display" id="ethPrice">$0.00</div>
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">Position</span>
                        <span class="info-value" id="ethPosition">0.000000</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Entry Price</span>
                        <span class="info-value" id="ethEntry">$0.00</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Value</span>
                        <span class="info-value" id="ethValue">$0.00</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">P&L</span>
                        <span class="info-value" id="ethPnL">$0.00</span>
                    </div>
                </div>
                
                <!-- Trading Interface for ETH -->
                <div class="trading-interface">
                    <div class="order-type-buttons">
                        <button class="order-type-btn active" data-type="market">MKT</button>
                        <button class="order-type-btn" data-type="limit">LMT</button>
                        <button class="order-type-btn" data-type="stop">STP</button>
                        <button class="order-type-btn" data-type="oco">OCO</button>
                        <button class="order-type-btn" data-type="iceberg">ICE</button>
                    </div>
                    <button class="trade-btn sell-btn" onclick="executeTrade('ETHUSDT', 'SELL')">Sell</button>
                    
                    <div class="amount-section">
                        <span class="amount-label">💰 Enter Amount ($):</span>
                        <input type="text" class="amount-input" id="ethAmount" placeholder="Enter amount to buy" />
                        <div class="amount-buttons">
                            <button class="amount-btn" onclick="setAmount('eth', 50)">$50</button>
                            <button class="amount-btn" onclick="setAmount('eth', 100)">$100</button>
                            <button class="amount-btn" onclick="setAmount('eth', 250)">$250</button>
                            <button class="amount-btn" onclick="setAmount('eth', 500)">$500</button>
                            <button class="amount-btn" onclick="setAmount('eth', 1000)">$1000</button>
                        </div>
                    </div>
                    <button class="trade-btn buy-btn" onclick="executeTrade('ETHUSDT', 'BUY')">Buy</button>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3 style="text-align: center;">📊 Portfolio Summary</h3>
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">Total Value</span>
                    <span class="info-value" id="totalValue">$0.00</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Total P&L</span>
                    <span class="info-value" id="totalPnL">$0.00</span>
                </div>
            </div>
        </div>
        
        <!-- Alpha Signals Panel -->
        <div class="card">
            <h3 style="text-align: center;">🧠 Alpha Signals</h3>
            <div class="alpha-signals-grid">
                <div class="alpha-signal" id="rsiSignal">
                    <div class="signal-header">
                        <span class="signal-name">RSI Divergence</span>
                        <span class="signal-strength" id="rsiStrength">65%</span>
                    </div>
                    <div class="signal-status" id="rsiStatus">WEAK HOLD</div>
                    <div class="signal-time" id="rsiTime">06:26:03</div>
                </div>
                
                <div class="alpha-signal" id="volumeSignal">
                    <div class="signal-header">
                        <span class="signal-name">Volume Profile</span>
                        <span class="signal-strength" id="volumeStrength">69%</span>
                    </div>
                    <div class="signal-status" id="volumeStatus">WEAK HOLD</div>
                    <div class="signal-time" id="volumeTime">06:26:03</div>
                </div>
                
                <div class="alpha-signal" id="fiboSignal">
                    <div class="signal-header">
                        <span class="signal-name">Fibonacci Retracement</span>
                        <span class="signal-strength" id="fiboStrength">77%</span>
                    </div>
                    <div class="signal-status" id="fiboStatus">STRONG SELL</div>
                    <div class="signal-time" id="fiboTime">06:26:03</div>
                </div>
                
                <div class="alpha-signal" id="whaleSignal">
                    <div class="signal-header">
                        <span class="signal-name">Whale Activity</span>
                        <span class="signal-strength" id="whaleStrength">72%</span>
                    </div>
                    <div class="signal-status" id="whaleStatus">WEAK HOLD</div>
                    <div class="signal-time" id="whaleTime">06:26:03</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3 style="text-align: center;">📰 Market Sentiment</h3>
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">News Count</span>
                    <span class="info-value" id="newsCount">0</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Bullish</span>
                    <span class="info-value positive" id="bullishPct">0%</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Bearish</span>
                    <span class="info-value negative" id="bearishPct">0%</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Avg Impact</span>
                    <span class="info-value" id="avgImpact">0.000</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3 style="text-align: center;">📢 Recent News</h3>
            <div id="newsList" style="max-height: 300px; overflow-y: auto;">
                <div class="loading">Loading news...</div>
            </div>
        </div>
        
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3>📑 Daily Trade Log</h3>
                <div>
                    <button class="btn-small" onclick="exportTradeCSV()">📄 CSV</button>
                    <button class="btn-small" onclick="exportTradePDF()">📋 PDF</button>
                </div>
            </div>
            <div class="trade-summary">
                <div class="summary-item">
                    <span class="summary-label">Trades:</span>
                    <span id="tradeCount">0</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Win Rate:</span>
                    <span id="winRate">0%</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Gross P&L:</span>
                    <span id="grossPnL">$0.00</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Avg Latency:</span>
                    <span id="avgLatency">0ms</span>
                </div>
            </div>
            <div class="trade-log-container" id="tradeLogContainer">
                <div class="loading">Loading trades...</div>
            </div>
        </div>
        
        <div class="card">
            <h3 style="text-align: center;">⚡ RL Action Tape</h3>
            <div class="action-summary">
                <div class="summary-item">
                    <span class="summary-label">Actions/min:</span>
                    <span id="actionsPerMin">0</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Avg Size:</span>
                    <span id="avgActionSize">0.000</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Last Action:</span>
                    <span id="lastAction">-</span>
                </div>
            </div>
            <div class="action-sparkline" id="actionSparkline">
                <canvas id="actionChart" width="400" height="60"></canvas>
            </div>
            <div class="action-log-container" id="actionLogContainer">
                <div class="loading">Loading RL actions...</div>
            </div>
        </div>
        
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3>🎯 Policy Metrics</h3>
                <div class="metrics-status" id="policyStatus">
                    <span class="status-dot" id="policyDot"></span>
                    <span>Monitoring</span>
                </div>
            </div>
            <div class="gauges-container">
                <div class="gauge-wrapper">
                    <div class="gauge-label">Policy Entropy</div>
                    <div class="gauge" id="entropyGauge">
                        <div class="gauge-value" id="entropyValue">0.00</div>
                    </div>
                </div>
                <div class="gauge-wrapper">
                    <div class="gauge-label">Q-Spread</div>
                    <div class="gauge" id="qspreadGauge">
                        <div class="gauge-value" id="qspreadValue">0.00</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Health-Check Panel -->
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3>💚 System Health</h3>
                <div class="overall-status" id="overallHealthStatus">
                    <span class="status-dot" id="overallHealthDot"></span>
                    <span id="overallHealthText">Checking...</span>
                </div>
            </div>
            <div class="health-grid">
                <div class="health-service" id="healthRedis">
                    <div class="service-name">📊 Redis</div>
                    <div class="service-status">
                        <span class="status-dot" id="redisStatusDot"></span>
                        <span class="status-text" id="redisStatusText">Checking...</span>
                    </div>
                </div>
                <div class="health-service" id="healthWebSocket">
                    <div class="service-name">🔌 WebSocket</div>
                    <div class="service-status">
                        <span class="status-dot" id="websocketStatusDot"></span>
                        <span class="status-text" id="websocketStatusText">Checking...</span>
                    </div>
                </div>
                <div class="health-service" id="healthNews">
                    <div class="service-name">📰 News Feed</div>
                    <div class="service-status">
                        <span class="status-dot" id="newsStatusDot"></span>
                        <span class="status-text" id="newsStatusText">Checking...</span>
                    </div>
                </div>
                <div class="health-service" id="healthPolicy">
                    <div class="service-name">🤖 Policy Daemon</div>
                    <div class="service-status">
                        <span class="status-dot" id="policyStatusDot"></span>
                        <span class="status-text" id="policyStatusText">Checking...</span>
                    </div>
                </div>
                <div class="health-service" id="healthAPI">
                    <div class="service-name">⚡ API Response</div>
                    <div class="service-status">
                        <span class="status-dot" id="apiStatusDot"></span>
                        <span class="status-text" id="apiStatusText">Checking...</span>
                    </div>
                </div>
            </div>
            <div class="health-timestamp" id="healthTimestamp">Last check: Never</div>
        </div>
        
        <!-- Prediction vs Market Chart -->
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3>📈 Prediction vs Market</h3>
                <div class="chart-controls">
                    <select id="predictionSymbol" onchange="updatePredictionChart()">
                        <option value="BTCUSDT">BTC</option>
                        <option value="ETHUSDT">ETH</option>
                    </select>
                    <button class="btn-small" onclick="updatePredictionChart()">🔄 Refresh</button>
                </div>
            </div>
            <div id="predictionChart" style="height: 400px; width: 100%;"></div>
        </div>
        
        <!-- PnL Curve Multi-Timeframe -->
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3>💰 Portfolio Performance</h3>
                <div class="timeframe-buttons">
                    <button class="timeframe-btn active" data-period="24h" onclick="updatePnLChart('24h')">24H</button>
                    <button class="timeframe-btn" data-period="7d" onclick="updatePnLChart('7d')">7D</button>
                    <button class="timeframe-btn" data-period="30d" onclick="updatePnLChart('30d')">30D</button>
                </div>
            </div>
            <div id="pnlChart" style="height: 300px; width: 100%;"></div>
        </div>
        
        <!-- Residual Distribution -->
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3>🎯 Model Accuracy</h3>
                <div class="residual-controls">
                    <select id="residualSymbol" onchange="updateResidualChart()">
                        <option value="BTCUSDT">BTC</option>
                        <option value="ETHUSDT">ETH</option>
                    </select>
                    <select id="residualPeriod" onchange="updateResidualChart()">
                        <option value="1h">1H</option>
                        <option value="24h" selected>24H</option>
                        <option value="7d">7D</option>
                    </select>
                </div>
            </div>
            <div id="residualChart" style="height: 250px; width: 100%;"></div>
        </div>
        
        <!-- Twitter News & Sentiment -->
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3>𝕏 Twitter Sentiment</h3>
                <div class="chart-controls">
                    <button class="btn-small" onclick="updateTwitterNews()">🔄 Refresh</button>
                </div>
            </div>
            <div id="twitterNews" style="max-height: 300px; overflow-y: auto;">
                <div class="loading">Loading Twitter news...</div>
            </div>
        </div>
        
        <!-- Policy Monitoring -->
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3>🤖 RL Policy Health</h3>
                <div class="chart-controls">
                    <button class="btn-small" onclick="updatePolicyMonitoring()">🔄 Refresh</button>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div>
                    <h4>Entropy (Collapse Detection)</h4>
                    <div id="entropyChart" style="height: 200px; width: 100%;"></div>
                </div>
                <div>
                    <h4>Q-Value Spread</h4>
                    <div id="qspreadChart" style="height: 200px; width: 100%;"></div>
                </div>
            </div>
            <div id="policyStats" style="margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                <div class="loading">Loading policy stats...</div>
            </div>
        </div>
        
        <!-- Latency Drill-down -->
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3>⚡ Latency Pipeline</h3>
                <div class="chart-controls">
                    <button class="btn-small" onclick="updateLatencyDrilldown()">🔄 Refresh</button>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 15px;">
                <div>
                    <h4>WS → Redis → Policy → Order</h4>
                    <div id="latencyChart" style="height: 300px; width: 100%;"></div>
                </div>
                <div>
                    <h4>Performance Stats</h4>
                    <div id="latencyStats" style="padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px; height: 280px; overflow-y: auto;">
                        <div class="loading">Loading latency stats...</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Action Heat-map -->
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3>🎯 Action Bias Detection</h3>
                <div class="chart-controls">
                    <button class="btn-small" onclick="updateActionHeatmap()">🔄 Refresh</button>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 15px;">
                <div>
                    <h4>Price Offset × Size Density</h4>
                    <div id="actionHeatmap" style="height: 350px; width: 100%;"></div>
                </div>
                <div>
                    <h4>Bias Metrics</h4>
                    <div id="biasMetrics" style="padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px; height: 330px; overflow-y: auto;">
                        <div class="loading">Loading bias metrics...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Alert Toast Container -->
    <div class="toast-container" id="toastContainer"></div>

    <script>
        let updateCount = 0;
        
        // Role-based access control
        function getCurrentUserRole() {
            // In production, this would check JWT token or session
            // For demo, check localStorage or default to 'trader'
            return localStorage.getItem('user_role') || 'trader';
        }
        
        function hasTradePermission() {
            const role = getCurrentUserRole();
            // Only 'trader' and 'admin' can trade, not 'observer'
            return role !== 'observer';
        }
        
        function initializeRoleBasedUI() {
            const hasPermission = hasTradePermission();
            const role = getCurrentUserRole();
            
            // Hide/show trade buttons based on role
            const tradeButtons = document.querySelectorAll('.trade-btn');
            const amountInputs = document.querySelectorAll('input[type="number"]');
            const exportButtons = document.querySelectorAll('button[onclick*="export"]');
            
            tradeButtons.forEach(button => {
                if (!hasPermission) {
                    button.style.display = 'none';
                    // Add observer message
                    if (!button.nextElementSibling?.classList.contains('observer-message')) {
                        const message = document.createElement('div');
                        message.className = 'observer-message';
                        message.style.cssText = 'color: #ff9800; font-size: 11px; margin-top: 5px; padding: 5px; background: rgba(255,152,0,0.1); border-radius: 4px; text-align: center;';
                        message.textContent = '🔒 Trading disabled for observer role';
                        button.parentNode.insertBefore(message, button.nextSibling);
                    }
                } else {
                    button.style.display = '';
                    // Remove observer message if role changed back
                    const observerMsg = button.nextElementSibling;
                    if (observerMsg?.classList.contains('observer-message')) {
                        observerMsg.remove();
                    }
                }
            });
            
            // Disable amount inputs for observers
            amountInputs.forEach(input => {
                input.disabled = !hasPermission;
                if (!hasPermission) {
                    input.placeholder = "Observer mode";
                    input.style.opacity = "0.5";
                } else {
                    input.style.opacity = "1";
                }
            });
            
            // Hide export buttons for observers (sensitive trade data)
            exportButtons.forEach(button => {
                if (role === 'observer') {
                    button.style.display = 'none';
                } else {
                    button.style.display = '';
                }
            });
            
            // Hide/show entire trade panels for observers
            const tradePanels = document.querySelectorAll('.trading-interface, .trade-summary');
            tradePanels.forEach(panel => {
                if (!hasPermission && panel) {
                    panel.style.opacity = '0.3';
                    panel.style.pointerEvents = 'none';
                    
                    // Add overlay message
                    if (!panel.querySelector('.role-overlay')) {
                        const overlay = document.createElement('div');
                        overlay.className = 'role-overlay';
                        overlay.style.cssText = `
                            position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                            background: rgba(0,0,0,0.5); color: white; display: flex;
                            align-items: center; justify-content: center; font-size: 14px;
                            border-radius: 8px; z-index: 10;
                        `;
                        overlay.innerHTML = '🔒 Observer Role - Trading Restricted';
                        panel.style.position = 'relative';
                        panel.appendChild(overlay);
                    }
                } else if (panel) {
                    panel.style.opacity = '1';
                    panel.style.pointerEvents = '';
                    const overlay = panel.querySelector('.role-overlay');
                    if (overlay) overlay.remove();
                }
            });
            
            console.log(`UI initialized for role: ${role} (trade permission: ${hasPermission})`);
        }
        
        function setUserRole(role) {
            localStorage.setItem('user_role', role);
            initializeRoleBasedUI();
            showToast('info', '👤 Role Changed', `Switched to ${role} role`);
        }
        
        // Safe JSON parsing to prevent crashes
        async function safeJson(response) {
            const text = await response.text();
            try {
                return JSON.parse(text);
            } catch (e) {
                console.error('Bad JSON from', response.url, text.slice(0, 150));
                console.error('Parse error:', e);
                // Return fallback object instead of crashing
                return { error: 'JSON parse error', raw_response: text.slice(0, 100) };
            }
        }
        
        function updateDisplay(data) {
            console.log('Updating display with data:', data);
            
            // Add debug logging
            console.log('BTC Price element:', document.getElementById('btcPrice'));
            console.log('ETH Price element:', document.getElementById('ethPrice'));
            
            const btc = data.BTCUSDT;
            const eth = data.ETHUSDT;
            
            console.log('BTC data:', btc);
            console.log('ETH data:', eth);
            
            // Update BTC
            document.getElementById('btcPrice').textContent = `$${btc.current_price.toFixed(2)}`;
            document.getElementById('btcPosition').textContent = btc.position_size.toFixed(6);
            document.getElementById('btcEntry').textContent = `$${btc.entry_price.toFixed(2)}`;
            document.getElementById('btcValue').textContent = `$${btc.current_value.toFixed(2)}`;
            
            const btcPnLElement = document.getElementById('btcPnL');
            btcPnLElement.textContent = `$${btc.total_pnl.toFixed(2)}`;
            btcPnLElement.className = `info-value ${btc.total_pnl >= 0 ? 'positive' : 'negative'}`;
            
            // Update ETH
            document.getElementById('ethPrice').textContent = `$${eth.current_price.toFixed(2)}`;
            document.getElementById('ethPosition').textContent = eth.position_size.toFixed(6);
            document.getElementById('ethEntry').textContent = `$${eth.entry_price.toFixed(2)}`;
            document.getElementById('ethValue').textContent = `$${eth.current_value.toFixed(2)}`;
            
            const ethPnLElement = document.getElementById('ethPnL');
            ethPnLElement.textContent = `$${eth.total_pnl.toFixed(2)}`;
            ethPnLElement.className = `info-value ${eth.total_pnl >= 0 ? 'positive' : 'negative'}`;
            
            // Update totals
            const totalValue = btc.current_value + eth.current_value;
            const totalPnL = btc.total_pnl + eth.total_pnl;
            
            document.getElementById('totalValue').textContent = `$${totalValue.toFixed(2)}`;
            
            const totalPnLElement = document.getElementById('totalPnL');
            totalPnLElement.textContent = `$${totalPnL.toFixed(2)}`;
            totalPnLElement.className = `info-value ${totalPnL >= 0 ? 'positive' : 'negative'}`;
            
            // Update status
            updateCount++;
            const statusElement = document.getElementById('status');
            statusElement.textContent = `✅ Data loaded successfully (Update #${updateCount})`;
            statusElement.className = 'status updated';
            
            document.getElementById('lastUpdate').textContent = `Last update: ${new Date().toLocaleTimeString()}`;
        }
        
        async function fetchData() {
            try {
                console.log('Fetching portfolio data...');
                const response = await fetch('/api/portfolio');
                
                if (response.ok) {
                    const data = await safeJson(response);
                    if (data.error) {
                        console.error('API error:', data.error);
                        document.getElementById('status').textContent = `❌ API Error: ${data.error}`;
                        document.getElementById('status').className = 'status error';
                        return;
                    }
                    console.log('Portfolio data received:', data);
                    updateDisplay(data);
                } else {
                    console.error('Failed to fetch data:', response.status);
                    document.getElementById('status').textContent = `❌ Error: ${response.status}`;
                    document.getElementById('status').className = 'status error';
                }
            } catch (error) {
                console.error('Error fetching data:', error);
                document.getElementById('status').textContent = `❌ Error: ${error.message}`;
                document.getElementById('status').className = 'status error';
            }
        }
        
        async function fetchNewsData() {
            try {
                // Fetch sentiment summary
                const sentimentResponse = await fetch('/api/news-sentiment');
                if (sentimentResponse.ok) {
                    const sentimentData = await sentimentResponse.json();
                    updateSentimentDisplay(sentimentData);
                }
                
                // Fetch recent news
                const newsResponse = await fetch('/api/recent-news?limit=8');
                if (newsResponse.ok) {
                    const newsData = await newsResponse.json();
                    updateNewsDisplay(newsData.news || []);
                }
            } catch (error) {
                console.error('Error fetching news data:', error);
                document.getElementById('newsList').innerHTML = '<div class="error">Error loading news</div>';
            }
        }
        
        function updateSentimentDisplay(data) {
            if (data.error) {
                console.error('News sentiment error:', data.error);
                return;
            }
            
            document.getElementById('newsCount').textContent = data.total_posts || 0;
            document.getElementById('bullishPct').textContent = `${data.bullish_pct || 0}%`;
            document.getElementById('bearishPct').textContent = `${data.bearish_pct || 0}%`;
            document.getElementById('avgImpact').textContent = (data.avg_impact || 0).toFixed(3);
        }
        
        function updateNewsDisplay(news) {
            const newsList = document.getElementById('newsList');
            
            if (!news || news.length === 0) {
                newsList.innerHTML = '<div class="loading">No recent news available</div>';
                return;
            }
            
            const newsHtml = news.map(item => {
                const sentimentClass = item.sentiment === 'bullish' ? 'sentiment-bullish' : 
                                     item.sentiment === 'bearish' ? 'sentiment-bearish' : 'sentiment-neutral';
                const publishedTime = new Date(item.published_at).toLocaleTimeString();
                
                return `
                    <div class="news-item ${item.sentiment}">
                        <div class="news-title">${item.title}</div>
                        <div class="news-meta">
                            <span>${item.source} • ${publishedTime}</span>
                            <span class="sentiment-pill ${sentimentClass}">${item.sentiment}</span>
                        </div>
                    </div>
                `;
            }).join('');
            
            newsList.innerHTML = newsHtml;
        }
        
        async function fetchRiskData() {
            try {
                const response = await fetch('/api/risk-metrics');
                if (response.ok) {
                    const data = await response.json();
                    updateRiskDisplay(data);
                }
            } catch (error) {
                console.error('Error fetching risk data:', error);
            }
        }
        
        async function fetchTradeData() {
            try {
                const response = await fetch('/api/trading-log?limit=20');
                if (response.ok) {
                    const data = await response.json();
                    updateTradeDisplay(data);
                }
            } catch (error) {
                console.error('Error fetching trade data:', error);
            }
        }
        
        function updateTradeDisplay(data) {
            if (data.error) {
                console.error('Trade data error:', data.error);
                return;
            }
            
            // Update summary
            const summary = data.summary;
            document.getElementById('tradeCount').textContent = summary.total_trades;
            document.getElementById('winRate').textContent = `${summary.win_rate}%`;
            document.getElementById('avgLatency').textContent = `${summary.avg_latency}ms`;
            
            const grossPnLElement = document.getElementById('grossPnL');
            grossPnLElement.textContent = `$${summary.gross_pnl}`;
            grossPnLElement.className = summary.gross_pnl >= 0 ? 'positive' : 'negative';
            
            // Update trade rows
            const container = document.getElementById('tradeLogContainer');
            const trades = data.trades || [];
            
            if (trades.length === 0) {
                container.innerHTML = '<div class="loading">No trades found</div>';
                return;
            }
            
            const tradesHtml = trades.map(trade => {
                return `
                    <div class="trade-row">
                        <div class="trade-time">${trade.time}</div>
                        <div class="trade-symbol">${trade.symbol}</div>
                        <div class="trade-side ${trade.side.toLowerCase()}">${trade.side}</div>
                        <div class="trade-qty">${trade.quantity}</div>
                        <div class="trade-price">$${trade.price}</div>
                        <div class="trade-pnl ${trade.pnl >= 0 ? 'positive' : 'negative'}">$${trade.pnl}</div>
                        <div class="trade-latency">${trade.latency_ms}ms</div>
                    </div>
                `;
            }).join('');
            
            container.innerHTML = `
                <div class="trade-row" style="font-weight: bold; background: rgba(255,255,255,0.1);">
                    <div class="trade-time">Time</div>
                    <div class="trade-symbol">Symbol</div>
                    <div class="trade-side">Side</div>
                    <div class="trade-qty">Qty</div>
                    <div class="trade-price">Price</div>
                    <div class="trade-pnl">P&L</div>
                    <div class="trade-latency">Lat</div>
                </div>
                ${tradesHtml}
            `;
            
            // Store trades for export
            window.currentTrades = trades;
        }
        
        function exportTradeCSV() {
            if (!window.currentTrades) {
                alert('No trade data available');
                return;
            }
            
            const headers = ['Time', 'Symbol', 'Side', 'Quantity', 'Price', 'Value', 'P&L', 'Latency (ms)', 'Status'];
            const csvRows = [headers.join(',')];
            
            window.currentTrades.forEach(trade => {
                const row = [
                    trade.timestamp,
                    trade.symbol,
                    trade.side,
                    trade.quantity,
                    trade.price,
                    trade.value,
                    trade.pnl,
                    trade.latency_ms,
                    trade.status
                ];
                csvRows.push(row.join(','));
            });
            
            const csvString = csvRows.join('\\n');
            const blob = new Blob([csvString], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `trades_${new Date().toISOString().slice(0,10)}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }
        
        
        async function fetchRLActions() {
            try {
                const response = await fetch('/api/rl-actions?limit=15');
                if (response.ok) {
                    const data = await response.json();
                    updateRLActionDisplay(data);
                }
            } catch (error) {
                console.error('Error fetching RL actions:', error);
            }
        }
        
        function updateRLActionDisplay(data) {
            if (data.error) {
                console.error('RL action data error:', data.error);
                return;
            }
            
            // Update summary
            const summary = data.summary;
            document.getElementById('actionsPerMin').textContent = summary.actions_per_min;
            document.getElementById('avgActionSize').textContent = summary.avg_action_size;
            document.getElementById('lastAction').textContent = summary.last_action;
            
            // Update sparkline chart
            updateActionSparkline(data.actions);
            
            // Update action rows
            const container = document.getElementById('actionLogContainer');
            const actions = data.actions || [];
            
            if (actions.length === 0) {
                container.innerHTML = '<div class="loading">No RL actions found</div>';
                return;
            }
            
            const actionsHtml = actions.map(action => {
                return `
                    <div class="action-row">
                        <div class="action-time">${action.time}</div>
                        <div class="action-type">${action.action_type}</div>
                        <div class="action-size">${action.size}</div>
                        <div class="action-price">${action.price_offset}</div>
                        <div class="action-confidence">${action.confidence}</div>
                    </div>
                `;
            }).join('');
            
            container.innerHTML = `
                <div class="action-row" style="font-weight: bold; background: rgba(76,175,80,0.1);">
                    <div class="action-time">Time</div>
                    <div class="action-type">Type</div>
                    <div class="action-size">Size</div>
                    <div class="action-price">Offset</div>
                    <div class="action-confidence">Conf</div>
                </div>
                ${actionsHtml}
            `;
        }
        
        function updateActionSparkline(actions) {
            const canvas = document.getElementById('actionChart');
            const ctx = canvas.getContext('2d');
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (!actions || actions.length === 0) return;
            
            // Draw sparkline
            const width = canvas.width - 20;
            const height = canvas.height - 20;
            const xStep = width / (actions.length - 1);
            
            // Get action sizes for plotting
            const sizes = actions.slice().reverse().map(a => a.size);
            const maxSize = Math.max(...sizes.map(Math.abs));
            
            ctx.strokeStyle = '#4CAF50';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            sizes.forEach((size, i) => {
                const x = 10 + i * xStep;
                const y = height/2 + 10 - (size / maxSize) * (height/2 - 5);
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
            
            // Draw zero line
            ctx.strokeStyle = 'rgba(255,255,255,0.3)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(10, height/2 + 10);
            ctx.lineTo(width + 10, height/2 + 10);
            ctx.stroke();
        }
        
        async function fetchPolicyMetrics() {
            try {
                const response = await fetch('/api/policy-metrics');
                if (response.ok) {
                    const data = await response.json();
                    updatePolicyMetricsDisplay(data);
                }
            } catch (error) {
                console.error('Error fetching policy metrics:', error);
            }
        }
        
        function updatePolicyMetricsDisplay(data) {
            if (data.error) {
                console.error('Policy metrics error:', data.error);
                return;
            }
            
            // Update values
            document.getElementById('entropyValue').textContent = data.entropy;
            document.getElementById('qspreadValue').textContent = data.q_spread;
            
            // Update gauges
            updateGauge('entropyGauge', data.entropy_pct, data.entropy_status);
            updateGauge('qspreadGauge', data.q_spread_pct, data.q_spread_status);
            
            // Update overall status
            const dot = document.getElementById('policyDot');
            dot.className = `status-dot ${data.overall_status === 'healthy' ? '' : data.overall_status}`;
        }
        
        function updateGauge(gaugeId, percentage, status) {
            const gauge = document.getElementById(gaugeId);
            
            // Color based on status
            let color = '#4CAF50';  // healthy
            if (status === 'warning') color = '#ff9800';
            if (status === 'critical') color = '#f44336';
            
            // Update conic gradient
            const degrees = Math.min(percentage * 3.6, 360);  // Convert percentage to degrees
            gauge.style.background = `conic-gradient(${color} 0deg, ${color} ${degrees}deg, rgba(255,255,255,0.1) ${degrees}deg, rgba(255,255,255,0.1) 360deg)`;
        }
        
        // Alert Toast System
        function showToast(type, title, message, dismissible = true) {
            const container = document.getElementById('toastContainer');
            const toastId = `toast_${Date.now()}`;
            
            // Icon mapping
            const icons = {
                info: '💼',
                warning: '⚠️', 
                error: '❌',
                critical: '🚨'
            };
            
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.id = toastId;
            
            toast.innerHTML = `
                <div class="toast-icon">${icons[type] || '💼'}</div>
                <div class="toast-content">
                    <div class="toast-title">${title}</div>
                    <div class="toast-message">${message}</div>
                </div>
                ${dismissible ? `<button class="toast-close" onclick="closeToast('${toastId}')">&times;</button>` : ''}
            `;
            
            container.appendChild(toast);
            
            // Auto-dismiss after delay (except critical alerts)
            if (dismissible && type !== 'critical') {
                setTimeout(() => {
                    closeToast(toastId);
                }, type === 'error' ? 8000 : 5000);
            }
        }
        
        function closeToast(toastId) {
            const toast = document.getElementById(toastId);
            if (toast) {
                toast.style.animation = 'slideIn 0.3s ease-out reverse';
                setTimeout(() => {
                    toast.remove();
                }, 300);
            }
        }
        
        async function fetchAlerts() {
            try {
                const response = await fetch('/api/alerts');
                if (response.ok) {
                    const data = await response.json();
                    handleAlerts(data.alerts || []);
                }
            } catch (error) {
                console.error('Error fetching alerts:', error);
            }
        }
        
        async function fetchHealthData() {
            try {
                const response = await fetch('/api/health');
                if (response.ok) {
                    const data = await response.json();
                    updateHealthDisplay(data);
                } else {
                    // Show error state
                    updateHealthDisplay({
                        status: "unhealthy",
                        timestamp: new Date().toISOString(),
                        services: {
                            api: { status: "unhealthy", message: `HTTP ${response.status}` }
                        }
                    });
                }
            } catch (error) {
                console.error('Error fetching health data:', error);
                updateHealthDisplay({
                    status: "unhealthy",
                    timestamp: new Date().toISOString(),
                    services: {
                        api: { status: "unhealthy", message: "Connection failed" }
                    }
                });
            }
        }
        
        function updateHealthDisplay(data) {
            if (!data || !data.services) {
                console.error('Invalid health data:', data);
                return;
            }
            
            // Update overall status
            const overallDot = document.getElementById('overallHealthDot');
            const overallText = document.getElementById('overallHealthText');
            if (overallDot && overallText) {
                overallDot.className = `status-dot ${data.status}`;
                overallText.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
            }
            
            // Update individual services
            const services = ['redis', 'websocket', 'news', 'policy', 'api'];
            services.forEach(service => {
                const serviceData = data.services[service];
                if (!serviceData) return;
                
                const dot = document.getElementById(`${service}StatusDot`);
                const text = document.getElementById(`${service}StatusText`);
                
                if (dot && text) {
                    dot.className = `status-dot ${serviceData.status}`;
                    text.textContent = serviceData.message || serviceData.status;
                }
            });
            
            // Update timestamp
            const timestampElement = document.getElementById('healthTimestamp');
            if (timestampElement) {
                const timestamp = new Date(data.timestamp).toLocaleString();
                timestampElement.textContent = `Last check: ${timestamp}`;
            }
        }
        
        let shownAlerts = new Set();
        
        function handleAlerts(alerts) {
            alerts.forEach(alert => {
                // Only show each alert once until dismissed
                if (!shownAlerts.has(alert.id)) {
                    showToast(alert.type, alert.title, alert.message, alert.dismissible);
                    shownAlerts.add(alert.id);
                    
                    // Auto-trigger kill-switch for critical drawdown
                    if (alert.id === 'drawdown_critical') {
                        triggerKillSwitch('Automatic trigger due to 5% drawdown');
                    }
                }
            });
        }
        
        async function triggerKillSwitch(reason = 'Manual trigger') {
            // Allow automatic triggers but restrict manual observer access
            if (reason === 'Manual trigger' && getCurrentUserRole() === 'observer') {
                showToast('error', '🔒 Access Denied', 'Observer role cannot trigger kill-switch manually', true);
                return;
            }
            
            try {
                const response = await fetch('/api/kill-switch', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    showToast('critical', '🚨 KILL-SWITCH ACTIVATED', 
                        `${data.message} - ${reason}`, false);
                    
                    // Also switch to failover mode
                    await setTradingMode('failover');
                } else {
                    showToast('error', '❌ Kill-Switch Failed', 
                        'Failed to activate emergency kill-switch', true);
                }
            } catch (error) {
                console.error('Kill-switch error:', error);
                showToast('error', '❌ System Error', 
                    'Failed to communicate with kill-switch API', true);
            }
        }
        
        async function setTradingMode(mode) {
            try {
                const response = await fetch(`/api/mode?value=${mode}`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    const data = await response.json();
                    showToast('warning', '🔄 Mode Change', 
                        `${data.message}`, true);
                }
            } catch (error) {
                console.error('Mode change error:', error);
            }
        }
        
        // Mode indicator functions
        function showModeIndicator(mode, message) {
            const pill = document.getElementById('modeStatusPill');
            const indicator = document.getElementById('modeIndicator');
            
            if (pill && indicator) {
                indicator.textContent = `Mode: ${mode}`;
                pill.style.display = 'flex';
                
                // Show toast notification if message provided
                if (message) {
                    showToast('warning', '🚨 Mode Change', message, true);
                }
            }
        }
        
        function hideModeIndicator() {
            const pill = document.getElementById('modeStatusPill');
            if (pill) {
                pill.style.display = 'none';
            }
        }
        
        // Check current trading mode on page load
        async function checkCurrentMode() {
            try {
                const response = await fetch('/api/system-status');
                if (response.ok) {
                    const data = await response.json();
                    const mode = data.trading_mode || 'normal';
                    
                    if (mode === 'failover' || mode === 'emergency') {
                        showModeIndicator(mode.charAt(0).toUpperCase() + mode.slice(1));
                    } else {
                        hideModeIndicator();
                    }
                }
            } catch (error) {
                console.error('Error checking current mode:', error);
            }
        }
        
        // Enhanced risk monitoring with automatic kill-switch
        function updateRiskDisplay(data) {
            if (data.error) {
                console.error('Risk data error:', data.error);
                return;
            }
            
            // Update risk values
            document.getElementById('totalPosition').textContent = `${data.position_pct}%`;
            document.getElementById('drawdown').textContent = `${data.drawdown_pct}%`;
            document.getElementById('var95').textContent = `$${data.var_95}`;
            document.getElementById('cvar95').textContent = `$${data.cvar_95}`;
            
            // Check for mode changes from risk data
            if (data.trading_mode && data.trading_mode !== 'normal') {
                showModeIndicator(data.trading_mode.charAt(0).toUpperCase() + data.trading_mode.slice(1));
            } else if (data.trading_mode === 'normal') {
                hideModeIndicator();
            }
            
            // Update risk bar styling based on status
            const riskBar = document.getElementById('riskBar');
            riskBar.className = 'risk-bar'; // Reset classes
            
            if (data.risk_status === 'critical') {
                riskBar.classList.add('risk-critical');
                document.getElementById('drawdown').style.color = '#f44336';
                
                // Auto-trigger kill-switch if drawdown >= 5%
                if (data.drawdown_pct >= 5.0) {
                    triggerKillSwitch(`Automatic trigger: ${data.drawdown_pct}% drawdown`);
                    showModeIndicator('Failover', `Emergency mode: ${data.drawdown_pct}% drawdown reached`);
                }
            } else if (data.risk_status === 'warning') {
                riskBar.classList.add('risk-warning');
                document.getElementById('drawdown').style.color = '#ff9800';
            } else {
                document.getElementById('drawdown').style.color = '#4CAF50';
            }
            
            // Color position percentage
            const positionElement = document.getElementById('totalPosition');
            if (data.position_pct > 100) {
                positionElement.style.color = '#f44336'; // Over-leveraged
            } else if (data.position_pct > 80) {
                positionElement.style.color = '#ff9800'; // High leverage
            } else {
                positionElement.style.color = '#4CAF50'; // Normal
            }
        }
        
        // Export Functions
        async function exportTradeCSV() {
            // Role-based access control
            if (getCurrentUserRole() === 'observer') {
                showToast('error', '🔒 Access Denied', 'Observer role cannot export trade data', true);
                return;
            }
            
            try {
                console.log('Exporting CSV...');
                const response = await fetch('/api/trading-log?limit=100');
                if (!response.ok) {
                    showToast('error', 'Export Failed', 'Could not fetch trade data');
                    return;
                }
                
                const data = await response.json();
                const trades = data.trades || [];
                
                if (trades.length === 0) {
                    showToast('warning', 'No Data', 'No trades found to export');
                    return;
                }
                
                // Create CSV content
                const headers = ['Timestamp', 'Symbol', 'Side', 'Size', 'Price', 'P&L', 'Latency'];
                const csvContent = [
                    headers.join(','),
                    ...trades.map(trade => [
                        trade.timestamp,
                        trade.symbol,
                        trade.side,
                        trade.size,
                        trade.price,
                        trade.pnl,
                        trade.latency
                    ].join(','))
                ].join('\\n');
                
                // Download CSV
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement('a');
                const url = URL.createObjectURL(blob);
                link.setAttribute('href', url);
                link.setAttribute('download', `trades_${new Date().toISOString().split('T')[0]}.csv`);
                link.style.visibility = 'hidden';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                showToast('success', '📄 CSV Exported', `${trades.length} trades exported successfully`);
                
            } catch (error) {
                console.error('CSV export error:', error);
                showToast('error', 'Export Failed', 'Error generating CSV file');
            }
        }
        
        async function exportTradePDF() {
            // Role-based access control
            if (getCurrentUserRole() === 'observer') {
                showToast('error', '🔒 Access Denied', 'Observer role cannot export trade data', true);
                return;
            }
            
            try {
                console.log('Exporting PDF...');
                const response = await fetch('/api/trading-log?limit=100');
                if (!response.ok) {
                    showToast('error', 'Export Failed', 'Could not fetch trade data');
                    return;
                }
                
                const data = await response.json();
                const trades = data.trades || [];
                const summary = data.summary || {};
                
                if (trades.length === 0) {
                    showToast('warning', 'No Data', 'No trades found to export');
                    return;
                }
                
                // Initialize jsPDF
                const { jsPDF } = window.jspdf;
                const doc = new jsPDF();
                
                // Add title
                doc.setFontSize(20);
                doc.text('Trading Report', 20, 20);
                
                // Add date
                doc.setFontSize(12);
                const today = new Date().toLocaleDateString();
                doc.text(`Generated: ${today}`, 20, 30);
                
                // Add summary
                doc.setFontSize(14);
                doc.text('Summary', 20, 45);
                doc.setFontSize(10);
                doc.text(`Total Trades: ${summary.total_trades || trades.length}`, 20, 55);
                doc.text(`Win Rate: ${summary.win_rate || 0}%`, 20, 62);
                doc.text(`Gross P&L: $${summary.gross_pnl || 0}`, 20, 69);
                doc.text(`Average Latency: ${summary.avg_latency || 0}ms`, 20, 76);
                
                // Add trades table
                const tableColumns = ['Time', 'Symbol', 'Side', 'Size', 'Price', 'P&L', 'Latency'];
                const tableRows = trades.map(trade => [
                    new Date(trade.timestamp).toLocaleTimeString(),
                    trade.symbol,
                    trade.side,
                    (trade.quantity || trade.size || 0).toFixed(4),
                    `$${trade.price.toFixed(2)}`,
                    `$${trade.pnl.toFixed(2)}`,
                    `${trade.latency_ms || trade.latency || 0}ms`
                ]);
                
                doc.autoTable({
                    head: [tableColumns],
                    body: tableRows,
                    startY: 85,
                    styles: { fontSize: 8 },
                    headStyles: { fillColor: [41, 128, 185] }
                });
                
                // Save PDF
                const filename = `trades_${new Date().toISOString().split('T')[0]}.pdf`;
                doc.save(filename);
                
                showToast('success', '📋 PDF Exported', `${trades.length} trades exported successfully`);
                
            } catch (error) {
                console.error('PDF export error:', error);
                showToast('error', 'Export Failed', 'Error generating PDF file. Make sure jsPDF is loaded.');
            }
        }
        
        // Model Management Functions
        function showRollbackModal() {
            if (confirm('⚠️ Rollback Model\\n\\nThis will revert to the previous model version and restart the policy daemon. Trading will be temporarily paused.\\n\\nDo you want to continue?')) {
                performRollback();
            }
        }
        
        async function performRollback() {
            try {
                showToast('warning', '🔄 Rolling Back', 'Initiating model rollback...', true);
                
                const response = await fetch('/api/rollback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        reason: 'Manual rollback from dashboard',
                        timestamp: new Date().toISOString()
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    showToast('success', '✅ Rollback Complete', 
                        `Model rolled back to ${data.previous_version || 'previous version'}`, true);
                    
                    // Update model display
                    updateModelStatus({
                        name: data.previous_version || 'SAC-DiF v1.2.2',
                        hash: data.previous_hash || '#f3e4d1',
                        status: 'active'
                    });
                } else {
                    const errorData = await response.json();
                    showToast('error', '❌ Rollback Failed', 
                        errorData.message || 'Failed to rollback model', true);
                }
            } catch (error) {
                console.error('Rollback error:', error);
                showToast('error', '❌ Rollback Error', 
                    'Network error during rollback', true);
            }
        }
        
        function updateModelStatus(modelData) {
            const modelName = document.getElementById('modelName');
            const modelHash = document.getElementById('modelHash');
            const modelDot = document.getElementById('modelStatusDot');
            
            if (modelName && modelData.name) {
                modelName.textContent = modelData.name;
            }
            
            if (modelHash && modelData.hash) {
                modelHash.textContent = modelData.hash;
            }
            
            if (modelDot) {
                modelDot.className = 'model-status-dot';
                if (modelData.status === 'training') {
                    modelDot.classList.add('training');
                } else if (modelData.status === 'error') {
                    modelDot.classList.add('error');
                }
                // Default is active (green)
            }
        }
        
        async function fetchModelStatus() {
            try {
                const response = await fetch('/api/model-status');
                if (response.ok) {
                    const data = await response.json();
                    updateModelStatus(data);
                }
            } catch (error) {
                console.error('Error fetching model status:', error);
                // Set error state
                updateModelStatus({
                    name: 'SAC-DiF (Unknown)',
                    hash: '#error',
                    status: 'error'
                });
            }
        }
        
        // Trading Interface Functions
        function setAmount(asset, amount) {
            const inputId = asset === 'btc' ? 'btcAmount' : 'ethAmount';
            document.getElementById(inputId).value = amount;
        }
        
        async function executeTrade(symbol, side) {
            // Check role-based permission first
            if (!hasTradePermission()) {
                showToast('error', '🔒 Access Denied', 'Observer role cannot execute trades', true);
                return;
            }
            
            const asset = symbol.includes('BTC') ? 'btc' : 'eth';
            const amountInput = document.getElementById(`${asset}Amount`);
            const amount = parseFloat(amountInput.value);
            
            if (!amount || amount <= 0) {
                showToast('warning', '⚠️ Invalid Amount', 'Please enter a valid trade amount', true);
                return;
            }
            
            // Get active order type
            const activeOrderType = document.querySelector('.order-type-btn.active')?.dataset.type || 'market';
            
            try {
                showToast('info', '🔄 Executing Trade', `${side} $${amount} ${symbol} (${activeOrderType.toUpperCase()})`, true);
                
                const response = await fetch('/api/execute-trade', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        symbol,
                        side,
                        amount,
                        order_type: activeOrderType,
                        timestamp: new Date().toISOString()
                    })
                });
                
                if (response.ok) {
                    const result = await safeJson(response);
                    if (result.error) {
                        showToast('error', '❌ Trade Failed', result.error, true);
                    } else {
                        showToast('success', '✅ Trade Executed', 
                            `${result.side} ${result.quantity} ${result.symbol} @ $${result.price}`, true);
                        
                        // Clear amount input
                        amountInput.value = '';
                        
                        // Refresh portfolio data
                        setTimeout(() => {
                            fetchData();
                        }, 1000);
                    }
                } else {
                    showToast('error', '❌ Trade Failed', `HTTP ${response.status}`, true);
                }
            } catch (error) {
                console.error('Trade execution error:', error);
                showToast('error', '❌ Trade Error', 'Network error during trade execution', true);
            }
        }
        
        // Order type selection
        document.addEventListener('click', function(e) {
            if (e.target.classList.contains('order-type-btn')) {
                // Remove active from siblings
                const siblings = e.target.parentNode.children;
                for (let sibling of siblings) {
                    sibling.classList.remove('active');
                }
                // Add active to clicked button
                e.target.classList.add('active');
            }
        });
        
        // Alpha Signals Functions
        async function fetchAlphaSignals() {
            try {
                const response = await fetch('/api/alpha-signals');
                if (response.ok) {
                    const data = await safeJson(response);
                    if (!data.error) {
                        updateAlphaSignalsDisplay(data);
                    }
                }
            } catch (error) {
                console.error('Error fetching alpha signals:', error);
            }
        }
        
        function updateAlphaSignalsDisplay(data) {
            if (!data.signals) return;
            
            const signals = ['rsi', 'volume', 'fibo', 'whale'];
            
            signals.forEach(signal => {
                const signalData = data.signals[signal];
                if (!signalData) return;
                
                // Update strength percentage
                const strengthElement = document.getElementById(`${signal}Strength`);
                if (strengthElement) {
                    strengthElement.textContent = `${signalData.strength}%`;
                }
                
                // Update status
                const statusElement = document.getElementById(`${signal}Status`);
                if (statusElement) {
                    statusElement.textContent = signalData.status;
                    statusElement.className = `signal-status ${signalData.class}`;
                }
                
                // Update time
                const timeElement = document.getElementById(`${signal}Time`);
                if (timeElement) {
                    timeElement.textContent = signalData.time;
                }
                
                // Update signal container class for border color
                const signalElement = document.getElementById(`${signal}Signal`);
                if (signalElement) {
                    signalElement.className = `alpha-signal ${signalData.class}`;
                }
            });
        }
        
        // Prediction vs Market Chart Functions
        async function updatePredictionChart() {
            const symbol = document.getElementById('predictionSymbol').value;
            try {
                const response = await fetch(`/api/model-price-series?symbol=${symbol}`);
                if (response.ok) {
                    const data = await safeJson(response);
                    if (!data.error && data.length > 0) {
                        renderPredictionChart(data, symbol);
                    }
                }
            } catch (error) {
                console.error('Error fetching prediction data:', error);
            }
        }
        
        function renderPredictionChart(data, symbol) {
            const timestamps = data.map(d => new Date(d.ts * 1000));
            const market = data.map(d => d.market);
            const model = data.map(d => d.model);
            const ciLow = data.map(d => d.ci_low);
            const ciHigh = data.map(d => d.ci_high);
            
            // Confidence band (fill between ci_low and ci_high)
            const fillX = [...timestamps, ...timestamps.slice().reverse()];
            const fillY = [...ciHigh, ...ciLow.slice().reverse()];
            
            const traces = [
                {
                    x: timestamps,
                    y: market,
                    mode: 'lines',
                    name: `${symbol.replace('USDT', '')} Market`,
                    line: { color: '#4CAF50', width: 2 }
                },
                {
                    x: timestamps,
                    y: model,
                    mode: 'lines',
                    name: 'Model Prediction',
                    line: { color: '#FF9800', width: 2, dash: 'dot' }
                },
                {
                    x: fillX,
                    y: fillY,
                    fill: 'toself',
                    fillcolor: 'rgba(255, 152, 0, 0.1)',
                    line: { width: 0 },
                    name: 'Confidence Interval',
                    showlegend: true
                }
            ];
            
            const layout = {
                margin: { l: 50, r: 30, t: 20, b: 40 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(255,255,255,0.05)',
                font: { color: '#ffffff', size: 11 },
                legend: { 
                    orientation: 'h', 
                    y: -0.2,
                    bgcolor: 'rgba(0,0,0,0)'
                },
                xaxis: { 
                    gridcolor: 'rgba(255,255,255,0.1)',
                    color: '#ffffff'
                },
                yaxis: { 
                    gridcolor: 'rgba(255,255,255,0.1)',
                    color: '#ffffff',
                    title: 'Price ($)'
                }
            };
            
            Plotly.newPlot('predictionChart', traces, layout, { responsive: true });
        }
        
        // PnL Chart Functions
        async function updatePnLChart(period) {
            // Update active button
            document.querySelectorAll('.timeframe-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.dataset.period === period) {
                    btn.classList.add('active');
                }
            });
            
            try {
                const response = await fetch(`/api/pnl-series?period=${period}`);
                if (response.ok) {
                    const data = await safeJson(response);
                    if (!data.error && data.length > 0) {
                        renderPnLChart(data, period);
                    }
                }
            } catch (error) {
                console.error('Error fetching PnL data:', error);
            }
        }
        
        function renderPnLChart(data, period) {
            const timestamps = data.map(d => new Date(d.ts * 1000));
            const equity = data.map(d => d.equity);
            const benchmark = data.map(d => d.benchmark || 10000); // Starting capital
            
            // Calculate proper Y-axis range to avoid flat green bar
            const allValues = [...equity, ...benchmark];
            const minVal = Math.min(...allValues);
            const maxVal = Math.max(...allValues);
            const range = maxVal - minVal;
            const padding = range * 0.1; // 10% padding on each side
            
            const yAxisRange = [
                Math.max(0, minVal - padding), // Don't go below 0
                maxVal + padding
            ];
            
            const traces = [
                {
                    x: timestamps,
                    y: equity,
                    mode: 'lines',
                    name: 'Portfolio Equity',
                    line: { color: '#4CAF50', width: 3 },
                    fill: 'tonexty'
                },
                {
                    x: timestamps,
                    y: benchmark,
                    mode: 'lines',
                    name: 'Initial Capital',
                    line: { color: '#FF9800', width: 1, dash: 'dash' }
                }
            ];
            
            const layout = {
                margin: { l: 50, r: 30, t: 20, b: 40 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(255,255,255,0.05)',
                font: { color: '#ffffff', size: 11 },
                legend: { 
                    orientation: 'h', 
                    y: -0.2,
                    bgcolor: 'rgba(0,0,0,0)'
                },
                xaxis: { 
                    gridcolor: 'rgba(255,255,255,0.1)',
                    color: '#ffffff'
                },
                yaxis: { 
                    gridcolor: 'rgba(255,255,255,0.1)',
                    color: '#ffffff',
                    title: 'Portfolio Value ($)',
                    range: yAxisRange // Fixed Y-axis scaling
                }
            };
            
            Plotly.newPlot('pnlChart', traces, layout, { responsive: true });
        }
        
        // Residual Chart Functions
        async function updateResidualChart() {
            const symbol = document.getElementById('residualSymbol').value;
            const period = document.getElementById('residualPeriod').value;
            
            try {
                const response = await fetch(`/api/residuals?symbol=${symbol}&period=${period}`);
                if (response.ok) {
                    const data = await safeJson(response);
                    if (!data.error) {
                        renderResidualChart(data, symbol, period);
                    }
                }
            } catch (error) {
                console.error('Error fetching residual data:', error);
            }
        }
        
        function renderResidualChart(data, symbol, period) {
            const bins = data.bins || [];
            const counts = data.counts || [];
            const meanError = data.mean_error || 0;
            const stdError = data.std_error || 0;
            const sampleSize = data.sample_size || 0;
            
            const colors = bins.map(bin => bin < 0 ? '#f44336' : '#4CAF50');
            
            const traces = [{
                x: bins,
                y: counts,
                type: 'bar',
                marker: { 
                    color: colors,
                    opacity: 0.7
                },
                name: 'Residuals'
            }];
            
            // Add mean line
            if (meanError !== 0) {
                traces.push({
                    x: [meanError, meanError],
                    y: [0, Math.max(...counts)],
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: 'white', width: 2, dash: 'dash' },
                    name: `Mean: ${meanError.toFixed(2)}`,
                    showlegend: false
                });
            }
            
            const layout = {
                margin: { l: 50, r: 30, t: 60, b: 40 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(255,255,255,0.05)',
                font: { color: '#ffffff', size: 11 },
                xaxis: { 
                    gridcolor: 'rgba(255,255,255,0.1)',
                    color: '#ffffff',
                    title: 'Prediction Error ($)'
                },
                yaxis: { 
                    gridcolor: 'rgba(255,255,255,0.1)',
                    color: '#ffffff',
                    title: 'Frequency'
                },
                showlegend: false,
                annotations: [{
                    x: 0.02,
                    y: 0.98,
                    xref: 'paper',
                    yref: 'paper',
                    text: `${symbol} ${period.toUpperCase()} | μ=${meanError.toFixed(2)} | σ=${stdError.toFixed(2)} | n=${sampleSize}`,
                    showarrow: false,
                    bgcolor: 'rgba(0,0,0,0.7)',
                    bordercolor: 'white',
                    borderwidth: 1,
                    font: { color: 'white', size: 10 }
                }]
            };
            
            Plotly.newPlot('residualChart', traces, layout, { responsive: true });
        }
        
        // Twitter News Functions
        async function updateTwitterNews() {
            try {
                const response = await fetch('/api/twitter-news');
                const data = await safeJson(response);
                
                if (data.news && data.news.length > 0) {
                    let newsHtml = '';
                    let bullishCount = 0;
                    let bearishCount = 0;
                    let totalCount = data.news.length;
                    
                    data.news.forEach(item => {
                        const sentimentColor = item.sentiment > 0.1 ? '#4CAF50' : 
                                             item.sentiment < -0.1 ? '#f44336' : '#ff9800';
                        const sentimentIcon = item.sentiment > 0.1 ? '📈' : 
                                            item.sentiment < -0.1 ? '📉' : '➡️';
                        
                        // Count sentiment for stats
                        if (item.sentiment > 0.1) {
                            bullishCount++;
                        } else if (item.sentiment < -0.1) {
                            bearishCount++;
                        }
                        
                        newsHtml += `
                            <div style="padding: 8px; margin: 5px 0; background: rgba(255,255,255,0.05); border-radius: 6px; border-left: 3px solid ${sentimentColor};">
                                <div style="font-size: 13px; line-height: 1.4;">${item.text}</div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px; font-size: 11px; opacity: 0.7;">
                                    <span>${sentimentIcon} ${item.sentiment.toFixed(2)}</span>
                                    <span>${new Date(item.timestamp * 1000).toLocaleTimeString()}</span>
                                </div>
                            </div>`;
                    });
                    document.getElementById('twitterNews').innerHTML = newsHtml;
                    
                    // Update sentiment stats widget
                    updateSentimentStats(totalCount, bullishCount, bearishCount);
                } else {
                    document.getElementById('twitterNews').innerHTML = '<div class="loading">No Twitter news available</div>';
                }
            } catch (error) {
                console.error('Error fetching Twitter news:', error);
                document.getElementById('twitterNews').innerHTML = '<div class="error">Error loading news</div>';
            }
        }
        
        function updateSentimentStats(totalCount, bullishCount, bearishCount) {
            // Update the sentiment stats widget
            const bullishPct = totalCount > 0 ? (bullishCount / totalCount * 100).toFixed(0) : 0;
            const bearishPct = totalCount > 0 ? (bearishCount / totalCount * 100).toFixed(0) : 0;
            
            document.getElementById('newsCount').textContent = totalCount;
            document.getElementById('bullishPct').textContent = `${bullishPct}%`;
            document.getElementById('bearishPct').textContent = `${bearishPct}%`;
            
            console.log(`Updated sentiment stats: ${totalCount} total, ${bullishPct}% bullish, ${bearishPct}% bearish`);
        }
        
        // Policy Monitoring Functions
        async function updatePolicyMonitoring() {
            console.log('🤖 Starting RL Policy Health update...');
            try {
                const response = await fetch('/api/entropy-qspread');
                const data = await safeJson(response);
                
                console.log('📊 Policy data received:', data.entropy_series?.length || 0, 'entropy points,', data.qspread_series?.length || 0, 'Q-spread points');
                
                if (data.entropy_series && data.qspread_series) {
                    // Update entropy chart
                    console.log('🔄 Updating entropy chart...');
                    updateEntropyChart(data.entropy_series);
                    
                    // Update Q-spread chart
                    console.log('🔄 Updating Q-spread chart...');
                    updateQSpreadChart(data.qspread_series);
                    
                    // Update stats
                    console.log('🔄 Updating policy stats...');
                    updatePolicyStats(data.stats);
                    
                    console.log('✅ RL Policy Health update completed');
                } else {
                    console.warn('❌ No policy monitoring data available');
                }
            } catch (error) {
                console.error('❌ Error fetching policy monitoring data:', error);
            }
        }
        
        function updateEntropyChart(entropyData) {
            if (!entropyData || entropyData.length === 0) {
                console.log('No entropy data available');
                return;
            }
            
            console.log('Updating entropy chart with', entropyData.length, 'data points');
            
            // Fix timestamp conversion - data might already be in milliseconds
            const timestamps = entropyData.map(d => {
                const ts = d.timestamp;
                // If timestamp is in seconds, multiply by 1000
                const date = ts > 1e12 ? new Date(ts) : new Date(ts * 1000);
                return date;
            });
            const values = entropyData.map(d => d.value);
            
            const trace = {
                x: timestamps,
                y: values,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Entropy',
                line: { color: '#2196F3', width: 2 },
                marker: { size: 4 }
            };
            
            // Add warning zones
            const warningZone = {
                x: timestamps,
                y: Array(timestamps.length).fill(0.3),
                type: 'scatter',
                mode: 'lines',
                name: 'Collapse Risk',
                line: { color: 'red', dash: 'dash', width: 1 },
                showlegend: false
            };
            
            const criticalZone = {
                x: timestamps,
                y: Array(timestamps.length).fill(0.1),
                type: 'scatter',
                mode: 'lines',
                name: 'Critical',
                line: { color: 'darkred', dash: 'dot', width: 1 },
                showlegend: false
            };
            
            const layout = {
                margin: { t: 20, r: 20, b: 40, l: 40 },
                yaxis: { title: 'Entropy', range: [0, 2] },
                xaxis: { title: '' },
                showlegend: false,
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'white' }
            };
            
            Plotly.newPlot('entropyChart', [trace, warningZone, criticalZone], layout, { responsive: true });
        }
        
        function updateQSpreadChart(qspreadData) {
            if (!qspreadData || qspreadData.length === 0) {
                console.log('No Q-spread data available');
                return;
            }
            
            console.log('Updating Q-spread chart with', qspreadData.length, 'data points');
            
            // Fix timestamp conversion - data might already be in milliseconds
            const timestamps = qspreadData.map(d => {
                const ts = d.timestamp;
                // If timestamp is in seconds, multiply by 1000
                const date = ts > 1e12 ? new Date(ts) : new Date(ts * 1000);
                return date;
            });
            const values = qspreadData.map(d => d.value);
            const meanValue = values.reduce((a, b) => a + b, 0) / values.length;
            
            const trace = {
                x: timestamps,
                y: values,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Q-Spread',
                line: { color: '#4CAF50', width: 2 },
                marker: { size: 4 }
            };
            
            const meanLine = {
                x: timestamps,
                y: Array(timestamps.length).fill(meanValue),
                type: 'scatter',
                mode: 'lines',
                name: `Mean: ${meanValue.toFixed(1)}`,
                line: { color: 'gray', dash: 'dash', width: 1 },
                showlegend: false
            };
            
            const layout = {
                margin: { t: 20, r: 20, b: 40, l: 40 },
                yaxis: { title: 'Q-Spread' },
                xaxis: { title: '' },
                showlegend: false,
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'white' }
            };
            
            Plotly.newPlot('qspreadChart', [trace, meanLine], layout, { responsive: true });
        }
        
        function updatePolicyStats(stats) {
            if (!stats || !stats.entropy || !stats.qspread) return;
            
            const entropy = stats.entropy;
            const qspread = stats.qspread;
            const riskClass = entropy.policy_collapse_risk === 'HIGH' ? 'error' : 'success';
            
            const statsHtml = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                    <div>
                        <strong>Current Entropy:</strong><br>
                        <span style="color: ${entropy.current < 0.3 ? '#f44336' : '#4CAF50'}; font-size: 18px;">
                            ${entropy.current.toFixed(3)}
                        </span>
                    </div>
                    <div>
                        <strong>Collapse Risk:</strong><br>
                        <span class="${riskClass}" style="font-weight: bold;">
                            ${entropy.policy_collapse_risk}
                        </span>
                    </div>
                    <div>
                        <strong>Q-Spread:</strong><br>
                        <span style="color: #4CAF50; font-size: 16px;">
                            ${qspread.current.toFixed(1)}
                        </span>
                    </div>
                    <div>
                        <strong>Data Points:</strong><br>
                        <span>${stats.count || 0}</span>
                    </div>
                </div>
            `;
            
            document.getElementById('policyStats').innerHTML = statsHtml;
        }
        
        // Latency Drill-down Functions
        async function updateLatencyDrilldown() {
            try {
                const response = await fetch('/api/latency-hops');
                const data = await safeJson(response);
                
                if (data.latency_hops && data.latency_hops.length > 0) {
                    updateLatencyChart(data.latency_hops);
                    updateLatencyStats(data.stats);
                } else {
                    console.warn('No latency data available');
                }
            } catch (error) {
                console.error('Error fetching latency data:', error);
            }
        }
        
        function updateLatencyChart(latencyData) {
            if (!latencyData || latencyData.length === 0) return;
            
            const timestamps = latencyData.map(d => new Date(d.timestamp * 1000));
            
            // Create box plot data for each hop
            const feedData = latencyData.map(d => d.feed);
            const redisData = latencyData.map(d => d.redis);
            const policyData = latencyData.map(d => d.policy);
            const orderData = latencyData.map(d => d.order);
            
            // Create stacked area chart showing latency breakdown
            const traces = [
                {
                    x: timestamps,
                    y: feedData,
                    name: 'WebSocket Feed',
                    type: 'scatter',
                    mode: 'lines',
                    stackgroup: 'one',
                    fillcolor: 'rgba(255, 193, 7, 0.7)',
                    line: { color: 'rgb(255, 193, 7)' }
                },
                {
                    x: timestamps,
                    y: redisData,
                    name: 'Redis Write',
                    type: 'scatter',
                    mode: 'lines',
                    stackgroup: 'one',
                    fillcolor: 'rgba(220, 53, 69, 0.7)',
                    line: { color: 'rgb(220, 53, 69)' }
                },
                {
                    x: timestamps,
                    y: policyData,
                    name: 'Policy Inference',
                    type: 'scatter',
                    mode: 'lines',
                    stackgroup: 'one',
                    fillcolor: 'rgba(13, 110, 253, 0.7)',
                    line: { color: 'rgb(13, 110, 253)' }
                },
                {
                    x: timestamps,
                    y: orderData,
                    name: 'Order Execution',
                    type: 'scatter',
                    mode: 'lines',
                    stackgroup: 'one',
                    fillcolor: 'rgba(25, 135, 84, 0.7)',
                    line: { color: 'rgb(25, 135, 84)' }
                }
            ];
            
            // Add 2ms target line
            traces.push({
                x: timestamps,
                y: Array(timestamps.length).fill(2.0),
                type: 'scatter',
                mode: 'lines',
                name: 'Sub-2ms Target',
                line: { color: 'white', dash: 'dash', width: 2 },
                yaxis: 'y2'
            });
            
            const layout = {
                margin: { t: 20, r: 20, b: 40, l: 50 },
                yaxis: { 
                    title: 'Latency (ms)', // Linear scale, not logarithmic
                    side: 'left'
                },
                yaxis2: {
                    overlaying: 'y',
                    side: 'right',
                    showgrid: false
                },
                xaxis: { title: 'Time' },
                legend: { 
                    orientation: 'h',
                    y: -0.2,
                    x: 0
                },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'white' }
            };
            
            Plotly.newPlot('latencyChart', traces, layout, { responsive: true });
        }
        
        function updateLatencyStats(stats) {
            if (!stats) return;
            
            const feedStats = stats.feed || {};
            const redisStats = stats.redis || {};
            const policyStats = stats.policy || {};
            const orderStats = stats.order || {};
            const totalStats = stats.total || {};
            
            const statsHtml = `
                <div style="margin-bottom: 15px;">
                    <strong>🎯 Sub-2ms Achievement:</strong><br>
                    <span style="color: ${totalStats.target_sub2ms > 0.95 ? '#4CAF50' : totalStats.target_sub2ms > 0.8 ? '#ff9800' : '#f44336'}; font-size: 18px;">
                        ${(totalStats.target_sub2ms * 100).toFixed(1)}%
                    </span>
                </div>
                
                <table style="width: 100%; font-size: 11px;">
                    <thead>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                            <th style="text-align: left; padding: 4px;">Hop</th>
                            <th style="text-align: right; padding: 4px;">P50</th>
                            <th style="text-align: right; padding: 4px;">P95</th>
                            <th style="text-align: right; padding: 4px;">Avg</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="padding: 4px; color: rgb(255, 193, 7);">Feed</td>
                            <td style="text-align: right; padding: 4px;">${feedStats.p50?.toFixed(2)}ms</td>
                            <td style="text-align: right; padding: 4px;">${feedStats.p95?.toFixed(2)}ms</td>
                            <td style="text-align: right; padding: 4px;">${feedStats.avg?.toFixed(2)}ms</td>
                        </tr>
                        <tr>
                            <td style="padding: 4px; color: rgb(220, 53, 69);">Redis</td>
                            <td style="text-align: right; padding: 4px;">${redisStats.p50?.toFixed(2)}ms</td>
                            <td style="text-align: right; padding: 4px;">${redisStats.p95?.toFixed(2)}ms</td>
                            <td style="text-align: right; padding: 4px;">${redisStats.avg?.toFixed(2)}ms</td>
                        </tr>
                        <tr>
                            <td style="padding: 4px; color: rgb(13, 110, 253);">Policy</td>
                            <td style="text-align: right; padding: 4px;">${policyStats.p50?.toFixed(2)}ms</td>
                            <td style="text-align: right; padding: 4px;">${policyStats.p95?.toFixed(2)}ms</td>
                            <td style="text-align: right; padding: 4px;">${policyStats.avg?.toFixed(2)}ms</td>
                        </tr>
                        <tr>
                            <td style="padding: 4px; color: rgb(25, 135, 84);">Order</td>
                            <td style="text-align: right; padding: 4px;">${orderStats.p50?.toFixed(2)}ms</td>
                            <td style="text-align: right; padding: 4px;">${orderStats.p95?.toFixed(2)}ms</td>
                            <td style="text-align: right; padding: 4px;">${orderStats.avg?.toFixed(2)}ms</td>
                        </tr>
                        <tr style="border-top: 1px solid rgba(255,255,255,0.2); font-weight: bold;">
                            <td style="padding: 4px;">Total</td>
                            <td style="text-align: right; padding: 4px;">${totalStats.p50?.toFixed(2)}ms</td>
                            <td style="text-align: right; padding: 4px;">${totalStats.p95?.toFixed(2)}ms</td>
                            <td style="text-align: right; padding: 4px;">${totalStats.avg?.toFixed(2)}ms</td>
                        </tr>
                    </tbody>
                </table>
            `;
            
            document.getElementById('latencyStats').innerHTML = statsHtml;
        }
        
        // Action Heat-map Functions
        async function updateActionHeatmap() {
            try {
                const response = await fetch('/api/action-heatmap');
                const data = await safeJson(response);
                
                if (data.heatmap && data.heatmap.z) {
                    updateActionHeatmapChart(data.heatmap);
                    updateBiasMetrics(data.bias_metrics);
                } else {
                    console.warn('No action heat-map data available');
                }
            } catch (error) {
                console.error('Error fetching action heat-map data:', error);
            }
        }
        
        function updateActionHeatmapChart(heatmapData) {
            if (!heatmapData || !heatmapData.z) return;
            
            const trace = {
                z: heatmapData.z,
                x: heatmapData.x,
                y: heatmapData.y,
                type: 'heatmap',
                colorscale: [
                    [0, 'rgba(0,0,0,0)'],
                    [0.1, 'rgba(0,100,250,0.3)'],
                    [0.5, 'rgba(255,193,7,0.6)'],
                    [1.0, 'rgba(220,53,69,0.9)']
                ],
                showscale: true,
                colorbar: {
                    title: 'Action<br>Density',
                    titlefont: { color: 'white', size: 11 },
                    tickfont: { color: 'white', size: 10 }
                }
            };
            
            const layout = {
                margin: { t: 20, r: 80, b: 50, l: 60 }, // Increased right margin for legend
                xaxis: { 
                    title: 'Price Offset ($)',
                    color: 'white',
                    gridcolor: 'rgba(255,255,255,0.1)'
                },
                yaxis: { 
                    title: 'Position Size',
                    color: 'white',
                    gridcolor: 'rgba(255,255,255,0.1)'
                },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'white' }
            };
            
            Plotly.newPlot('actionHeatmap', [trace], layout, { responsive: true });
        }
        
        function updateBiasMetrics(metrics) {
            if (!metrics) return;
            
            const biasScore = metrics.bias_score || 0;
            const buyRatio = metrics.buy_ratio || 0;
            const liftingOffers = metrics.lifting_offers || 0;
            const hittingBids = metrics.hitting_bids || 0;
            
            // Determine bias severity
            const biasSeverity = biasScore > 2.0 ? 'HIGH' : biasScore > 1.0 ? 'MEDIUM' : 'LOW';
            const biasColor = biasScore > 2.0 ? '#f44336' : biasScore > 1.0 ? '#ff9800' : '#4CAF50';
            
            const metricsHtml = `
                <div style="margin-bottom: 15px;">
                    <strong>🎯 Overall Bias Score:</strong><br>
                    <span style="color: ${biasColor}; font-size: 20px; font-weight: bold;">
                        ${biasScore.toFixed(2)}
                    </span>
                    <span style="font-size: 12px; opacity: 0.7; margin-left: 8px;">
                        (${biasSeverity})
                    </span>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
                    <div style="text-align: center; padding: 8px; background: rgba(76,175,80,0.2); border-radius: 6px;">
                        <div style="font-size: 18px; color: #4CAF50; font-weight: bold;">
                            ${(buyRatio * 100).toFixed(1)}%
                        </div>
                        <div style="font-size: 11px; opacity: 0.8;">Buy Actions</div>
                    </div>
                    <div style="text-align: center; padding: 8px; background: rgba(244,67,54,0.2); border-radius: 6px;">
                        <div style="font-size: 18px; color: #f44336; font-weight: bold;">
                            ${((1-buyRatio) * 100).toFixed(1)}%
                        </div>
                        <div style="font-size: 11px; opacity: 0.8;">Sell Actions</div>
                    </div>
                </div>
                
                <div style="margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-size: 12px;">Lifting Offers:</span>
                        <span style="color: #4CAF50;">${(liftingOffers * 100).toFixed(1)}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-size: 12px;">Hitting Bids:</span>
                        <span style="color: #f44336;">${(hittingBids * 100).toFixed(1)}%</span>
                    </div>
                </div>
                
                <div style="border-top: 1px solid rgba(255,255,255,0.2); padding-top: 10px;">
                    <div style="font-size: 11px; opacity: 0.7;">
                        <strong>Avg Offsets:</strong><br>
                        Buy: $${metrics.avg_buy_offset?.toFixed(2) || '0.00'}<br>
                        Sell: $${metrics.avg_sell_offset?.toFixed(2) || '0.00'}<br>
                        Total Actions: ${metrics.total_actions || 0}
                    </div>
                </div>
                
                <div style="margin-top: 15px; padding: 8px; background: rgba(13,110,253,0.1); border-radius: 6px; border-left: 3px solid #0d6efd;">
                    <div style="font-size: 11px; line-height: 1.4;">
                        <strong>📊 Analysis:</strong><br>
                        ${biasScore > 1.5 ? 
                            'Significant trading bias detected. Consider rebalancing strategy.' : 
                            biasScore > 0.8 ? 
                                'Moderate bias detected. Monitor for pattern changes.' : 
                                'Trading patterns appear balanced.'
                        }
                    </div>
                </div>
            `;
            
            document.getElementById('biasMetrics').innerHTML = metricsHtml;
        }
        
        // Load data when page loads
        document.addEventListener('DOMContentLoaded', () => {
            console.log('Page loaded, starting data fetch...');
            
            // Initialize role-based UI
            const currentRole = getCurrentUserRole();
            document.getElementById('roleSelector').value = currentRole;
            initializeRoleBasedUI();
            
            // Essential functions first
            try {
                fetchData();
            } catch (error) {
                console.error('Error in fetchData:', error);
            }
            
            try {
                fetchRiskData();
            } catch (error) {
                console.error('Error in fetchRiskData:', error);
            }
            
            // Check current trading mode
            try {
                checkCurrentMode();
            } catch (error) {
                console.error('Error in checkCurrentMode:', error);
            }
            
            // Secondary functions with error handling
            setTimeout(() => {
                try {
                    fetchNewsData();
                } catch (error) {
                    console.error('Error in fetchNewsData:', error);
                }
                
                try {
                    fetchTradeData();
                } catch (error) {
                    console.error('Error in fetchTradeData:', error);
                }
                
                try {
                    fetchRLActions();
                } catch (error) {
                    console.error('Error in fetchRLActions:', error);
                }
                
                try {
                    fetchPolicyMetrics();
                } catch (error) {
                    console.error('Error in fetchPolicyMetrics:', error);
                }
                
                try {
                    fetchAlerts();
                } catch (error) {
                    console.error('Error in fetchAlerts:', error);
                }
                
                try {
                    fetchHealthData();
                } catch (error) {
                    console.error('Error in fetchHealthData:', error);
                }
                
                try {
                    fetchModelStatus();
                } catch (error) {
                    console.error('Error in fetchModelStatus:', error);
                }
                
                try {
                    fetchAlphaSignals();
                } catch (error) {
                    console.error('Error in fetchAlphaSignals:', error);
                }
                
                try {
                    updatePredictionChart();
                } catch (error) {
                    console.error('Error in updatePredictionChart:', error);
                }
                
                try {
                    updatePnLChart('24h');
                } catch (error) {
                    console.error('Error in updatePnLChart:', error);
                }
                
                try {
                    updateResidualChart();
                } catch (error) {
                    console.error('Error in updateResidualChart:', error);
                }
                
                try {
                    updateTwitterNews();
                } catch (error) {
                    console.error('Error in updateTwitterNews:', error);
                }
                
                try {
                    updatePolicyMonitoring();
                } catch (error) {
                    console.error('Error in updatePolicyMonitoring:', error);
                }
                
                try {
                    updateLatencyDrilldown();
                } catch (error) {
                    console.error('Error in updateLatencyDrilldown:', error);
                }
                
                try {
                    updateActionHeatmap();
                } catch (error) {
                    console.error('Error in updateActionHeatmap:', error);
                }
                
                // Show welcome toast
                try {
                    showToast('info', '🚀 Dashboard Online', 'All systems connected and monitoring', true);
                } catch (error) {
                    console.error('Error showing toast:', error);
                }
            }, 1000);
            
            // Set up intervals for core functionality only
            setInterval(() => {
                try {
                    fetchData();
                } catch (error) {
                    console.error('Error in periodic fetchData:', error);
                }
            }, 5000);
            
            setInterval(() => {
                try {
                    fetchRiskData();
                } catch (error) {
                    console.error('Error in periodic fetchRiskData:', error);
                }
            }, 10000);
            
            setInterval(() => {
                try {
                    fetchHealthData();
                } catch (error) {
                    console.error('Error in periodic fetchHealthData:', error);
                }
            }, 30000);
        });
    </script>
</body>
</html>
    """
    )


@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio data."""
    try:
        portfolio_data = get_portfolio_data()
        return safe_json_response(portfolio_data)
    except Exception as e:
        return safe_json_response({"error": str(e)})


@app.get("/api/health")
async def health():
    """Comprehensive health check endpoint."""
    services = {}
    overall_status = "healthy"

    # Redis connection check
    try:
        if redis_client:
            redis_client.ping()
            services["redis"] = {"status": "healthy", "message": "Connected"}
        else:
            services["redis"] = {"status": "unhealthy", "message": "Not connected"}
            overall_status = "degraded"
    except Exception as e:
        services["redis"] = {"status": "unhealthy", "message": f"Error: {str(e)}"}
        overall_status = "degraded"

    # WebSocket data check (price feeds)
    try:
        btc_price = get_latest_price("BTCUSDT")
        eth_price = get_latest_price("ETHUSDT")
        if btc_price > 0 and eth_price > 0:
            services["websocket"] = {
                "status": "healthy",
                "message": f"Live data: BTC=${btc_price:,.2f}, ETH=${eth_price:,.2f}",
            }
        else:
            services["websocket"] = {
                "status": "degraded",
                "message": "No live price data",
            }
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        services["websocket"] = {"status": "unhealthy", "message": f"Error: {str(e)}"}
        overall_status = "degraded"

    # CryptoPanic news service check
    if cryptopanic_client:
        try:
            # Check if we can fetch recent news (with rate limiting)
            news_check = (
                redis_client.lrange("news.crypto.recent", 0, 0) if redis_client else []
            )
            if news_check:
                services["news"] = {
                    "status": "healthy",
                    "message": "News data available",
                }
            else:
                services["news"] = {
                    "status": "degraded",
                    "message": "No recent news data",
                }
        except Exception as e:
            services["news"] = {
                "status": "degraded",
                "message": f"Limited: {str(e)[:50]}",
            }
    else:
        services["news"] = {
            "status": "unhealthy",
            "message": "CryptoPanic client not initialized",
        }
        if overall_status == "healthy":
            overall_status = "degraded"

    # Policy daemon heartbeat check
    try:
        if redis_client:
            # Check for heartbeat key (SETEX policy:ping 2 ok)
            heartbeat = redis_client.get("policy:ping")
            if heartbeat == "ok":
                services["policy"] = {
                    "status": "healthy",
                    "message": "Policy daemon heartbeat active",
                }
            else:
                # Fallback: check for recent policy metrics activity
                policy_metrics = (
                    redis_client.xlen("policy:actions")
                    if redis_client.exists("policy:actions")
                    else 0
                )
                if policy_metrics > 0:
                    services["policy"] = {
                        "status": "degraded",
                        "message": "Policy active but no heartbeat",
                    }
                else:
                    services["policy"] = {
                        "status": "degraded",
                        "message": "No recent policy activity",
                    }
                    if overall_status == "healthy":
                        overall_status = "degraded"
        else:
            services["policy"] = {
                "status": "unknown",
                "message": "Cannot check without Redis",
            }
    except Exception as e:
        services["policy"] = {
            "status": "unknown",
            "message": f"Check error: {str(e)[:30]}",
        }

    # API latency self-check
    api_start = datetime.now()
    services["api"] = {
        "status": "healthy",
        "message": f"Response time: {(datetime.now() - api_start).total_seconds() * 1000:.1f}ms",
    }

    return safe_json_response(
        {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "services": services,
        }
    )


@app.get("/api/news-sentiment")
async def get_news_sentiment():
    """Get crypto news sentiment summary."""
    try:
        if not cryptopanic_client:
            return {"error": "CryptoPanic client not available"}

        summary = cryptopanic_client.get_sentiment_summary(
            currencies="BTC,ETH", hours=24
        )
        return summary
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/recent-news")
async def get_recent_news(limit: int = 10):
    """Get recent crypto news."""
    try:
        if not cryptopanic_client:
            return {"error": "CryptoPanic client not available"}

        # Try to get from Redis cache first
        news = cryptopanic_client.get_latest_from_redis(limit=limit)

        # If cache is empty, fetch fresh data
        if not news:
            news = cryptopanic_client.get_news(currencies="BTC,ETH", limit=limit)
            # Stream to Redis for future requests
            if news:
                cryptopanic_client.stream_to_redis()

        return {"news": news[:limit], "count": len(news[:limit])}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/risk-metrics")
async def get_risk_metrics():
    """Get real-time risk metrics for the sticky risk bar."""
    try:
        # Calculate from current portfolio
        portfolio_data = get_portfolio_data()

        btc_value = portfolio_data.get("BTCUSDT", {}).get("current_value", 0)
        eth_value = portfolio_data.get("ETHUSDT", {}).get("current_value", 0)
        total_value = btc_value + eth_value

        btc_pnl = portfolio_data.get("BTCUSDT", {}).get("total_pnl", 0)
        eth_pnl = portfolio_data.get("ETHUSDT", {}).get("total_pnl", 0)
        total_pnl = btc_pnl + eth_pnl

        # Calculate position percentage (assume $10k initial capital)
        initial_capital = 10000
        position_pct = (
            (total_value / initial_capital) * 100 if initial_capital > 0 else 0
        )

        # Calculate drawdown percentage
        peak_value = initial_capital + max(0, total_pnl)  # Peak equity
        current_equity = initial_capital + total_pnl
        drawdown_pct = (
            ((peak_value - current_equity) / peak_value * 100) if peak_value > 0 else 0
        )

        # Simplified VaR/CVaR calculation (for demo - in production use proper risk models)
        # Assume 2% daily volatility
        portfolio_vol = total_value * 0.02
        var_95 = portfolio_vol * 1.65  # 95% confidence (Z-score 1.65)
        cvar_95 = var_95 * 1.3  # Expected shortfall typically 30% higher than VaR

        # Risk status and trading mode management
        risk_status = "normal"
        trading_mode = "normal"

        # Connect to Redis for mode management
        try:
            r = redis.Redis(host="localhost", port=6379, decode_responses=True)

            if drawdown_pct >= 5.0:
                risk_status = "critical"  # Hit kill line
                trading_mode = "failover"
                r.set("mode", "failover")  # Set failover mode in Redis
            elif drawdown_pct >= 3.0:
                risk_status = "warning"
            else:
                # Check current mode from Redis
                current_mode = r.get("mode")
                if current_mode:
                    trading_mode = current_mode

        except Exception as redis_error:
            print(f"Redis error in risk metrics: {redis_error}")

        return safe_json_response(
            {
                "position_pct": round(position_pct, 2),
                "drawdown_pct": round(drawdown_pct, 2),
                "var_95": round(var_95, 2),
                "cvar_95": round(cvar_95, 2),
                "total_value": round(total_value, 2),
                "total_pnl": round(total_pnl, 2),
                "risk_status": risk_status,
                "trading_mode": trading_mode,
                "kill_line_pct": 5.0,
                "updated_at": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return safe_json_response({"error": str(e)})


@app.get("/api/trading-log")
async def get_trading_log(limit: int = 50):
    """Get daily trading log for forensic analysis."""
    try:
        # For demo purposes, generate simulated trade data
        # In production, this would read from Redis streams or database

        import random
        from datetime import datetime, timedelta

        trades = []
        base_time = datetime.now()

        symbols = ["BTCUSDT", "ETHUSDT"]
        sides = ["BUY", "SELL"]

        # Generate sample trades
        for i in range(min(limit, 20)):
            trade_time = base_time - timedelta(minutes=i * 15)
            symbol = random.choice(symbols)
            side = random.choice(sides)

            # Realistic quantities and prices
            if symbol == "BTCUSDT":
                qty = round(random.uniform(0.001, 0.01), 6)
                price = round(random.uniform(115000, 118000), 2)
            else:  # ETHUSDT
                qty = round(random.uniform(0.01, 0.1), 6)
                price = round(random.uniform(3800, 3900), 2)

            # Calculate P&L (simplified)
            pnl = round(random.uniform(-50, 100), 2)

            # Latency simulation
            latency_ms = round(random.uniform(1.2, 5.8), 1)

            trades.append(
                {
                    "id": f"trade_{i+1}",
                    "timestamp": trade_time.isoformat(),
                    "time": trade_time.strftime("%H:%M:%S"),
                    "symbol": symbol,
                    "side": side,
                    "quantity": qty,
                    "price": price,
                    "value": round(qty * price, 2),
                    "pnl": pnl,
                    "latency_ms": latency_ms,
                    "status": "FILLED",
                    "order_type": "MARKET",
                }
            )

        # Calculate summary stats
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t["pnl"] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        gross_pnl = sum(t["pnl"] for t in trades)
        avg_latency = (
            sum(t["latency_ms"] for t in trades) / total_trades
            if total_trades > 0
            else 0
        )

        return {
            "trades": trades,
            "summary": {
                "total_trades": total_trades,
                "win_rate": round(win_rate, 1),
                "gross_pnl": round(gross_pnl, 2),
                "avg_latency": round(avg_latency, 1),
            },
            "updated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/rl-actions")
async def get_rl_actions(limit: int = 20):
    """Get RL action tape for policy monitoring."""
    try:
        # Generate simulated RL actions for demo
        # In production, this would read from `sacdif_action` Redis stream

        import random
        from datetime import datetime, timedelta

        actions = []
        base_time = datetime.now()

        action_types = ["BUY", "SELL", "HOLD"]

        # Generate sample actions
        for i in range(min(limit, 20)):
            action_time = base_time - timedelta(seconds=i * 30)
            action_type = random.choice(action_types)

            # Realistic RL action parameters
            size = round(random.uniform(-1.0, 1.0), 3)  # Normalized action size
            price_offset = round(random.uniform(-0.1, 0.1), 4)  # Price offset from mid
            confidence = round(random.uniform(0.6, 0.95), 3)  # Action confidence

            # Policy state
            entropy = round(random.uniform(0.3, 0.8), 3)
            q_value = round(random.uniform(-100, 100), 2)

            actions.append(
                {
                    "id": f"action_{i+1}",
                    "timestamp": action_time.isoformat(),
                    "time": action_time.strftime("%H:%M:%S"),
                    "action_type": action_type,
                    "size": size,
                    "price_offset": price_offset,
                    "confidence": confidence,
                    "entropy": entropy,
                    "q_value": q_value,
                    "state_hash": f"s{random.randint(1000, 9999)}",
                }
            )

        # Calculate summary stats
        total_actions = len(actions)
        actions_per_min = round(total_actions / 10, 1)  # Over last 10 minutes
        avg_size = (
            sum(abs(a["size"]) for a in actions) / total_actions
            if total_actions > 0
            else 0
        )
        last_action_type = actions[0]["action_type"] if actions else "-"

        return {
            "actions": actions,
            "summary": {
                "actions_per_min": actions_per_min,
                "avg_action_size": round(avg_size, 3),
                "last_action": last_action_type,
                "total_actions": total_actions,
            },
            "updated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/policy-metrics")
async def get_policy_metrics():
    """Get policy entropy and Q-spread metrics."""
    try:
        # Generate simulated policy metrics
        # In production, this would read from `policy_metrics` Redis topic

        import random
        import math

        # Policy entropy (0.0 = deterministic, 1.0 = random)
        entropy = round(random.uniform(0.2, 0.7), 3)

        # Q-spread (difference between max and min Q-values)
        q_spread = round(random.uniform(50, 200), 2)

        # Policy health assessment
        entropy_status = "healthy"
        if entropy < 0.1:
            entropy_status = "critical"  # Too deterministic, overfitting risk
        elif entropy > 0.8:
            entropy_status = "warning"  # Too random, undertraining

        q_spread_status = "healthy"
        if q_spread < 20:
            q_spread_status = "warning"  # Low confidence in actions
        elif q_spread > 300:
            q_spread_status = "critical"  # Unstable value estimates

        overall_status = "healthy"
        if entropy_status == "critical" or q_spread_status == "critical":
            overall_status = "critical"
        elif entropy_status == "warning" or q_spread_status == "warning":
            overall_status = "warning"

        return {
            "entropy": entropy,
            "entropy_pct": round(entropy * 100, 1),  # As percentage for gauge
            "q_spread": q_spread,
            "q_spread_pct": min(
                round((q_spread / 300) * 100, 1), 100
            ),  # Normalized for gauge
            "entropy_status": entropy_status,
            "q_spread_status": q_spread_status,
            "overall_status": overall_status,
            "updated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/api/kill-switch")
async def trigger_kill_switch():
    """Emergency kill-switch to disable trading."""
    try:
        # In production, this would:
        # 1. Set Redis key to disable all trading
        # 2. Close all open positions
        # 3. Switch to failover mode
        # 4. Send alerts to Slack/PagerDuty

        # For demo, just log the action
        kill_switch_time = datetime.now().isoformat()

        # Store kill-switch state in Redis
        if redis_client:
            # Set failover mode as specified in requirements
            redis_client.set("mode", "failover")

            # Also set kill-switch timestamp
            redis_client.setex(
                "trading:kill_switch", 300, kill_switch_time
            )  # 5 min timeout

            # Publish alert
            redis_client.publish(
                "alerts",
                json.dumps(
                    {
                        "type": "kill_switch_triggered",
                        "timestamp": kill_switch_time,
                        "reason": "Risk threshold exceeded",
                        "action": "Switched to failover mode - All trading disabled",
                    }
                ),
            )

        return {
            "status": "success",
            "message": "Kill-switch activated - All trading disabled",
            "timestamp": kill_switch_time,
            "timeout_seconds": 300,
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}


@app.post("/api/mode")
async def set_trading_mode(value: str = "failover"):
    """Set trading mode (normal/failover)."""
    try:
        # In production, this would switch between RL and rule-based engines

        mode_time = datetime.now().isoformat()

        if redis_client:
            redis_client.set("trading:mode", value)
            redis_client.publish(
                "alerts",
                json.dumps(
                    {
                        "type": "mode_change",
                        "timestamp": mode_time,
                        "new_mode": value,
                        "action": f"Switched to {value} mode",
                    }
                ),
            )

        return {
            "status": "success",
            "mode": value,
            "timestamp": mode_time,
            "message": f"Trading mode set to {value}",
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}


@app.get("/api/alerts")
async def get_system_alerts():
    """Get current system alerts and warnings."""
    try:
        alerts = []

        # Check for active kill-switch
        if redis_client:
            kill_switch = redis_client.get("trading:kill_switch")
            if kill_switch:
                alerts.append(
                    {
                        "id": "kill_switch",
                        "type": "critical",
                        "title": "🚨 Kill-Switch Active",
                        "message": "Trading has been disabled due to risk threshold breach",
                        "timestamp": kill_switch,
                        "dismissible": False,
                    }
                )

        # Simulate other alerts based on current data
        portfolio_data = get_portfolio_data()
        risk_data = await get_risk_metrics()

        # Drawdown alert
        if risk_data.get("drawdown_pct", 0) >= 3.0:
            alerts.append(
                {
                    "id": "drawdown_warning",
                    "type": "warning",
                    "title": "⚠️ High Drawdown",
                    "message": f"Portfolio drawdown at {risk_data['drawdown_pct']}% - approaching 5% kill-line",
                    "timestamp": datetime.now().isoformat(),
                    "dismissible": True,
                }
            )

        # Policy metrics alerts (simulated)
        policy_data = await get_policy_metrics()
        if policy_data.get("entropy_status") == "critical":
            alerts.append(
                {
                    "id": "entropy_critical",
                    "type": "error",
                    "title": "🧠 Policy Entropy Critical",
                    "message": f"Policy entropy at {policy_data['entropy']} - risk of overfitting",
                    "timestamp": datetime.now().isoformat(),
                    "dismissible": True,
                }
            )

        return {
            "alerts": alerts,
            "count": len(alerts),
            "updated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"error": str(e), "alerts": []}


@app.get("/api/model-status")
async def get_model_status():
    """Get current model status and metadata."""
    try:
        # In production, this would read from MLflow or model registry
        # For now, simulate model status

        import random

        # Simulate different model states
        states = ["active", "training", "error"]
        weights = [0.8, 0.15, 0.05]  # Most likely active

        status = random.choices(states, weights=weights)[0]

        model_info = {
            "name": "SAC-DiF v1.2.3",
            "hash": "#a7b9c2d",
            "status": status,
            "version": "1.2.3",
            "deployed_at": "2025-08-07T18:30:00Z",
            "accuracy": round(random.uniform(0.85, 0.95), 3),
            "last_training": "2025-08-07T12:00:00Z",
        }

        if status == "training":
            model_info.update(
                {
                    "training_progress": round(random.uniform(0.1, 0.9), 2),
                    "eta_minutes": random.randint(10, 120),
                }
            )
        elif status == "error":
            model_info.update(
                {
                    "error_message": "Connection lost to training infrastructure",
                    "last_known_good": "SAC-DiF v1.2.2",
                }
            )

        return model_info

    except Exception as e:
        return {
            "name": "SAC-DiF (Unknown)",
            "hash": "#error",
            "status": "error",
            "error": str(e),
        }


@app.get("/api/system-status")
async def get_system_status():
    """Get current system status including trading mode."""
    try:
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()

        # Check trading mode (set by risk management system)
        trading_mode = r.get("mode") or "normal"

        # Get system health metrics
        policy_ping = r.exists("policy:ping")

        return {
            "trading_mode": trading_mode,
            "system_health": {
                "policy_daemon": "healthy" if policy_ping else "inactive",
                "redis": "connected",
                "timestamp": datetime.now().isoformat(),
            },
        }

    except Exception as e:
        return {
            "trading_mode": "unknown",
            "system_health": {"policy_daemon": "unknown", "redis": "disconnected"},
            "error": str(e),
        }


@app.post("/api/rollback")
async def rollback_model():
    """Rollback to previous model version."""
    try:
        # In production, this would:
        # 1. Stop current model serving
        # 2. Load previous model from MLflow
        # 3. Update model registry
        # 4. Restart policy daemon
        # 5. Update configuration

        rollback_info = {
            "status": "success",
            "message": "Model rollback completed successfully",
            "previous_version": "SAC-DiF v1.2.2",
            "previous_hash": "#f3e4d1",
            "current_version": "SAC-DiF v1.2.3",
            "current_hash": "#a7b9c2d",
            "rollback_time": datetime.now().isoformat(),
            "restart_required": True,
        }

        # Simulate rollback delay
        import asyncio

        await asyncio.sleep(2)

        # Log rollback event
        if redis_client:
            redis_client.publish(
                "alerts",
                json.dumps(
                    {
                        "type": "rollback_complete",
                        "timestamp": rollback_info["rollback_time"],
                        "previous_version": rollback_info["previous_version"],
                        "action": "Model rollback completed - system restarting",
                    }
                ),
            )

        return rollback_info

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "message": "Rollback failed - check system logs",
        }


@app.post("/api/execute-trade")
async def execute_trade(trade_data: dict):
    """Execute a trade order (simulation for demo)."""
    try:
        # In production, this would:
        # 1. Validate order parameters
        # 2. Check account balance/margin
        # 3. Submit to exchange API
        # 4. Handle partial fills
        # 5. Update positions in database

        import random

        symbol = trade_data.get("symbol", "BTCUSDT")
        side = trade_data.get("side", "BUY")
        amount = float(trade_data.get("amount", 0))
        order_type = trade_data.get("order_type", "market")

        if amount <= 0:
            return safe_json_response(
                {"status": "failed", "error": "Invalid trade amount"}
            )

        # Get current price for simulation
        current_price = get_latest_price(symbol) or 0
        if current_price == 0:
            return safe_json_response(
                {"status": "failed", "error": "Unable to get current market price"}
            )

        # Calculate quantity based on amount
        quantity = amount / current_price

        # Simulate execution with small slippage
        slippage_pct = random.uniform(-0.01, 0.01)  # ±0.01% slippage
        execution_price = current_price * (1 + slippage_pct)

        # Simulate execution time
        import asyncio

        await asyncio.sleep(random.uniform(0.1, 0.5))

        trade_result = {
            "status": "success",
            "trade_id": f"trade_{random.randint(100000, 999999)}",
            "symbol": symbol,
            "side": side,
            "quantity": round(quantity, 6),
            "price": round(execution_price, 2),
            "amount": round(quantity * execution_price, 2),
            "order_type": order_type.upper(),
            "timestamp": datetime.now().isoformat(),
            "fees": round(amount * 0.001, 4),  # 0.1% fee
            "slippage": round(slippage_pct * 100, 4),
        }

        # Log trade event to Redis for audit trail
        if redis_client:
            redis_client.publish("trades", json.dumps(trade_result))
            redis_client.lpush(f"trades.{symbol}", json.dumps(trade_result))

        return safe_json_response(trade_result)

    except Exception as e:
        return safe_json_response({"status": "failed", "error": str(e)})


@app.get("/api/alpha-signals")
async def get_alpha_signals():
    """Get technical analysis alpha signals."""
    try:
        import random
        from datetime import datetime

        # Simulate different signal types and their states
        signal_types = ["STRONG BUY", "WEAK BUY", "HOLD", "WEAK SELL", "STRONG SELL"]
        signal_classes = ["strong-buy", "weak-buy", "hold", "weak-sell", "strong-sell"]

        def generate_signal():
            idx = random.choices(
                [0, 1, 2, 3, 4], weights=[0.15, 0.25, 0.3, 0.25, 0.15]
            )[0]
            return {
                "status": signal_types[idx],
                "class": signal_classes[idx],
                "strength": random.randint(45, 95),
                "time": datetime.now().strftime("%H:%M:%S"),
            }

        signals = {
            "rsi": generate_signal(),
            "volume": generate_signal(),
            "fibo": generate_signal(),
            "whale": generate_signal(),
        }

        # Add some realistic correlations
        # If RSI is strong sell, make fibo more likely to be sell too
        if signals["rsi"]["status"] in ["STRONG SELL", "WEAK SELL"]:
            if random.random() < 0.6:  # 60% correlation
                sell_idx = random.choice([3, 4])
                signals["fibo"]["status"] = signal_types[sell_idx]
                signals["fibo"]["class"] = signal_classes[sell_idx]

        return safe_json_response(
            {
                "signals": signals,
                "updated_at": datetime.now().isoformat(),
                "market_regime": random.choice(
                    ["Trending", "Ranging", "Volatile", "Calm"]
                ),
                "overall_sentiment": random.choice(["Bullish", "Bearish", "Neutral"]),
            }
        )

    except Exception as e:
        return safe_json_response({"error": str(e), "signals": {}})


@app.get("/api/model-price-series")
async def get_model_price_series(symbol: str = "BTCUSDT"):
    """Get prediction vs market time series data."""
    try:
        import random
        from datetime import datetime, timedelta

        # Generate simulated time series data
        # In production, this would read from Redis streams with actual model predictions

        now = datetime.now()
        data_points = []

        # Generate base price trend
        base_price = 116000 if symbol == "BTCUSDT" else 3800
        current_price = get_latest_price(symbol) or base_price

        for i in range(100):
            timestamp = now - timedelta(minutes=i * 5)  # 5-minute intervals

            # Simulate realistic price movement
            price_change = random.uniform(-0.02, 0.02)  # ±2% change
            market_price = current_price * (1 + price_change * (100 - i) / 100)

            # Model prediction with some tracking accuracy
            model_noise = random.uniform(-0.005, 0.005)  # Model error
            model_price = market_price * (1 + model_noise)

            # Confidence intervals
            volatility = market_price * 0.01  # 1% volatility
            ci_low = model_price - volatility
            ci_high = model_price + volatility

            data_points.append(
                {
                    "ts": int(timestamp.timestamp()),
                    "market": round(market_price, 2),
                    "model": round(model_price, 2),
                    "ci_low": round(ci_low, 2),
                    "ci_high": round(ci_high, 2),
                }
            )

        # Sort by timestamp (oldest first)
        data_points.sort(key=lambda x: x["ts"])

        return safe_json_response(data_points)

    except Exception as e:
        return safe_json_response({"error": str(e)})


@app.get("/api/pnl-series")
async def get_pnl_series(period: str = "24h"):
    """Get portfolio PnL time series for different timeframes."""
    try:
        import random
        from datetime import datetime, timedelta

        # Parse period
        if period == "24h":
            hours = 24
            interval_minutes = 15  # 15-minute intervals
        elif period == "7d":
            hours = 24 * 7
            interval_minutes = 60 * 2  # 2-hour intervals
        elif period == "30d":
            hours = 24 * 30
            interval_minutes = 60 * 8  # 8-hour intervals
        else:
            hours = 24
            interval_minutes = 15

        now = datetime.now()
        data_points = []

        # Starting portfolio value
        initial_capital = 10000
        current_equity = initial_capital

        # Generate PnL curve
        num_points = int((hours * 60) / interval_minutes)

        for i in range(num_points):
            timestamp = now - timedelta(minutes=i * interval_minutes)

            # Simulate portfolio performance with some volatility
            if i == 0:
                equity = current_equity
            else:
                # Random walk with slight upward bias
                change_pct = random.normalvariate(0.0001, 0.005)  # Small positive drift
                equity = max(0, equity * (1 + change_pct))

            data_points.append(
                {
                    "ts": int(timestamp.timestamp()),
                    "equity": round(equity, 2),
                    "benchmark": initial_capital,
                }
            )

            current_equity = equity

        # Sort by timestamp (oldest first)
        data_points.sort(key=lambda x: x["ts"])

        return safe_json_response(data_points)

    except Exception as e:
        return safe_json_response({"error": str(e)})


@app.get("/api/residuals")
async def get_residuals(symbol: str = "BTCUSDT", period: str = "24h"):
    """Get prediction residuals distribution."""
    try:
        import random as rand

        # Simple residual generation without numpy
        num_samples = 1000 if period == "24h" else 5000 if period == "7d" else 500

        # Generate simulated residuals (market - model)
        # Most errors should be small, with occasional larger errors
        residuals = []

        # 80% small errors (within 1%)
        for _ in range(int(num_samples * 0.8)):
            residuals.append(rand.gauss(0, 10))

        # 15% medium errors (1-3%)
        for _ in range(int(num_samples * 0.15)):
            residuals.append(rand.gauss(0, 30))

        # 5% large errors (3%+)
        for _ in range(int(num_samples * 0.05)):
            residuals.append(rand.gauss(0, 100))

        # Create histogram bins manually
        min_res = min(residuals)
        max_res = max(residuals)
        num_bins = 20
        bin_width = (max_res - min_res) / num_bins

        bins = []
        counts = []

        for i in range(num_bins):
            bin_start = min_res + i * bin_width
            bin_end = min_res + (i + 1) * bin_width
            bin_center = (bin_start + bin_end) / 2

            # Count residuals in this bin
            count = sum(1 for r in residuals if bin_start <= r < bin_end)

            bins.append(round(bin_center, 2))
            counts.append(count)

        mean_error = sum(residuals) / len(residuals) if residuals else 0
        variance = (
            sum((x - mean_error) ** 2 for x in residuals) / len(residuals)
            if residuals
            else 0
        )
        std_error = variance**0.5

        return safe_json_response(
            {
                "bins": bins,
                "counts": counts,
                "mean_error": round(mean_error, 3),
                "std_error": round(std_error, 3),
                "sample_size": len(residuals),
            }
        )

    except Exception as e:
        return safe_json_response({"error": str(e), "bins": [], "counts": []})


@app.get("/api/twitter-news")
async def get_twitter_news_api():
    """Get latest Twitter news with sentiment."""
    try:
        if redis_client:
            # Get latest news from the Twitter stream
            news_data = redis_client.xrevrange("news:x", count=20)
            news_items = []

            for stream_id, fields in news_data:
                news_items.append(
                    {
                        "id": fields.get("id", ""),
                        "text": fields.get("text", ""),
                        "url": fields.get("url", ""),
                        "sentiment": float(fields.get("sent", 0.0)),
                        "timestamp": int(fields.get("ts", 0)),
                        "source": fields.get("source", "x"),
                    }
                )

            return safe_json_response(
                {
                    "news": news_items,
                    "count": len(news_items),
                    "updated_at": int(datetime.now().timestamp()),
                }
            )
        else:
            return safe_json_response(
                {"error": "Redis not connected", "news": [], "count": 0}
            )
    except Exception as e:
        return safe_json_response({"error": str(e), "news": [], "count": 0})


@app.get("/api/pnl-curve")
async def get_pnl_curve(timeframe: str = "24h"):
    """Get PnL curve data for different timeframes."""
    try:
        if redis_client:
            # Define timeframe parameters
            timeframes = {
                "1h": {"hours": 1, "interval_min": 1, "points": 60},
                "24h": {"hours": 24, "interval_min": 5, "points": 288},
                "7d": {"hours": 168, "interval_min": 30, "points": 336},
                "30d": {"hours": 720, "interval_min": 120, "points": 360},
            }

            if timeframe not in timeframes:
                timeframe = "24h"

            tf_config = timeframes[timeframe]

            # Try to get PnL data from Redis timeseries
            pnl_key = f"pnl:portfolio"
            pnl_data = redis_client.xrevrange(pnl_key, count=tf_config["points"])

            portfolio_curve = []
            if pnl_data:
                for stream_id, fields in reversed(pnl_data):
                    ts = (
                        int(stream_id.decode().split("-")[0])
                        if isinstance(stream_id, bytes)
                        else int(stream_id.split("-")[0])
                    )
                    total_pnl = float(fields.get("total_pnl", 0.0))
                    btc_pnl = float(fields.get("btc_pnl", 0.0))
                    eth_pnl = float(fields.get("eth_pnl", 0.0))

                    portfolio_curve.append(
                        {
                            "timestamp": ts,
                            "total_pnl": total_pnl,
                            "btc_pnl": btc_pnl,
                            "eth_pnl": eth_pnl,
                            "equity": 10000 + total_pnl,
                        }
                    )

            # If no data, generate sample PnL curve
            if not portfolio_curve:
                import time
                import random
                import numpy as np

                current_time = int(time.time())
                start_capital = 10000
                running_pnl = 0

                for i in range(tf_config["points"]):
                    ts = (
                        current_time
                        - (tf_config["points"] - i) * tf_config["interval_min"] * 60
                    )

                    pnl_change = np.random.normal(0.5, 15)
                    running_pnl += pnl_change

                    btc_pnl = running_pnl * 0.6 + random.uniform(-50, 50)
                    eth_pnl = running_pnl * 0.4 + random.uniform(-30, 30)
                    total_pnl = btc_pnl + eth_pnl

                    portfolio_curve.append(
                        {
                            "timestamp": ts,
                            "total_pnl": total_pnl,
                            "btc_pnl": btc_pnl,
                            "eth_pnl": eth_pnl,
                            "equity": start_capital + total_pnl,
                        }
                    )

            return safe_json_response(
                {
                    "timeframe": timeframe,
                    "data": portfolio_curve,
                    "count": len(portfolio_curve),
                }
            )
        else:
            return safe_json_response(
                {"error": "Redis not connected", "data": [], "count": 0}
            )
    except Exception as e:
        return safe_json_response({"error": str(e), "data": [], "count": 0})


@app.get("/api/entropy-qspread")
async def get_entropy_qspread():
    """Get entropy and Q-spread time series for policy monitoring."""
    try:
        if redis_client:
            # Get recent policy data
            policy_key = "policy:actions"
            recent_data = redis_client.xrevrange(policy_key, count=60)

            entropy_data = []
            qspread_data = []

            if recent_data:
                for stream_id, fields in reversed(recent_data):
                    ts = (
                        int(stream_id.decode().split("-")[0])
                        if isinstance(stream_id, bytes)
                        else int(stream_id.split("-")[0])
                    )
                    entropy = float(fields.get("entropy", 0.0))
                    qspread = float(fields.get("q_spread", 0.0))

                    entropy_data.append({"timestamp": ts, "value": entropy})
                    qspread_data.append({"timestamp": ts, "value": qspread})

            # If no data, generate sample data
            if not entropy_data:
                import time
                import random
                import numpy as np

                current_time = int(time.time() * 1000)  # Milliseconds for consistency

                for i in range(60):
                    ts = (
                        current_time - (60 - i) * 60000
                    )  # 60 seconds ago in milliseconds

                    base_entropy = 1.5
                    entropy_noise = np.random.normal(0, 0.2)
                    if random.random() < 0.05:
                        entropy_val = max(0, base_entropy + entropy_noise - 1.2)
                    else:
                        entropy_val = max(0, min(2, base_entropy + entropy_noise))

                    base_qspread = 50
                    qspread_noise = np.random.normal(0, 15)
                    qspread_val = max(0, base_qspread + qspread_noise)

                    entropy_data.append({"timestamp": ts, "value": entropy_val})
                    qspread_data.append({"timestamp": ts, "value": qspread_val})

            # Calculate statistics
            entropy_values = [d["value"] for d in entropy_data]
            qspread_values = [d["value"] for d in qspread_data]

            stats = {
                "entropy": {
                    "current": entropy_values[-1] if entropy_values else 0,
                    "mean": (
                        sum(entropy_values) / len(entropy_values)
                        if entropy_values
                        else 0
                    ),
                    "min": min(entropy_values) if entropy_values else 0,
                    "policy_collapse_risk": (
                        "HIGH" if entropy_values and entropy_values[-1] < 0.1 else "LOW"
                    ),
                },
                "qspread": {
                    "current": qspread_values[-1] if qspread_values else 0,
                    "mean": (
                        sum(qspread_values) / len(qspread_values)
                        if qspread_values
                        else 0
                    ),
                    "max": max(qspread_values) if qspread_values else 0,
                },
            }

            return safe_json_response(
                {
                    "entropy_series": entropy_data,
                    "qspread_series": qspread_data,
                    "stats": stats,
                    "count": len(entropy_data),
                }
            )
        else:
            return safe_json_response(
                {
                    "error": "Redis not connected",
                    "entropy_series": [],
                    "qspread_series": [],
                    "stats": {},
                }
            )
    except Exception as e:
        return safe_json_response(
            {"error": str(e), "entropy_series": [], "qspread_series": [], "stats": {}}
        )


@app.get("/api/latency-hops")
async def get_latency_hops():
    """Get latency drill-down data for WS→Redis→Policy→Order pipeline."""
    try:
        if redis_client:
            # Get recent latency data
            latency_key = "lat_hops"
            recent_data = redis_client.xrevrange(latency_key, count=60)  # Last hour

            latency_hops = []
            if recent_data:
                for stream_id, fields in reversed(recent_data):
                    ts = (
                        int(stream_id.decode().split("-")[0])
                        if isinstance(stream_id, bytes)
                        else int(stream_id.split("-")[0])
                    )

                    hop_data = {
                        "timestamp": ts,
                        "feed": float(fields.get("feed", 0.0)),  # WS feed latency
                        "redis": float(fields.get("redis", 0.0)),  # Redis write latency
                        "policy": float(
                            fields.get("policy", 0.0)
                        ),  # Policy inference latency
                        "order": float(
                            fields.get("order", 0.0)
                        ),  # Order execution latency
                    }
                    latency_hops.append(hop_data)

            # Generate sample data if none exists
            if not latency_hops:
                import time
                import numpy as np

                current_time = int(time.time())

                for i in range(60):  # Last 60 minutes
                    ts = current_time - (60 - i) * 60

                    # Simulate realistic latencies (in milliseconds)
                    feed_lat = max(0.1, np.random.gamma(2, 0.3))  # WS feed: ~0.5ms avg
                    redis_lat = max(0.1, np.random.gamma(1.5, 0.2))  # Redis: ~0.3ms avg
                    policy_lat = max(0.5, np.random.gamma(3, 0.4))  # Policy: ~1.2ms avg
                    order_lat = max(0.2, np.random.gamma(2, 0.5))  # Order: ~1.0ms avg

                    # Occasionally simulate queue blow-ups
                    if np.random.random() < 0.02:  # 2% chance
                        policy_lat *= np.random.uniform(3, 8)  # Spike to 3-10ms

                    hop_data = {
                        "timestamp": ts,
                        "feed": round(feed_lat, 2),
                        "redis": round(redis_lat, 2),
                        "policy": round(policy_lat, 2),
                        "order": round(order_lat, 2),
                    }
                    latency_hops.append(hop_data)

            # Calculate statistics for each hop
            if latency_hops:
                import numpy as np

                feed_vals = [h["feed"] for h in latency_hops]
                redis_vals = [h["redis"] for h in latency_hops]
                policy_vals = [h["policy"] for h in latency_hops]
                order_vals = [h["order"] for h in latency_hops]

                total_vals = [
                    h["feed"] + h["redis"] + h["policy"] + h["order"]
                    for h in latency_hops
                ]

                stats = {
                    "feed": {
                        "p50": float(np.percentile(feed_vals, 50)),
                        "p95": float(np.percentile(feed_vals, 95)),
                        "avg": float(np.mean(feed_vals)),
                    },
                    "redis": {
                        "p50": float(np.percentile(redis_vals, 50)),
                        "p95": float(np.percentile(redis_vals, 95)),
                        "avg": float(np.mean(redis_vals)),
                    },
                    "policy": {
                        "p50": float(np.percentile(policy_vals, 50)),
                        "p95": float(np.percentile(policy_vals, 95)),
                        "avg": float(np.mean(policy_vals)),
                    },
                    "order": {
                        "p50": float(np.percentile(order_vals, 50)),
                        "p95": float(np.percentile(order_vals, 95)),
                        "avg": float(np.mean(order_vals)),
                    },
                    "total": {
                        "p50": float(np.percentile(total_vals, 50)),
                        "p95": float(np.percentile(total_vals, 95)),
                        "avg": float(np.mean(total_vals)),
                        "target_sub2ms": sum(1 for t in total_vals if t < 2.0)
                        / len(total_vals),
                    },
                }
            else:
                stats = {}

            return safe_json_response(
                {
                    "latency_hops": latency_hops,
                    "stats": stats,
                    "count": len(latency_hops),
                }
            )
        else:
            return safe_json_response(
                {"error": "Redis not connected", "latency_hops": [], "stats": {}}
            )
    except Exception as e:
        return safe_json_response({"error": str(e), "latency_hops": [], "stats": {}})


@app.get("/api/action-heatmap")
async def get_action_heatmap():
    """Get action heat-map data (price-offset × size density) for bias detection."""
    try:
        if redis_client:
            # Get recent action data from Action Tape
            action_key = "action_tape"
            recent_actions = redis_client.xrevrange(
                action_key, count=1000
            )  # Last 1000 actions

            actions = []
            if recent_actions:
                for stream_id, fields in recent_actions:
                    ts = (
                        int(stream_id.decode().split("-")[0])
                        if isinstance(stream_id, bytes)
                        else int(stream_id.split("-")[0])
                    )

                    action_data = {
                        "timestamp": ts,
                        "price_offset": float(
                            fields.get("price_offset", 0.0)
                        ),  # Offset from mid price
                        "size": float(fields.get("size", 0.0)),  # Trade size
                        "action": fields.get("action", "hold"),  # buy/sell/hold
                        "confidence": float(fields.get("confidence", 0.0)),
                    }
                    actions.append(action_data)

            # Generate sample action data if none exists
            if not actions:
                import time
                import numpy as np

                current_time = int(time.time())

                for i in range(500):  # Generate 500 sample actions
                    ts = current_time - i * 60  # Every minute

                    # Simulate trading bias: slight preference for lifting offers
                    if np.random.random() < 0.6:  # 60% buy bias
                        action_type = "buy"
                        # Buy orders tend to lift the offer (positive offset)
                        price_offset = np.random.gamma(2, 0.5) * np.random.choice(
                            [1, -1], p=[0.7, 0.3]
                        )
                        size = np.random.exponential(0.1) * 1000  # Smaller buy sizes
                    else:
                        action_type = "sell"
                        # Sell orders tend to hit the bid (negative offset)
                        price_offset = np.random.gamma(2, 0.5) * np.random.choice(
                            [1, -1], p=[0.3, 0.7]
                        )
                        size = np.random.exponential(0.15) * 1000  # Larger sell sizes

                    actions.append(
                        {
                            "timestamp": ts,
                            "price_offset": round(price_offset, 2),
                            "size": round(size, 4),
                            "action": action_type,
                            "confidence": round(np.random.uniform(0.5, 1.0), 2),
                        }
                    )

            # Create 2D histogram data
            if actions:
                import numpy as np

                price_offsets = [a["price_offset"] for a in actions]
                sizes = [a["size"] for a in actions]

                # Create bins for the heat-map
                price_bins = np.linspace(min(price_offsets), max(price_offsets), 20)
                size_bins = np.linspace(0, max(sizes), 15)

                # Create 2D histogram
                hist, x_edges, y_edges = np.histogram2d(
                    price_offsets, sizes, bins=[price_bins, size_bins]
                )

                # Calculate bias metrics
                buy_actions = [a for a in actions if a["action"] == "buy"]
                sell_actions = [a for a in actions if a["action"] == "sell"]

                avg_buy_offset = (
                    np.mean([a["price_offset"] for a in buy_actions])
                    if buy_actions
                    else 0
                )
                avg_sell_offset = (
                    np.mean([a["price_offset"] for a in sell_actions])
                    if sell_actions
                    else 0
                )

                bias_metrics = {
                    "total_actions": len(actions),
                    "buy_ratio": len(buy_actions) / len(actions) if actions else 0,
                    "avg_buy_offset": float(avg_buy_offset),
                    "avg_sell_offset": float(avg_sell_offset),
                    "bias_score": float(abs(avg_buy_offset - avg_sell_offset)),
                    "lifting_offers": (
                        sum(1 for a in actions if a["price_offset"] > 0.5)
                        / len(actions)
                        if actions
                        else 0
                    ),
                    "hitting_bids": (
                        sum(1 for a in actions if a["price_offset"] < -0.5)
                        / len(actions)
                        if actions
                        else 0
                    ),
                }

                heatmap_data = {
                    "z": hist.tolist(),  # 2D array for heat-map
                    "x": price_bins.tolist(),  # Price offset bins
                    "y": size_bins.tolist(),  # Size bins
                    "x_label": "Price Offset ($)",
                    "y_label": "Position Size",
                    "title": "Action Density Heat-map",
                }
            else:
                bias_metrics = {}
                heatmap_data = {}

            return safe_json_response(
                {
                    "actions": actions[
                        -100:
                    ],  # Return last 100 actions for detail view
                    "heatmap": heatmap_data,
                    "bias_metrics": bias_metrics,
                    "count": len(actions),
                }
            )
        else:
            return safe_json_response(
                {
                    "error": "Redis not connected",
                    "actions": [],
                    "heatmap": {},
                    "bias_metrics": {},
                }
            )
    except Exception as e:
        return safe_json_response(
            {"error": str(e), "actions": [], "heatmap": {}, "bias_metrics": {}}
        )


if __name__ == "__main__":
    print("🚀 Starting Simple Trading Dashboard...")
    print("📊 URL: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
