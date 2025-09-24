#!/usr/bin/env python3
"""
Real-time Trading Dashboard
A modern web interface for monitoring live trading data, P&L, and market metrics.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import redis
import logging
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import os
import requests
import hmac
import hashlib
from urllib.parse import urlencode

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)


@dataclass
class MarketTick:
    """Market tick data structure"""

    timestamp: float
    symbol: str
    price: float
    quantity: float


@dataclass
class PnLData:
    """P&L data structure"""

    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    current_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float


@dataclass
class AlphaSignal:
    """Alpha signal data structure"""

    symbol: str
    edge_bps: float
    confidence: float
    model_name: str
    timestamp: float


class TradingDataManager:
    """Manages real-time trading data from various sources"""

    def __init__(self):
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        self.current_prices = {}
        self.pnl_data = {}
        self.alpha_signals = {}
        self.model_running = True
        self.trading_allocation = 100  # Default: trade with 100% of assets
        self.trading_active = False  # Separate flag for trading control

        # Binance API credentials
        self.binance_api_key = os.getenv("BINANCE_API_KEY")
        self.binance_secret_key = os.getenv("BINANCE_SECRET_KEY")

        # Initialize portfolio with $100 each for BTC and ETH
        # Get current prices dynamically
        btc_current_price = self.get_latest_price("BTCUSDT") or 118586.99
        eth_current_price = self.get_latest_price("ETHUSDT") or 3402.81

        # $100 worth of each asset
        btc_position_size = 100.0 / btc_current_price
        eth_position_size = 100.0 / eth_current_price

        self.portfolio = {
            "BTCUSDT": {
                "position_size": btc_position_size,
                "entry_price": btc_current_price,
            },
            "ETHUSDT": {
                "position_size": eth_position_size,
                "entry_price": eth_current_price,
            },
        }

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from Redis"""
        try:
            tick_data = redis_client.lindex(f"market.raw.crypto.{symbol}", -1)
            if tick_data:
                tick = json.loads(tick_data)
                return float(tick["price"])
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
        return None

    def get_price_history(self, symbol: str, limit: int = 100) -> List[MarketTick]:
        """Get price history from Redis"""
        try:
            tick_data_list = redis_client.lrange(
                f"market.raw.crypto.{symbol}", -limit, -1
            )
            ticks = []
            for tick_data in tick_data_list:
                tick = json.loads(tick_data)
                ticks.append(
                    MarketTick(
                        timestamp=tick["ts"],
                        symbol=symbol,
                        price=float(tick["price"]),
                        quantity=float(tick["qty"]),
                    )
                )
            return ticks
        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {e}")
            return []

    def calculate_pnl(self, symbol: str) -> PnLData:
        """Calculate P&L for a symbol with current portfolio"""
        current_price = self.get_latest_price(symbol)
        if not current_price:
            current_price = 0.0

        # Get current position from portfolio
        position_data = self.portfolio.get(
            symbol, {"position_size": 0.0, "entry_price": 0.0}
        )
        position_size = position_data["position_size"]
        entry_price = position_data["entry_price"]

        # Calculate P&L and current value
        current_value = current_price * position_size
        unrealized_pnl = (current_price - entry_price) * position_size
        realized_pnl = 0.0  # No realized P&L for demo
        total_pnl = unrealized_pnl + realized_pnl

        return PnLData(
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            current_value=current_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_pnl=total_pnl,
        )

    def sell_position(
        self, symbol: str, percentage: float = None, amount: float = None
    ) -> Dict[str, Any]:
        """Sell position by percentage or amount"""
        if symbol not in self.portfolio:
            return {"success": False, "error": "No position found"}

        current_position = self.portfolio[symbol]["position_size"]
        current_price = self.get_latest_price(symbol)

        if not current_price:
            return {"success": False, "error": "Unable to get current price"}

        # Calculate sell amount
        if percentage is not None:
            sell_amount = current_position * (percentage / 100)
        elif amount is not None:
            sell_amount = min(amount, current_position)
        else:
            return {"success": False, "error": "Must specify percentage or amount"}

        if sell_amount <= 0:
            return {"success": False, "error": "Invalid sell amount"}

        # Update position
        new_position = current_position - sell_amount
        self.portfolio[symbol]["position_size"] = new_position

        # Calculate realized P&L
        entry_price = self.portfolio[symbol]["entry_price"]
        realized_pnl = (current_price - entry_price) * sell_amount

        logger.info(
            f"Sold {sell_amount:.6f} {symbol} at ${current_price:.2f} for ${realized_pnl:.2f} P&L"
        )

        return {
            "success": True,
            "symbol": symbol,
            "sold_amount": sell_amount,
            "remaining_position": new_position,
            "sell_price": current_price,
            "realized_pnl": realized_pnl,
            "percentage_sold": (
                percentage if percentage else (sell_amount / current_position * 100)
            ),
        }

    def buy_position(self, symbol: str, dollar_amount: float) -> Dict[str, Any]:
        """Buy/add position with dollar amount"""
        if dollar_amount <= 0:
            return {"success": False, "error": "Dollar amount must be positive"}

        current_price = self.get_latest_price(symbol)
        if not current_price:
            return {"success": False, "error": "Unable to get current price"}

        # Calculate how much crypto to buy
        crypto_amount = dollar_amount / current_price

        # Update position
        if symbol not in self.portfolio:
            self.portfolio[symbol] = {
                "position_size": 0.0,
                "entry_price": current_price,
            }

        current_position = self.portfolio[symbol]["position_size"]
        current_entry_price = self.portfolio[symbol]["entry_price"]

        # Calculate weighted average entry price
        total_value_before = current_position * current_entry_price
        total_value_after = total_value_before + dollar_amount
        new_position_size = current_position + crypto_amount

        if new_position_size > 0:
            new_entry_price = total_value_after / new_position_size
        else:
            new_entry_price = current_price

        # Update portfolio
        self.portfolio[symbol] = {
            "position_size": new_position_size,
            "entry_price": new_entry_price,
        }

        logger.info(
            f"Bought ${dollar_amount:.2f} worth of {symbol} ({crypto_amount:.6f}) at ${current_price:.2f}"
        )

        return {
            "success": True,
            "symbol": symbol,
            "dollar_amount": dollar_amount,
            "crypto_amount": crypto_amount,
            "buy_price": current_price,
            "new_position_size": new_position_size,
            "new_entry_price": new_entry_price,
            "previous_position": current_position,
        }

    def toggle_model(self) -> Dict[str, Any]:
        """Toggle model on/off"""
        self.model_running = not self.model_running
        logger.info(f"Model {'started' if self.model_running else 'stopped'}")
        return {
            "success": True,
            "model_running": self.model_running,
            "status": "started" if self.model_running else "stopped",
        }

    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        return {
            "model_running": self.model_running,
            "status": "running" if self.model_running else "stopped",
        }

    def reset_portfolio(self) -> Dict[str, Any]:
        """Reset portfolio to exactly $100 for each coin"""
        try:
            # Get current prices
            btc_current_price = self.get_latest_price("BTCUSDT")
            eth_current_price = self.get_latest_price("ETHUSDT")

            if not btc_current_price or not eth_current_price:
                return {"success": False, "error": "Unable to get current prices"}

            # Calculate position sizes for exactly $100 each
            btc_position_size = 100.0 / btc_current_price
            eth_position_size = 100.0 / eth_current_price

            # Reset portfolio
            self.portfolio = {
                "BTCUSDT": {
                    "position_size": btc_position_size,
                    "entry_price": btc_current_price,
                },
                "ETHUSDT": {
                    "position_size": eth_position_size,
                    "entry_price": eth_current_price,
                },
            }

            logger.info(
                f"Portfolio reset: BTC={btc_position_size:.6f} at ${btc_current_price:.2f}, ETH={eth_position_size:.6f} at ${eth_current_price:.2f}"
            )

            return {
                "success": True,
                "btc_position": btc_position_size,
                "btc_price": btc_current_price,
                "eth_position": eth_position_size,
                "eth_price": eth_current_price,
                "message": "Portfolio reset to $100 for each coin",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_total_portfolio_value(self) -> float:
        """Get total portfolio value in USD"""
        total_value = 0.0
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            pnl_data = self.calculate_pnl(symbol)
            total_value += pnl_data.current_value
        return total_value

    def set_trading_allocation(self, percentage: float) -> Dict[str, Any]:
        """Set trading allocation percentage"""
        if percentage < 5 or percentage > 100:
            return {
                "success": False,
                "error": "Trading allocation must be between 5% and 100%",
            }

        self.trading_allocation = percentage
        logger.info(f"Trading allocation set to {percentage}%")

        return {
            "success": True,
            "trading_allocation": percentage,
            "message": f"Trading allocation set to {percentage}%",
        }

    def get_trading_allocation(self) -> Dict[str, Any]:
        """Get current trading allocation"""
        total_value = self.get_total_portfolio_value()
        available_capital = total_value * (self.trading_allocation / 100)

        # Calculate total profit percentage
        total_invested = 0.0
        total_current_value = 0.0

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            if symbol in self.portfolio:
                position_data = self.portfolio[symbol]
                position_size = position_data["position_size"]
                entry_price = position_data["entry_price"]

                # Calculate how much was invested (position size * entry price)
                invested = position_size * entry_price
                total_invested += invested

                # Get current value
                pnl_data = self.calculate_pnl(symbol)
                total_current_value += pnl_data.current_value

        # Calculate profit percentage
        if total_invested > 0:
            profit_percentage = (
                (total_current_value - total_invested) / total_invested
            ) * 100
        else:
            profit_percentage = 0.0

        return {
            "trading_allocation": self.trading_allocation,
            "total_portfolio_value": total_value,
            "available_capital": available_capital,
            "total_invested": total_invested,
            "total_current_value": total_current_value,
            "profit_percentage": profit_percentage,
            "status": "active" if self.model_running else "inactive",
        }

    def get_alpha_signals(self, symbol: str) -> List[AlphaSignal]:
        """Get alpha signals (demo - you can integrate with your alpha models)"""
        current_price = self.get_latest_price(symbol)
        if not current_price:
            return []

        # Demo alpha signals - replace with real model outputs
        import random

        signals = []

        models = ["momentum", "mean_reversion", "orderbook_pressure", "ensemble"]
        for model in models:
            edge_bps = random.uniform(-20, 20)
            confidence = random.uniform(0.5, 0.95)

            signals.append(
                AlphaSignal(
                    symbol=symbol,
                    edge_bps=edge_bps,
                    confidence=confidence,
                    model_name=model,
                    timestamp=time.time(),
                )
            )

        return signals

    def _generate_binance_signature(self, query_string: str) -> str:
        """Generate Binance API signature"""
        return hmac.new(
            self.binance_secret_key.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def get_binance_account_info(self) -> Dict[str, Any]:
        """Get Binance account information"""
        if not self.binance_api_key:
            return {
                "success": False,
                "error": "Binance API key not configured",
                "instructions": "Add BINANCE_API_KEY to your .env file",
            }

        if not self.binance_secret_key:
            return {
                "success": False,
                "error": "Binance secret key not configured",
                "instructions": "Add BINANCE_SECRET_KEY to your .env file. You can get this from your Binance account settings.",
            }

        try:
            # Binance API endpoint
            url = "https://api.binance.com/api/v3/account"

            # Create query parameters
            timestamp = int(time.time() * 1000)
            params = {"timestamp": timestamp, "recvWindow": 5000}

            # Create query string
            query_string = urlencode(params)

            # Generate signature
            signature = self._generate_binance_signature(query_string)
            params["signature"] = signature

            # Make API request
            headers = {"X-MBX-APIKEY": self.binance_api_key}

            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Extract relevant balances
                balances = {}
                total_value = 0.0

                for balance in data.get("balances", []):
                    asset = balance["asset"]
                    free = float(balance["free"])
                    locked = float(balance["locked"])

                    if asset in ["BTC", "ETH", "USDT"] and (free > 0 or locked > 0):
                        balances[asset] = {
                            "free": f"{free:.8f}",
                            "locked": f"{locked:.8f}",
                        }

                        # Simple price estimation for total value
                        if asset == "BTC":
                            total_value += free * 120000  # Approximate BTC price
                        elif asset == "ETH":
                            total_value += free * 3600  # Approximate ETH price
                        elif asset == "USDT":
                            total_value += free

                # Ensure we have entries for all expected assets
                for asset in ["BTC", "ETH", "USDT"]:
                    if asset not in balances:
                        balances[asset] = {"free": "0.00000000", "locked": "0.00000000"}

                return {
                    "success": True,
                    "balances": balances,
                    "total_wallet_balance": f"{total_value:.2f}",
                    "message": "Real Binance account data",
                }
            else:
                error_msg = f"Binance API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('msg', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text}"

                return {
                    "success": False,
                    "error": error_msg,
                    "fallback_data": {
                        "balances": {
                            "BTC": {"free": "0.5", "locked": "0.0"},
                            "ETH": {"free": "5.0", "locked": "0.0"},
                            "USDT": {"free": "10000.0", "locked": "0.0"},
                        },
                        "total_wallet_balance": "50000.0",
                        "message": "Demo mode - API call failed",
                    },
                }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timeout - Binance API is slow",
                "fallback_data": {
                    "balances": {
                        "BTC": {"free": "0.5", "locked": "0.0"},
                        "ETH": {"free": "5.0", "locked": "0.0"},
                        "USDT": {"free": "10000.0", "locked": "0.0"},
                    },
                    "total_wallet_balance": "50000.0",
                    "message": "Demo mode - timeout",
                },
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback_data": {
                    "balances": {
                        "BTC": {"free": "0.5", "locked": "0.0"},
                        "ETH": {"free": "5.0", "locked": "0.0"},
                        "USDT": {"free": "10000.0", "locked": "0.0"},
                    },
                    "total_wallet_balance": "50000.0",
                    "message": "Demo mode - error occurred",
                },
            }

    def start_trading(self) -> Dict[str, Any]:
        """Start trading system"""
        try:
            self.trading_active = True
            self.model_running = True
            logger.info("Trading system started")
            return {
                "success": True,
                "message": "Trading system started successfully",
                "trading_active": self.trading_active,
                "model_running": self.model_running,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop_trading(self) -> Dict[str, Any]:
        """Stop trading system"""
        try:
            self.trading_active = False
            self.model_running = False
            logger.info("Trading system stopped")
            return {
                "success": True,
                "message": "Trading system stopped successfully",
                "trading_active": self.trading_active,
                "model_running": self.model_running,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        return {
            "trading_active": self.trading_active,
            "model_running": self.model_running,
            "trading_allocation": self.trading_allocation,
            "status": "active" if self.trading_active else "stopped",
        }

    def get_portfolio_performance(self) -> Dict[str, Any]:
        """Get portfolio performance data for charts"""
        try:
            # Calculate current portfolio value
            total_portfolio_value = 0.0
            total_market_value = 0.0

            for symbol, position in self.portfolio.items():
                current_price = self.get_latest_price(symbol) or position["entry_price"]
                position_value = position["position_size"] * current_price
                total_portfolio_value += position_value

                # Market value (if we had invested the same amount in index)
                entry_value = position["position_size"] * position["entry_price"]
                total_market_value += entry_value

            # Create time series data
            timestamps = []
            portfolio_values = []
            market_values = []

            # Get historical data from Redis (simplified)
            for i in range(100):  # Last 100 data points
                timestamp = time.time() - (100 - i) * 60  # 1-minute intervals
                timestamps.append(timestamp * 1000)  # Convert to milliseconds

                # Simulate portfolio vs market performance
                portfolio_values.append(total_portfolio_value * (1 + 0.001 * i))
                market_values.append(total_market_value * (1 + 0.0008 * i))

            return {
                "timestamps": timestamps,
                "portfolio_values": portfolio_values,
                "market_values": market_values,
                "current_portfolio_value": total_portfolio_value,
                "current_market_value": total_market_value,
            }
        except Exception as e:
            logger.error(f"Error getting portfolio performance: {e}")
            return {"error": str(e)}

    def get_pnl_performance(self) -> Dict[str, Any]:
        """Get P&L performance data for charts"""
        try:
            # Calculate P&L for each symbol
            total_pnl = 0.0
            pnl_history = []
            timestamps = []

            for symbol, position in self.portfolio.items():
                current_price = self.get_latest_price(symbol) or position["entry_price"]
                position_value = position["position_size"] * current_price
                entry_value = position["position_size"] * position["entry_price"]
                pnl = position_value - entry_value
                total_pnl += pnl

            # Create time series data
            for i in range(100):  # Last 100 data points
                timestamp = time.time() - (100 - i) * 60  # 1-minute intervals
                timestamps.append(timestamp * 1000)  # Convert to milliseconds

                # Simulate P&L progression
                pnl_history.append(total_pnl * (i / 100))

            return {
                "timestamps": timestamps,
                "pnl_values": pnl_history,
                "current_pnl": total_pnl,
                "total_return_pct": (total_pnl / 200)
                * 100,  # Assuming $200 initial investment
            }
        except Exception as e:
            logger.error(f"Error getting P&L performance: {e}")
            return {"error": str(e)}

    def get_position_performance(self) -> Dict[str, Any]:
        """Get position sizing data for charts"""
        try:
            # Calculate current position sizes
            position_sizes = []
            timestamps = []
            total_position_value = 0.0

            for symbol, position in self.portfolio.items():
                current_price = self.get_latest_price(symbol) or position["entry_price"]
                position_value = position["position_size"] * current_price
                total_position_value += position_value

            # Create time series data
            for i in range(100):  # Last 100 data points
                timestamp = time.time() - (100 - i) * 60  # 1-minute intervals
                timestamps.append(timestamp * 1000)  # Convert to milliseconds

                # Simulate position sizing changes
                position_sizes.append(total_position_value * (0.8 + 0.4 * (i / 100)))

            return {
                "timestamps": timestamps,
                "position_sizes": position_sizes,
                "current_position_size": total_position_value,
                "symbols": list(self.portfolio.keys()),
            }
        except Exception as e:
            logger.error(f"Error getting position performance: {e}")
            return {"error": str(e)}


# Global data manager
data_manager = TradingDataManager()

# FastAPI app
app = FastAPI(title="Real-time Trading Dashboard")


@app.get("/")
async def get_dashboard():
    """Serve the main dashboard HTML"""
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-time Trading Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            min-height: 100vh;
            padding: 20px;
        }

        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-rows: auto auto 1fr;
            gap: 20px;
            height: calc(100vh - 40px);
        }

        .header {
            text-align: center;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            position: relative;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        .status {
            display: inline-block;
            padding: 8px 16px;
            background: #4CAF50;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }

        .header-controls {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
        }

        .refresh-btn {
            background: #2196F3;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .refresh-btn:hover {
            background: #1976D2;
            transform: translateY(-2px);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .metric-card h3 {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #ffffff;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            padding-bottom: 10px;
        }

        .pnl-positive {
            color: #4CAF50;
        }

        .pnl-negative {
            color: #f44336;
        }

        .price-display {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }

        .chart-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            min-height: 350px;
        }

        .charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .trading-charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        
        .performance-chart {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .chart-title {
            color: white;
            text-align: center;
            margin-bottom: 10px;
            font-size: 18px;
            font-weight: bold;
        }

        .alpha-signals {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .alpha-signal {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }

        .confidence-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #f44336, #FF9800, #4CAF50);
            transition: width 0.3s ease;
        }

        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 0.9em;
        }

        .connected {
            background: #4CAF50;
            color: white;
        }

        .disconnected {
            background: #f44336;
            color: white;
        }

        .trade-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .trade-stat {
            text-align: center;
            padding: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }

        .trade-stat .value {
            font-size: 1.5em;
            font-weight: bold;
            display: block;
            margin-bottom: 5px;
        }

        .trade-stat .label {
            font-size: 0.9em;
            opacity: 0.8;
        }

        .trading-controls {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .control-section {
            margin-bottom: 15px;
        }

        .control-section h4 {
            margin-bottom: 10px;
            color: #ffffff;
            font-size: 1em;
        }

        .sell-controls, .buy-controls {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .sell-input, .buy-input {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .sell-input input, .buy-input input {
            flex: 1;
            padding: 8px 12px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            font-size: 0.9em;
        }

        .sell-input input::placeholder, .buy-input input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }

        .percentage-buttons {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 5px;
        }

        .percentage-btn, .control-btn {
            padding: 8px 12px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.8em;
            text-align: center;
        }

        .percentage-btn:hover, .control-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }

        .sell-btn {
            background: #ff4444;
            border-color: #ff4444;
        }

        .sell-btn:hover {
            background: #ff6666;
        }

        .buy-btn {
            background: #4CAF50;
            border-color: #4CAF50;
        }

        .buy-btn:hover {
            background: #66bb6a;
        }

        .model-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .model-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
        }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            animation: pulse 2s infinite;
        }

        .status-indicator.stopped {
            background: #f44336;
            animation: none;
        }

        .start-btn {
            background: #4CAF50;
            border-color: #4CAF50;
        }

        .start-btn:hover {
            background: #66bb6a;
        }

        .stop-btn {
            background: #f44336;
            border-color: #f44336;
        }

        .stop-btn:hover {
            background: #ef5350;
        }

        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.8; }
            100% { transform: scale(1); opacity: 1; }
        }

        .price-update {
            animation: pulse 0.5s ease-in-out;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <div class="header-controls">
                <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh</button>
                <button class="refresh-btn" onclick="clearPortfolio()">🗑️ Clear</button>
            </div>
            <h1>🚀 Real-time Trading Dashboard</h1>
            <div class="status" id="systemStatus">System Online</div>
            <div class="model-controls">
                <div class="model-status">
                    <div class="status-indicator" id="modelIndicator"></div>
                    <span id="modelStatus">Model Running</span>
                </div>
                <button class="control-btn stop-btn" id="toggleModelBtn" onclick="toggleModel()">Stop Model</button>
            </div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <h3>📊 BTCUSDT Position</h3>
                <div class="price-display" id="btcPrice">$0.00</div>
                <div class="trade-info">
                    <div class="trade-stat">
                        <span class="value" id="PositionSize">0.00</span>
                        <span class="label">Position</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value" id="EntryPrice">$0.00</span>
                        <span class="label">Entry</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value" id="CurrentValue">$0.00</span>
                        <span class="label">Current Value</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value pnl-positive" id="UnrealizedPnl">$0.00</span>
                        <span class="label">Unrealized P&L</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value" id="TotalPnl">$0.00</span>
                        <span class="label">Total P&L</span>
                    </div>
                </div>
                <div class="trading-controls">
                    <div class="control-section">
                        <h4>💰 Sell Position</h4>
                        <div class="sell-controls">
                            <div class="sell-input">
                                <input type="number" id="btcSellAmount" placeholder="Amount to sell" step="0.000001">
                                <button class="control-btn sell-btn" onclick="sellByAmount('BTCUSDT')">Sell</button>
                            </div>
                            <div class="percentage-buttons">
                                <div class="percentage-btn" onclick="sellByPercentage('BTCUSDT', 5)">5%</div>
                                <div class="percentage-btn" onclick="sellByPercentage('BTCUSDT', 10)">10%</div>
                                <div class="percentage-btn" onclick="sellByPercentage('BTCUSDT', 25)">25%</div>
                                <div class="percentage-btn" onclick="sellByPercentage('BTCUSDT', 50)">50%</div>
                                <div class="percentage-btn" onclick="sellByPercentage('BTCUSDT', 75)">75%</div>
                                <div class="percentage-btn" onclick="sellByPercentage('BTCUSDT', 100)">100%</div>
                            </div>
                        </div>
                    </div>
                    <div class="control-section">
                        <h4>🚀 Buy More Position</h4>
                        <div class="buy-controls">
                            <div class="buy-input">
                                <input type="number" id="btcBuyAmount" placeholder="Dollar amount to buy" step="1" min="1">
                                <button class="control-btn buy-btn" onclick="buyPosition('BTCUSDT')">Buy</button>
                            </div>
                            <div class="percentage-buttons">
                                <div class="percentage-btn" onclick="buyWithAmount('BTCUSDT', 10)">$10</div>
                                <div class="percentage-btn" onclick="buyWithAmount('BTCUSDT', 25)">$25</div>
                                <div class="percentage-btn" onclick="buyWithAmount('BTCUSDT', 50)">$50</div>
                                <div class="percentage-btn" onclick="buyWithAmount('BTCUSDT', 100)">$100</div>
                                <div class="percentage-btn" onclick="buyWithAmount('BTCUSDT', 200)">$200</div>
                                <div class="percentage-btn" onclick="buyWithAmount('BTCUSDT', 500)">$500</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="metric-card">
                <h3>📊 ETHUSDT Position</h3>
                <div class="price-display" id="ethPrice">$0.00</div>
                <div class="trade-info">
                    <div class="trade-stat">
                        <span class="value" id="ethPositionSize">0.00</span>
                        <span class="label">Position</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value" id="ethEntryPrice">$0.00</span>
                        <span class="label">Entry</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value" id="ethCurrentValue">$0.00</span>
                        <span class="label">Current Value</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value pnl-positive" id="ethUnrealizedPnl">$0.00</span>
                        <span class="label">Unrealized P&L</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value" id="ethTotalPnl">$0.00</span>
                        <span class="label">Total P&L</span>
                    </div>
                </div>
                <div class="trading-controls">
                    <div class="control-section">
                        <h4>💰 Sell Position</h4>
                        <div class="sell-controls">
                            <div class="sell-input">
                                <input type="number" id="ethSellAmount" placeholder="Amount to sell" step="0.000001">
                                <button class="control-btn sell-btn" onclick="sellByAmount('ETHUSDT')">Sell</button>
                            </div>
                            <div class="percentage-buttons">
                                <div class="percentage-btn" onclick="sellByPercentage('ETHUSDT', 5)">5%</div>
                                <div class="percentage-btn" onclick="sellByPercentage('ETHUSDT', 10)">10%</div>
                                <div class="percentage-btn" onclick="sellByPercentage('ETHUSDT', 25)">25%</div>
                                <div class="percentage-btn" onclick="sellByPercentage('ETHUSDT', 50)">50%</div>
                                <div class="percentage-btn" onclick="sellByPercentage('ETHUSDT', 75)">75%</div>
                                <div class="percentage-btn" onclick="sellByPercentage('ETHUSDT', 100)">100%</div>
                            </div>
                        </div>
                    </div>
                    <div class="control-section">
                        <h4>🚀 Buy More Position</h4>
                        <div class="buy-controls">
                            <div class="buy-input">
                                <input type="number" id="ethBuyAmount" placeholder="Dollar amount to buy" step="1" min="1">
                                <button class="control-btn buy-btn" onclick="buyPosition('ETHUSDT')">Buy</button>
                            </div>
                            <div class="percentage-buttons">
                                <div class="percentage-btn" onclick="buyWithAmount('ETHUSDT', 10)">$10</div>
                                <div class="percentage-btn" onclick="buyWithAmount('ETHUSDT', 25)">$25</div>
                                <div class="percentage-btn" onclick="buyWithAmount('ETHUSDT', 50)">$50</div>
                                <div class="percentage-btn" onclick="buyWithAmount('ETHUSDT', 100)">$100</div>
                                <div class="percentage-btn" onclick="buyWithAmount('ETHUSDT', 200)">$200</div>
                                <div class="percentage-btn" onclick="buyWithAmount('ETHUSDT', 500)">$500</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="metric-card">
                <h3>🧠 Alpha Signals</h3>
                <div class="alpha-signals" id="alphaSignals">
                    <!-- Alpha signals will be populated here -->
                </div>
            </div>

            <div class="metric-card">
                <h3>⚙️ Trading Settings</h3>
                <div class="trade-info">
                    <div class="trade-stat">
                        <span class="value" id="totalPortfolioValue">$0.00</span>
                        <span class="label">Total Portfolio</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value" id="totalInvested">$0.00</span>
                        <span class="label">Total Invested</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value pnl-positive" id="profitPercentage">0.00%</span>
                        <span class="label">Profit %</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value" id="availableCapital">$0.00</span>
                        <span class="label">Available Capital</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value" id="tradingAllocation">100%</span>
                        <span class="label">Trading Allocation</span>
                    </div>
                </div>
                <div class="trading-controls">
                    <div class="control-section">
                        <h4>📈 Trading Allocation</h4>
                        <div class="percentage-buttons">
                            <div class="percentage-btn" onclick="setTradingAllocation(5)">5%</div>
                            <div class="percentage-btn" onclick="setTradingAllocation(10)">10%</div>
                            <div class="percentage-btn" onclick="setTradingAllocation(25)">25%</div>
                            <div class="percentage-btn" onclick="setTradingAllocation(50)">50%</div>
                            <div class="percentage-btn" onclick="setTradingAllocation(75)">75%</div>
                            <div class="percentage-btn" onclick="setTradingAllocation(100)">100%</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>🏦 Binance Account</h3>
                <div class="trade-info" id="binanceAccountInfo">
                    <div class="trade-stat">
                        <span class="value" id="binanceBTC">0.00</span>
                        <span class="label">BTC Balance</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value" id="binanceETH">0.00</span>
                        <span class="label">ETH Balance</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value" id="binanceUSDT">$0.00</span>
                        <span class="label">USDT Balance</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value" id="binanceTotal">$0.00</span>
                        <span class="label">Total Balance</span>
                    </div>
                </div>
                <div class="trading-controls">
                    <div class="control-section">
                        <h4>🔄 Account Actions</h4>
                        <div class="percentage-buttons">
                            <button class="control-btn" onclick="refreshBinanceAccount()">Refresh Balance</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>🎯 Trading Control</h3>
                <div class="trade-info">
                    <div class="trade-stat">
                        <span class="value" id="tradingStatus">Stopped</span>
                        <span class="label">Trading Status</span>
                    </div>
                    <div class="trade-stat">
                        <span class="value" id="modelRunning">Stopped</span>
                        <span class="label">Model Status</span>
                    </div>
                </div>
                <div class="trading-controls">
                    <div class="control-section">
                        <h4>⚡ Trading Controls</h4>
                        <div class="percentage-buttons">
                            <button class="control-btn start-btn" id="startTradingBtn" onclick="startTrading()">Start Trading</button>
                            <button class="control-btn stop-btn" id="stopTradingBtn" onclick="stopTrading()">Stop Trading</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <h3>₿ BTCUSDT Real-time Chart</h3>
                <div id="btcChart" style="width: 100%; height: 300px;"></div>
            </div>
            <div class="chart-container">
                <h3>Ξ ETHUSDT Real-time Chart</h3>
                <div id="ethChart" style="width: 100%; height: 300px;"></div>
            </div>
        </div>
        
        <!-- Trading Performance Charts -->
        <div class="trading-charts-grid">
            <div class="performance-chart">
                <div class="chart-title">📊 Portfolio Value vs Market Performance</div>
                <div id="portfolioChart" style="width: 100%; height: 350px;"></div>
            </div>
            <div class="performance-chart">
                <div class="chart-title">💰 Real-time P&L Performance</div>
                <div id="pnlChart" style="width: 100%; height: 350px;"></div>
            </div>
        </div>
        
        <!-- Position Sizing Chart -->
        <div style="margin-top: 20px;">
            <div class="performance-chart">
                <div class="chart-title">🎯 Position Sizing & Risk Metrics</div>
                <div id="positionChart" style="width: 100%; height: 300px;"></div>
            </div>
        </div>
        
        <!-- Long-term Performance Charts -->
        <div style="margin-top: 30px;">
            <h2 style="text-align: center; color: white; margin-bottom: 20px;">📊 Long-term Performance Analysis</h2>
            
            <div class="trading-charts-grid">
                <div class="performance-chart">
                    <div class="chart-title">⏱️ Hourly Performance (Last 24 Hours)</div>
                    <div id="hourlyChart" style="width: 100%; height: 350px;"></div>
                </div>
                <div class="performance-chart">
                    <div class="chart-title">📅 5-Hour Performance (Last 5 Days)</div>
                    <div id="fiveHourChart" style="width: 100%; height: 350px;"></div>
                </div>
            </div>
            
            <div style="margin-top: 20px;">
                <div class="performance-chart">
                    <div class="chart-title">📈 Daily Performance (Last 30 Days)</div>
                    <div id="dailyChart" style="width: 100%; height: 350px;"></div>
                </div>
            </div>
        </div>
    </div>

    <div class="connection-status connected" id="connectionStatus">🟢 Connected</div>

    <script>
        class TradingDashboard {
            constructor() {
                this.ws = null;
                this.priceData = {
                    BTCUSDT: {x: [], y: []},
                    ETHUSDT: {x: [], y: []}
                };
                this.maxDataPoints = 100;
                
                // Trading performance data
                this.portfolioData = {
                    timestamps: [],
                    portfolioValue: [],
                    marketValue: [],
                    pnl: [],
                    positions: [],
                    riskMetrics: []
                };
                
                // Long-term performance data
                this.longTermData = {
                    hourly: { timestamps: [], portfolioValue: [], pnl: [] },
                    fiveHour: { timestamps: [], portfolioValue: [], pnl: [] },
                    daily: { timestamps: [], portfolioValue: [], pnl: [] }
                };
                
                // Real-time update interval
                this.updateInterval = null;
                
                this.initWebSocket();
                this.initCharts();
                this.startRealTimeUpdates();
            }

            initWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                this.ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.updateConnectionStatus(true);
                };
                
                this.ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.handleMessage(data);
                    } catch (error) {
                        console.error('Error parsing WebSocket message:', error);
                    }
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.updateConnectionStatus(false);
                    setTimeout(() => this.initWebSocket(), 5000);
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus(false);
                };
            }
            
            startRealTimeUpdates() {
                // Load initial data
                this.loadInitialData();
                
                // Start real-time updates every second
                this.updateInterval = setInterval(() => {
                    this.updateAllCharts();
                    this.fetchLatestData();
                }, 1000);
            }
            
            stopRealTimeUpdates() {
                if (this.updateInterval) {
                    clearInterval(this.updateInterval);
                    this.updateInterval = null;
                }
            }
            
            async loadInitialData() {
                // Load initial price data and update display
                try {
                    const [btcResponse, ethResponse] = await Promise.all([
                        fetch('/api/data/BTCUSDT'),
                        fetch('/api/data/ETHUSDT')
                    ]);
                    
                    if (btcResponse.ok && ethResponse.ok) {
                        const btcData = await btcResponse.json();
                        const ethData = await ethResponse.json();
                        
                        // Update price displays
                        this.updatePriceDisplay('BTC', btcData.current_price);
                        this.updatePriceDisplay('ETH', ethData.current_price);
                        
                        // Update price history for charts
                        this.updatePriceHistory('BTCUSDT', btcData.price_history);
                        this.updatePriceHistory('ETHUSDT', ethData.price_history);
                        
                        // Update position data
                        this.updatePositionData('BTCUSDT', btcData.pnl_data);
                        this.updatePositionData('ETHUSDT', ethData.pnl_data);
                    }
                } catch (error) {
                    console.error('Error loading initial data:', error);
                }
            }
            
            async fetchLatestData() {
                // Fetch latest price data every second
                try {
                    const [btcResponse, ethResponse] = await Promise.all([
                        fetch('/api/data/BTCUSDT'),
                        fetch('/api/data/ETHUSDT')
                    ]);
                    
                    if (btcResponse.ok && ethResponse.ok) {
                        const btcData = await btcResponse.json();
                        const ethData = await ethResponse.json();
                        
                        // Update price displays
                        this.updatePriceDisplay('BTC', btcData.current_price);
                        this.updatePriceDisplay('ETH', ethData.current_price);
                        
                        // Add new price data to charts
                        this.addPriceData('BTCUSDT', btcData.current_price);
                        this.addPriceData('ETHUSDT', ethData.current_price);
                        
                        // Update position data
                        this.updatePositionData('BTCUSDT', btcData.pnl_data);
                        this.updatePositionData('ETHUSDT', ethData.pnl_data);
                    }
                } catch (error) {
                    console.error('Error fetching latest data:', error);
                }
            }
            
            updatePriceDisplay(symbol, price) {
                // Update price displays for BTC and ETH
                if (symbol === 'BTC') {
                    const btcPriceElement = document.getElementById('btcPrice');
                    if (btcPriceElement) {
                        btcPriceElement.textContent = `$${price.toFixed(2)}`;
                    }
                } else if (symbol === 'ETH') {
                    const ethPriceElement = document.getElementById('ethPrice');
                    if (ethPriceElement) {
                        ethPriceElement.textContent = `$${price.toFixed(2)}`;
                    }
                }
            }
            
            updatePriceHistory(symbol, priceHistory) {
                if (priceHistory && priceHistory.length > 0) {
                    const times = [];
                    const prices = [];
                    
                    // Get last 50 price points
                    const recentHistory = priceHistory.slice(-50);
                    
                    recentHistory.forEach(item => {
                        times.push(new Date(item.timestamp));
                        prices.push(item.price);
                    });
                    
                    this.priceData[symbol] = { x: times, y: prices };
                    
                    // Refresh the chart
                    this.refreshChart(symbol);
                }
            }
            
            addPriceData(symbol, price) {
                const now = new Date();
                
                // Add new data point
                this.priceData[symbol].x.push(now);
                this.priceData[symbol].y.push(price);
                
                // Keep only last 100 data points
                if (this.priceData[symbol].x.length > 100) {
                    this.priceData[symbol].x.shift();
                    this.priceData[symbol].y.shift();
                }
                
                // Refresh the chart
                this.refreshChart(symbol);
            }
            
            updatePositionData(symbol, pnlData) {
                if (!pnlData) return;
                
                if (symbol === 'BTCUSDT') {
                    // Update BTC position data
                    const positionElement = document.getElementById('PositionSize');
                    const entryElement = document.getElementById('EntryPrice');
                    const currentValueElement = document.getElementById('CurrentValue');
                    const unrealizedPnlElement = document.getElementById('UnrealizedPnl');
                    const totalPnlElement = document.getElementById('TotalPnl');
                    
                    if (positionElement) positionElement.textContent = pnlData.position_size?.toFixed(6) || '0.000000';
                    if (entryElement) entryElement.textContent = `$${pnlData.entry_price?.toFixed(2) || '0.00'}`;
                    if (currentValueElement) currentValueElement.textContent = `$${pnlData.current_value?.toFixed(2) || '0.00'}`;
                    if (unrealizedPnlElement) {
                        const unrealizedPnl = pnlData.unrealized_pnl || 0;
                        unrealizedPnlElement.textContent = `$${unrealizedPnl.toFixed(2)}`;
                        unrealizedPnlElement.className = unrealizedPnl >= 0 ? 'value pnl-positive' : 'value pnl-negative';
                    }
                    if (totalPnlElement) {
                        const totalPnl = pnlData.total_pnl || 0;
                        totalPnlElement.textContent = `$${totalPnl.toFixed(2)}`;
                        totalPnlElement.className = totalPnl >= 0 ? 'value pnl-positive' : 'value pnl-negative';
                    }
                } else if (symbol === 'ETHUSDT') {
                    // Update ETH position data
                    const positionElement = document.getElementById('ethPositionSize');
                    const entryElement = document.getElementById('ethEntryPrice');
                    const currentValueElement = document.getElementById('ethCurrentValue');
                    const unrealizedPnlElement = document.getElementById('ethUnrealizedPnl');
                    const totalPnlElement = document.getElementById('ethTotalPnl');
                    
                    if (positionElement) positionElement.textContent = pnlData.position_size?.toFixed(6) || '0.000000';
                    if (entryElement) entryElement.textContent = `$${pnlData.entry_price?.toFixed(2) || '0.00'}`;
                    if (currentValueElement) currentValueElement.textContent = `$${pnlData.current_value?.toFixed(2) || '0.00'}`;
                    if (unrealizedPnlElement) {
                        const unrealizedPnl = pnlData.unrealized_pnl || 0;
                        unrealizedPnlElement.textContent = `$${unrealizedPnl.toFixed(2)}`;
                        unrealizedPnlElement.className = unrealizedPnl >= 0 ? 'value pnl-positive' : 'value pnl-negative';
                    }
                    if (totalPnlElement) {
                        const totalPnl = pnlData.total_pnl || 0;
                        totalPnlElement.textContent = `$${totalPnl.toFixed(2)}`;
                        totalPnlElement.className = totalPnl >= 0 ? 'value pnl-positive' : 'value pnl-negative';
                    }
                }
            }
            
            updateAllCharts() {
                // Update all charts with real-time data
                this.updatePortfolioChart();
                this.updatePnLChart();
                this.updatePositionChart();
                this.updateLongTermCharts();
            }
            }

            handleMessage(data) {
                try {
                    if (data.type === 'price_update') {
                        this.updatePriceDisplay(data.symbol, data.price);
                        this.updatePriceChart(data.symbol, data.price, data.timestamp);
                    } else if (data.type === 'pnl_update') {
                        this.updatePnLDisplay(data.symbol, data.pnl_data);
                        // Update P&L chart
                        if (data.pnl_data && data.pnl_data.total_pnl !== undefined) {
                            this.updatePnLChart(data.timestamp || Date.now(), data.pnl_data.total_pnl);
                        }
                    } else if (data.type === 'alpha_signals') {
                        this.updateAlphaSignals(data.signals);
                    } else if (data.type === 'trading_allocation') {
                        this.updateTradingAllocation(data.allocation);
                        // Update portfolio chart
                        if (data.allocation) {
                            const portfolioValue = data.allocation.total_portfolio_value || 100000;
                            const marketValue = data.allocation.total_invested || 0;
                            this.updatePortfolioChart(data.timestamp || Date.now(), portfolioValue, marketValue);
                        }
                    } else if (data.type === 'position_update') {
                        // New message type for position sizing updates
                        if (data.position_data) {
                            this.updatePositionChart(data.timestamp || Date.now(), data.position_data.total_position_size || 0);
                        }
                    }
                } catch (error) {
                    console.error('Error handling WebSocket message:', error, data);
                }
            }

            updatePriceDisplay(symbol, price) {
                const elementId = symbol === 'BTCUSDT' ? 'btcPrice' : 'ethPrice';
                const element = document.getElementById(elementId);
                if (element) {
                    element.textContent = `$${price.toFixed(2)}`;
                    element.classList.add('price-update');
                    setTimeout(() => element.classList.remove('price-update'), 500);
                }
            }

            updatePnLDisplay(symbol, pnlData) {
                const prefix = symbol === 'BTCUSDT' ? '' : 'eth';
                
                const elements = {
                    positionSize: document.getElementById(prefix + 'PositionSize'),
                    entryPrice: document.getElementById(prefix + 'EntryPrice'),
                    currentValue: document.getElementById(prefix + 'CurrentValue'),
                    unrealizedPnl: document.getElementById(prefix + 'UnrealizedPnl'),
                    totalPnl: document.getElementById(prefix + 'TotalPnl')
                };


                if (elements.positionSize) elements.positionSize.textContent = pnlData.position_size.toFixed(4);
                if (elements.entryPrice) elements.entryPrice.textContent = `$${pnlData.entry_price.toFixed(2)}`;
                if (elements.currentValue) elements.currentValue.textContent = `$${pnlData.current_value.toFixed(2)}`;
                if (elements.unrealizedPnl) {
                    elements.unrealizedPnl.textContent = `$${pnlData.unrealized_pnl.toFixed(2)}`;
                    elements.unrealizedPnl.className = pnlData.unrealized_pnl >= 0 ? 'value pnl-positive' : 'value pnl-negative';
                }
                if (elements.totalPnl) {
                    elements.totalPnl.textContent = `$${pnlData.total_pnl.toFixed(2)}`;
                    elements.totalPnl.className = pnlData.total_pnl >= 0 ? 'value pnl-positive' : 'value pnl-negative';
                }
            }

            updateAlphaSignals(signals) {
                const container = document.getElementById('alphaSignals');
                container.innerHTML = '';
                
                signals.forEach(signal => {
                    const signalDiv = document.createElement('div');
                    signalDiv.className = 'alpha-signal';
                    
                    const edgeColor = signal.edge_bps >= 0 ? '#4CAF50' : '#f44336';
                    
                    signalDiv.innerHTML = `
                        <div style="font-weight: bold; margin-bottom: 5px;">${signal.model_name}</div>
                        <div style="color: ${edgeColor}; font-size: 1.2em; font-weight: bold;">
                            ${signal.edge_bps.toFixed(1)} bps
                        </div>
                        <div style="margin-top: 5px; font-size: 0.9em;">
                            Confidence: ${(signal.confidence * 100).toFixed(1)}%
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${signal.confidence * 100}%"></div>
                        </div>
                    `;
                    
                    container.appendChild(signalDiv);
                });
            }

            updatePriceChart(symbol, price, timestamp) {
                const data = this.priceData[symbol];
                const time = new Date(timestamp);
                
                data.x.push(time);
                data.y.push(price);
                
                if (data.x.length > this.maxDataPoints) {
                    data.x.shift();
                    data.y.shift();
                }
                
                this.refreshChart(symbol);
            }

            async initCharts() {
                // Initialize charts with historical data
                await this.loadHistoricalData('BTCUSDT', 'btcChart', '#f7931a', '₿ BTCUSDT Price Movement');
                await this.loadHistoricalData('ETHUSDT', 'ethChart', '#627eea', 'Ξ ETHUSDT Price Movement');
                
                // Initialize trading performance charts
                this.initPortfolioChart();
                this.initPnLChart();
                this.initPositionChart();
                
                // Initialize long-term charts
                this.initLongTermCharts();
            }

            async loadHistoricalData(symbol, chartId, color, title) {
                try {
                    const response = await fetch(`/api/data/${symbol}`);
                    const data = await response.json();
                    
                    if (data.price_history && data.price_history.length > 0) {
                        // Clear existing data
                        this.priceData[symbol] = {x: [], y: []};
                        
                        // Load historical data
                        data.price_history.forEach(tick => {
                            this.priceData[symbol].x.push(new Date(tick.timestamp));
                            this.priceData[symbol].y.push(tick.price);
                        });
                        
                        // Initialize chart with historical data
                        this.initChart(symbol, chartId, color, title);
                        
                        console.log(`Loaded ${data.price_history.length} historical data points for ${symbol}`);
                    } else {
                        // Initialize empty chart
                        this.initChart(symbol, chartId, color, title);
                    }
                } catch (error) {
                    console.error(`Error loading historical data for ${symbol}:`, error);
                    // Initialize empty chart as fallback
                    this.initChart(symbol, chartId, color, title);
                }
            }

            initChart(symbol, chartId, color, title) {
                const layout = {
                    title: {
                        text: title,
                        font: {
                            color: '#ffffff',
                            size: 16
                        }
                    },
                    xaxis: {
                        title: 'Time',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    yaxis: {
                        title: 'Price (USD)',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: {
                        color: '#ffffff'
                    },
                    showlegend: false,
                    margin: {
                        l: 60,
                        r: 30,
                        t: 60,
                        b: 60
                    }
                };
                
                const trace = {
                    x: this.priceData[symbol].x,
                    y: this.priceData[symbol].y,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: symbol,
                    line: {
                        color: color,
                        width: 3
                    },
                    marker: {
                        color: color,
                        size: 4
                    }
                };
                
                Plotly.newPlot(chartId, [trace], layout, {responsive: true});
            }

            refreshChart(symbol) {
                const chartId = symbol === 'BTCUSDT' ? 'btcChart' : 'ethChart';
                const color = symbol === 'BTCUSDT' ? '#f7931a' : '#627eea';
                
                const trace = {
                    x: this.priceData[symbol].x,
                    y: this.priceData[symbol].y,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: symbol,
                    line: {
                        color: color,
                        width: 3
                    },
                    marker: {
                        color: color,
                        size: 4
                    }
                };
                
                Plotly.redraw(chartId, [trace]);
            }

            initPortfolioChart() {
                const layout = {
                    title: {
                        text: 'Portfolio Value vs Market Performance',
                        font: { color: '#ffffff', size: 16 }
                    },
                    xaxis: {
                        title: 'Time',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    yaxis: {
                        title: 'Value ($)',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#ffffff' },
                    legend: {
                        font: { color: '#ffffff' }
                    }
                };
                
                // Add sample data to make charts visible
                const now = new Date();
                const sampleTimes = [];
                const portfolioValues = [];
                const marketValues = [];
                
                for (let i = 0; i < 20; i++) {
                    const time = new Date(now - (20 - i) * 60 * 1000); // 1-minute intervals
                    sampleTimes.push(time);
                    portfolioValues.push(100000 + Math.random() * 5000 + i * 200); // Growing portfolio
                    marketValues.push(100000 + Math.random() * 3000 + i * 150); // Growing market
                }
                
                const portfolioTrace = {
                    x: sampleTimes,
                    y: portfolioValues,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Portfolio Value',
                    line: { color: '#4CAF50', width: 3 }
                };
                
                const marketTrace = {
                    x: sampleTimes,
                    y: marketValues,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Market Value',
                    line: { color: '#2196F3', width: 3 }
                };
                
                Plotly.newPlot('portfolioChart', [portfolioTrace, marketTrace], layout, {responsive: true});
            }

            initPnLChart() {
                const layout = {
                    title: {
                        text: 'Real-time P&L Performance',
                        font: { color: '#ffffff', size: 16 }
                    },
                    xaxis: {
                        title: 'Time',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    yaxis: {
                        title: 'P&L ($)',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#ffffff' },
                    legend: {
                        font: { color: '#ffffff' }
                    }
                };
                
                // Add sample P&L data
                const now = new Date();
                const sampleTimes = [];
                const pnlValues = [];
                
                for (let i = 0; i < 20; i++) {
                    const time = new Date(now - (20 - i) * 60 * 1000); // 1-minute intervals
                    sampleTimes.push(time);
                    // Simulate P&L growth with some volatility
                    pnlValues.push(i * 250 + Math.random() * 1000 - 500); // Growing P&L with noise
                }
                
                const pnlTrace = {
                    x: sampleTimes,
                    y: pnlValues,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Cumulative P&L',
                    line: { color: '#FF9800', width: 3 },
                    marker: { color: '#FF9800', size: 4 }
                };
                
                Plotly.newPlot('pnlChart', [pnlTrace], layout, {responsive: true});
            }

            initPositionChart() {
                const layout = {
                    title: {
                        text: 'Position Sizing & Risk Metrics',
                        font: { color: '#ffffff', size: 16 }
                    },
                    xaxis: {
                        title: 'Time',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    yaxis: {
                        title: 'Position Size ($)',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#ffffff' },
                    legend: {
                        font: { color: '#ffffff' }
                    }
                };
                
                // Add sample position data
                const now = new Date();
                const sampleTimes = [];
                const positionValues = [];
                
                for (let i = 0; i < 20; i++) {
                    const time = new Date(now - (20 - i) * 60 * 1000); // 1-minute intervals
                    sampleTimes.push(time);
                    // Simulate position sizing changes
                    const basePosition = 50000;
                    const variation = Math.sin(i * 0.5) * 10000; // Oscillating position sizes
                    positionValues.push(basePosition + variation + Math.random() * 2000);
                }
                
                const positionTrace = {
                    x: sampleTimes,
                    y: positionValues,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Position Size',
                    line: { color: '#9C27B0', width: 3 },
                    marker: { color: '#9C27B0', size: 4 }
                };
                
                Plotly.newPlot('positionChart', [positionTrace], layout, {responsive: true});
            }

            updatePortfolioChart(timestamp, portfolioValue, marketValue) {
                const time = new Date(timestamp || Date.now());
                
                // Add new data point
                this.portfolioData.timestamps.push(time);
                this.portfolioData.portfolioValue.push(portfolioValue || (100000 + Math.random() * 5000));
                this.portfolioData.marketValue.push(marketValue || (100000 + Math.random() * 3000));
                
                // Keep only last 100 data points
                if (this.portfolioData.timestamps.length > 100) {
                    this.portfolioData.timestamps.shift();
                    this.portfolioData.portfolioValue.shift();
                    this.portfolioData.marketValue.shift();
                }
                
                const portfolioTrace = {
                    x: this.portfolioData.timestamps,
                    y: this.portfolioData.portfolioValue,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Portfolio Value',
                    line: { color: '#4CAF50', width: 3 }
                };
                
                const marketTrace = {
                    x: this.portfolioData.timestamps,
                    y: this.portfolioData.marketValue,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Market Value',
                    line: { color: '#2196F3', width: 3 }
                };
                
                Plotly.redraw('portfolioChart', [portfolioTrace, marketTrace]);
            }

            updatePnLChart(timestamp, pnl) {
                const time = new Date(timestamp || Date.now());
                
                // Add new P&L data point
                this.portfolioData.pnl.push(pnl || (Math.random() * 1000 - 500));
                
                // Keep only last 100 data points
                if (this.portfolioData.pnl.length > 100) {
                    this.portfolioData.pnl.shift();
                }
                
                const pnlTrace = {
                    x: this.portfolioData.timestamps,
                    y: this.portfolioData.pnl,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Cumulative P&L',
                    line: { color: '#FF9800', width: 3 },
                    marker: { color: '#FF9800', size: 4 }
                };
                
                Plotly.redraw('pnlChart', [pnlTrace]);
            }

            updatePositionChart(timestamp, positionSize) {
                const time = new Date(timestamp || Date.now());
                
                // Add new position data point
                const basePosition = 50000;
                const variation = Math.sin(Date.now() * 0.001) * 10000;
                this.portfolioData.positions.push(positionSize || (basePosition + variation + Math.random() * 2000));
                
                // Keep only last 100 data points
                if (this.portfolioData.positions.length > 100) {
                    this.portfolioData.positions.shift();
                }
                
                const positionTrace = {
                    x: this.portfolioData.timestamps,
                    y: this.portfolioData.positions,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Position Size',
                    line: { color: '#9C27B0', width: 3 },
                    marker: { color: '#9C27B0', size: 4 }
                };
                
                Plotly.redraw('positionChart', [positionTrace]);
            }
            
            initLongTermCharts() {
                this.initHourlyChart();
                this.initFiveHourChart();
                this.initDailyChart();
            }
            
            initHourlyChart() {
                const layout = {
                    title: {
                        text: 'Hourly Performance (Last 24 Hours)',
                        font: { color: '#ffffff', size: 16 }
                    },
                    xaxis: {
                        title: 'Time',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    yaxis: {
                        title: 'Value ($)',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#ffffff' },
                    legend: { font: { color: '#ffffff' } }
                };
                
                // Generate sample hourly data (last 24 hours)
                const now = new Date();
                const hourlyTimes = [];
                const hourlyPortfolio = [];
                const hourlyPnL = [];
                
                for (let i = 0; i < 24; i++) {
                    const time = new Date(now - (24 - i) * 60 * 60 * 1000); // 1-hour intervals
                    hourlyTimes.push(time);
                    hourlyPortfolio.push(100000 + Math.random() * 8000 + i * 100);
                    hourlyPnL.push(i * 150 + Math.random() * 500 - 250);
                }
                
                const portfolioTrace = {
                    x: hourlyTimes,
                    y: hourlyPortfolio,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Portfolio Value',
                    line: { color: '#4CAF50', width: 3 },
                    marker: { color: '#4CAF50', size: 5 }
                };
                
                const pnlTrace = {
                    x: hourlyTimes,
                    y: hourlyPnL,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Hourly P&L',
                    line: { color: '#FF9800', width: 3 },
                    marker: { color: '#FF9800', size: 5 },
                    yaxis: 'y2'
                };
                
                const layoutWithSecondAxis = {
                    ...layout,
                    yaxis2: {
                        title: 'P&L ($)',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)',
                        overlaying: 'y',
                        side: 'right'
                    }
                };
                
                Plotly.newPlot('hourlyChart', [portfolioTrace, pnlTrace], layoutWithSecondAxis, {responsive: true});
            }
            
            initFiveHourChart() {
                const layout = {
                    title: {
                        text: '5-Hour Performance (Last 5 Days)',
                        font: { color: '#ffffff', size: 16 }
                    },
                    xaxis: {
                        title: 'Time',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    yaxis: {
                        title: 'Value ($)',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#ffffff' },
                    legend: { font: { color: '#ffffff' } }
                };
                
                // Generate sample 5-hour data (last 5 days)
                const now = new Date();
                const fiveHourTimes = [];
                const fiveHourPortfolio = [];
                const fiveHourPnL = [];
                
                for (let i = 0; i < 24; i++) { // 24 x 5-hour periods = 5 days
                    const time = new Date(now - (24 - i) * 5 * 60 * 60 * 1000); // 5-hour intervals
                    fiveHourTimes.push(time);
                    fiveHourPortfolio.push(100000 + Math.random() * 12000 + i * 250);
                    fiveHourPnL.push(i * 300 + Math.random() * 1000 - 500);
                }
                
                const portfolioTrace = {
                    x: fiveHourTimes,
                    y: fiveHourPortfolio,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Portfolio Value',
                    line: { color: '#2196F3', width: 3 },
                    marker: { color: '#2196F3', size: 5 }
                };
                
                const pnlTrace = {
                    x: fiveHourTimes,
                    y: fiveHourPnL,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '5-Hour P&L',
                    line: { color: '#E91E63', width: 3 },
                    marker: { color: '#E91E63', size: 5 },
                    yaxis: 'y2'
                };
                
                const layoutWithSecondAxis = {
                    ...layout,
                    yaxis2: {
                        title: 'P&L ($)',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)',
                        overlaying: 'y',
                        side: 'right'
                    }
                };
                
                Plotly.newPlot('fiveHourChart', [portfolioTrace, pnlTrace], layoutWithSecondAxis, {responsive: true});
            }
            
            initDailyChart() {
                const layout = {
                    title: {
                        text: 'Daily Performance (Last 30 Days)',
                        font: { color: '#ffffff', size: 16 }
                    },
                    xaxis: {
                        title: 'Date',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    yaxis: {
                        title: 'Value ($)',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#ffffff' },
                    legend: { font: { color: '#ffffff' } }
                };
                
                // Generate sample daily data (last 30 days)
                const now = new Date();
                const dailyTimes = [];
                const dailyPortfolio = [];
                const dailyPnL = [];
                
                for (let i = 0; i < 30; i++) {
                    const time = new Date(now - (30 - i) * 24 * 60 * 60 * 1000); // 1-day intervals
                    dailyTimes.push(time);
                    dailyPortfolio.push(100000 + Math.random() * 15000 + i * 400);
                    dailyPnL.push(i * 500 + Math.random() * 2000 - 1000);
                }
                
                const portfolioTrace = {
                    x: dailyTimes,
                    y: dailyPortfolio,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Portfolio Value',
                    line: { color: '#9C27B0', width: 3 },
                    marker: { color: '#9C27B0', size: 5 }
                };
                
                const pnlTrace = {
                    x: dailyTimes,
                    y: dailyPnL,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Daily P&L',
                    line: { color: '#FF5722', width: 3 },
                    marker: { color: '#FF5722', size: 5 },
                    yaxis: 'y2'
                };
                
                const layoutWithSecondAxis = {
                    ...layout,
                    yaxis2: {
                        title: 'P&L ($)',
                        color: '#ffffff',
                        gridcolor: 'rgba(255,255,255,0.1)',
                        overlaying: 'y',
                        side: 'right'
                    }
                };
                
                Plotly.newPlot('dailyChart', [portfolioTrace, pnlTrace], layoutWithSecondAxis, {responsive: true});
            }
            
            updateLongTermCharts() {
                // Update long-term charts with new data
                // This would integrate with real trading data
                const now = new Date();
                const currentValue = 100000 + Math.random() * 5000;
                const currentPnL = Math.random() * 1000 - 500;
                
                // Update hourly data
                this.longTermData.hourly.timestamps.push(now);
                this.longTermData.hourly.portfolioValue.push(currentValue);
                this.longTermData.hourly.pnl.push(currentPnL);
                
                // Keep only last 24 hours
                if (this.longTermData.hourly.timestamps.length > 24) {
                    this.longTermData.hourly.timestamps.shift();
                    this.longTermData.hourly.portfolioValue.shift();
                    this.longTermData.hourly.pnl.shift();
                }
                
                // Update 5-hour data (every 5 updates)
                if (this.longTermData.hourly.timestamps.length % 5 === 0) {
                    this.longTermData.fiveHour.timestamps.push(now);
                    this.longTermData.fiveHour.portfolioValue.push(currentValue);
                    this.longTermData.fiveHour.pnl.push(currentPnL);
                    
                    // Keep only last 5 days (24 x 5-hour periods)
                    if (this.longTermData.fiveHour.timestamps.length > 24) {
                        this.longTermData.fiveHour.timestamps.shift();
                        this.longTermData.fiveHour.portfolioValue.shift();
                        this.longTermData.fiveHour.pnl.shift();
                    }
                }
                
                // Update daily data (every 24 updates)
                if (this.longTermData.hourly.timestamps.length % 24 === 0) {
                    this.longTermData.daily.timestamps.push(now);
                    this.longTermData.daily.portfolioValue.push(currentValue);
                    this.longTermData.daily.pnl.push(currentPnL);
                    
                    // Keep only last 30 days
                    if (this.longTermData.daily.timestamps.length > 30) {
                        this.longTermData.daily.timestamps.shift();
                        this.longTermData.daily.portfolioValue.shift();
                        this.longTermData.daily.pnl.shift();
                    }
                }
            }

            updateConnectionStatus(connected) {
                const statusElement = document.getElementById('connectionStatus');
                if (connected) {
                    statusElement.textContent = '🟢 Connected';
                    statusElement.className = 'connection-status connected';
                } else {
                    statusElement.textContent = '🔴 Disconnected';
                    statusElement.className = 'connection-status disconnected';
                }
            }

            updateTradingAllocation(allocation) {
                const totalPortfolioValue = document.getElementById('totalPortfolioValue');
                const totalInvested = document.getElementById('totalInvested');
                const profitPercentage = document.getElementById('profitPercentage');
                const availableCapital = document.getElementById('availableCapital');
                const tradingAllocation = document.getElementById('tradingAllocation');

                if (totalPortfolioValue) totalPortfolioValue.textContent = `$${allocation.total_portfolio_value.toFixed(2)}`;
                if (totalInvested) totalInvested.textContent = `$${allocation.total_invested.toFixed(2)}`;
                if (profitPercentage) {
                    profitPercentage.textContent = `${allocation.profit_percentage >= 0 ? '+' : ''}${allocation.profit_percentage.toFixed(2)}%`;
                    profitPercentage.className = allocation.profit_percentage >= 0 ? 'value pnl-positive' : 'value pnl-negative';
                }
                if (availableCapital) availableCapital.textContent = `$${allocation.available_capital.toFixed(2)}`;
                if (tradingAllocation) tradingAllocation.textContent = `${allocation.trading_allocation}%`;
            }
        }

        // Dashboard control functions
        async function refreshDashboard() {
            try {
                // Refresh the page
                window.location.reload();
            } catch (error) {
                console.error('Error refreshing dashboard:', error);
            }
        }

        async function clearPortfolio() {
            if (!confirm('Are you sure you want to reset the portfolio to $100 each for BTC and ETH?')) {
                return;
            }
            
            try {
                const response = await fetch('/api/portfolio/reset', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert('Portfolio reset successfully!');
                    // The dashboard will update automatically via WebSocket
                } else {
                    alert(`Error: ${result.error}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        // Trading allocation function
        async function setTradingAllocation(percentage) {
            try {
                const response = await fetch('/api/trading/allocation', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ percentage: percentage })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert(`Trading allocation set to ${percentage}%`);
                    // Refresh trading allocation display
                    const allocationResponse = await fetch('/api/trading/allocation');
                    const allocationData = await allocationResponse.json();
                    // The dashboard will receive the update via WebSocket
                } else {
                    alert(`Error: ${result.error}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        // Trading functions
        async function buyPosition(symbol) {
            const inputId = symbol === 'BTCUSDT' ? 'btcBuyAmount' : 'ethBuyAmount';
            const amount = parseFloat(document.getElementById(inputId).value);
            
            if (!amount || amount <= 0) {
                alert('Please enter a valid dollar amount');
                return;
            }
            
            try {
                const response = await fetch(`/api/buy/${symbol}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ dollar_amount: amount })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert(`Successfully bought $${result.dollar_amount} worth of ${symbol}\\n` +
                          `Amount: ${result.crypto_amount.toFixed(6)}\\n` +
                          `Price: $${result.buy_price.toFixed(2)}\\n` +
                          `New Entry Price: $${result.new_entry_price.toFixed(2)}`);
                    document.getElementById(inputId).value = '';
                } else {
                    alert(`Error: ${result.error}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        async function buyWithAmount(symbol, amount) {
            try {
                const response = await fetch(`/api/buy/${symbol}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ dollar_amount: amount })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert(`Successfully bought $${result.dollar_amount} worth of ${symbol}\\n` +
                          `Amount: ${result.crypto_amount.toFixed(6)}\\n` +
                          `Price: $${result.buy_price.toFixed(2)}\\n` +
                          `New Entry Price: $${result.new_entry_price.toFixed(2)}`);
                } else {
                    alert(`Error: ${result.error}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        async function sellByPercentage(symbol, percentage) {
            try {
                const response = await fetch(`/api/sell/${symbol}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ percentage: percentage })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert(`Successfully sold ${percentage}% of ${symbol}\\n` +
                          `Amount: ${result.sold_amount.toFixed(6)}\\n` +
                          `Price: $${result.sell_price.toFixed(2)}\\n` +
                          `P&L: $${result.realized_pnl.toFixed(2)}`);
                } else {
                    alert(`Error: ${result.error}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        async function sellByAmount(symbol) {
            const inputId = symbol === 'BTCUSDT' ? 'btcSellAmount' : 'ethSellAmount';
            const amount = parseFloat(document.getElementById(inputId).value);
            
            if (!amount || amount <= 0) {
                alert('Please enter a valid amount');
                return;
            }
            
            try {
                const response = await fetch(`/api/sell/${symbol}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ amount: amount })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert(`Successfully sold ${result.sold_amount.toFixed(6)} ${symbol}\\n` +
                          `Price: $${result.sell_price.toFixed(2)}\\n` +
                          `P&L: $${result.realized_pnl.toFixed(2)}`);
                    document.getElementById(inputId).value = '';
                } else {
                    alert(`Error: ${result.error}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        async function toggleModel() {
            try {
                const response = await fetch('/api/model/toggle', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                const result = await response.json();
                
                if (result.success) {
                    updateModelStatus(result.model_running);
                } else {
                    alert('Error toggling model');
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        function updateModelStatus(isRunning) {
            const indicator = document.getElementById('modelIndicator');
            const status = document.getElementById('modelStatus');
            const button = document.getElementById('toggleModelBtn');
            
            if (isRunning) {
                indicator.className = 'status-indicator';
                status.textContent = 'Model Running';
                button.textContent = 'Stop Model';
                button.className = 'control-btn stop-btn';
            } else {
                indicator.className = 'status-indicator stopped';
                status.textContent = 'Model Stopped';
                button.textContent = 'Start Model';
                button.className = 'control-btn start-btn';
            }
        }

        // Binance account functions
        async function refreshBinanceAccount() {
            try {
                const response = await fetch('/api/binance/account');
                const result = await response.json();
                
                if (result.success) {
                    const binanceBTC = document.getElementById('binanceBTC');
                    const binanceETH = document.getElementById('binanceETH');
                    const binanceUSDT = document.getElementById('binanceUSDT');
                    const binanceTotal = document.getElementById('binanceTotal');
                    
                    if (binanceBTC) binanceBTC.textContent = result.balances.BTC.free;
                    if (binanceETH) binanceETH.textContent = result.balances.ETH.free;
                    if (binanceUSDT) binanceUSDT.textContent = `$${parseFloat(result.balances.USDT.free).toFixed(2)}`;
                    if (binanceTotal) binanceTotal.textContent = `$${parseFloat(result.total_wallet_balance).toFixed(2)}`;
                    
                    if (result.message) {
                        alert(result.message);
                    }
                } else {
                    // Handle error case with fallback data
                    if (result.fallback_data) {
                        const binanceBTC = document.getElementById('binanceBTC');
                        const binanceETH = document.getElementById('binanceETH');
                        const binanceUSDT = document.getElementById('binanceUSDT');
                        const binanceTotal = document.getElementById('binanceTotal');
                        
                        if (binanceBTC) binanceBTC.textContent = result.fallback_data.balances.BTC.free;
                        if (binanceETH) binanceETH.textContent = result.fallback_data.balances.ETH.free;
                        if (binanceUSDT) binanceUSDT.textContent = `$${parseFloat(result.fallback_data.balances.USDT.free).toFixed(2)}`;
                        if (binanceTotal) binanceTotal.textContent = `$${parseFloat(result.fallback_data.total_wallet_balance).toFixed(2)}`;
                        
                        alert(`${result.error}\\n\\nShowing demo data instead.\\n\\n${result.instructions || ''}`);
                    } else {
                        alert(`Error: ${result.error}\\n\\n${result.instructions || ''}`);
                    }
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        // Trading control functions
        async function startTrading() {
            try {
                const response = await fetch('/api/trading/start', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                const result = await response.json();
                
                if (result.success) {
                    updateTradingStatus(result.trading_active, result.model_running);
                    alert(result.message);
                } else {
                    alert(`Error: ${result.error}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        async function stopTrading() {
            try {
                const response = await fetch('/api/trading/stop', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                const result = await response.json();
                
                if (result.success) {
                    updateTradingStatus(result.trading_active, result.model_running);
                    alert(result.message);
                } else {
                    alert(`Error: ${result.error}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        function updateTradingStatus(tradingActive, modelRunning) {
            const tradingStatus = document.getElementById('tradingStatus');
            const modelStatus = document.getElementById('modelRunning');
            const startBtn = document.getElementById('startTradingBtn');
            const stopBtn = document.getElementById('stopTradingBtn');
            
            if (tradingStatus) {
                tradingStatus.textContent = tradingActive ? 'Active' : 'Stopped';
                tradingStatus.className = tradingActive ? 'value pnl-positive' : 'value pnl-negative';
            }
            
            if (modelStatus) {
                modelStatus.textContent = modelRunning ? 'Running' : 'Stopped';
                modelStatus.className = modelRunning ? 'value pnl-positive' : 'value pnl-negative';
            }
            
            if (startBtn) {
                startBtn.disabled = tradingActive;
                startBtn.style.opacity = tradingActive ? '0.5' : '1';
            }
            
            if (stopBtn) {
                stopBtn.disabled = !tradingActive;
                stopBtn.style.opacity = !tradingActive ? '0.5' : '1';
            }
        }

        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', async () => {
            console.log('Dashboard loading...');
            const dashboard = new TradingDashboard();
            
            // Initialize data safely, handling errors for individual endpoints
            try {
                // Load model status
                try {
                    const modelResponse = await fetch('/api/model/status');
                    if (modelResponse.ok) {
                        const modelResult = await modelResponse.json();
                        updateModelStatus(modelResult.model_running);
                    }
                } catch (error) {
                    console.warn('Model status failed:', error);
                }
                
                // Load portfolio data
                try {
                    const portfolioResponse = await fetch('/api/portfolio');
                    if (portfolioResponse.ok) {
                        const portfolioResult = await portfolioResponse.json();
                        
                        // Update portfolio display for each symbol
                        for (const [symbol, pnlData] of Object.entries(portfolioResult.portfolio)) {
                            console.log(`Updating ${symbol} display:`, pnlData);
                            
                            // Update price display
                            if (symbol === 'BTCUSDT') {
                                const btcPriceElement = document.getElementById('btcPrice');
                                if (btcPriceElement) {
                                    btcPriceElement.textContent = `$${pnlData.current_price.toFixed(2)}`;
                                }
                                
                                // Update BTC position data
                                const positionElement = document.getElementById('PositionSize');
                                const entryElement = document.getElementById('EntryPrice');
                                const currentValueElement = document.getElementById('CurrentValue');
                                const unrealizedPnlElement = document.getElementById('UnrealizedPnl');
                                const totalPnlElement = document.getElementById('TotalPnl');
                                
                                if (positionElement) positionElement.textContent = pnlData.position_size.toFixed(6);
                                if (entryElement) entryElement.textContent = `$${pnlData.entry_price.toFixed(2)}`;
                                if (currentValueElement) currentValueElement.textContent = `$${pnlData.current_value.toFixed(2)}`;
                                if (unrealizedPnlElement) {
                                    unrealizedPnlElement.textContent = `$${pnlData.unrealized_pnl.toFixed(2)}`;
                                    unrealizedPnlElement.className = pnlData.unrealized_pnl >= 0 ? 'value pnl-positive' : 'value pnl-negative';
                                }
                                if (totalPnlElement) {
                                    totalPnlElement.textContent = `$${pnlData.total_pnl.toFixed(2)}`;
                                    totalPnlElement.className = pnlData.total_pnl >= 0 ? 'value pnl-positive' : 'value pnl-negative';
                                }
                            } else if (symbol === 'ETHUSDT') {
                                const ethPriceElement = document.getElementById('ethPrice');
                                if (ethPriceElement) {
                                    ethPriceElement.textContent = `$${pnlData.current_price.toFixed(2)}`;
                                }
                                
                                // Update ETH position data
                                const positionElement = document.getElementById('ethPositionSize');
                                const entryElement = document.getElementById('ethEntryPrice');
                                const currentValueElement = document.getElementById('ethCurrentValue');
                                const unrealizedPnlElement = document.getElementById('ethUnrealizedPnl');
                                const totalPnlElement = document.getElementById('ethTotalPnl');
                                
                                if (positionElement) positionElement.textContent = pnlData.position_size.toFixed(6);
                                if (entryElement) entryElement.textContent = `$${pnlData.entry_price.toFixed(2)}`;
                                if (currentValueElement) currentValueElement.textContent = `$${pnlData.current_value.toFixed(2)}`;
                                if (unrealizedPnlElement) {
                                    unrealizedPnlElement.textContent = `$${pnlData.unrealized_pnl.toFixed(2)}`;
                                    unrealizedPnlElement.className = pnlData.unrealized_pnl >= 0 ? 'value pnl-positive' : 'value pnl-negative';
                                }
                                if (totalPnlElement) {
                                    totalPnlElement.textContent = `$${pnlData.total_pnl.toFixed(2)}`;
                                    totalPnlElement.className = pnlData.total_pnl >= 0 ? 'value pnl-positive' : 'value pnl-negative';
                                }
                            }
                        }
                        
                        console.log('Portfolio data loaded successfully:', portfolioResult.portfolio);
                    }
                } catch (error) {
                    console.error('Portfolio data failed:', error);
                }
                
                // Load Binance account data (optional)
                try {
                    const binanceResponse = await fetch('/api/binance/account');
                    if (binanceResponse.ok) {
                        const binanceResult = await binanceResponse.json();
                        
                        if (binanceResult.success) {
                            const binanceBTC = document.getElementById('binanceBTC');
                            const binanceETH = document.getElementById('binanceETH');
                            const binanceUSDT = document.getElementById('binanceUSDT');
                            const binanceTotal = document.getElementById('binanceTotal');
                            
                            if (binanceBTC) binanceBTC.textContent = binanceResult.balances.BTC.free;
                            if (binanceETH) binanceETH.textContent = binanceResult.balances.ETH.free;
                            if (binanceUSDT) binanceUSDT.textContent = `$${parseFloat(binanceResult.balances.USDT.free).toFixed(2)}`;
                            if (binanceTotal) binanceTotal.textContent = `$${parseFloat(binanceResult.total_wallet_balance).toFixed(2)}`;
                        }
                    }
                } catch (error) {
                    console.warn('Binance account failed (optional):', error);
                }
                
                // Load trading status
                try {
                    const tradingResponse = await fetch('/api/trading/status');
                    if (tradingResponse.ok) {
                        const tradingResult = await tradingResponse.json();
                        updateTradingStatus(tradingResult.trading_active, tradingResult.model_running);
                    }
                } catch (error) {
                    console.warn('Trading status failed:', error);
                }
                
                console.log('✅ Dashboard initialization complete');
            } catch (error) {
                console.error('❌ Dashboard initialization failed:', error);
            }
        });
    </script>
</body>
</html>
    """
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data updates"""
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Get real-time data for both symbols
            symbols = ["BTCUSDT", "ETHUSDT"]

            for symbol in symbols:
                # Get latest price
                latest_price = data_manager.get_latest_price(symbol)
                if latest_price:
                    # Get the actual timestamp from Redis data
                    try:
                        tick_data = redis_client.lindex(
                            f"market.raw.crypto.{symbol}", -1
                        )
                        if tick_data:
                            tick = json.loads(tick_data)
                            timestamp = tick["ts"]
                        else:
                            timestamp = time.time() * 1000
                    except:
                        timestamp = time.time() * 1000

                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "price_update",
                                "symbol": symbol,
                                "price": latest_price,
                                "timestamp": timestamp,
                            }
                        )
                    )

                    logger.debug(
                        f"Sent price update: {symbol} = ${latest_price:.2f} at {timestamp}"
                    )

                # Get P&L data
                pnl_data = data_manager.calculate_pnl(symbol)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "pnl_update",
                            "symbol": symbol,
                            "pnl_data": asdict(pnl_data),
                        }
                    )
                )

            # Get alpha signals
            alpha_signals = data_manager.get_alpha_signals("BTCUSDT")
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "alpha_signals",
                        "signals": [asdict(signal) for signal in alpha_signals],
                    }
                )
            )

            # Get trading allocation
            trading_allocation = data_manager.get_trading_allocation()
            await websocket.send_text(
                json.dumps(
                    {"type": "trading_allocation", "allocation": trading_allocation}
                )
            )

            # Wait before next update
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis_connected": True,
        "data_sources": ["Redis", "Binance Feed"],
    }


@app.get("/api/data/{symbol}")
async def get_symbol_data(symbol: str):
    """Get historical data for a symbol"""
    price_history = data_manager.get_price_history(symbol)
    pnl_data = data_manager.calculate_pnl(symbol)
    alpha_signals = data_manager.get_alpha_signals(symbol)

    return {
        "symbol": symbol,
        "current_price": data_manager.get_latest_price(symbol),
        "price_history": [asdict(tick) for tick in price_history],
        "pnl_data": asdict(pnl_data),
        "alpha_signals": [asdict(signal) for signal in alpha_signals],
        "timestamp": time.time(),
    }


@app.post("/api/sell/{symbol}")
async def sell_position(symbol: str, request: Request):
    """Sell position by percentage or amount"""
    try:
        body = await request.json()
        percentage = body.get("percentage")
        amount = body.get("amount")

        result = data_manager.sell_position(
            symbol, percentage=percentage, amount=amount
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/buy/{symbol}")
async def buy_position(symbol: str, request: Request):
    """Buy/add position with dollar amount"""
    try:
        body = await request.json()
        dollar_amount = body.get("dollar_amount")

        if dollar_amount is None:
            return {"success": False, "error": "Dollar amount is required"}

        result = data_manager.buy_position(symbol, dollar_amount)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/model/toggle")
async def toggle_model():
    """Toggle model on/off"""
    result = data_manager.toggle_model()
    return result


@app.get("/api/model/status")
async def get_model_status():
    """Get current model status"""
    return data_manager.get_model_status()


@app.post("/api/portfolio/reset")
async def reset_portfolio():
    """Reset portfolio to exactly $100 for each coin"""
    result = data_manager.reset_portfolio()
    return result


@app.post("/api/trading/allocation")
async def set_trading_allocation(request: Request):
    """Set trading allocation percentage"""
    try:
        body = await request.json()
        percentage = body.get("percentage")

        if percentage is None:
            return {"success": False, "error": "Percentage is required"}

        result = data_manager.set_trading_allocation(percentage)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/trading/allocation")
async def get_trading_allocation():
    """Get current trading allocation"""
    return data_manager.get_trading_allocation()


@app.get("/api/portfolio")
async def get_portfolio():
    """Get current portfolio status"""
    portfolio = {}
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        pnl_data = data_manager.calculate_pnl(symbol)
        portfolio[symbol] = asdict(pnl_data)

    return {
        "portfolio": portfolio,
        "model_status": data_manager.get_model_status(),
        "timestamp": time.time(),
    }


@app.get("/api/binance/account")
async def get_binance_account():
    """Get Binance account information"""
    return data_manager.get_binance_account_info()


@app.post("/api/trading/start")
async def start_trading():
    """Start trading system"""
    return data_manager.start_trading()


@app.post("/api/trading/stop")
async def stop_trading():
    """Stop trading system"""
    return data_manager.stop_trading()


@app.get("/api/trading/status")
async def get_trading_status():
    """Get current trading status"""
    return data_manager.get_trading_status()


@app.get("/api/performance/portfolio")
async def get_portfolio_performance():
    """Get portfolio performance data for charts"""
    try:
        # Get portfolio data from data manager
        portfolio_data = data_manager.get_portfolio_performance()
        return portfolio_data
    except Exception as e:
        logger.error(f"Error getting portfolio performance: {e}")
        return {"error": str(e)}


@app.get("/api/performance/pnl")
async def get_pnl_performance():
    """Get P&L performance data for charts"""
    try:
        # Get P&L data from data manager
        pnl_data = data_manager.get_pnl_performance()
        return pnl_data
    except Exception as e:
        logger.error(f"Error getting P&L performance: {e}")
        return {"error": str(e)}


@app.get("/api/performance/positions")
async def get_position_performance():
    """Get position sizing data for charts"""
    try:
        # Get position data from data manager
        position_data = data_manager.get_position_performance()
        return position_data
    except Exception as e:
        logger.error(f"Error getting position performance: {e}")
        return {"error": str(e)}


def main():
    """Main entry point"""
    logger.info("Starting Real-time Trading Dashboard...")

    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("✅ Redis connection successful")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
