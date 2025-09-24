#!/usr/bin/env python3
"""
Simple Fix Dashboard - Basic working version
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import redis
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Fixed Trading Dashboard")

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
            market_key = f"market.raw.crypto.{symbol}"
            latest_data = redis_client.lrange(market_key, -1, -1)
            if latest_data:
                data = json.loads(latest_data[0])
                return float(data.get("price", 0.0))
    except Exception as e:
        print(f"Error getting price for {symbol}: {e}")
    return 0.0


def get_twitter_news(limit=10):
    """Get latest Twitter news from Redis stream."""
    try:
        if redis_client:
            # Get latest news from the Twitter stream
            news_data = redis_client.xrevrange("news:x", count=limit)
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

            return news_items
    except Exception as e:
        print(f"Error getting Twitter news: {e}")
    return []


@app.get("/")
async def dashboard():
    """Serve the fixed dashboard."""
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fixed Trading Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            backdrop-filter: blur(10px);
        }
        .price {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .status {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            background: rgba(76, 175, 80, 0.3);
        }
    </style>
</head>
<body>
    <h1>🚀 Fixed Trading Dashboard</h1>
    
    <div class="status" id="status">Loading...</div>
    
    <div class="card">
        <h2>₿ Bitcoin (BTCUSDT)</h2>
        <div class="price" id="btcPrice">$0.00</div>
        <div>Position: <span id="btcPosition">0.000000</span></div>
        <div>P&L: <span id="btcPnL">$0.00</span></div>
    </div>
    
    <div class="card">
        <h2>Ξ Ethereum (ETHUSDT)</h2>
        <div class="price" id="ethPrice">$0.00</div>
        <div>Position: <span id="ethPosition">0.000000</span></div>
        <div>P&L: <span id="ethPnL">$0.00</span></div>
    </div>

    <script>
        console.log('Dashboard JavaScript loaded');
        
        async function fetchAndUpdate() {
            console.log('Fetching data...');
            document.getElementById('status').textContent = 'Fetching...';
            
            try {
                const response = await fetch('/api/portfolio');
                console.log('Response status:', response.status);
                
                if (response.ok) {
                    const data = await response.json();
                    console.log('Data received:', data);
                    
                    // Update BTC
                    const btc = data.BTCUSDT;
                    if (btc) {
                        document.getElementById('btcPrice').textContent = `$${btc.current_price.toFixed(2)}`;
                        document.getElementById('btcPosition').textContent = btc.position_size.toFixed(6);
                        document.getElementById('btcPnL').textContent = `$${btc.total_pnl.toFixed(2)}`;
                    }
                    
                    // Update ETH
                    const eth = data.ETHUSDT;
                    if (eth) {
                        document.getElementById('ethPrice').textContent = `$${eth.current_price.toFixed(2)}`;
                        document.getElementById('ethPosition').textContent = eth.position_size.toFixed(6);
                        document.getElementById('ethPnL').textContent = `$${eth.total_pnl.toFixed(2)}`;
                    }
                    
                    document.getElementById('status').textContent = `✅ Updated at ${new Date().toLocaleTimeString()}`;
                } else {
                    document.getElementById('status').textContent = `❌ Error: ${response.status}`;
                }
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('status').textContent = `❌ Error: ${error.message}`;
            }
        }
        
        // Start when page loads
        document.addEventListener('DOMContentLoaded', () => {
            console.log('DOM loaded, starting updates...');
            fetchAndUpdate();
            setInterval(fetchAndUpdate, 5000);
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
        # Get prices
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
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/twitter-news")
async def get_twitter_news_api():
    """Get latest Twitter news with sentiment."""
    try:
        news_items = get_twitter_news(limit=20)
        return {
            "news": news_items,
            "count": len(news_items),
            "updated_at": int(datetime.now().timestamp()),
        }
    except Exception as e:
        return {"error": str(e), "news": [], "count": 0}


@app.get("/api/model-price-series")
async def get_model_price_series(symbol: str = "BTC"):
    """Get prediction vs market price series with confidence bands."""
    try:
        if redis_client:
            # Get price series data from Redis
            price_key = f"price:{symbol}"
            rows = redis_client.xrevrange(price_key, count=500)

            data_points = []
            for stream_id, fields in rows:
                # Extract timestamp from stream ID (format: timestamp-sequence)
                ts = (
                    int(stream_id.decode().split("-")[0])
                    if isinstance(stream_id, bytes)
                    else int(stream_id.split("-")[0])
                )

                data_points.append(
                    {
                        "ts": ts,
                        "market": float(fields.get("price", 0.0)),
                        "model": float(
                            fields.get("pred", fields.get("price", 0.0))
                        ),  # Use prediction or fallback to price
                        "ci_low": float(fields.get("ci_low", fields.get("price", 0.0)))
                        * 0.98,  # Simulate confidence interval
                        "ci_high": float(
                            fields.get("ci_high", fields.get("price", 0.0))
                        )
                        * 1.02,
                    }
                )

            # If no data, create sample data for demonstration
            if not data_points:
                import time
                import random

                current_time = int(time.time())
                base_price = 100000 if symbol == "BTC" else 3500

                for i in range(100):
                    ts = current_time - (100 - i) * 300  # 5-minute intervals
                    market_price = base_price + random.uniform(-1000, 1000)
                    model_pred = market_price + random.uniform(-200, 200)

                    data_points.append(
                        {
                            "ts": ts,
                            "market": market_price,
                            "model": model_pred,
                            "ci_low": model_pred * 0.98,
                            "ci_high": model_pred * 1.02,
                        }
                    )

            return {
                "data": data_points[-100:],  # Return last 100 points
                "symbol": symbol,
                "count": len(data_points[-100:]),
            }
        else:
            return {"error": "Redis not connected", "data": [], "count": 0}
    except Exception as e:
        return {"error": str(e), "data": [], "count": 0}


@app.get("/api/residuals")
async def get_residuals(symbol: str = "BTC", period: str = "24h"):
    """Get residual distribution (market - model prediction)."""
    try:
        if redis_client:
            # Get data for residual calculation
            price_key = f"price:{symbol}"

            # Define lookback based on period
            period_hours = {"1h": 1, "24h": 24, "7d": 168}
            hours = period_hours.get(period, 24)
            count = hours * 12  # 12 data points per hour (5-min intervals)

            rows = redis_client.xrevrange(price_key, count=count)

            residuals = []
            for stream_id, fields in rows:
                market = float(fields.get("price", 0.0))
                model = float(fields.get("pred", market))  # Use prediction or fallback
                residual = market - model
                residuals.append(residual)

            # If no data, create sample residuals for demonstration
            if not residuals:
                import random
                import numpy as np

                # Generate normal distribution of residuals with some bias
                residuals = np.random.normal(0, 100, 200).tolist()
                residuals.extend(
                    np.random.normal(50, 150, 50).tolist()
                )  # Add some bias

            # Create histogram bins
            import numpy as np

            residuals = np.array(residuals)

            # Calculate histogram
            counts, bin_edges = np.histogram(residuals, bins=20)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Calculate statistics
            stats = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
                "median": float(np.median(residuals)),
                "count": len(residuals),
            }

            histogram_data = [
                {"bin_center": float(center), "count": int(count)}
                for center, count in zip(bin_centers, counts)
            ]

            return {
                "residuals": residuals.tolist()[:100],  # Return sample of raw residuals
                "histogram": histogram_data,
                "stats": stats,
                "symbol": symbol,
                "period": period,
            }
        else:
            return {
                "error": "Redis not connected",
                "residuals": [],
                "histogram": [],
                "stats": {},
            }
    except Exception as e:
        return {"error": str(e), "residuals": [], "histogram": [], "stats": {}}


@app.get("/api/pnl-curve")
async def get_pnl_curve(timeframe: str = "24h"):
    """Get PnL curve data for different timeframes (1h, 24h, 7d, 30d)."""
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
                for stream_id, fields in reversed(
                    pnl_data
                ):  # Reverse to get chronological order
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
                            "equity": 10000
                            + total_pnl,  # Assuming $10k starting capital
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

                    # Generate realistic PnL movement (random walk with slight upward bias)
                    pnl_change = np.random.normal(0.5, 15)  # Slight positive bias
                    running_pnl += pnl_change

                    # Simulate individual coin PnL
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

            # Calculate performance metrics
            if portfolio_curve:
                pnl_values = [p["total_pnl"] for p in portfolio_curve]
                equity_values = [p["equity"] for p in portfolio_curve]

                # Calculate returns
                returns = []
                for i in range(1, len(equity_values)):
                    ret = (equity_values[i] - equity_values[i - 1]) / equity_values[
                        i - 1
                    ]
                    returns.append(ret)

                import numpy as np

                returns = np.array(returns)

                metrics = {
                    "total_return": (
                        (equity_values[-1] - equity_values[0]) / equity_values[0]
                        if equity_values
                        else 0
                    ),
                    "max_drawdown": calculate_max_drawdown(equity_values),
                    "sharpe_ratio": calculate_sharpe_ratio(returns),
                    "win_rate": (
                        len([r for r in returns if r > 0]) / len(returns)
                        if returns.size > 0
                        else 0
                    ),
                    "volatility": float(np.std(returns)) if returns.size > 0 else 0,
                    "current_pnl": pnl_values[-1] if pnl_values else 0,
                }
            else:
                metrics = {}

            return {
                "timeframe": timeframe,
                "data": portfolio_curve,
                "metrics": metrics,
                "count": len(portfolio_curve),
            }
        else:
            return {"error": "Redis not connected", "data": [], "metrics": {}}
    except Exception as e:
        return {"error": str(e), "data": [], "metrics": {}}


def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve."""
    try:
        import numpy as np

        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return float(np.min(drawdown))
    except:
        return 0.0


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio from returns."""
    try:
        import numpy as np

        if len(returns) == 0:
            return 0.0
        excess_return = np.mean(returns) - risk_free_rate / 252  # Daily risk-free rate
        if np.std(returns) == 0:
            return 0.0
        return float(excess_return / np.std(returns) * np.sqrt(252))  # Annualized
    except:
        return 0.0


@app.get("/api/entropy-qspread")
async def get_entropy_qspread():
    """Get entropy and Q-spread time series for policy monitoring."""
    try:
        if redis_client:
            # Get recent policy data
            policy_key = "policy:actions"
            recent_data = redis_client.xrevrange(policy_key, count=60)  # Last hour

            entropy_data = []
            qspread_data = []

            if recent_data:
                for stream_id, fields in reversed(recent_data):
                    ts = (
                        int(stream_id.decode().split("-")[0])
                        if isinstance(stream_id, bytes)
                        else int(stream_id.split("-")[0])
                    )

                    # Extract policy entropy and Q-spread if available
                    entropy = float(fields.get("entropy", 0.0))
                    qspread = float(fields.get("q_spread", 0.0))

                    entropy_data.append({"timestamp": ts, "value": entropy})
                    qspread_data.append({"timestamp": ts, "value": qspread})

            # If no data, generate sample data
            if not entropy_data:
                import time
                import random
                import numpy as np

                current_time = int(time.time())

                for i in range(60):  # Last 60 minutes
                    ts = current_time - (60 - i) * 60

                    # Simulate entropy (0 to 2, policy collapse when near 0)
                    base_entropy = 1.5
                    entropy_noise = np.random.normal(0, 0.2)
                    # Occasionally simulate policy collapse
                    if random.random() < 0.05:  # 5% chance of collapse
                        entropy_val = max(0, base_entropy + entropy_noise - 1.2)
                    else:
                        entropy_val = max(0, min(2, base_entropy + entropy_noise))

                    # Simulate Q-spread (difference between max and min Q-values)
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

            return {
                "entropy_series": entropy_data,
                "qspread_series": qspread_data,
                "stats": stats,
                "count": len(entropy_data),
            }
        else:
            return {
                "error": "Redis not connected",
                "entropy_series": [],
                "qspread_series": [],
                "stats": {},
            }
    except Exception as e:
        return {
            "error": str(e),
            "entropy_series": [],
            "qspread_series": [],
            "stats": {},
        }


if __name__ == "__main__":
    print("🚀 Starting Fixed Trading Dashboard...")
    print("📊 URL: http://localhost:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)
