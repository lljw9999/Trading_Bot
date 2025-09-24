#!/usr/bin/env python3
"""
Enhanced Trading Dashboard

Building on the working simple dashboard, adding more features and data visualization.
"""

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
import uvicorn
import redis
import json
import asyncio
from datetime import datetime, timedelta
import requests
import random
import math
import sys
import os
import numpy as np
import logging

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from layers.layer1_signal_generation.news_sentiment import NewsIntegrationEngine
except ImportError:
    print("⚠️ News sentiment module not available")
    NewsIntegrationEngine = None

try:
    from layers.layer2_feature_engineering.temporal_fusion_transformer import (
        TFTCryptoPredictor,
    )
except ImportError:
    print("⚠️ TFT module not available")
    TFTCryptoPredictor = None

try:
    from layers.layer2_feature_engineering.graph_neural_networks import (
        GCNCryptoAnalyzer,
    )
except ImportError:
    print("⚠️ GCN module not available")
    GCNCryptoAnalyzer = None

try:
    from layers.layer3_risk_management.expected_shortfall_evt import (
        ComprehensiveRiskManager,
    )
except ImportError:
    print("⚠️ ES/EVT risk management module not available")
    ComprehensiveRiskManager = None

try:
    from layers.layer3_risk_management.copula_correlation_modeling import CopulaAnalyzer
except ImportError:
    print("⚠️ Copula correlation modeling module not available")
    CopulaAnalyzer = None

try:
    from layers.layer4_strategies.statistical_arbitrage import (
        StatisticalArbitrageEngine,
    )
except ImportError:
    print("⚠️ Statistical arbitrage module not available")
    StatisticalArbitrageEngine = None

try:
    from layers.layer4_strategies.smart_order_routing import (
        SmartOrderRouter,
        ExecutionConfig,
    )
except ImportError:
    print("⚠️ Smart order routing module not available")
    SmartOrderRouter = None
    ExecutionConfig = None

try:
    from layers.layer3_risk_management.enhanced_risk_harmonizer import (
        EnhancedRiskHarmonizer,
        CostModel,
    )
except ImportError:
    print("⚠️ Enhanced risk harmonizer module not available")
    EnhancedRiskHarmonizer = None
    CostModel = None

try:
    from layers.layer5_monitoring.slippage_monitor import SlippageMonitor
except ImportError:
    print("⚠️ Slippage monitor module not available")
    SlippageMonitor = None


def clean_data_for_json(data):
    """Clean data to ensure JSON serialization compatibility"""
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, (np.floating, float)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, np.ndarray):
        return clean_data_for_json(data.tolist())
    else:
        return data


app = FastAPI(title="Enhanced Trading Dashboard")

# Initialize Redis connection
try:
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    redis_client.ping()
    print("✅ Redis connected successfully")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    redis_client = None

# Initialize News Integration Engine
news_engine = None
if NewsIntegrationEngine:
    try:
        news_engine = NewsIntegrationEngine()
        print("✅ News sentiment engine initialized")
    except Exception as e:
        print(f"⚠️ News sentiment engine initialization failed: {e}")
        news_engine = None

# Initialize TFT Predictor
tft_predictor = None
if TFTCryptoPredictor:
    try:
        tft_predictor = TFTCryptoPredictor()
        print("✅ TFT predictor initialized")
    except Exception as e:
        print(f"⚠️ TFT predictor initialization failed: {e}")
        tft_predictor = None

# Initialize GCN Analyzer
gcn_analyzer = None
if GCNCryptoAnalyzer:
    try:
        gcn_analyzer = GCNCryptoAnalyzer()
        print("✅ GCN analyzer initialized")
    except Exception as e:
        print(f"⚠️ GCN analyzer initialization failed: {e}")
        gcn_analyzer = None

# Initialize Risk Manager
risk_manager = None
if ComprehensiveRiskManager:
    try:
        risk_manager = ComprehensiveRiskManager()
        print("✅ ES/EVT risk manager initialized")
    except Exception as e:
        print(f"⚠️ ES/EVT risk manager initialization failed: {e}")
        risk_manager = None

# Initialize Copula Analyzer
copula_analyzer = None
if CopulaAnalyzer:
    try:
        copula_analyzer = CopulaAnalyzer()
        print("✅ Copula analyzer initialized")
    except Exception as e:
        print(f"⚠️ Copula analyzer initialization failed: {e}")
        copula_analyzer = None

# Initialize Statistical Arbitrage Engine
stat_arb_engine = None
if StatisticalArbitrageEngine:
    try:
        stat_arb_engine = StatisticalArbitrageEngine()
        print("✅ Statistical arbitrage engine initialized")
    except Exception as e:
        print(f"⚠️ Statistical arbitrage engine initialization failed: {e}")
        stat_arb_engine = None

# Initialize Enhanced Smart Order Router
smart_order_router = None
if SmartOrderRouter and ExecutionConfig:
    try:
        execution_config = ExecutionConfig(
            max_spread_bps=8.0, min_edge_threshold_bps=12.0, target_latency_ms=100.0
        )
        smart_order_router = SmartOrderRouter(execution_config)
        print("✅ Enhanced smart order router initialized")
    except Exception as e:
        print(f"⚠️ Smart order router initialization failed: {e}")
        smart_order_router = None

# Initialize Enhanced Risk Harmonizer
enhanced_risk_harmonizer = None
if EnhancedRiskHarmonizer and CostModel:
    try:
        cost_model = CostModel(
            spread_cost_bps=2.5,
            commission_bps=1.0,
            slippage_bps=3.0,
            market_impact_factor=0.3,
        )
        enhanced_risk_harmonizer = EnhancedRiskHarmonizer(cost_model)
        print("✅ Enhanced risk harmonizer initialized")
    except Exception as e:
        print(f"⚠️ Enhanced risk harmonizer initialization failed: {e}")
        enhanced_risk_harmonizer = None

# Initialize Slippage Monitor
slippage_monitor = None
if SlippageMonitor:
    try:
        slippage_monitor = SlippageMonitor()
        print("✅ Slippage monitor initialized")
    except Exception as e:
        print(f"⚠️ Slippage monitor initialization failed: {e}")
        slippage_monitor = None


def get_latest_price(symbol):
    """Get latest price from Redis."""
    try:
        if redis_client:
            price_key = f"price:{symbol}"
            price_data = redis_client.get(price_key)
            if price_data:
                data = json.loads(price_data)
                return data.get("price", 0.0)
    except Exception as e:
        print(f"Error getting price for {symbol}: {e}")
    return 0.0


def get_portfolio_positions():
    """Get current portfolio positions from Redis."""
    try:
        positions = redis_client.get("portfolio_positions")
        if positions:
            return json.loads(positions)
        else:
            # Initialize default positions
            default_positions = {
                "BTCUSDT": {"position": 0.000849, "entry_price": 117800.0},
                "ETHUSDT": {"position": 0.027927, "entry_price": 3580.82},
            }
            redis_client.set("portfolio_positions", json.dumps(default_positions))
            return default_positions
    except Exception as e:
        logger.error(f"Error getting positions from Redis: {e}")
        # Fallback to defaults
        return {
            "BTCUSDT": {"position": 0.000849, "entry_price": 117800.0},
            "ETHUSDT": {"position": 0.027927, "entry_price": 3580.82},
        }


def update_portfolio_position(symbol, position_change, price):
    """Update portfolio position in Redis."""
    try:
        positions = get_portfolio_positions()
        current_pos = positions[symbol]["position"]
        current_entry = positions[symbol]["entry_price"]

        # Calculate new weighted average entry price if buying
        if position_change > 0:
            total_cost = (current_pos * current_entry) + (position_change * price)
            new_position = current_pos + position_change
            new_entry_price = (
                total_cost / new_position if new_position > 0 else current_entry
            )
        else:
            # Selling - keep same entry price
            new_position = max(0, current_pos + position_change)  # Can't go negative
            new_entry_price = current_entry

        positions[symbol] = {"position": new_position, "entry_price": new_entry_price}

        redis_client.set("portfolio_positions", json.dumps(positions))
        logger.info(
            f"Updated {symbol} position: {new_position:.6f} @ ${new_entry_price:.2f}"
        )
        return True
    except Exception as e:
        logger.error(f"Error updating position: {e}")
        return False


def add_trading_log_entry(action, asset, amount, price, success=True, message=""):
    """Add entry to trading log."""
    try:
        if not redis_client:
            return False

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,  # 'buy' or 'sell'
            "asset": asset,
            "amount": amount,
            "price": price,
            "success": success,
            "message": message,
        }

        # Get existing log
        trading_log_key = "trading_log"
        existing_log = redis_client.get(trading_log_key)

        if existing_log:
            log_entries = json.loads(existing_log)
        else:
            log_entries = []

        # Add new entry
        log_entries.append(log_entry)

        # Keep only last 50 entries
        if len(log_entries) > 50:
            log_entries = log_entries[-50:]

        # Store back in Redis with 7 day expiry
        redis_client.setex(trading_log_key, 604800, json.dumps(log_entries))

        logger.info(f"Added trading log entry: {action} {asset} ${amount}")
        return True

    except Exception as e:
        logger.error(f"Error adding trading log entry: {e}")
        return False


def get_trading_log():
    """Get recent trading log entries."""
    try:
        if not redis_client:
            return []

        trading_log_key = "trading_log"
        existing_log = redis_client.get(trading_log_key)

        if existing_log:
            log_entries = json.loads(existing_log)
            # Return most recent first
            return sorted(log_entries, key=lambda x: x["timestamp"], reverse=True)
        else:
            return []

    except Exception as e:
        logger.error(f"Error getting trading log: {e}")
        return []


def get_portfolio_data():
    """Get enhanced portfolio data."""
    # Get current prices
    btc_price = get_latest_price("BTCUSDT") or 117852.81
    eth_price = get_latest_price("ETHUSDT") or 3579.83

    # Get positions from Redis
    positions = get_portfolio_positions()
    btc_position = positions["BTCUSDT"]["position"]
    eth_position = positions["ETHUSDT"]["position"]
    btc_entry = positions["BTCUSDT"]["entry_price"]
    eth_entry = positions["ETHUSDT"]["entry_price"]

    btc_value = btc_position * btc_price
    eth_value = eth_position * eth_price

    btc_pnl = btc_value - (btc_position * btc_entry)
    eth_pnl = eth_value - (eth_position * eth_entry)

    # Calculate additional metrics
    btc_pnl_pct = (
        (btc_pnl / (btc_position * btc_entry)) * 100 if btc_position > 0 else 0
    )
    eth_pnl_pct = (
        (eth_pnl / (eth_position * eth_entry)) * 100 if eth_position > 0 else 0
    )

    portfolio = {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "current_price": btc_price,
            "position_size": btc_position,
            "entry_price": btc_entry,
            "current_value": btc_value,
            "total_pnl": btc_pnl,
            "pnl_percentage": btc_pnl_pct,
            "price_change_24h": random.uniform(-5, 5),  # Simulated 24h change
            "volume_24h": random.uniform(20000, 50000),  # Simulated volume
        },
        "ETHUSDT": {
            "symbol": "ETHUSDT",
            "current_price": eth_price,
            "position_size": eth_position,
            "entry_price": eth_entry,
            "current_value": eth_value,
            "total_pnl": eth_pnl,
            "pnl_percentage": eth_pnl_pct,
            "price_change_24h": random.uniform(-5, 5),  # Simulated 24h change
            "volume_24h": random.uniform(15000, 35000),  # Simulated volume
        },
    }

    return portfolio


def get_market_metrics():
    """Get additional market metrics."""
    return {
        "total_market_cap": random.uniform(2500000000000, 3000000000000),  # $2.5T - $3T
        "btc_dominance": random.uniform(40, 45),  # 40-45%
        "fear_greed_index": random.randint(20, 80),  # 20-80
        "active_addresses": random.randint(800000, 1200000),  # 800k-1.2M
        "network_hash_rate": random.uniform(400, 600),  # 400-600 EH/s
    }


def get_trading_signals():
    """Get current trading signals."""
    signals = []

    # Generate comprehensive alpha signals
    signal_types = ["BUY", "SELL", "HOLD"]
    signal_strengths = ["WEAK", "MODERATE", "STRONG"]
    indicators = [
        "RSI Divergence",
        "MACD Cross",
        "Bollinger Bands",
        "Moving Average",
        "Volume Profile",
        "Momentum",
        "Stochastic",
        "Williams %R",
        "Fibonacci Retracement",
        "Support/Resistance",
        "News Sentiment",
        "On-Chain Metrics",
        "Whale Activity",
        "Exchange Flow",
    ]

    # Generate 8-10 signals for comprehensive display
    for i in range(random.randint(8, 10)):
        signal = {
            "indicator": random.choice(indicators),
            "signal": random.choice(signal_types),
            "strength": random.choice(signal_strengths),
            "confidence": random.uniform(60, 95),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "symbol": random.choice(["BTCUSDT", "ETHUSDT", "MARKET"]),
            "score": random.uniform(-1, 1),  # For model comparison
        }
        signals.append(signal)

    return signals


def get_performance_history():
    """Get performance history for charts using real portfolio data."""
    try:
        # Get real portfolio data
        portfolio_data = get_portfolio_data()
        btc_data = portfolio_data["BTCUSDT"]
        eth_data = portfolio_data["ETHUSDT"]

        current_portfolio_value = btc_data["current_value"] + eth_data["current_value"]
        current_pnl = btc_data["total_pnl"] + eth_data["total_pnl"]

        # Try to get historical data from Redis first
        history_key = "portfolio_history"
        if redis_client:
            try:
                stored_history = redis_client.get(history_key)
                if stored_history:
                    history = json.loads(stored_history)

                    # Add current data point
                    current_point = {
                        "timestamp": datetime.now().isoformat(),
                        "portfolio_value": current_portfolio_value,
                        "pnl": current_pnl,
                        "btc_price": btc_data["current_price"],
                        "eth_price": eth_data["current_price"],
                    }

                    # Keep only last 24 hours of data
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    history = [
                        h
                        for h in history
                        if datetime.fromisoformat(
                            h["timestamp"].replace("Z", "+00:00").replace("+00:00", "")
                        )
                        > cutoff_time
                    ]
                    history.append(current_point)

                    # Store updated history
                    redis_client.setex(
                        history_key, 86400, json.dumps(history)
                    )  # 24 hour expiry

                    return history
            except Exception as e:
                logger.error(f"Error getting portfolio history from Redis: {e}")

        # Fallback: Generate realistic history based on current data
        history = []
        base_time = datetime.now() - timedelta(hours=24)
        base_portfolio_value = max(
            180, current_portfolio_value - random.uniform(10, 30)
        )

        for i in range(24):
            timestamp = base_time + timedelta(hours=i)

            # Create realistic progression to current values
            progress = i / 23.0  # 0 to 1
            portfolio_value = (
                base_portfolio_value
                + (current_portfolio_value - base_portfolio_value) * progress
            )
            portfolio_value += random.uniform(-5, 5)  # Add some noise

            pnl = portfolio_value - 200  # Assume initial investment of $200

            history.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "portfolio_value": portfolio_value,
                    "pnl": pnl,
                    "btc_price": btc_data["current_price"]
                    + random.uniform(-2000, 2000),
                    "eth_price": eth_data["current_price"] + random.uniform(-200, 200),
                }
            )

        # Store in Redis for future use
        if redis_client:
            try:
                redis_client.setex(history_key, 86400, json.dumps(history))
            except Exception as e:
                logger.error(f"Error storing portfolio history: {e}")

        return history

    except Exception as e:
        logger.error(f"Error generating performance history: {e}")
        # Return minimal fallback data
        return [
            {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": 200,
                "pnl": 0,
                "btc_price": 117852.81,
                "eth_price": 3579.83,
            }
        ]


def get_news_sentiment():
    """Get current news sentiment data."""
    if news_engine:
        try:
            # Get sentiment for main trading pairs
            btc_sentiment = news_engine.get_current_sentiment("BTCUSDT")
            eth_sentiment = news_engine.get_current_sentiment("ETHUSDT")
            market_sentiment = news_engine.get_market_sentiment()

            # If no real data, generate demo data
            if not btc_sentiment:
                btc_sentiment = generate_demo_sentiment("BTCUSDT")
            if not eth_sentiment:
                eth_sentiment = generate_demo_sentiment("ETHUSDT")
            if not market_sentiment:
                market_sentiment = generate_demo_market_sentiment()

            return {
                "BTCUSDT": btc_sentiment,
                "ETHUSDT": eth_sentiment,
                "market_sentiment": market_sentiment,
                "last_update": datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"Error getting news sentiment: {e}")

    # Fallback to demo data
    return {
        "BTCUSDT": generate_demo_sentiment("BTCUSDT"),
        "ETHUSDT": generate_demo_sentiment("ETHUSDT"),
        "market_sentiment": generate_demo_market_sentiment(),
        "last_update": datetime.now().isoformat(),
    }


def generate_demo_sentiment(symbol):
    """Generate demo sentiment data."""
    sentiment_score = random.uniform(-0.8, 0.8)
    return {
        "symbol": symbol,
        "sentiment_score": sentiment_score,
        "confidence": random.uniform(0.6, 0.95),
        "news_count": random.randint(5, 15),
        "positive_news": random.randint(2, 8),
        "negative_news": random.randint(1, 5),
        "neutral_news": random.randint(2, 6),
        "key_phrases": random.sample(
            [
                "institutional adoption",
                "market rally",
                "price discovery",
                "technical breakout",
                "resistance level",
                "whale activity",
                "DeFi innovation",
                "blockchain upgrade",
                "regulatory clarity",
            ],
            3,
        ),
        "top_sources": random.sample(
            ["coindesk", "cointelegraph", "decrypt", "bitcoinmagazine"], 2
        ),
        "timestamp": datetime.now().isoformat(),
    }


def generate_demo_market_sentiment():
    """Generate demo market sentiment data."""
    return {
        "market_sentiment": random.uniform(-0.5, 0.5),
        "market_confidence": random.uniform(0.7, 0.9),
        "active_symbols": 8,
        "total_news": random.randint(20, 50),
        "timestamp": datetime.now().isoformat(),
    }


def get_recent_news():
    """Get recent crypto news."""
    # Generate sample news items
    news_items = []
    current_time = datetime.now()

    sample_news = [
        {
            "title": "Bitcoin Institutional Adoption Reaches New Heights",
            "source": "CoinDesk",
            "sentiment": 0.7,
            "timestamp": current_time - timedelta(minutes=15),
            "url": "https://coindesk.com/bitcoin-institutional",
        },
        {
            "title": "Ethereum Layer 2 Solutions Show Record Growth",
            "source": "Cointelegraph",
            "sentiment": 0.5,
            "timestamp": current_time - timedelta(minutes=45),
            "url": "https://cointelegraph.com/ethereum-layer2",
        },
        {
            "title": "Regulatory Clarity Boosts Crypto Market Confidence",
            "source": "Decrypt",
            "sentiment": 0.6,
            "timestamp": current_time - timedelta(hours=1),
            "url": "https://decrypt.co/regulatory-clarity",
        },
        {
            "title": "DeFi Protocol Launches Revolutionary Yield Mechanism",
            "source": "Bitcoin Magazine",
            "sentiment": 0.4,
            "timestamp": current_time - timedelta(hours=2),
            "url": "https://bitcoinmagazine.com/defi-yield",
        },
        {
            "title": "Market Analysis: Technical Indicators Signal Bullish Trend",
            "source": "CoinDesk",
            "sentiment": 0.3,
            "timestamp": current_time - timedelta(hours=3),
            "url": "https://coindesk.com/market-analysis",
        },
    ]

    return [
        {
            "title": news["title"],
            "source": news["source"],
            "sentiment_score": news["sentiment"],
            "timestamp": news["timestamp"].isoformat(),
            "url": news["url"],
            "time_ago": format_time_ago(news["timestamp"]),
        }
        for news in sample_news
    ]


def format_time_ago(timestamp):
    """Format timestamp as 'time ago' string."""
    now = datetime.now()
    diff = now - timestamp

    if diff.total_seconds() < 3600:  # Less than 1 hour
        minutes = int(diff.total_seconds() / 60)
        return f"{minutes}m ago"
    elif diff.total_seconds() < 86400:  # Less than 1 day
        hours = int(diff.total_seconds() / 3600)
        return f"{hours}h ago"
    else:
        days = diff.days
        return f"{days}d ago"


def get_market_model_comparison_5s():
    """Get 5-second market vs model comparison data."""
    data = []
    base_time = datetime.now() - timedelta(minutes=5)  # Last 5 minutes of 5-second data

    for i in range(60):  # 60 data points (5 minutes * 12 points per minute)
        timestamp = base_time + timedelta(seconds=i * 5)

        # Simulate market price (with noise)
        btc_market = 117800 + random.uniform(-200, 200) + math.sin(i * 0.1) * 100
        eth_market = 3580 + random.uniform(-50, 50) + math.sin(i * 0.1) * 25

        # Simulate model predictions (smoother, slightly leading)
        btc_model = 117800 + random.uniform(-100, 100) + math.sin((i + 2) * 0.1) * 80
        eth_model = 3580 + random.uniform(-25, 25) + math.sin((i + 2) * 0.1) * 20

        data.append(
            {
                "timestamp": timestamp.isoformat(),
                "btc_market": btc_market,
                "btc_model": btc_model,
                "eth_market": eth_market,
                "eth_model": eth_model,
                "market_accuracy": random.uniform(0.75, 0.95),
                "model_confidence": random.uniform(0.8, 0.98),
            }
        )

    return data


def get_market_model_comparison_1h():
    """Get 1-hour market vs model comparison data."""
    data = []
    base_time = datetime.now() - timedelta(hours=24)  # Last 24 hours

    for i in range(24):  # 24 hourly data points
        timestamp = base_time + timedelta(hours=i)

        # Simulate hourly market data with trend
        btc_market = 117000 + (i * 50) + random.uniform(-500, 500)
        eth_market = 3500 + (i * 5) + random.uniform(-100, 100)

        # Model predictions (better at capturing trends)
        btc_model = 117000 + (i * 45) + random.uniform(-200, 200)
        eth_model = 3500 + (i * 4.5) + random.uniform(-50, 50)

        data.append(
            {
                "timestamp": timestamp.isoformat(),
                "btc_market": btc_market,
                "btc_model": btc_model,
                "eth_market": eth_market,
                "eth_model": eth_model,
                "market_accuracy": random.uniform(0.80, 0.95),
                "model_confidence": random.uniform(0.85, 0.98),
            }
        )

    return data


def get_market_model_comparison_1d():
    """Get daily market vs model comparison data."""
    data = []
    base_time = datetime.now() - timedelta(days=30)  # Last 30 days

    for i in range(30):  # 30 daily data points
        timestamp = base_time + timedelta(days=i)

        # Simulate daily market data with volatility cycles
        btc_market = (
            110000 + (i * 300) + random.uniform(-2000, 2000) + math.sin(i * 0.2) * 3000
        )
        eth_market = (
            3200 + (i * 15) + random.uniform(-200, 200) + math.sin(i * 0.2) * 300
        )

        # Model predictions (good at long-term trends)
        btc_model = (
            110000
            + (i * 280)
            + random.uniform(-1000, 1000)
            + math.sin((i + 1) * 0.2) * 2500
        )
        eth_model = (
            3200 + (i * 14) + random.uniform(-100, 100) + math.sin((i + 1) * 0.2) * 250
        )

        data.append(
            {
                "timestamp": timestamp.isoformat(),
                "btc_market": btc_market,
                "btc_model": btc_model,
                "eth_market": eth_market,
                "eth_model": eth_model,
                "market_accuracy": random.uniform(0.85, 0.98),
                "model_confidence": random.uniform(0.90, 0.99),
            }
        )

    return data


def get_tft_predictions():
    """Get TFT model predictions."""
    if tft_predictor:
        try:
            # Get predictions for both BTC and ETH
            btc_predictions = tft_predictor.get_latest_predictions("BTCUSDT")
            eth_predictions = tft_predictor.get_latest_predictions("ETHUSDT")

            # If no cached predictions, generate new ones
            if not btc_predictions:
                btc_predictions = tft_predictor.predict_future("BTCUSDT", hours_ahead=6)
                if btc_predictions:
                    tft_predictor.store_predictions(btc_predictions)

            if not eth_predictions:
                eth_predictions = tft_predictor.predict_future("ETHUSDT", hours_ahead=6)
                if eth_predictions:
                    tft_predictor.store_predictions(eth_predictions)

            return {
                "BTCUSDT": btc_predictions,
                "ETHUSDT": eth_predictions,
                "last_update": datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"Error getting TFT predictions: {e}")

    # Fallback to demo data
    return generate_demo_tft_predictions()


def get_gcn_analysis():
    """Get GCN cross-asset relationship analysis."""
    if gcn_analyzer:
        try:
            # Get cached analysis or generate new one
            analysis = gcn_analyzer.get_stored_analysis()
            if not analysis:
                analysis = gcn_analyzer.analyze_cross_asset_relationships()

            return analysis
        except Exception as e:
            print(f"Error getting GCN analysis: {e}")

    # Fallback to demo data
    return generate_demo_gcn_analysis()


def get_asset_relationships():
    """Get asset relationship matrix."""
    if gcn_analyzer:
        try:
            analysis = gcn_analyzer.get_stored_analysis()
            if analysis and "relationships" in analysis:
                return analysis["relationships"]
        except Exception as e:
            print(f"Error getting asset relationships: {e}")

    # Fallback to demo data
    return generate_demo_relationships()


def get_portfolio_diversification():
    """Get portfolio diversification analysis."""
    if gcn_analyzer:
        try:
            portfolio_assets = ["BTCUSDT", "ETHUSDT"]
            return gcn_analyzer.predict_portfolio_relationships(portfolio_assets)
        except Exception as e:
            print(f"Error getting portfolio diversification: {e}")

    # Fallback to demo data
    return generate_demo_diversification()


def get_risk_analysis():
    """Get comprehensive risk analysis."""
    if risk_manager:
        try:
            # Get risk analysis for main portfolio assets
            btc_analysis = risk_manager.get_stored_analysis("BTCUSDT")
            if not btc_analysis:
                btc_analysis = risk_manager.analyze_comprehensive_risk("BTCUSDT")

            eth_analysis = risk_manager.get_stored_analysis("ETHUSDT")
            if not eth_analysis:
                eth_analysis = risk_manager.analyze_comprehensive_risk("ETHUSDT")

            return {
                "BTCUSDT": btc_analysis,
                "ETHUSDT": eth_analysis,
                "last_update": datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"Error getting risk analysis: {e}")

    # Fallback to demo data
    return generate_demo_risk_analysis()


def get_portfolio_risk():
    """Get portfolio-level risk analysis."""
    if risk_manager:
        try:
            portfolio_symbols = ["BTCUSDT", "ETHUSDT"]
            weights = [0.6, 0.4]  # 60% BTC, 40% ETH
            return risk_manager.calculate_portfolio_risk(portfolio_symbols, weights)
        except Exception as e:
            print(f"Error getting portfolio risk: {e}")

    # Fallback to demo data
    return generate_demo_portfolio_risk()


def get_risk_metrics():
    """Get summary risk metrics."""
    if risk_manager:
        try:
            btc_metrics = (
                risk_manager.redis_client.get("risk_metrics_BTCUSDT")
                if risk_manager.redis_client
                else None
            )
            eth_metrics = (
                risk_manager.redis_client.get("risk_metrics_ETHUSDT")
                if risk_manager.redis_client
                else None
            )

            result = {}
            if btc_metrics:
                result["BTCUSDT"] = json.loads(btc_metrics)
            if eth_metrics:
                result["ETHUSDT"] = json.loads(eth_metrics)

            return result if result else generate_demo_risk_metrics()
        except Exception as e:
            print(f"Error getting risk metrics: {e}")

    # Fallback to demo data
    return generate_demo_risk_metrics()


def get_copula_analysis():
    """Get copula dependency analysis."""
    if copula_analyzer:
        try:
            # Get cached analysis or generate new one
            btc_eth_analysis = copula_analyzer.get_stored_analysis("BTCUSDT-ETHUSDT")
            if not btc_eth_analysis:
                btc_eth_analysis = copula_analyzer.analyze_pairwise_dependencies(
                    "BTCUSDT", "ETHUSDT"
                )

            return btc_eth_analysis
        except Exception as e:
            print(f"Error getting copula analysis: {e}")

    # Fallback to demo data
    return generate_demo_copula_analysis()


def get_portfolio_dependencies():
    """Get portfolio dependency analysis."""
    if copula_analyzer:
        try:
            portfolio_assets = ["BTCUSDT", "ETHUSDT"]
            return copula_analyzer.analyze_portfolio_dependencies(portfolio_assets)
        except Exception as e:
            print(f"Error getting portfolio dependencies: {e}")

    # Fallback to demo data
    return generate_demo_portfolio_dependencies()


def get_dependency_matrix():
    """Get dependency correlation matrix."""
    if copula_analyzer:
        try:
            portfolio_deps = copula_analyzer.analyze_portfolio_dependencies(
                ["BTCUSDT", "ETHUSDT"]
            )
            return portfolio_deps.get("portfolio_summary", {})
        except Exception as e:
            print(f"Error getting dependency matrix: {e}")

    # Fallback to demo data
    return generate_demo_dependency_matrix()


def get_statistical_arbitrage():
    """Get statistical arbitrage analysis."""
    if stat_arb_engine:
        try:
            # Get cached analysis or generate new one
            analysis = stat_arb_engine.get_stored_analysis()
            if not analysis:
                analysis = stat_arb_engine.run_comprehensive_analysis()

            return analysis
        except Exception as e:
            return {
                "error": str(e),
                "pairs_trading": {"viable_pairs": 0, "top_pairs": []},
                "cross_asset_arbitrage": {
                    "total_opportunities": 0,
                    "opportunities": [],
                },
                "strategy_recommendations": [],
                "risk_metrics": {},
            }
    else:
        # Return demo data when engine not available
        return {
            "timestamp": datetime.now().isoformat(),
            "analysis_period": "252 days",
            "assets_analyzed": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"],
            "pairs_trading": {
                "viable_pairs": 2,
                "top_pairs": [
                    {
                        "pair": "BTCUSDT-ETHUSDT",
                        "correlation": 0.854,
                        "formation_score": 87.3,
                        "cointegration_p_value": 0.023,
                        "half_life": 5.2,
                        "current_signal": {
                            "signal": "LONG_SPREAD",
                            "confidence": 0.73,
                            "z_score": -2.15,
                            "reasoning": ["Spread 2.15 std below mean - long spread"],
                        },
                    },
                    {
                        "pair": "ETHUSDT-ADAUSDT",
                        "correlation": 0.712,
                        "formation_score": 68.9,
                        "cointegration_p_value": 0.041,
                        "half_life": 8.7,
                        "current_signal": {
                            "signal": "SHORT_SPREAD",
                            "confidence": 0.64,
                            "z_score": 2.31,
                            "reasoning": ["Spread 2.31 std above mean - short spread"],
                        },
                    },
                ],
            },
            "cross_asset_arbitrage": {
                "total_opportunities": 4,
                "momentum_opportunities": 1,
                "volatility_opportunities": 1,
                "cross_exchange_opportunities": 2,
                "opportunities": [
                    {
                        "type": "cross_exchange",
                        "asset": "BTCUSDT",
                        "buy_exchange": "Binance",
                        "sell_exchange": "Coinbase",
                        "profit_pct": 0.12,
                        "confidence": 0.85,
                        "reasoning": "Price gap: 0.125% between Binance and Coinbase",
                    },
                    {
                        "type": "momentum_reversal",
                        "asset": "ETHUSDT",
                        "signal": "LONG",
                        "confidence": 0.71,
                        "reasoning": "Strong short-term momentum vs weak long-term, oversold RSI",
                    },
                ],
            },
            "strategy_recommendations": [
                {
                    "type": "pairs_trading",
                    "pair": "BTCUSDT-ETHUSDT",
                    "signal": "LONG_SPREAD",
                    "confidence": 0.73,
                    "reasoning": "Spread 2.15 std below mean - long spread",
                },
                {
                    "type": "cross_exchange",
                    "asset": "BTCUSDT",
                    "signal": "ARBITRAGE",
                    "confidence": 0.85,
                    "reasoning": "Price gap: 0.125% between Binance and Coinbase",
                },
            ],
            "risk_metrics": {
                "average_correlation": 0.623,
                "max_correlation": 0.854,
                "pairs_avg_half_life": 6.95,
                "high_confidence_opportunities": 3,
                "pairs_risk_score": 0.20,
            },
        }


def get_slippage_monitoring():
    """Get real-time slippage monitoring data."""
    if slippage_monitor:
        try:
            return slippage_monitor.get_monitoring_dashboard_data()
        except Exception as e:
            return {"error": str(e)}
    else:
        # Return demo slippage monitoring data
        return {
            "market_conditions": {
                "volatility_regime": "NORMAL",
                "liquidity_score": 0.75,
                "spread_environment": "NORMAL",
                "market_stress_level": 0.2,
                "timestamp": datetime.now().isoformat(),
            },
            "performance_analysis": {
                "total_executions": 47,
                "avg_slippage_bps": 2.8,
                "median_slippage_bps": 2.1,
                "max_slippage_bps": 8.3,
                "std_slippage_bps": 1.9,
                "positive_slippage_rate": 0.53,
                "high_slippage_rate": 0.04,
                "venue_breakdown": {
                    "binance": {"count": 18, "avg_slippage": 2.3},
                    "coinbase": {"count": 15, "avg_slippage": 3.1},
                    "alpaca": {"count": 14, "avg_slippage": 3.2},
                },
            },
            "recent_alerts": [
                {
                    "alert_type": "HIGH_SLIPPAGE",
                    "symbol": "ETHUSDT",
                    "venue": "coinbase",
                    "slippage_bps": 12.4,
                    "threshold_bps": 10.0,
                    "severity": "MEDIUM",
                    "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                }
            ],
            "venue_performance": {
                "binance": {"avg_slippage_bps": 2.3, "execution_count": 18},
                "coinbase": {"avg_slippage_bps": 3.1, "execution_count": 15},
                "alpaca": {"avg_slippage_bps": 3.2, "execution_count": 14},
            },
            "total_executions": 47,
            "alert_count": 3,
            "last_updated": datetime.now().isoformat(),
        }


def get_execution_optimization():
    """Get execution optimization recommendations."""
    if smart_order_router:
        try:
            # Generate demo optimization data
            symbols = ["BTCUSDT", "ETHUSDT", "NVDA"]
            optimization_data = {}

            for symbol in symbols:
                optimization_data[symbol] = {
                    "success": True,
                    "order_type": "marketable_limit",
                    "venue": "binance" if "USDT" in symbol else "alpaca",
                    "predicted_edge_bps": 20.0,
                    "execution_costs": {"total_cost_bps": 5.5},
                    "confidence": 0.75,
                }

            return {
                "optimization_plans": optimization_data,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"error": str(e)}
    else:
        return {
            "optimization_plans": {
                "BTCUSDT": {
                    "success": True,
                    "order_type": "marketable_limit",
                    "execution_price": 45002.1,
                    "predicted_edge_bps": 18.5,
                    "execution_costs": {
                        "spread_cost_bps": 2.5,
                        "expected_slippage_bps": 2.1,
                        "market_impact_bps": 0.8,
                        "total_cost_bps": 5.4,
                    },
                    "venue": "binance",
                    "confidence": 0.78,
                },
                "ETHUSDT": {
                    "success": True,
                    "order_type": "limit",
                    "execution_price": 3001.5,
                    "predicted_edge_bps": 15.2,
                    "execution_costs": {"total_cost_bps": 7.1},
                    "venue": "coinbase",
                    "confidence": 0.71,
                },
            },
            "timestamp": datetime.now().isoformat(),
        }


def get_enhanced_risk_metrics():
    """Get enhanced risk metrics with cost analysis."""
    if enhanced_risk_harmonizer:
        try:
            return {
                "enhanced_risk_metrics": {
                    "BTCUSDT": {
                        "raw_edge_bps": 18.3,
                        "net_edge_bps": 12.8,
                        "total_cost_bps": 5.5,
                        "position_size_usd": 8500,
                        "kelly_fraction": 0.127,
                    }
                },
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"error": str(e)}
    else:
        return {
            "enhanced_risk_metrics": {
                "BTCUSDT": {
                    "raw_edge_bps": 18.3,
                    "cost_adjusted_edge_bps": 12.8,
                    "net_edge_bps": 12.8,
                    "total_cost_bps": 5.5,
                    "position_size_usd": 8500,
                    "position_pct": 0.085,
                    "kelly_fraction": 0.127,
                    "confidence": 0.75,
                },
                "ETHUSDT": {
                    "raw_edge_bps": 15.7,
                    "net_edge_bps": 9.1,
                    "total_cost_bps": 6.6,
                    "position_size_usd": 6200,
                    "kelly_fraction": 0.098,
                },
            },
            "cost_model": {
                "spread_cost_bps": 2.5,
                "slippage_bps": 3.0,
                "total_avg_cost_bps": 5.3,
            },
            "timestamp": datetime.now().isoformat(),
        }


def generate_demo_tft_predictions():
    """Generate demo TFT predictions."""
    current_time = datetime.now()

    btc_base = 117800
    eth_base = 3580

    btc_predictions = []
    eth_predictions = []

    for i in range(6):  # 6 hour forecast
        timestamp = current_time + timedelta(hours=i + 1)

        # BTC predictions with trend and uncertainty
        btc_pred = btc_base + (i * 50) + random.uniform(-200, 300)
        btc_lower = btc_pred - random.uniform(100, 300)
        btc_upper = btc_pred + random.uniform(100, 300)

        # ETH predictions
        eth_pred = eth_base + (i * 2) + random.uniform(-20, 30)
        eth_lower = eth_pred - random.uniform(10, 30)
        eth_upper = eth_pred + random.uniform(10, 30)

        btc_predictions.append(btc_pred)
        eth_predictions.append(eth_pred)

    timestamps = [(current_time + timedelta(hours=i + 1)).isoformat() for i in range(6)]

    return {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "timestamps": timestamps,
            "predictions": btc_predictions,
            "lower_bound": [p - random.uniform(100, 300) for p in btc_predictions],
            "upper_bound": [p + random.uniform(100, 300) for p in btc_predictions],
            "confidence_interval": 80,
            "model_type": "TFT",
            "prediction_horizon": 6,
        },
        "ETHUSDT": {
            "symbol": "ETHUSDT",
            "timestamps": timestamps,
            "predictions": eth_predictions,
            "lower_bound": [p - random.uniform(10, 30) for p in eth_predictions],
            "upper_bound": [p + random.uniform(10, 30) for p in eth_predictions],
            "confidence_interval": 80,
            "model_type": "TFT",
            "prediction_horizon": 6,
        },
        "last_update": datetime.now().isoformat(),
    }


def generate_demo_gcn_analysis():
    """Generate demo GCN analysis."""
    assets = [
        "BTCUSDT",
        "ETHUSDT",
        "ADAUSDT",
        "DOTUSDT",
        "LINKUSDT",
        "LTCUSDT",
        "XRPUSDT",
        "BCHUSDT",
        "BNBUSDT",
        "SOLUSDT",
    ]

    # Generate random predictions
    predictions = {}
    for asset in assets:
        base_price = {
            "BTCUSDT": 117800,
            "ETHUSDT": 3580,
            "ADAUSDT": 0.5,
            "DOTUSDT": 8.0,
            "LINKUSDT": 15.0,
        }.get(asset, 100)
        pred_values = [base_price * (1 + random.uniform(-0.05, 0.05)) for _ in range(6)]

        predictions[asset] = {
            "current_price": base_price,
            "predictions": pred_values,
            "prediction_hours": list(range(1, 7)),
            "expected_return": random.uniform(-0.03, 0.05),
            "volatility_forecast": random.uniform(0.02, 0.08),
        }

    # Generate relationships
    strong_correlations = [
        {
            "asset1": "BTCUSDT",
            "asset2": "ETHUSDT",
            "strength": 0.85,
            "relationship_type": "positive",
        },
        {
            "asset1": "ETHUSDT",
            "asset2": "ADAUSDT",
            "strength": 0.75,
            "relationship_type": "moderate",
        },
        {
            "asset1": "BTCUSDT",
            "asset2": "LTCUSDT",
            "strength": 0.78,
            "relationship_type": "moderate",
        },
    ]

    return {
        "timestamp": datetime.now().isoformat(),
        "model_type": "Graph Convolutional Network",
        "assets": assets,
        "predictions": predictions,
        "relationships": {
            "strong_correlations": strong_correlations,
            "average_connectivity": 0.65,
        },
        "graph_metrics": {
            "num_nodes": len(assets),
            "num_edges": 15,
            "density": 0.33,
            "clustering_coefficient": 0.42,
            "centrality_scores": {asset: random.uniform(0.1, 0.9) for asset in assets},
        },
        "market_insights": {
            "most_connected_asset": "BTCUSDT",
            "market_cohesion": 0.65,
            "volatility_cluster": ["BTCUSDT", "ETHUSDT"],
            "growth_leaders": ["ADAUSDT", "DOTUSDT"],
        },
    }


def generate_demo_relationships():
    """Generate demo relationship data."""
    return {
        "strong_correlations": [
            {
                "asset1": "BTCUSDT",
                "asset2": "ETHUSDT",
                "strength": 0.85,
                "relationship_type": "positive",
            },
            {
                "asset1": "ETHUSDT",
                "asset2": "ADAUSDT",
                "strength": 0.75,
                "relationship_type": "moderate",
            },
        ],
        "average_connectivity": 0.65,
    }


def generate_demo_diversification():
    """Generate demo diversification analysis."""
    return {
        "portfolio_assets": ["BTCUSDT", "ETHUSDT"],
        "relationships": {
            "BTCUSDT-ETHUSDT": {
                "strength": 0.85,
                "diversification_benefit": 0.15,
                "recommendation": "maintain",
            }
        },
        "diversification_score": 0.15,
        "timestamp": datetime.now().isoformat(),
    }


def generate_demo_risk_analysis():
    """Generate demo risk analysis."""
    assets = ["BTCUSDT", "ETHUSDT"]
    base_prices = {"BTCUSDT": 117800, "ETHUSDT": 3580}

    result = {}
    for asset in assets:
        base_price = base_prices[asset]

        result[asset] = {
            "symbol": asset,
            "timestamp": datetime.now().isoformat(),
            "risk_metrics": {
                "current_price": base_price,
                "daily_volatility": random.uniform(0.02, 0.05),
                "annualized_volatility": random.uniform(0.4, 0.8),
                "var_95_historical": base_price * random.uniform(0.03, 0.08),
                "es_95_historical": base_price * random.uniform(0.04, 0.12),
                "var_95_evt": base_price * random.uniform(0.035, 0.09),
                "es_95_evt": base_price * random.uniform(0.045, 0.15),
                "tail_risk_ratio": random.uniform(1.2, 1.8),
            },
            "expected_shortfall": {
                "95.0%": {
                    "historical": {
                        "es": random.uniform(0.04, 0.08),
                        "var": random.uniform(0.03, 0.06),
                        "confidence_level": 0.95,
                    },
                    "parametric_normal": {
                        "es": random.uniform(0.035, 0.075),
                        "var": random.uniform(0.025, 0.055),
                    },
                    "cornish_fisher": {
                        "es": random.uniform(0.042, 0.085),
                        "var": random.uniform(0.032, 0.065),
                    },
                }
            },
            "extreme_value_theory": {
                "peaks_over_threshold": {
                    "fitted": True,
                    "threshold": random.uniform(0.02, 0.05),
                    "parameters": {
                        "shape": random.uniform(-0.2, 0.3),
                        "scale": random.uniform(0.01, 0.03),
                    },
                }
            },
            "recommendations": [
                "📊 Moderate tail risk detected",
                "⚡ Consider position size adjustment",
            ],
        }

    result["last_update"] = datetime.now().isoformat()
    return result


def generate_demo_portfolio_risk():
    """Generate demo portfolio risk analysis."""
    return {
        "timestamp": datetime.now().isoformat(),
        "portfolio_composition": {"BTCUSDT": 0.6, "ETHUSDT": 0.4},
        "portfolio_risk": {
            "portfolio_var_95": random.uniform(8000, 12000),
            "portfolio_es_95": random.uniform(10000, 15000),
            "sum_individual_var": random.uniform(9000, 14000),
            "sum_individual_es": random.uniform(12000, 18000),
        },
        "diversification_benefit": {
            "var_reduction": random.uniform(0.1, 0.3),
            "es_reduction": random.uniform(0.15, 0.35),
        },
    }


def generate_demo_risk_metrics():
    """Generate demo risk metrics."""
    return {
        "BTCUSDT": {
            "current_price": 117800,
            "daily_volatility": 0.035,
            "var_95_historical": 4500,
            "es_95_historical": 6200,
            "tail_risk_ratio": 1.38,
        },
        "ETHUSDT": {
            "current_price": 3580,
            "daily_volatility": 0.042,
            "var_95_historical": 180,
            "es_95_historical": 245,
            "tail_risk_ratio": 1.36,
        },
    }


def generate_demo_copula_analysis():
    """Generate demo copula analysis."""
    return {
        "asset_pair": "BTCUSDT-ETHUSDT",
        "timestamp": datetime.now().isoformat(),
        "copula_models": {
            "gaussian": {
                "type": "gaussian",
                "parameters": {"rho": 0.78},
                "log_likelihood": -125.4,
                "aic": 252.8,
                "bic": 256.2,
                "fitted": True,
            },
            "student_t": {
                "type": "student_t",
                "parameters": {"rho": 0.76, "nu": 4.5},
                "log_likelihood": -122.1,
                "aic": 248.2,
                "bic": 254.8,
                "fitted": True,
            },
            "clayton": {
                "type": "clayton",
                "parameters": {"theta": 1.8},
                "log_likelihood": -130.2,
                "aic": 262.4,
                "bic": 265.8,
                "fitted": True,
            },
            "best_copula": "student_t",
            "model_comparison": {
                "gaussian": {"aic": 252.8, "bic": 256.2},
                "student_t": {"aic": 248.2, "bic": 254.8},
                "clayton": {"aic": 262.4, "bic": 265.8},
            },
        },
        "dependency_measures": {
            "pearson_correlation": 0.75,
            "spearman_correlation": 0.73,
            "kendall_tau": 0.52,
            "upper_tail_dependence": 0.31,
            "lower_tail_dependence": 0.28,
            "mutual_information": 0.45,
        },
        "simulations": {
            "copula_type": "student_t",
            "n_simulations": 1000,
            "simulated_pearson": 0.74,
            "simulated_spearman": 0.72,
        },
        "recommendations": [
            "📊 Student-t copula: Symmetric tail dependence detected",
            "🔗 Strong dependency detected (ρ=0.730) - consider diversification",
            "📈 Moderate upper tail dependence - assets move together in bull markets",
        ],
    }


def generate_demo_portfolio_dependencies():
    """Generate demo portfolio dependencies."""
    return {
        "timestamp": datetime.now().isoformat(),
        "assets": ["BTCUSDT", "ETHUSDT"],
        "pairwise_analysis": {"BTCUSDT-ETHUSDT": generate_demo_copula_analysis()},
        "portfolio_summary": {
            "average_correlation": 0.73,
            "max_correlation": 0.73,
            "min_correlation": 0.73,
            "dependency_matrix": [[1.0, 0.73], [0.73, 1.0]],
        },
    }


def generate_demo_dependency_matrix():
    """Generate demo dependency matrix."""
    return {
        "average_correlation": 0.73,
        "max_correlation": 0.73,
        "min_correlation": 0.73,
        "dependency_matrix": [[1.0, 0.73], [0.73, 1.0]],
    }


@app.get("/")
async def enhanced_dashboard():
    """Serve the enhanced dashboard."""
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Real-time Trading Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
            color: #ffffff;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 20px;
            position: relative;
        }
        
        .header h1 {
            font-size: 2.2em;
            margin-bottom: 15px;
            background: linear-gradient(45deg, #4CAF50, #2196F3, #FF9800);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header-controls {
            position: absolute;
            top: 0;
            right: 20px;
            display: flex;
            gap: 10px;
        }
        
        .header-btn {
            background: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: bold;
            transition: background 0.2s ease;
        }
        
        .header-btn:hover {
            background: #1976D2;
        }
        
        .refresh-btn {
            background: #00BCD4;
        }
        
        .refresh-btn:hover {
            background: #0097A7;
        }
        
        .system-status {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-bottom: 15px;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-light {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            box-shadow: 0 0 8px rgba(76, 175, 80, 0.6);
        }
        
        .model-controls {
            display: flex;
            gap: 10px;
            margin-left: 20px;
        }
        
        .model-btn {
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            font-weight: bold;
            transition: all 0.2s ease;
        }
        
        .model-running {
            background: #4CAF50;
            color: white;
        }
        
        .model-stop {
            background: #f44336;
            color: white;
        }
        
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 10px 20px;
            background: rgba(15, 15, 35, 0.8);
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        
        .status-item {
            text-align: center;
        }
        
        .status-label {
            font-size: 12px;
            opacity: 0.8;
            display: block;
        }
        
        .status-value {
            font-size: 16px;
            font-weight: bold;
            display: block;
            margin-top: 2px;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .position-card {
            background: rgba(15, 15, 35, 0.8);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .position-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 15px;
            font-size: 16px;
            font-weight: bold;
        }
        
        .trading-controls {
            margin-top: 15px;
        }
        
        .control-section {
            margin-bottom: 15px;
        }
        
        .control-label {
            font-size: 13px;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 8px;
            display: block;
        }
        
        .button-group {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }
        
        .trade-btn {
            background: rgba(15, 15, 35, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.2s ease;
            min-width: 45px;
        }
        
        .trade-btn:hover {
            background: rgba(15, 15, 35, 0.9);
        }
        
        .sell-btn {
            background: rgba(244, 67, 54, 0.3);
            border-color: #f44336;
        }
        
        .sell-btn:hover {
            background: rgba(244, 67, 54, 0.5);
        }
        
        .buy-btn {
            background: rgba(76, 175, 80, 0.3);
            border-color: #4CAF50;
        }
        
        .buy-btn:hover {
            background: rgba(76, 175, 80, 0.5);
        }
        
        .action-btn {
            background: #f44336;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: bold;
            margin-top: 8px;
            width: 100%;
        }
        
        .action-btn.buy-action {
            background: #4CAF50;
        }
        
        .allocation-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-bottom: 15px;
        }
        
        .allocation-btn {
            background: rgba(15, 15, 35, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 8px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            text-align: center;
            transition: all 0.2s ease;
        }
        
        .allocation-btn:hover {
            background: rgba(33, 150, 243, 0.5);
        }
        
        .card {
            background: rgba(15, 15, 35, 0.8);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .symbol {
            font-size: 20px;
            font-weight: bold;
        }
        
        .price-change {
            font-size: 14px;
            padding: 2px 8px;
            border-radius: 4px;
        }
        
        .price-display {
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            margin: 15px 0;
            color: #4CAF50;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 15px;
        }
        
        .metric-item {
            text-align: center;
            padding: 8px;
            background: rgba(15, 15, 35, 0.8);
            border-radius: 8px;
        }
        
        .metric-label {
            font-size: 11px;
            opacity: 0.8;
            display: block;
        }
        
        .metric-value {
            font-size: 14px;
            font-weight: bold;
            display: block;
            margin-top: 2px;
        }
        
        .positive {
            color: #4CAF50;
        }
        
        .negative {
            color: #f44336;
        }
        
        .neutral {
            color: #FF9800;
        }
        
        .signals-container {
            max-height: 300px;
            overflow-y: auto;
            padding-right: 8px;
        }
        
        .signals-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .signals-container::-webkit-scrollbar-track {
            background: rgba(15, 15, 35, 0.8);
            border-radius: 3px;
        }
        
        .signals-container::-webkit-scrollbar-thumb {
            background: rgba(15, 15, 35, 0.9);
            border-radius: 3px;
        }
        
        .signals-container::-webkit-scrollbar-thumb:hover {
            background: rgba(15, 15, 35, 0.9);
        }
        
        .signal-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            margin: 5px 0;
            background: rgba(15, 15, 35, 0.7);
            border-radius: 6px;
            border-left: 3px solid;
        }
        
        .signal-buy {
            border-left-color: #4CAF50;
        }
        
        .signal-sell {
            border-left-color: #f44336;
        }
        
        .signal-hold {
            border-left-color: #FF9800;
        }
        
        .chart-container {
            background: rgba(26, 26, 46, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }
        
        .chart-title {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
        }
        
        .bottom-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .loading {
            color: #FF9800;
        }
        
        .success {
            color: #4CAF50;
        }
        
        .error {
            color: #f44336;
        }
        
        /* News Sentiment Styles */
        .sentiment-gauge {
            text-align: center;
            padding: 20px;
        }
        
        .sentiment-score {
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 8px;
        }
        
        .sentiment-label {
            font-size: 12px;
            opacity: 0.8;
        }
        
        .sentiment-breakdown {
            margin: 15px 0;
        }
        
        .sentiment-bar {
            height: 8px;
            background: rgba(15, 15, 35, 0.8);
            border-radius: 4px;
            margin: 8px 0;
            overflow: hidden;
            position: relative;
        }
        
        .positive-bar, .negative-bar {
            height: 100%;
            transition: width 0.3s ease;
        }
        
        .positive-bar {
            background: #4CAF50;
            float: left;
        }
        
        .negative-bar {
            background: #f44336;
            float: right;
        }
        
        .sentiment-counts {
            font-size: 12px;
            text-align: center;
            margin: 5px 0;
            opacity: 0.8;
        }
        
        .key-phrases {
            font-size: 11px;
            opacity: 0.7;
            margin-top: 10px;
            line-height: 1.4;
        }
        
        .news-feed-container {
            margin-top: 20px;
        }
        
        .news-feed {
            max-height: 300px;
            overflow-y: auto;
            background: rgba(15, 15, 35, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 15px;
        }
        
        .news-item {
            padding: 12px;
            margin: 8px 0;
            background: rgba(22, 33, 62, 0.5);
            border-radius: 6px;
            border-left: 3px solid;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.2s ease;
        }
        
        .news-item:hover {
            background: rgba(22, 33, 62, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transform: translateX(3px);
        }
        
        .news-positive {
            border-left-color: #4CAF50;
        }
        
        .news-negative {
            border-left-color: #f44336;
        }
        
        .news-neutral {
            border-left-color: #FF9800;
        }
        
        .news-title {
            font-size: 13px;
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
        
        .news-source {
            font-weight: bold;
        }
        
        .news-time {
            opacity: 0.6;
        }
        
        .sentiment-indicator {
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: bold;
        }
        
        .sentiment-positive {
            background: rgba(76, 175, 80, 0.3);
            color: #4CAF50;
        }
        
        .sentiment-negative {
            background: rgba(244, 67, 54, 0.3);
            color: #f44336;
        }
        
        .sentiment-neutral {
            background: rgba(255, 152, 0, 0.3);
            color: #FF9800;
        }
        
        @media (max-width: 1200px) {
            .main-grid {
                grid-template-columns: 1fr 1fr;
            }
            
            .header-controls {
                position: relative;
                top: auto;
                right: auto;
                margin-bottom: 10px;
            }
        }
        
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            
            .system-status {
                flex-direction: column;
                gap: 10px;
            }
            
            .model-controls {
                margin-left: 0;
            }
            
            .button-group {
                grid-template-columns: repeat(3, 1fr);
            }
            
            .allocation-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-controls">
                <button class="header-btn refresh-btn" onclick="refreshDashboard()">🔄 Refresh</button>
                <button class="header-btn" onclick="clearData()">🗑️ Clear</button>
            </div>
            <h1>🚀 Real-time Trading Dashboard</h1>
            
            <div class="system-status">
                <div class="status-indicator">
                    <div class="status-light"></div>
                    <span>System Online</span>
                </div>
                <div class="model-controls">
                    <button class="model-btn model-running" onclick="toggleModel()">Model Running</button>
                    <button class="model-btn model-stop" onclick="stopModel()">Stop Model</button>
                </div>
            </div>
            
            <div id="status" class="loading">Loading enhanced data...</div>
        </div>
        
        <div class="main-grid">
            <!-- BTC Position Card -->
            <div class="position-card">
                <div class="position-header">
                    <span>₿ BTCUSDT Position</span>
                </div>
                <div class="price-display" id="btcPrice">$0.00</div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <span class="metric-label">Position</span>
                        <span class="metric-value" id="btcPosition">0.00</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Entry</span>
                        <span class="metric-value" id="btcEntry">$0.00</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Current Value</span>
                        <span class="metric-value" id="btcValue">$0.00</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Unrealized P&L</span>
                        <span class="metric-value" id="btcUnrealizedPnL">$0.00</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Total P&L</span>
                        <span class="metric-value" id="btcTotalPnL">$0.00</span>
                    </div>
                </div>
                
                <div class="trading-controls">
                    <div class="control-section">
                        <span class="control-label">💰 Sell Position</span>
                        <div style="margin: 10px 0;">
                            <label for="btc-sell-amount" style="display: block; margin-bottom: 5px; font-weight: bold;">📉 Enter Percentage (%):</label>
                            <input type="number" id="btc-sell-amount" placeholder="Enter % to sell" min="1" max="100" step="1" 
                                   style="width: 100%; padding: 8px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px;">
                        </div>
                        <div class="button-group">
                            <button class="trade-btn sell-btn" onclick="setSellAmount('BTC', 5)">5%</button>
                            <button class="trade-btn sell-btn" onclick="setSellAmount('BTC', 10)">10%</button>
                            <button class="trade-btn sell-btn" onclick="setSellAmount('BTC', 25)">25%</button>
                            <button class="trade-btn sell-btn" onclick="setSellAmount('BTC', 50)">50%</button>
                            <button class="trade-btn sell-btn" onclick="setSellAmount('BTC', 75)">75%</button>
                            <button class="trade-btn sell-btn" onclick="setSellAmount('BTC', 100)">100%</button>
                        </div>
                        <button class="action-btn" onclick="executeSell('BTC')">Sell</button>
                    </div>
                    
                    <div class="control-section">
                        <span class="control-label">🚀 Buy More Position</span>
                        <div style="margin: 10px 0;">
                            <label for="btc-buy-amount" style="display: block; margin-bottom: 5px; font-weight: bold;">💰 Enter Amount ($):</label>
                            <input type="number" id="btc-buy-amount" placeholder="Enter amount to buy" min="10" step="10" 
                                   style="width: 100%; padding: 8px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px;">
                        </div>
                        <div class="button-group">
                            <button class="trade-btn buy-btn" onclick="setBuyAmount('BTC', 10)">$10</button>
                            <button class="trade-btn buy-btn" onclick="setBuyAmount('BTC', 25)">$25</button>
                            <button class="trade-btn buy-btn" onclick="setBuyAmount('BTC', 50)">$50</button>
                            <button class="trade-btn buy-btn" onclick="setBuyAmount('BTC', 100)">$100</button>
                            <button class="trade-btn buy-btn" onclick="setBuyAmount('BTC', 200)">$200</button>
                            <button class="trade-btn buy-btn" onclick="setBuyAmount('BTC', 500)">$500</button>
                        </div>
                        <button class="action-btn buy-action" onclick="executeBuy('BTC')">Buy</button>
                    </div>
                </div>
            </div>
            
            <!-- ETH Position Card -->
            <div class="position-card">
                <div class="position-header">
                    <span>Ξ ETHUSDT Position</span>
                </div>
                <div class="price-display" id="ethPrice">$0.00</div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <span class="metric-label">Position</span>
                        <span class="metric-value" id="ethPosition">0.00</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Entry</span>
                        <span class="metric-value" id="ethEntry">$0.00</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Current Value</span>
                        <span class="metric-value" id="ethValue">$0.00</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Unrealized P&L</span>
                        <span class="metric-value" id="ethUnrealizedPnL">$0.00</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Total P&L</span>
                        <span class="metric-value" id="ethTotalPnL">$0.00</span>
                    </div>
                </div>
                
                <div class="trading-controls">
                    <div class="control-section">
                        <span class="control-label">💰 Sell Position</span>
                        <div style="margin: 10px 0;">
                            <label for="eth-sell-amount" style="display: block; margin-bottom: 5px; font-weight: bold;">📉 Enter Percentage (%):</label>
                            <input type="number" id="eth-sell-amount" placeholder="Enter % to sell" min="1" max="100" step="1" 
                                   style="width: 100%; padding: 8px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px;">
                        </div>
                        <div class="button-group">
                            <button class="trade-btn sell-btn" onclick="setSellAmount('ETH', 5)">5%</button>
                            <button class="trade-btn sell-btn" onclick="setSellAmount('ETH', 10)">10%</button>
                            <button class="trade-btn sell-btn" onclick="setSellAmount('ETH', 25)">25%</button>
                            <button class="trade-btn sell-btn" onclick="setSellAmount('ETH', 50)">50%</button>
                            <button class="trade-btn sell-btn" onclick="setSellAmount('ETH', 75)">75%</button>
                            <button class="trade-btn sell-btn" onclick="setSellAmount('ETH', 100)">100%</button>
                        </div>
                        <button class="action-btn" onclick="executeSell('ETH')">Sell</button>
                    </div>
                    
                    <div class="control-section">
                        <span class="control-label">🚀 Buy More Position</span>
                        <div style="margin: 10px 0;">
                            <label for="eth-buy-amount" style="display: block; margin-bottom: 5px; font-weight: bold;">💰 Enter Amount ($):</label>
                            <input type="number" id="eth-buy-amount" placeholder="Enter amount to buy" min="10" step="10" 
                                   style="width: 100%; padding: 8px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px;">
                        </div>
                        <div class="button-group">
                            <button class="trade-btn buy-btn" onclick="setBuyAmount('ETH', 10)">$10</button>
                            <button class="trade-btn buy-btn" onclick="setBuyAmount('ETH', 25)">$25</button>
                            <button class="trade-btn buy-btn" onclick="setBuyAmount('ETH', 50)">$50</button>
                            <button class="trade-btn buy-btn" onclick="setBuyAmount('ETH', 100)">$100</button>
                            <button class="trade-btn buy-btn" onclick="setBuyAmount('ETH', 200)">$200</button>
                            <button class="trade-btn buy-btn" onclick="setBuyAmount('ETH', 500)">$500</button>
                        </div>
                        <button class="action-btn buy-action" onclick="executeBuy('ETH')">Buy</button>
                    </div>
                </div>
            </div>
            
            <!-- Alpha Signals Card -->
            <div class="position-card">
                <div class="position-header">
                    <span>🧠 Alpha Signals</span>
                </div>
                <div class="signals-container" id="signalsContainer">
                    Loading signals...
                </div>
            </div>
            
            <!-- Trading Settings Card -->
            <div class="position-card">
                <div class="position-header">
                    <span>⚙️ Trading Settings</span>
                </div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <span class="metric-label">Total Portfolio</span>
                        <span class="metric-value" id="totalPortfolio">$0.00</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Total Invested</span>
                        <span class="metric-value" id="totalInvested">$0.00</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Profit %</span>
                        <span class="metric-value" id="profitPercent">0.00%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Available Capital</span>
                        <span class="metric-value" id="availableCapital">$0.00</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Trading Allocation</span>
                        <span class="metric-value" id="tradingAllocation">100%</span>
                    </div>
                </div>
                
                <div class="trading-controls">
                    <div class="control-section">
                        <span class="control-label">📈 Trading Allocation</span>
                        <div class="allocation-grid">
                            <button class="allocation-btn" onclick="setAllocation(5)">5%</button>
                            <button class="allocation-btn" onclick="setAllocation(10)">10%</button>
                            <button class="allocation-btn" onclick="setAllocation(25)">25%</button>
                            <button class="allocation-btn" onclick="setAllocation(50)">50%</button>
                            <button class="allocation-btn" onclick="setAllocation(75)">75%</button>
                            <button class="allocation-btn" onclick="setAllocation(100)">100%</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📈 Portfolio Performance (Last 24 Hours)</div>
            <div id="performanceChart" style="width: 100%; height: 400px;"></div>
        </div>
        
        <!-- News Sentiment Section -->
        <div class="chart-container">
            <div class="chart-title">📰 Market Sentiment & News Analysis</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                <!-- Market Sentiment Overview -->
                <div class="card">
                    <div class="card-header">
                        <span style="font-size: 16px; font-weight: bold;">📊 Market Sentiment</span>
                    </div>
                    <div id="marketSentiment">
                        <div class="sentiment-gauge" id="sentimentGauge">
                            <div class="sentiment-score" id="marketSentimentScore">0.0</div>
                            <div class="sentiment-label">Overall Sentiment</div>
                        </div>
                        <div class="metrics-grid" style="margin-top: 15px;">
                            <div class="metric-item">
                                <span class="metric-label">Confidence</span>
                                <span class="metric-value" id="marketConfidence">0%</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">News Count</span>
                                <span class="metric-value" id="totalNewsCount">0</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- BTC Sentiment -->
                <div class="card">
                    <div class="card-header">
                        <span style="font-size: 16px; font-weight: bold;">₿ BTC Sentiment</span>
                    </div>
                    <div id="btcSentiment">
                        <div class="sentiment-score" id="btcSentimentScore">0.0</div>
                        <div class="sentiment-breakdown">
                            <div class="sentiment-bar">
                                <div class="positive-bar" id="btcPositiveBar"></div>
                                <div class="negative-bar" id="btcNegativeBar"></div>
                            </div>
                            <div class="sentiment-counts" id="btcSentimentCounts">0+ 0- 0=</div>
                        </div>
                        <div class="key-phrases" id="btcKeyPhrases">Loading...</div>
                    </div>
                </div>
                
                <!-- ETH Sentiment -->
                <div class="card">
                    <div class="card-header">
                        <span style="font-size: 16px; font-weight: bold;">Ξ ETH Sentiment</span>
                    </div>
                    <div id="ethSentiment">
                        <div class="sentiment-score" id="ethSentimentScore">0.0</div>
                        <div class="sentiment-breakdown">
                            <div class="sentiment-bar">
                                <div class="positive-bar" id="ethPositiveBar"></div>
                                <div class="negative-bar" id="ethNegativeBar"></div>
                            </div>
                            <div class="sentiment-counts" id="ethSentimentCounts">0+ 0- 0=</div>
                        </div>
                        <div class="key-phrases" id="ethKeyPhrases">Loading...</div>
                    </div>
                </div>
            </div>
            
            <!-- Recent News Feed -->
            <div class="news-feed-container">
                <div style="font-size: 16px; font-weight: bold; margin-bottom: 15px;">📱 Recent Crypto News</div>
                <div class="news-feed" id="newsFeed">
                    Loading recent news...
                </div>
            </div>
        </div>

        <!-- TFT Prediction Charts -->
        <div class="chart-container">
            <div class="chart-title">🔮 Temporal Fusion Transformer (TFT) Predictions</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                <div class="chart-container">
                    <div class="chart-title">₿ BTC TFT Predictions (6 Hours)</div>
                    <div id="btcTftChart" style="width: 100%; height: 300px;"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Ξ ETH TFT Predictions (6 Hours)</div>
                    <div id="ethTftChart" style="width: 100%; height: 300px;"></div>
                </div>
            </div>
        </div>

        <!-- GCN Cross-Asset Analysis -->
        <div class="section">
            <h2>🔗 Cross-Asset Relationship Analysis (GCN)</h2>
            <div class="charts-row">
                <div class="chart-container">
                    <div class="chart-title">🔗 Asset Correlation Network</div>
                    <div id="assetNetworkChart" style="width: 100%; height: 400px;"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">📊 Portfolio Diversification Analysis</div>
                    <div id="diversificationChart" style="width: 100%; height: 400px;"></div>
                </div>
            </div>
            <div class="charts-row">
                <div class="chart-container">
                    <div class="chart-title">📈 Cross-Asset Predictions</div>
                    <div id="crossAssetChart" style="width: 100%; height: 300px;"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">⚡ Market Insights</div>
                    <div id="marketInsightsContainer" style="padding: 20px; background: rgba(15, 15, 35, 0.8); border-radius: 10px;">
                        <div id="marketInsights">Loading market insights...</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ES/EVT Risk Management -->
        <div class="section">
            <h2>⚠️ Risk Management: Expected Shortfall & EVT</h2>
            <div class="charts-row">
                <div class="chart-container">
                    <div class="chart-title">📊 VaR vs Expected Shortfall</div>
                    <div id="varEsChart" style="width: 100%; height: 400px;"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">🎯 Extreme Value Theory Analysis</div>
                    <div id="evtChart" style="width: 100%; height: 400px;"></div>
                </div>
            </div>
            <div class="charts-row">
                <div class="chart-container">
                    <div class="chart-title">💼 Portfolio Risk Metrics</div>
                    <div id="portfolioRiskChart" style="width: 100%; height: 300px;"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">📈 Risk Analysis Summary</div>
                    <div id="riskSummaryContainer" style="padding: 20px; background: rgba(15, 15, 35, 0.8); border-radius: 10px;">
                        <div id="riskSummary">Loading risk analysis...</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Copula Correlation Modeling -->
        <div class="section">
            <h2>🔗 Copula-based Correlation Modeling</h2>
            <div class="charts-row">
                <div class="chart-container">
                    <div class="chart-title">📊 Copula Model Comparison</div>
                    <div id="copulaComparisonChart" style="width: 100%; height: 400px;"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">🎯 Dependency Measures</div>
                    <div id="dependencyMeasuresChart" style="width: 100%; height: 400px;"></div>
                </div>
            </div>
            <div class="charts-row">
                <div class="chart-container">
                    <div class="chart-title">🔗 Tail Dependence Analysis</div>
                    <div id="tailDependenceChart" style="width: 100%; height: 300px;"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">📈 Copula Insights</div>
                    <div id="copulaInsightsContainer" style="padding: 20px; background: rgba(15, 15, 35, 0.8); border-radius: 10px;">
                        <div id="copulaInsights">Loading copula analysis...</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Market vs Model Comparison Charts -->
        <div class="chart-container">
            <div class="chart-title">📊 Market vs Model Performance Analysis</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                <!-- 5-Second Comparison Charts -->
                <div class="chart-container">
                    <div class="chart-title">⚡ 5-Second: BTC Market vs Model</div>
                    <div id="btc5sChart" style="width: 100%; height: 250px;"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">⚡ 5-Second: ETH Market vs Model</div>
                    <div id="eth5sChart" style="width: 100%; height: 250px;"></div>
                </div>
                
                <!-- 1-Hour Comparison Charts -->
                <div class="chart-container">
                    <div class="chart-title">🕐 1-Hour: BTC Market vs Model</div>
                    <div id="btc1hChart" style="width: 100%; height: 250px;"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">🕐 1-Hour: ETH Market vs Model</div>
                    <div id="eth1hChart" style="width: 100%; height: 250px;"></div>
                </div>
                
                <!-- 1-Day Comparison Charts -->
                <div class="chart-container">
                    <div class="chart-title">📅 Daily: BTC Market vs Model</div>
                    <div id="btc1dChart" style="width: 100%; height: 250px;"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">📅 Daily: ETH Market vs Model</div>
                    <div id="eth1dChart" style="width: 100%; height: 250px;"></div>
                </div>
            </div>
        </div>

        <div class="bottom-grid">
            <div class="chart-container">
                <div class="chart-title">₿ BTC Price Trend</div>
                <div id="btcChart" style="width: 100%; height: 300px;"></div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Ξ ETH Price Trend</div>
                <div id="ethChart" style="width: 100%; height: 300px;"></div>
            </div>
        </div>
    </div>

    <!-- Trading Log Section -->
    <div class="chart-container">
        <div class="chart-title">📋 Trading Log</div>
        <div id="tradingLog" style="max-height: 200px; overflow-y: auto; background: rgba(15, 15, 35, 0.8); border-radius: 8px; padding: 15px;">
            <div id="tradingLogContent">
                <div style="color: #888; text-align: center; padding: 20px;">
                    Loading trading activity...
                </div>
            </div>
        </div>
    </div>

    <script>
        let updateCount = 0;
        let performanceData = [];
        
        function formatNumber(num, decimals = 2) {
            if (num >= 1e12) return (num / 1e12).toFixed(1) + 'T';
            if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
            if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
            if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
            return num.toFixed(decimals);
        }
        
        // Trading state variables
        let selectedSellAmount = { BTC: 0, ETH: 0 };
        let selectedBuyAmount = { BTC: 0, ETH: 0 };
        let tradingAllocation = 100;
        
        function updatePortfolioDisplay(data) {
            console.log('Updating portfolio display:', data);
            
            const btc = data.BTCUSDT;
            const eth = data.ETHUSDT;
            
            // Update BTC
            document.getElementById('btcPrice').textContent = `$${btc.current_price.toFixed(2)}`;
            document.getElementById('btcPosition').textContent = btc.position_size.toFixed(6);
            document.getElementById('btcEntry').textContent = `$${btc.entry_price.toFixed(2)}`;
            document.getElementById('btcValue').textContent = `$${btc.current_value.toFixed(2)}`;
            
            // Update P&L values
            const btcUnrealizedPnL = document.getElementById('btcUnrealizedPnL');
            btcUnrealizedPnL.textContent = `$${btc.total_pnl.toFixed(2)}`;
            btcUnrealizedPnL.className = `metric-value ${btc.total_pnl >= 0 ? 'positive' : 'negative'}`;
            
            const btcTotalPnL = document.getElementById('btcTotalPnL');
            btcTotalPnL.textContent = `$${btc.total_pnl.toFixed(2)}`;
            btcTotalPnL.className = `metric-value ${btc.total_pnl >= 0 ? 'positive' : 'negative'}`;
            
            // Update ETH
            document.getElementById('ethPrice').textContent = `$${eth.current_price.toFixed(2)}`;
            document.getElementById('ethPosition').textContent = eth.position_size.toFixed(6);
            document.getElementById('ethEntry').textContent = `$${eth.entry_price.toFixed(2)}`;
            document.getElementById('ethValue').textContent = `$${eth.current_value.toFixed(2)}`;
            
            // Update P&L values
            const ethUnrealizedPnL = document.getElementById('ethUnrealizedPnL');
            ethUnrealizedPnL.textContent = `$${eth.total_pnl.toFixed(2)}`;
            ethUnrealizedPnL.className = `metric-value ${eth.total_pnl >= 0 ? 'positive' : 'negative'}`;
            
            const ethTotalPnL = document.getElementById('ethTotalPnL');
            ethTotalPnL.textContent = `$${eth.total_pnl.toFixed(2)}`;
            ethTotalPnL.className = `metric-value ${eth.total_pnl >= 0 ? 'positive' : 'negative'}`;
            
            // Update portfolio summary
            const totalValue = btc.current_value + eth.current_value;
            const totalPnL = btc.total_pnl + eth.total_pnl;
            const totalInvested = (btc.position_size * btc.entry_price) + (eth.position_size * eth.entry_price);
            const profitPct = totalInvested > 0 ? (totalPnL / totalInvested) * 100 : 0;
            
            document.getElementById('totalPortfolio').textContent = `$${totalValue.toFixed(2)}`;
            document.getElementById('totalInvested').textContent = `$${totalInvested.toFixed(2)}`;
            
            const profitElement = document.getElementById('profitPercent');
            profitElement.textContent = `${profitPct.toFixed(2)}%`;
            profitElement.className = `metric-value ${profitPct >= 0 ? 'positive' : 'negative'}`;
            
            // Simulate available capital
            const availableCapital = 100000 - totalInvested;
            document.getElementById('availableCapital').textContent = `$${availableCapital.toFixed(2)}`;
            document.getElementById('tradingAllocation').textContent = `${tradingAllocation}%`;
        }
        
        // Trading functions
        function sellPosition(asset, percentage) {
            selectedSellAmount[asset] = percentage;
            console.log(`Selected to sell ${percentage}% of ${asset}`);
            
            // Highlight selected button
            document.querySelectorAll(`button[onclick*="sellPosition('${asset}',"]`).forEach(btn => {
                btn.style.background = btn.textContent === `${percentage}%` ? 
                    'rgba(244, 67, 54, 0.7)' : 'rgba(244, 67, 54, 0.3)';
            });
        }
        
        function buyAmount(asset, amount) {
            selectedBuyAmount[asset] = amount;
            console.log(`Selected to buy $${amount} of ${asset}`);
            
            // Highlight selected button
            document.querySelectorAll(`button[onclick*="buyAmount('${asset}',"]`).forEach(btn => {
                btn.style.background = btn.textContent === `$${amount}` ? 
                    'rgba(76, 175, 80, 0.7)' : 'rgba(76, 175, 80, 0.3)';
            });
        }
        
        function setSellAmount(asset, percentage) {
            selectedSellAmount[asset] = percentage;
            
            // Update the input field
            const inputId = asset === 'BTC' ? 'btc-sell-amount' : 'eth-sell-amount';
            document.getElementById(inputId).value = percentage;
            
            // Highlight selected button
            document.querySelectorAll(`button[onclick*="setSellAmount('${asset}',"]`).forEach(btn => {
                btn.style.background = 'rgba(244, 67, 54, 0.3)';
            });
            event.target.style.background = 'rgba(244, 67, 54, 0.8)';
            console.log(`Set sell amount: ${percentage}% for ${asset}`);
        }

        async function executeSell(asset) {
            // Get percentage from input field first, then fallback to selectedSellAmount
            const inputId = asset === 'BTC' ? 'btc-sell-amount' : 'eth-sell-amount';
            const inputPercentage = parseFloat(document.getElementById(inputId).value) || 0;
            
            const percentage = inputPercentage > 0 ? inputPercentage : selectedSellAmount[asset];
            
            if (percentage > 0 && percentage <= 100) {
                console.log(`Executing sell: ${percentage}% of ${asset}`);
                
                try {
                    const response = await fetch('/api/trade/sell', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            asset: asset,
                            percentage: percentage
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        alert(`✅ ${result.message}`);
                        
                        // Clear input and reset selection
                        document.getElementById(inputId).value = '';
                        selectedSellAmount[asset] = 0;
                        
                        // Reset button highlights
                        document.querySelectorAll(`button[onclick*="setSellAmount('${asset}',"]`).forEach(btn => {
                            btn.style.background = 'rgba(244, 67, 54, 0.3)';
                        });
                        
                        // Refresh portfolio data immediately
                        setTimeout(fetchAllData, 500);
                    } else {
                        alert(`❌ Error: ${result.error}`);
                    }
                } catch (error) {
                    console.error('Sell trade error:', error);
                    alert(`❌ Network error: ${error.message}`);
                }
            } else {
                alert('Please enter a valid percentage (1-100%) to sell or select a preset percentage');
            }
        }
        
        function setBuyAmount(asset, amount) {
            selectedBuyAmount[asset] = amount;
            
            // Update the input field
            const inputId = asset === 'BTC' ? 'btc-buy-amount' : 'eth-buy-amount';
            document.getElementById(inputId).value = amount;
            
            // Highlight selected button
            document.querySelectorAll(`button[onclick*="setBuyAmount('${asset}',"]`).forEach(btn => {
                btn.style.background = 'rgba(76, 175, 80, 0.3)';
            });
            event.target.style.background = 'rgba(76, 175, 80, 0.8)';
            console.log(`Set buy amount: $${amount} for ${asset}`);
        }

        async function executeBuy(asset) {
            // Get amount from input field first, then fallback to selectedBuyAmount
            const inputId = asset === 'BTC' ? 'btc-buy-amount' : 'eth-buy-amount';
            const inputAmount = parseFloat(document.getElementById(inputId).value) || 0;
            
            const amount = inputAmount > 0 ? inputAmount : selectedBuyAmount[asset];
            
            if (amount > 0) {
                console.log(`Executing buy: $${amount} of ${asset}`);
                
                try {
                    const response = await fetch('/api/trade/buy', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            asset: asset,
                            amount: amount
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        alert(`✅ ${result.message}`);
                        
                        // Clear input and reset selection
                        document.getElementById(inputId).value = '';
                        selectedBuyAmount[asset] = 0;
                        
                        // Reset button highlights
                        document.querySelectorAll(`button[onclick*="setBuyAmount('${asset}',"]`).forEach(btn => {
                            btn.style.background = 'rgba(76, 175, 80, 0.3)';
                        });
                        
                        // Refresh portfolio data immediately
                        setTimeout(fetchAllData, 500);
                    } else {
                        alert(`❌ Error: ${result.error}`);
                    }
                } catch (error) {
                    console.error('Buy trade error:', error);
                    alert(`❌ Network error: ${error.message}`);
                }
            } else {
                alert('Please enter an amount to buy or select a preset amount');
            }
        }
        
        function setAllocation(percentage) {
            tradingAllocation = percentage;
            document.getElementById('tradingAllocation').textContent = `${percentage}%`;
            console.log(`Set trading allocation to ${percentage}%`);
            
            // Highlight selected allocation button
            document.querySelectorAll('.allocation-btn').forEach(btn => {
                btn.style.background = btn.textContent === `${percentage}%` ? 
                    'rgba(33, 150, 243, 0.7)' : 'rgba(255, 255, 255, 0.1)';
            });
        }
        
        // Header control functions
        function refreshDashboard() {
            console.log('Refreshing dashboard...');
            location.reload();
        }
        
        function clearData() {
            console.log('Clearing data...');
            if (confirm('Are you sure you want to clear all data?')) {
                // Reset trading selections
                selectedSellAmount = { BTC: 0, ETH: 0 };
                selectedBuyAmount = { BTC: 0, ETH: 0 };
                
                // Reset button highlights
                document.querySelectorAll('.trade-btn, .allocation-btn').forEach(btn => {
                    btn.style.background = btn.classList.contains('sell-btn') ? 'rgba(244, 67, 54, 0.3)' :
                                         btn.classList.contains('buy-btn') ? 'rgba(76, 175, 80, 0.3)' :
                                         'rgba(255, 255, 255, 0.1)';
                });
            }
        }
        
        function toggleModel() {
            console.log('Toggling model...');
            alert('Model toggle functionality would be implemented here');
        }
        
        function stopModel() {
            console.log('Stopping model...');
            alert('Model stop functionality would be implemented here');
        }
        
        function updateMarketMetrics(metrics) {
            // Market metrics can be displayed in the alpha signals or settings card
            console.log('Market metrics updated:', metrics);
        }
        
        function updateTradingSignals(signals) {
            const container = document.getElementById('signalsContainer');
            container.innerHTML = '';
            
            signals.forEach(signal => {
                const signalDiv = document.createElement('div');
                signalDiv.className = `signal-item signal-${signal.signal.toLowerCase()}`;
                signalDiv.innerHTML = `
                    <div>
                        <strong>${signal.indicator}</strong><br>
                        <small>${signal.strength} ${signal.signal}</small>
                    </div>
                    <div style="text-align: right;">
                        <strong>${signal.confidence.toFixed(0)}%</strong><br>
                        <small>${signal.timestamp}</small>
                    </div>
                `;
                container.appendChild(signalDiv);
            });
        }
        
        function updateNewsSentiment(sentimentData) {
            console.log('Updating news sentiment:', sentimentData);
            
            // Update market sentiment
            if (sentimentData.market_sentiment) {
                const marketSentiment = sentimentData.market_sentiment;
                document.getElementById('marketSentimentScore').textContent = marketSentiment.market_sentiment.toFixed(2);
                document.getElementById('marketConfidence').textContent = `${(marketSentiment.market_confidence * 100).toFixed(0)}%`;
                document.getElementById('totalNewsCount').textContent = marketSentiment.total_news;
                
                // Color market sentiment score
                const scoreElement = document.getElementById('marketSentimentScore');
                if (marketSentiment.market_sentiment > 0.1) {
                    scoreElement.className = 'sentiment-score positive';
                } else if (marketSentiment.market_sentiment < -0.1) {
                    scoreElement.className = 'sentiment-score negative';
                } else {
                    scoreElement.className = 'sentiment-score neutral';
                }
            }
            
            // Update BTC sentiment
            if (sentimentData.BTCUSDT) {
                updateSymbolSentiment('btc', sentimentData.BTCUSDT);
            }
            
            // Update ETH sentiment
            if (sentimentData.ETHUSDT) {
                updateSymbolSentiment('eth', sentimentData.ETHUSDT);
            }
        }
        
        function updateSymbolSentiment(symbol, sentimentData) {
            const scoreElement = document.getElementById(`${symbol}SentimentScore`);
            const countsElement = document.getElementById(`${symbol}SentimentCounts`);
            const phrasesElement = document.getElementById(`${symbol}KeyPhrases`);
            const positiveBar = document.getElementById(`${symbol}PositiveBar`);
            const negativeBar = document.getElementById(`${symbol}NegativeBar`);
            
            // Update sentiment score
            scoreElement.textContent = sentimentData.sentiment_score.toFixed(2);
            
            // Color sentiment score
            if (sentimentData.sentiment_score > 0.1) {
                scoreElement.className = 'sentiment-score positive';
            } else if (sentimentData.sentiment_score < -0.1) {
                scoreElement.className = 'sentiment-score negative';
            } else {
                scoreElement.className = 'sentiment-score neutral';
            }
            
            // Update sentiment counts
            countsElement.textContent = `${sentimentData.positive_news}+ ${sentimentData.negative_news}- ${sentimentData.neutral_news}=`;
            
            // Update key phrases
            phrasesElement.textContent = sentimentData.key_phrases ? sentimentData.key_phrases.join(', ') : 'No key phrases';
            
            // Update sentiment bars
            const totalNews = sentimentData.positive_news + sentimentData.negative_news + sentimentData.neutral_news;
            if (totalNews > 0) {
                const positivePercentage = (sentimentData.positive_news / totalNews) * 100;
                const negativePercentage = (sentimentData.negative_news / totalNews) * 100;
                
                positiveBar.style.width = `${positivePercentage}%`;
                negativeBar.style.width = `${negativePercentage}%`;
            }
        }
        
        function updateNewsFeed(newsItems) {
            const newsFeed = document.getElementById('newsFeed');
            newsFeed.innerHTML = '';
            
            newsItems.forEach(news => {
                const newsDiv = document.createElement('div');
                
                // Determine sentiment class
                let sentimentClass = 'news-neutral';
                let sentimentText = 'Neutral';
                if (news.sentiment_score > 0.2) {
                    sentimentClass = 'news-positive';
                    sentimentText = 'Positive';
                } else if (news.sentiment_score < -0.2) {
                    sentimentClass = 'news-negative';
                    sentimentText = 'Negative';
                }
                
                newsDiv.className = `news-item ${sentimentClass}`;
                newsDiv.innerHTML = `
                    <div class="news-title">${news.title}</div>
                    <div class="news-meta">
                        <div>
                            <span class="news-source">${news.source}</span> • 
                            <span class="news-time">${news.time_ago}</span>
                        </div>
                        <div class="sentiment-indicator sentiment-${sentimentText.toLowerCase()}">
                            ${sentimentText}
                        </div>
                    </div>
                `;
                
                // Add click handler to open news article
                newsDiv.style.cursor = 'pointer';
                newsDiv.addEventListener('click', () => {
                    window.open(news.url, '_blank');
                });
                
                newsFeed.appendChild(newsDiv);
            });
        }
        
        function createPerformanceChart(history) {
            const timestamps = history.map(h => h.timestamp);
            const portfolioValues = history.map(h => h.portfolio_value);
            const pnlValues = history.map(h => h.pnl);
            
            const trace1 = {
                x: timestamps,
                y: portfolioValues,
                type: 'scatter',
                mode: 'lines',
                name: 'Portfolio Value',
                line: { color: '#4CAF50', width: 3 }
            };
            
            const trace2 = {
                x: timestamps,
                y: pnlValues,
                type: 'scatter',
                mode: 'lines',
                name: 'P&L',
                line: { color: '#FF9800', width: 2 },
                yaxis: 'y2'
            };
            
            const layout = {
                title: false,
                paper_bgcolor: '#1a1a2e',
                plot_bgcolor: '#16213e',
                font: { color: '#ffffff' },
                legend: { font: { color: '#ffffff' } },
                xaxis: {
                    color: '#ffffff',
                    gridcolor: 'rgba(255,255,255,0.1)'
                },
                yaxis: {
                    title: 'Portfolio Value ($)',
                    color: '#ffffff',
                    gridcolor: 'rgba(255,255,255,0.1)'
                },
                yaxis2: {
                    title: 'P&L ($)',
                    color: '#ffffff',
                    overlaying: 'y',
                    side: 'right'
                },
                margin: { l: 60, r: 60, t: 20, b: 40 }
            };
            
            Plotly.newPlot('performanceChart', [trace1, trace2], layout, {responsive: true});
        }
        
        function createPriceChart(chartId, history, symbol, color) {
            const timestamps = history.map(h => h.timestamp);
            const prices = history.map(h => symbol === 'BTC' ? h.btc_price : h.eth_price);
            
            const trace = {
                x: timestamps,
                y: prices,
                type: 'scatter',
                mode: 'lines',
                name: `${symbol} Price`,
                line: { color: color, width: 3 }
            };
            
            const layout = {
                title: false,
                paper_bgcolor: 'rgba(22, 33, 62, 0.9)',
                plot_bgcolor: 'rgba(26, 26, 46, 0.8)',
                font: { color: '#ffffff' },
                showlegend: false,
                xaxis: {
                    color: '#ffffff',
                    gridcolor: 'rgba(255,255,255,0.1)'
                },
                yaxis: {
                    title: 'Price ($)',
                    color: '#ffffff',
                    gridcolor: 'rgba(255,255,255,0.1)'
                },
                margin: { l: 60, r: 30, t: 20, b: 40 }
            };
            
            Plotly.newPlot(chartId, [trace], layout, {responsive: true});
        }
        
        function createMarketModelChart(chartId, data, symbol, timeframe) {
            const timestamps = data.map(d => d.timestamp);
            const marketPrices = data.map(d => symbol === 'BTC' ? d.btc_market : d.eth_market);
            const modelPrices = data.map(d => symbol === 'BTC' ? d.btc_model : d.eth_model);
            
            const marketTrace = {
                x: timestamps,
                y: marketPrices,
                type: 'scatter',
                mode: 'lines',
                name: `${symbol} Market`,
                line: { color: symbol === 'BTC' ? '#f7931a' : '#627eea', width: 2 }
            };
            
            const modelTrace = {
                x: timestamps,
                y: modelPrices,
                type: 'scatter',
                mode: 'lines',
                name: `${symbol} Model`,
                line: { color: symbol === 'BTC' ? '#ff6b35' : '#8b5cf6', width: 2, dash: 'dot' }
            };
            
            const layout = {
                title: false,
                paper_bgcolor: 'rgba(22, 33, 62, 0.9)',
                plot_bgcolor: 'rgba(26, 26, 46, 0.8)',
                font: { color: '#ffffff', size: 10 },
                legend: { 
                    font: { color: '#ffffff', size: 10 },
                    orientation: 'h',
                    x: 0,
                    y: 1.1,
                    bgcolor: 'rgba(255,255,255,0.8)'
                },
                xaxis: {
                    color: '#ffffff',
                    gridcolor: 'rgba(0,0,0,0.1)',
                    showticklabels: timeframe === '5s'
                },
                yaxis: {
                    title: 'Price ($)',
                    color: '#ffffff',
                    gridcolor: 'rgba(0,0,0,0.1)',
                    titlefont: { size: 10 }
                },
                margin: { l: 50, r: 20, t: 30, b: 30 }
            };
            
            Plotly.newPlot(chartId, [marketTrace, modelTrace], layout, {responsive: true});
        }
        
        function updateMarketModelCharts(data5s, data1h, data1d) {
            console.log('Updating market vs model charts...');
            
            // 5-second charts
            if (data5s) {
                createMarketModelChart('btc5sChart', data5s, 'BTC', '5s');
                createMarketModelChart('eth5sChart', data5s, 'ETH', '5s');
            }
            
            // 1-hour charts
            if (data1h) {
                createMarketModelChart('btc1hChart', data1h, 'BTC', '1h');
                createMarketModelChart('eth1hChart', data1h, 'ETH', '1h');
            }
            
            // 1-day charts
            if (data1d) {
                createMarketModelChart('btc1dChart', data1d, 'BTC', '1d');
                createMarketModelChart('eth1dChart', data1d, 'ETH', '1d');
            }
        }
        
        function createTFTPredictionChart(chartId, tftData, symbol) {
            if (!tftData) return;
            
            const timestamps = tftData.timestamps.map(ts => new Date(ts));
            const predictions = tftData.predictions;
            const lowerBound = tftData.lower_bound;
            const upperBound = tftData.upper_bound;
            
            // Prediction line
            const predictionTrace = {
                x: timestamps,
                y: predictions,
                type: 'scatter',
                mode: 'lines+markers',
                name: `${symbol} TFT Prediction`,
                line: { color: symbol === 'BTC' ? '#f7931a' : '#627eea', width: 3 },
                marker: { size: 6 }
            };
            
            // Confidence bands
            const upperTrace = {
                x: timestamps,
                y: upperBound,
                type: 'scatter',
                mode: 'lines',
                name: 'Upper Bound (90%)',
                line: { color: 'rgba(255, 0, 0, 0.3)', width: 1 },
                fill: 'tonexty',
                fillcolor: 'rgba(255, 0, 0, 0.1)'
            };
            
            const lowerTrace = {
                x: timestamps,
                y: lowerBound,
                type: 'scatter',
                mode: 'lines',
                name: 'Lower Bound (10%)',
                line: { color: 'rgba(255, 0, 0, 0.3)', width: 1 }
            };
            
            const layout = {
                title: false,
                paper_bgcolor: 'rgba(22, 33, 62, 0.9)',
                plot_bgcolor: 'rgba(26, 26, 46, 0.8)',
                font: { color: '#ffffff', size: 10 },
                legend: { 
                    font: { color: '#ffffff', size: 9 },
                    orientation: 'h',
                    x: 0,
                    y: 1.1,
                    bgcolor: 'rgba(255,255,255,0.8)'
                },
                xaxis: {
                    title: 'Time',
                    color: '#ffffff',
                    gridcolor: 'rgba(0,0,0,0.1)',
                    titlefont: { size: 10 }
                },
                yaxis: {
                    title: 'Price ($)',
                    color: '#ffffff',
                    gridcolor: 'rgba(0,0,0,0.1)',
                    titlefont: { size: 10 }
                },
                margin: { l: 50, r: 20, t: 30, b: 40 },
                annotations: [{
                    text: `Confidence: ${tftData.confidence_interval}%`,
                    xref: 'paper',
                    yref: 'paper',
                    x: 0.02,
                    y: 0.98,
                    showarrow: false,
                    font: { color: '#ffffff', size: 9 }
                }]
            };
            
            Plotly.newPlot(chartId, [lowerTrace, upperTrace, predictionTrace], layout, {responsive: true});
        }
        
        function updateTFTCharts(tftData) {
            console.log('Updating TFT prediction charts...', tftData);
            
            if (tftData.BTCUSDT) {
                console.log('Creating BTC TFT chart');
                createTFTPredictionChart('btcTftChart', tftData.BTCUSDT, 'BTC');
            } else {
                console.log('No BTC TFT data found');
            }
            
            if (tftData.ETHUSDT) {
                console.log('Creating ETH TFT chart');
                createTFTPredictionChart('ethTftChart', tftData.ETHUSDT, 'ETH');
            } else {
                console.log('No ETH TFT data found');
            }
        }
        
        function createAssetNetworkChart(gcnData) {
            if (!gcnData || !gcnData.relationships) return;
            
            const correlations = gcnData.relationships.strong_correlations || [];
            const assets = gcnData.assets || [];
            
            // Create network graph
            const nodes = assets.map((asset, i) => ({
                x: Math.cos(2 * Math.PI * i / assets.length),
                y: Math.sin(2 * Math.PI * i / assets.length),
                text: asset.replace('USDT', ''),
                mode: 'markers+text',
                marker: {
                    size: 20,
                    color: '#4CAF50',
                    line: { width: 2, color: 'white' }
                },
                textposition: 'top center',
                type: 'scatter',
                name: asset
            }));
            
            // Create edges
            const edges = [];
            correlations.forEach(corr => {
                const idx1 = assets.indexOf(corr.asset1);
                const idx2 = assets.indexOf(corr.asset2);
                if (idx1 !== -1 && idx2 !== -1) {
                    edges.push({
                        x: [Math.cos(2 * Math.PI * idx1 / assets.length), Math.cos(2 * Math.PI * idx2 / assets.length), null],
                        y: [Math.sin(2 * Math.PI * idx1 / assets.length), Math.sin(2 * Math.PI * idx2 / assets.length), null],
                        mode: 'lines',
                        line: {
                            width: corr.strength * 5,
                            color: corr.strength > 0.8 ? '#FF6B6B' : '#4ECDC4'
                        },
                        type: 'scatter',
                        hoverinfo: 'none',
                        showlegend: false
                    });
                }
            });
            
            const layout = {
                title: { text: '🔗 Asset Correlation Network', font: { color: 'white', size: 16 } },
                plot_bgcolor: 'rgba(26, 26, 46, 0.8)',
                paper_bgcolor: 'rgba(22, 33, 62, 0.9)',
                font: { color: 'white' },
                xaxis: { visible: false },
                yaxis: { visible: false },
                showlegend: false,
                margin: { t: 50, r: 20, b: 20, l: 20 }
            };
            
            Plotly.newPlot('assetNetworkChart', [...edges, ...nodes], layout, {responsive: true});
        }
        
        function createDiversificationChart(divData) {
            if (!divData || !divData.relationships) return;
            
            const relationships = Object.entries(divData.relationships);
            const assets = relationships.map(([pair, data]) => pair);
            const strengths = relationships.map(([pair, data]) => data.strength);
            const benefits = relationships.map(([pair, data]) => data.diversification_benefit);
            
            const trace1 = {
                x: assets,
                y: strengths,
                type: 'bar',
                name: 'Correlation Strength',
                marker: { color: '#FF6B6B' }
            };
            
            const trace2 = {
                x: assets,
                y: benefits,
                type: 'bar',
                name: 'Diversification Benefit',
                marker: { color: '#4ECDC4' },
                yaxis: 'y2'
            };
            
            const layout = {
                title: { text: '📊 Portfolio Diversification', font: { color: 'white', size: 16 } },
                plot_bgcolor: 'rgba(26, 26, 46, 0.8)',
                paper_bgcolor: 'rgba(22, 33, 62, 0.9)',
                font: { color: 'white' },
                xaxis: { title: 'Asset Pairs', color: 'white' },
                yaxis: { title: 'Correlation Strength', color: 'white' },
                yaxis2: {
                    title: 'Diversification Benefit',
                    overlaying: 'y',
                    side: 'right',
                    color: 'white'
                },
                barmode: 'group',
                margin: { t: 50, r: 50, b: 50, l: 50 }
            };
            
            Plotly.newPlot('diversificationChart', [trace1, trace2], layout, {responsive: true});
        }
        
        function createCrossAssetChart(gcnData) {
            if (!gcnData || !gcnData.predictions) return;
            
            const assets = Object.keys(gcnData.predictions).slice(0, 5); // Top 5 assets
            const traces = [];
            
            assets.forEach((asset, i) => {
                const pred = gcnData.predictions[asset];
                if (pred && pred.predictions) {
                    traces.push({
                        x: pred.prediction_hours,
                        y: pred.predictions,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: asset.replace('USDT', ''),
                        line: { width: 2 }
                    });
                }
            });
            
            const layout = {
                title: { text: '📈 Cross-Asset Predictions', font: { color: 'white', size: 16 } },
                plot_bgcolor: 'rgba(26, 26, 46, 0.8)',
                paper_bgcolor: 'rgba(22, 33, 62, 0.9)',
                font: { color: 'white' },
                xaxis: { title: 'Hours Ahead', color: 'white' },
                yaxis: { title: 'Predicted Price', color: 'white' },
                margin: { t: 50, r: 20, b: 50, l: 50 }
            };
            
            Plotly.newPlot('crossAssetChart', traces, layout, {responsive: true});
        }
        
        function updateMarketInsights(gcnData) {
            if (!gcnData || !gcnData.market_insights) return;
            
            const insights = gcnData.market_insights;
            const html = `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <strong>🎯 Most Connected:</strong><br>
                        ${insights.most_connected_asset?.replace('USDT', '') || 'N/A'}
                    </div>
                    <div>
                        <strong>🤝 Market Cohesion:</strong><br>
                        ${(insights.market_cohesion * 100).toFixed(1)}%
                    </div>
                    <div>
                        <strong>📈 Growth Leaders:</strong><br>
                        ${insights.growth_leaders?.map(a => a.replace('USDT', '')).join(', ') || 'N/A'}
                    </div>
                    <div>
                        <strong>⚡ Volatility Cluster:</strong><br>
                        ${insights.volatility_cluster?.map(a => a.replace('USDT', '')).join(', ') || 'N/A'}
                    </div>
                </div>
            `;
            
            document.getElementById('marketInsights').innerHTML = html;
        }
        
        function updateGCNCharts(gcnData) {
            console.log('Updating GCN charts...', gcnData);
            
            createAssetNetworkChart(gcnData);
            createDiversificationChart(gcnData);
            createCrossAssetChart(gcnData);
            updateMarketInsights(gcnData);
        }
        
        function createVarEsChart(riskData) {
            if (!riskData) return;
            
            const assets = Object.keys(riskData);
            const vars = [];
            const ess = [];
            const evtVars = [];
            const evtEss = [];
            
            assets.forEach(asset => {
                const metrics = riskData[asset]?.risk_metrics;
                if (metrics) {
                    vars.push(metrics.var_95_historical || 0);
                    ess.push(metrics.es_95_historical || 0);
                    evtVars.push(metrics.var_95_evt || 0);
                    evtEss.push(metrics.es_95_evt || 0);
                }
            });
            
            const trace1 = {
                x: assets.map(a => a.replace('USDT', '')),
                y: vars,
                type: 'bar',
                name: 'VaR (Historical)',
                marker: { color: '#FF6B6B' }
            };
            
            const trace2 = {
                x: assets.map(a => a.replace('USDT', '')),
                y: ess,
                type: 'bar',
                name: 'Expected Shortfall',
                marker: { color: '#4ECDC4' }
            };
            
            const trace3 = {
                x: assets.map(a => a.replace('USDT', '')),
                y: evtVars,
                type: 'bar',
                name: 'VaR (EVT)',
                marker: { color: '#45B7D1' }
            };
            
            const layout = {
                title: { text: '📊 VaR vs Expected Shortfall', font: { color: 'white', size: 16 } },
                plot_bgcolor: 'rgba(26, 26, 46, 0.8)',
                paper_bgcolor: 'rgba(22, 33, 62, 0.9)',
                font: { color: 'white' },
                xaxis: { title: 'Assets', color: 'white' },
                yaxis: { title: 'Risk Amount ($)', color: 'white' },
                barmode: 'group',
                margin: { t: 50, r: 20, b: 50, l: 60 }
            };
            
            Plotly.newPlot('varEsChart', [trace1, trace2, trace3], layout, {responsive: true});
        }
        
        function createEVTChart(riskData) {
            console.log('Creating EVT Chart with data:', riskData);
            if (!riskData) {
                console.log('No risk data provided');
                return;
            }
            
            const assets = Object.keys(riskData);
            const traces = [];
            
            assets.forEach((asset, i) => {
                const analysis = riskData[asset];
                let params = null;
                
                // Try peaks_over_threshold first, then fallback to block_maxima
                if (analysis?.extreme_value_theory?.peaks_over_threshold?.fitted) {
                    params = analysis.extreme_value_theory.peaks_over_threshold.parameters;
                } else if (analysis?.extreme_value_theory?.block_maxima?.fitted) {
                    params = analysis.extreme_value_theory.block_maxima.parameters;
                }
                
                if (params) {
                    // Create EVT distribution visualization
                    traces.push({
                        x: [i],
                        y: [params.shape],
                        type: 'scatter',
                        mode: 'markers',
                        name: asset.replace('USDT', ''),
                        marker: {
                            size: Math.abs(params.scale) * 1000,
                            color: params.shape > 0 ? '#FF6B6B' : '#4ECDC4',
                            opacity: 0.7
                        },
                        text: `Shape: ${params.shape.toFixed(3)}<br>Scale: ${params.scale.toFixed(3)}<br>Location: ${params.location.toFixed(3)}`,
                        hoverinfo: 'text+name'
                    });
                }
            });
            
            const layout = {
                title: { text: '🎯 EVT Shape Parameters', font: { color: 'white', size: 16 } },
                plot_bgcolor: 'rgba(26, 26, 46, 0.8)',
                paper_bgcolor: 'rgba(22, 33, 62, 0.9)',
                font: { color: 'white' },
                xaxis: { 
                    title: 'Assets', 
                    color: 'white',
                    tickmode: 'array',
                    tickvals: [0, 1],
                    ticktext: assets.map(a => a.replace('USDT', ''))
                },
                yaxis: { title: 'Shape Parameter', color: 'white' },
                margin: { t: 50, r: 20, b: 50, l: 60 }
            };
            
            Plotly.newPlot('evtChart', traces, layout, {responsive: true});
        }
        
        function createPortfolioRiskChart(portfolioRisk) {
            if (!portfolioRisk || !portfolioRisk.portfolio_risk) return;
            
            const risk = portfolioRisk.portfolio_risk;
            const benefit = portfolioRisk.diversification_benefit;
            
            const trace1 = {
                x: ['Portfolio VaR', 'Portfolio ES'],
                y: [risk.portfolio_var_95, risk.portfolio_es_95],
                type: 'bar',
                name: 'Portfolio Risk',
                marker: { color: '#FF6B6B' }
            };
            
            const trace2 = {
                x: ['Individual VaR Sum', 'Individual ES Sum'],
                y: [risk.sum_individual_var, risk.sum_individual_es],
                type: 'bar',
                name: 'Sum of Individual',
                marker: { color: '#4ECDC4' }
            };
            
            const layout = {
                title: { text: '💼 Portfolio Diversification', font: { color: 'white', size: 16 } },
                plot_bgcolor: 'rgba(26, 26, 46, 0.8)',
                paper_bgcolor: 'rgba(22, 33, 62, 0.9)',
                font: { color: 'white' },
                xaxis: { title: 'Risk Measures', color: 'white' },
                yaxis: { title: 'Risk Amount ($)', color: 'white' },
                barmode: 'group',
                margin: { t: 50, r: 20, b: 50, l: 60 }
            };
            
            Plotly.newPlot('portfolioRiskChart', [trace1, trace2], layout, {responsive: true});
        }
        
        function updateRiskSummary(riskData) {
            if (!riskData) return;
            
            const assets = Object.keys(riskData).filter(k => k !== 'last_update');
            let html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">';
            
            assets.forEach(asset => {
                const metrics = riskData[asset]?.risk_metrics;
                if (metrics) {
                    html += `
                        <div>
                            <strong>📊 ${asset.replace('USDT', '')}:</strong><br>
                            VaR (95%): $${metrics.var_95_historical?.toFixed(0) || 'N/A'}<br>
                            ES (95%): $${metrics.es_95_historical?.toFixed(0) || 'N/A'}<br>
                            Tail Ratio: ${metrics.tail_risk_ratio?.toFixed(2) || 'N/A'}
                        </div>
                    `;
                }
            });
            
            html += '</div>';
            document.getElementById('riskSummary').innerHTML = html;
        }
        
        function updateRiskCharts(riskData, portfolioRisk) {
            console.log('Updating risk management charts...');
            
            createVarEsChart(riskData);
            createEVTChart(riskData);
            createPortfolioRiskChart(portfolioRisk);
            updateRiskSummary(riskData);
        }
        
        function createCopulaComparisonChart(copulaData) {
            if (!copulaData || !copulaData.copula_models) return;
            
            const models = copulaData.copula_models;
            const comparison = models.model_comparison || {};
            
            const copulaNames = Object.keys(comparison);
            const aicValues = copulaNames.map(name => comparison[name]?.aic || 0);
            const bicValues = copulaNames.map(name => comparison[name]?.bic || 0);
            const logLikValues = copulaNames.map(name => comparison[name]?.log_likelihood || 0);
            
            const trace1 = {
                x: copulaNames,
                y: aicValues,
                type: 'bar',
                name: 'AIC',
                marker: { color: '#FF6B6B' }
            };
            
            const trace2 = {
                x: copulaNames,
                y: bicValues,
                type: 'bar',
                name: 'BIC',
                marker: { color: '#4ECDC4' }
            };
            
            const layout = {
                title: { text: '📊 Copula Model Comparison', font: { color: 'white', size: 16 } },
                plot_bgcolor: 'rgba(26, 26, 46, 0.8)',
                paper_bgcolor: 'rgba(22, 33, 62, 0.9)',
                font: { color: 'white' },
                xaxis: { title: 'Copula Models', color: 'white' },
                yaxis: { title: 'Information Criteria', color: 'white' },
                barmode: 'group',
                margin: { t: 50, r: 20, b: 50, l: 60 }
            };
            
            Plotly.newPlot('copulaComparisonChart', [trace1, trace2], layout, {responsive: true});
        }
        
        function createDependencyMeasuresChart(copulaData) {
            if (!copulaData || !copulaData.dependency_measures) return;
            
            const measures = copulaData.dependency_measures;
            
            const measureNames = ['Pearson', 'Spearman', 'Kendall Tau', 'Mutual Info'];
            const measureValues = [
                measures.pearson_correlation || 0,
                measures.spearman_correlation || 0, 
                measures.kendall_tau || 0,
                measures.mutual_information || 0
            ];
            
            const trace = {
                x: measureNames,
                y: measureValues,
                type: 'bar',
                marker: { 
                    color: measureValues.map(v => v > 0.5 ? '#FF6B6B' : '#4ECDC4'),
                    opacity: 0.8
                }
            };
            
            const layout = {
                title: { text: '🎯 Dependency Measures', font: { color: 'white', size: 16 } },
                plot_bgcolor: 'rgba(26, 26, 46, 0.8)',
                paper_bgcolor: 'rgba(22, 33, 62, 0.9)',
                font: { color: 'white' },
                xaxis: { title: 'Dependency Measures', color: 'white' },
                yaxis: { title: 'Correlation Value', color: 'white', range: [-1, 1] },
                margin: { t: 50, r: 20, b: 50, l: 60 }
            };
            
            Plotly.newPlot('dependencyMeasuresChart', [trace], layout, {responsive: true});
        }
        
        function createTailDependenceChart(copulaData) {
            console.log('Creating Tail Dependence Chart with data:', copulaData);
            if (!copulaData || !copulaData.dependency_measures) {
                console.log('No copula data or dependency measures found');
                return;
            }
            
            const measures = copulaData.dependency_measures;
            
            const tailTypes = ['Upper Tail', 'Lower Tail'];
            const tailValues = [
                measures.upper_tail_dependence || 0,
                measures.lower_tail_dependence || 0
            ];
            
            const trace = {
                x: tailTypes,
                y: tailValues,
                type: 'bar',
                marker: { 
                    color: ['#FF6B6B', '#45B7D1'],
                    opacity: 0.8
                }
            };
            
            const layout = {
                title: { text: '🔗 Tail Dependence', font: { color: 'white', size: 16 } },
                plot_bgcolor: 'rgba(26, 26, 46, 0.8)',
                paper_bgcolor: 'rgba(22, 33, 62, 0.9)',
                font: { color: 'white' },
                xaxis: { title: 'Tail Type', color: 'white' },
                yaxis: { title: 'Dependence Strength', color: 'white', range: [0, 1] },
                margin: { t: 50, r: 20, b: 50, l: 60 }
            };
            
            Plotly.newPlot('tailDependenceChart', [trace], layout, {responsive: true});
        }
        
        function updateCopulaInsights(copulaData) {
            if (!copulaData) return;
            
            const bestCopula = copulaData.copula_models?.best_copula || 'Unknown';
            const measures = copulaData.dependency_measures || {};
            const recommendations = copulaData.recommendations || [];
            
            let html = `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                    <div>
                        <strong>🏆 Best Model:</strong><br>
                        ${bestCopula.charAt(0).toUpperCase() + bestCopula.slice(1)} Copula
                    </div>
                    <div>
                        <strong>📊 Spearman ρ:</strong><br>
                        ${(measures.spearman_correlation || 0).toFixed(3)}
                    </div>
                    <div>
                        <strong>⬆️ Upper Tail:</strong><br>
                        ${(measures.upper_tail_dependence || 0).toFixed(3)}
                    </div>
                    <div>
                        <strong>⬇️ Lower Tail:</strong><br>
                        ${(measures.lower_tail_dependence || 0).toFixed(3)}
                    </div>
                </div>
                <div>
                    <strong>💡 Recommendations:</strong><br>
                    ${recommendations.slice(0, 2).map(rec => `• ${rec}`).join('<br>')}
                </div>
            `;
            
            document.getElementById('copulaInsights').innerHTML = html;
        }
        
        function updateCopulaCharts(copulaData) {
            console.log('Updating copula correlation charts...');
            
            createCopulaComparisonChart(copulaData);
            createDependencyMeasuresChart(copulaData);
            createTailDependenceChart(copulaData);
            updateCopulaInsights(copulaData);
        }
        
        function updateTradingLog(tradingLogData) {
            console.log('Updating trading log...');
            
            const tradingLogContent = document.getElementById('tradingLogContent');
            
            if (!tradingLogData.log_entries || tradingLogData.log_entries.length === 0) {
                tradingLogContent.innerHTML = '<div style="color: #888; text-align: center; padding: 20px;">No trading activity yet</div>';
                return;
            }
            
            // Build HTML for trading log entries
            let logHtml = '';
            tradingLogData.log_entries.slice(0, 10).forEach(entry => {  // Show only last 10 entries
                const timestamp = new Date(entry.timestamp).toLocaleString();
                const successIcon = entry.success ? '✅' : '❌';
                const actionIcon = entry.action === 'buy' ? '📈' : '📉';
                const actionColor = entry.action === 'buy' ? '#00ff88' : '#ff6b6b';
                
                logHtml += `
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 16px;">${successIcon}${actionIcon}</span>
                            <span style="color: ${actionColor}; font-weight: bold; text-transform: uppercase;">${entry.action}</span>
                            <span style="color: #fff;">${entry.asset}</span>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: #fff; font-weight: bold;">$${entry.amount.toFixed(2)}</div>
                            <div style="color: #888; font-size: 11px;">${timestamp}</div>
                        </div>
                    </div>
                `;
            });
            
            tradingLogContent.innerHTML = logHtml;
        }
        
        async function fetchAllData() {
            try {
                console.log('Fetching enhanced data...');
                
                const [portfolioResponse, metricsResponse, signalsResponse, historyResponse, sentimentResponse, newsResponse, model5sResponse, model1hResponse, model1dResponse, tftResponse, gcnResponse, divResponse, riskResponse, portfolioRiskResponse, copulaResponse, tradingLogResponse] = await Promise.all([
                    fetch('/api/portfolio'),
                    fetch('/api/market-metrics'),
                    fetch('/api/trading-signals'),
                    fetch('/api/performance-history'),
                    fetch('/api/news-sentiment'),
                    fetch('/api/recent-news'),
                    fetch('/api/market-model-5s'),
                    fetch('/api/market-model-1h'),
                    fetch('/api/market-model-1d'),
                    fetch('/api/tft-predictions'),
                    fetch('/api/gcn-analysis'),
                    fetch('/api/portfolio-diversification'),
                    fetch('/api/risk-analysis'),
                    fetch('/api/portfolio-risk'),
                    fetch('/api/copula-analysis'),
                    fetch('/api/trading-log')
                ]);
                
                if (portfolioResponse.ok) {
                    const portfolioData = await portfolioResponse.json();
                    updatePortfolioDisplay(portfolioData);
                }
                
                if (metricsResponse.ok) {
                    const metricsData = await metricsResponse.json();
                    updateMarketMetrics(metricsData);
                }
                
                if (signalsResponse.ok) {
                    const signalsData = await signalsResponse.json();
                    updateTradingSignals(signalsData);
                }
                
                if (historyResponse.ok) {
                    const historyData = await historyResponse.json();
                    createPerformanceChart(historyData);
                    createPriceChart('btcChart', historyData, 'BTC', '#f7931a');
                    createPriceChart('ethChart', historyData, 'ETH', '#627eea');
                }
                
                if (sentimentResponse.ok) {
                    const sentimentData = await sentimentResponse.json();
                    updateNewsSentiment(sentimentData);
                }
                
                if (newsResponse.ok) {
                    const newsData = await newsResponse.json();
                    updateNewsFeed(newsData);
                }
                
                // Update market vs model charts
                let model5sData, model1hData, model1dData;
                if (model5sResponse.ok) {
                    model5sData = await model5sResponse.json();
                }
                if (model1hResponse.ok) {
                    model1hData = await model1hResponse.json();
                }
                if (model1dResponse.ok) {
                    model1dData = await model1dResponse.json();
                }
                
                updateMarketModelCharts(model5sData, model1hData, model1dData);
                
                // Update TFT prediction charts
                if (tftResponse.ok) {
                    const tftData = await tftResponse.json();
                    console.log('TFT data received:', tftData);
                    updateTFTCharts(tftData);
                } else {
                    console.error('TFT API failed:', tftResponse.status);
                }
                
                if (gcnResponse.ok) {
                    const gcnData = await gcnResponse.json();
                    updateGCNCharts(gcnData);
                    createCrossAssetChart(gcnData);
                }
                
                if (divResponse.ok) {
                    const divData = await divResponse.json();
                    createDiversificationChart(divData);
                }
                
                // Update risk management charts
                if (riskResponse.ok && portfolioRiskResponse.ok) {
                    const riskData = await riskResponse.json();
                    const portfolioRiskData = await portfolioRiskResponse.json();
                    updateRiskCharts(riskData, portfolioRiskData);
                }
                
                // Update copula correlation charts
                if (copulaResponse.ok) {
                    const copulaData = await copulaResponse.json();
                    updateCopulaCharts(copulaData);
                }
                
                // Update trading log
                if (tradingLogResponse.ok) {
                    const tradingLogData = await tradingLogResponse.json();
                    updateTradingLog(tradingLogData);
                }
                
                updateCount++;
                document.getElementById('status').textContent = `✅ Complete data with TFT, GCN, ES/EVT, Copula analysis loaded (Update #${updateCount})`;
                document.getElementById('status').className = 'success';
                
            } catch (error) {
                console.error('Error fetching enhanced data:', error);
                document.getElementById('status').textContent = `❌ Error: ${error.message}`;
                document.getElementById('status').className = 'error';
            }
        }
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', () => {
            console.log('Enhanced dashboard loading...');
            fetchAllData();
            
            // Update every 10 seconds (to reduce server load with more complex data)
            setInterval(fetchAllData, 10000);
        });
    </script>
</body>
</html>
    """
    )


@app.get("/api/portfolio")
async def get_portfolio():
    """Get enhanced portfolio data."""
    try:
        portfolio_data = get_portfolio_data()
        return portfolio_data
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/market-metrics")
async def get_market_metrics_api():
    """Get market metrics."""
    try:
        return get_market_metrics()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/trading-signals")
async def get_trading_signals_api():
    """Get trading signals."""
    try:
        return get_trading_signals()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/performance-history")
async def get_performance_history_api():
    """Get performance history."""
    try:
        return get_performance_history()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/trading-log")
async def get_trading_log_api():
    """Get trading log entries."""
    try:
        return {"log_entries": get_trading_log()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/news-sentiment")
async def get_news_sentiment_api():
    """Get news sentiment data."""
    try:
        return get_news_sentiment()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/recent-news")
async def get_recent_news_api():
    """Get recent crypto news."""
    try:
        return get_recent_news()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/market-model-5s")
async def get_market_model_5s_api():
    """Get 5-second market vs model comparison data."""
    try:
        return get_market_model_comparison_5s()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/market-model-1h")
async def get_market_model_1h_api():
    """Get 1-hour market vs model comparison data."""
    try:
        return get_market_model_comparison_1h()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/market-model-1d")
async def get_market_model_1d_api():
    """Get daily market vs model comparison data."""
    try:
        return get_market_model_comparison_1d()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/tft-predictions")
async def get_tft_predictions_api():
    """Get TFT model predictions."""
    try:
        result = get_tft_predictions()
        return clean_data_for_json(result)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/gcn-analysis")
async def get_gcn_analysis_api():
    """Get GCN cross-asset relationship analysis."""
    try:
        result = get_gcn_analysis()
        return clean_data_for_json(result)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/asset-relationships")
async def get_asset_relationships_api():
    """Get asset relationship matrix."""
    try:
        return get_asset_relationships()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolio-diversification")
async def get_portfolio_diversification_api():
    """Get portfolio diversification analysis."""
    try:
        return get_portfolio_diversification()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/risk-analysis")
async def get_risk_analysis_api():
    """Get comprehensive risk analysis."""
    try:
        return get_risk_analysis()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolio-risk")
async def get_portfolio_risk_api():
    """Get portfolio-level risk analysis."""
    try:
        return get_portfolio_risk()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/risk-metrics")
async def get_risk_metrics_api():
    """Get summary risk metrics."""
    try:
        return get_risk_metrics()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/copula-analysis")
async def get_copula_analysis_api():
    """Get copula dependency analysis."""
    try:
        result = get_copula_analysis()
        return clean_data_for_json(result)
    except Exception as e:
        logger.error(f"Copula analysis API error: {e}")
        return {"error": str(e)}


@app.get("/api/portfolio-dependencies")
async def get_portfolio_dependencies_api():
    """Get portfolio dependency analysis."""
    try:
        return get_portfolio_dependencies()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/dependency-matrix")
async def get_dependency_matrix_api():
    """Get dependency correlation matrix."""
    try:
        return get_dependency_matrix()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/statistical-arbitrage")
async def get_statistical_arbitrage_api():
    """Get statistical arbitrage analysis."""
    try:
        return get_statistical_arbitrage()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/slippage-monitoring")
async def get_slippage_monitoring_api():
    """Get real-time slippage monitoring data."""
    try:
        return get_slippage_monitoring()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/execution-optimization")
async def get_execution_optimization_api():
    """Get execution optimization recommendations."""
    try:
        return get_execution_optimization()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/enhanced-risk-metrics")
async def get_enhanced_risk_metrics_api():
    """Get enhanced risk metrics with cost analysis."""
    try:
        return get_enhanced_risk_metrics()
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/trade/buy")
async def execute_buy_trade(request: Request):
    """Execute a buy trade."""
    try:
        body = await request.json()
        asset = body.get("asset")  # 'BTC' or 'ETH'
        amount_usd = float(body.get("amount", 0))

        if not asset or amount_usd <= 0:
            return {"success": False, "error": "Invalid asset or amount"}

        # Map asset to symbol
        symbol = f"{asset}USDT"

        # Get current price with fallback
        current_price = get_latest_price(symbol)
        if not current_price:
            # Use fallback prices
            if symbol == "BTCUSDT":
                current_price = 117852.81
            elif symbol == "ETHUSDT":
                current_price = 3579.83
            else:
                return {"success": False, "error": f"Could not get price for {symbol}"}

        # Calculate position change
        position_change = amount_usd / current_price

        # Update position
        success = update_portfolio_position(symbol, position_change, current_price)

        if success:
            # Add to trading log
            add_trading_log_entry(
                action="buy",
                asset=asset,
                amount=amount_usd,
                price=current_price,
                success=True,
                message=f"Bought {position_change:.6f} units",
            )

            return {
                "success": True,
                "message": f"Bought ${amount_usd:.2f} worth of {asset} ({position_change:.6f} units) at ${current_price:.2f}",
                "trade": {
                    "asset": asset,
                    "amount_usd": amount_usd,
                    "units": position_change,
                    "price": current_price,
                },
            }
        else:
            # Add failed trade to log
            add_trading_log_entry(
                action="buy",
                asset=asset,
                amount=amount_usd,
                price=current_price,
                success=False,
                message="Failed to update position",
            )
            return {"success": False, "error": "Failed to update position"}

    except Exception as e:
        logger.error(f"Buy trade error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/trade/sell")
async def execute_sell_trade(request: Request):
    """Execute a sell trade."""
    try:
        body = await request.json()
        asset = body.get("asset")  # 'BTC' or 'ETH'
        percentage = float(body.get("percentage", 0))

        if not asset or percentage <= 0 or percentage > 100:
            return {"success": False, "error": "Invalid asset or percentage"}

        # Map asset to symbol
        symbol = f"{asset}USDT"

        # Get current position
        positions = get_portfolio_positions()
        current_position = positions[symbol]["position"]

        if current_position <= 0:
            return {"success": False, "error": f"No {asset} position to sell"}

        # Calculate position change (negative for sell)
        position_change = -(current_position * percentage / 100)

        # Get current price with fallback
        current_price = get_latest_price(symbol)
        if not current_price:
            # Use fallback prices
            if symbol == "BTCUSDT":
                current_price = 117852.81
            elif symbol == "ETHUSDT":
                current_price = 3579.83
            else:
                return {"success": False, "error": f"Could not get price for {symbol}"}

        # Update position
        success = update_portfolio_position(symbol, position_change, current_price)

        if success:
            amount_usd = abs(position_change) * current_price

            # Add to trading log
            add_trading_log_entry(
                action="sell",
                asset=asset,
                amount=amount_usd,
                price=current_price,
                success=True,
                message=f"Sold {percentage}% ({abs(position_change):.6f} units)",
            )

            return {
                "success": True,
                "message": f"Sold {percentage}% of {asset} position ({abs(position_change):.6f} units) for ${amount_usd:.2f} at ${current_price:.2f}",
                "trade": {
                    "asset": asset,
                    "percentage": percentage,
                    "units": abs(position_change),
                    "amount_usd": amount_usd,
                    "price": current_price,
                },
            }
        else:
            # Add failed trade to log
            add_trading_log_entry(
                action="sell",
                asset=asset,
                amount=0,
                price=current_price,
                success=False,
                message="Failed to update position",
            )
            return {"success": False, "error": "Failed to update position"}

    except Exception as e:
        logger.error(f"Sell trade error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    features = [
        "Enhanced Portfolio",
        "Market Metrics",
        "Trading Signals",
        "Performance Charts",
        "News Sentiment",
    ]
    if news_engine:
        features.append("Real-time News Integration")
    if tft_predictor:
        features.append("Temporal Fusion Transformer")
    if gcn_analyzer:
        features.append("Graph Neural Networks")
    if risk_manager:
        features.append("Expected Shortfall & EVT Risk Management")
    if copula_analyzer:
        features.append("Copula Correlation Modeling")
    if stat_arb_engine:
        features.append("Statistical Arbitrage")

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis_connected": redis_client is not None,
        "news_engine_active": news_engine is not None,
        "tft_predictor_active": tft_predictor is not None,
        "gcn_analyzer_active": gcn_analyzer is not None,
        "risk_manager_active": risk_manager is not None,
        "copula_analyzer_active": copula_analyzer is not None,
        "stat_arb_engine_active": stat_arb_engine is not None,
        "features": features,
    }


if __name__ == "__main__":
    print("🚀 Starting Enhanced Trading Dashboard...")
    print("📊 URL: http://localhost:8000")
    print(
        "✨ Features: Portfolio tracking, Market metrics, Trading signals, Performance charts"
    )
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
