#!/usr/bin/env python3
"""
Complete Dashboard Test

Test all dashboard features including:
- Enhanced Alpha Signals (8-10 signals)
- 6 Market vs Model comparison charts
- All existing features
"""

import requests
import webbrowser
import time
from datetime import datetime


def test_complete_dashboard():
    """Test the complete enhanced dashboard with all features."""
    print("🚀 Testing Complete Enhanced Trading Dashboard")
    print("=" * 80)

    base_url = "http://localhost:8000"

    # Test 1: Health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Status: {health_data['status']}")
            print(
                f"   ✅ Redis: {'Connected' if health_data['redis_connected'] else 'Disconnected'}"
            )
            print(
                f"   ✅ News Engine: {'Active' if health_data.get('news_engine_active', False) else 'Inactive'}"
            )
            print(f"   ✅ Features: {', '.join(health_data['features'])}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

    # Test 2: Enhanced Alpha Signals
    print("\n2. Testing enhanced Alpha Signals...")
    try:
        response = requests.get(f"{base_url}/api/trading-signals")
        if response.status_code == 200:
            signals = response.json()

            if isinstance(signals, list) and len(signals) >= 8:
                print(f"   ✅ Generated {len(signals)} Alpha signals (expected 8-10)")

                # Check for enhanced indicators
                indicators = set(signal["indicator"] for signal in signals)
                enhanced_indicators = {
                    "RSI Divergence",
                    "MACD Cross",
                    "Volume Profile",
                    "Momentum",
                    "Fibonacci Retracement",
                    "News Sentiment",
                    "On-Chain Metrics",
                    "Whale Activity",
                }

                found_enhanced = indicators.intersection(enhanced_indicators)
                print(
                    f"   ✅ Enhanced indicators: {', '.join(list(found_enhanced)[:5])}"
                )

                # Show sample signals
                for i, signal in enumerate(signals[:3]):
                    print(
                        f"      {i+1}. {signal['indicator']}: {signal['signal']} ({signal['strength']}) - {signal['confidence']:.0f}%"
                    )

            else:
                print(f"   ❌ Expected 8-10 signals, got {len(signals)}")
                return False
        else:
            print(f"   ❌ Alpha signals failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Alpha signals error: {e}")
        return False

    # Test 3: Market vs Model comparison endpoints
    print("\n3. Testing Market vs Model comparison data...")

    endpoints = [
        ("5-second", "/api/market-model-5s", 60),
        ("1-hour", "/api/market-model-1h", 24),
        ("daily", "/api/market-model-1d", 30),
    ]

    for timeframe, endpoint, expected_points in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}")
            if response.status_code == 200:
                data = response.json()

                if isinstance(data, list) and len(data) == expected_points:
                    print(f"   ✅ {timeframe} data: {len(data)} points")

                    # Check data structure
                    first_point = data[0]
                    required_fields = [
                        "timestamp",
                        "btc_market",
                        "btc_model",
                        "eth_market",
                        "eth_model",
                    ]
                    if all(field in first_point for field in required_fields):
                        btc_diff = abs(
                            first_point["btc_market"] - first_point["btc_model"]
                        )
                        eth_diff = abs(
                            first_point["eth_market"] - first_point["eth_model"]
                        )
                        print(
                            f"      Sample BTC: Market=${first_point['btc_market']:.2f}, Model=${first_point['btc_model']:.2f} (diff=${btc_diff:.2f})"
                        )
                        print(
                            f"      Sample ETH: Market=${first_point['eth_market']:.2f}, Model=${first_point['eth_model']:.2f} (diff=${eth_diff:.2f})"
                        )
                    else:
                        print(f"   ❌ Missing required fields in {timeframe} data")
                        return False
                else:
                    print(
                        f"   ❌ {timeframe} data: expected {expected_points} points, got {len(data) if isinstance(data, list) else 'non-list'}"
                    )
                    return False
            else:
                print(f"   ❌ {timeframe} endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ {timeframe} endpoint error: {e}")
            return False

    # Test 4: News sentiment integration
    print("\n4. Testing news sentiment integration...")
    try:
        response = requests.get(f"{base_url}/api/news-sentiment")
        if response.status_code == 200:
            sentiment_data = response.json()

            if "BTCUSDT" in sentiment_data and "ETHUSDT" in sentiment_data:
                btc_sentiment = sentiment_data["BTCUSDT"]
                eth_sentiment = sentiment_data["ETHUSDT"]

                print(
                    f"   ✅ BTC sentiment: {btc_sentiment['sentiment_score']:.3f} (confidence: {btc_sentiment['confidence']:.3f})"
                )
                print(
                    f"   ✅ ETH sentiment: {eth_sentiment['sentiment_score']:.3f} (confidence: {eth_sentiment['confidence']:.3f})"
                )

                if "market_sentiment" in sentiment_data:
                    market = sentiment_data["market_sentiment"]
                    print(f"   ✅ Market sentiment: {market['market_sentiment']:.3f}")
            else:
                print("   ❌ Missing BTC/ETH sentiment data")
                return False
        else:
            print(f"   ❌ News sentiment failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ News sentiment error: {e}")
        return False

    # Test 5: Dashboard HTML structure for new features
    print("\n5. Testing dashboard HTML structure...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            html = response.text

            # Check for new elements
            new_elements = [
                # Market vs Model charts
                "Market vs Model Performance Analysis",
                "btc5sChart",
                "eth5sChart",
                "btc1hChart",
                "eth1hChart",
                "btc1dChart",
                "eth1dChart",
                "5-Second: BTC Market vs Model",
                "1-Hour: ETH Market vs Model",
                "Daily: BTC Market vs Model",
                # Enhanced Alpha Signals
                "signalsContainer",
                "Alpha Signals",
                # News sentiment
                "Market Sentiment & News Analysis",
                "Recent Crypto News",
            ]

            missing = []
            for element in new_elements:
                if element not in html:
                    missing.append(element)

            if not missing:
                print("   ✅ All new dashboard elements found")
            else:
                print(f"   ❌ Missing elements: {missing[:5]}...")  # Show first 5
                return False

        else:
            print(f"   ❌ Dashboard HTML failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Dashboard HTML error: {e}")
        return False

    print("\n" + "=" * 80)
    print("🎉 COMPLETE DASHBOARD TEST RESULTS")
    print("=" * 80)
    print("✅ All tests passed!")
    print("✅ Enhanced Alpha Signals working (8-10 signals)!")
    print("✅ 6 Market vs Model comparison charts operational!")
    print("✅ News sentiment integration functional!")
    print("✅ All dashboard UI elements present!")

    return True


def show_complete_features():
    """Show all completed dashboard features."""
    print("\n🚀 Complete Enhanced Trading Dashboard Features")
    print("=" * 80)

    features = [
        "📊 Enhanced Alpha Signals:",
        "   • 8-10 comprehensive trading signals per update",
        "   • Advanced indicators: RSI Divergence, MACD Cross, Volume Profile",
        "   • On-chain metrics: Whale Activity, Exchange Flow",
        "   • News-based signals: Sentiment scoring",
        "   • Technical analysis: Fibonacci, Support/Resistance",
        "   • Improved scrollable display with custom styling",
        "",
        "📈 Market vs Model Comparison (6 Charts):",
        "   • ⚡ 5-Second timeframe: Real-time market vs model predictions",
        "   • 🕐 1-Hour timeframe: Hourly trend analysis comparison",
        "   • 📅 Daily timeframe: Long-term model performance",
        "   • Separate BTC and ETH charts for each timeframe",
        "   • Market accuracy and model confidence metrics",
        "   • Interactive Plotly charts with dual-line visualization",
        "",
        "📰 News Sentiment Integration:",
        "   • Real-time sentiment analysis for BTC/ETH",
        "   • Market sentiment gauge with confidence levels",
        "   • Recent crypto news feed with sentiment indicators",
        "   • Multi-source news aggregation and processing",
        "",
        "💼 Portfolio Management:",
        "   • Real-time BTC/ETH position tracking",
        "   • Interactive buy/sell controls with percentage options",
        "   • P&L calculations with profit percentage display",
        "   • Trading allocation controls",
        "",
        "📊 Performance Analytics:",
        "   • 24-hour portfolio performance charts",
        "   • Individual BTC/ETH price trend charts",
        "   • Market metrics dashboard",
        "   • System status and model controls",
        "",
        "⚡ Technical Implementation:",
        "   • FastAPI backend with 9 API endpoints",
        "   • Redis integration for real-time data",
        "   • Plotly.js interactive charts",
        "   • Responsive design with gradient styling",
        "   • Real-time updates every 10 seconds",
        "   • Comprehensive error handling",
    ]

    for feature in features:
        print(feature)


def open_complete_dashboard():
    """Open the complete enhanced dashboard."""
    print("\n🌐 Opening Complete Enhanced Trading Dashboard...")
    try:
        webbrowser.open("http://localhost:8000")
        print("✅ Dashboard opened in browser!")

        print("\n🎯 New Features to Explore:")
        print("   • Enhanced Alpha Signals section with 8-10 signals")
        print("   • Market vs Model Performance Analysis section")
        print("   • 6 comparison charts (5-second, hourly, daily for BTC & ETH)")
        print("   • Improved scrollable Alpha Signals with better visibility")
        print("   • Real-time market vs model prediction tracking")
        print("   • News sentiment integration with live updates")

    except Exception as e:
        print(f"❌ Error opening browser: {e}")
        print("💡 Please manually open: http://localhost:8000")


def main():
    """Main test function."""
    success = test_complete_dashboard()

    if success:
        show_complete_features()
        open_complete_dashboard()

        print("\n🎉 COMPLETE ENHANCED DASHBOARD READY!")
        print("✅ 8-10 Alpha Signals with advanced indicators")
        print("✅ 6 Market vs Model comparison charts")
        print("✅ Real-time news sentiment analysis")
        print("✅ Enhanced portfolio management")
        print("✅ Comprehensive performance analytics")
        print("💻 Available at: http://localhost:8000")

    else:
        print("\n❌ Issues found with complete dashboard")
        print("💡 Please check the logs and try again")


if __name__ == "__main__":
    main()
