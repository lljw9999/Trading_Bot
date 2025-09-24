#!/usr/bin/env python3
"""
Test Enhanced Dashboard

Comprehensive test of all enhanced dashboard features.
"""

import requests
import webbrowser
import time


def test_enhanced_dashboard():
    """Test all enhanced dashboard features."""
    print("🚀 Testing Enhanced Trading Dashboard")
    print("=" * 60)

    # Test 1: Health check
    print("1. Testing health check...")
    try:
        response = requests.get("http://localhost:8000/api/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Status: {health_data['status']}")
            print(
                f"   ✅ Redis: {'Connected' if health_data['redis_connected'] else 'Disconnected'}"
            )
            print(f"   ✅ Features: {', '.join(health_data['features'])}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

    # Test 2: Portfolio data
    print("\n2. Testing enhanced portfolio data...")
    try:
        response = requests.get("http://localhost:8000/api/portfolio")
        if response.status_code == 200:
            data = response.json()

            btc = data.get("BTCUSDT", {})
            eth = data.get("ETHUSDT", {})

            if btc and eth:
                print(
                    f"   ✅ BTC: ${btc['current_price']:.2f} (24h: {btc['price_change_24h']:+.2f}%)"
                )
                print(
                    f"      Position: {btc['position_size']:.6f}, P&L: ${btc['total_pnl']:+.2f} ({btc['pnl_percentage']:+.2f}%)"
                )
                print(f"      Volume: ${btc['volume_24h']:,.0f}")

                print(
                    f"   ✅ ETH: ${eth['current_price']:.2f} (24h: {eth['price_change_24h']:+.2f}%)"
                )
                print(
                    f"      Position: {eth['position_size']:.6f}, P&L: ${eth['total_pnl']:+.2f} ({eth['pnl_percentage']:+.2f}%)"
                )
                print(f"      Volume: ${eth['volume_24h']:,.0f}")

                total_value = btc["current_value"] + eth["current_value"]
                total_pnl = btc["total_pnl"] + eth["total_pnl"]
                print(f"   ✅ Total: ${total_value:.2f}, P&L: ${total_pnl:+.2f}")
            else:
                print("   ❌ Portfolio data incomplete")
                return False
        else:
            print(f"   ❌ Portfolio data failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Portfolio data error: {e}")
        return False

    # Test 3: Market metrics
    print("\n3. Testing market metrics...")
    try:
        response = requests.get("http://localhost:8000/api/market-metrics")
        if response.status_code == 200:
            metrics = response.json()

            print(f"   ✅ Market Cap: ${metrics['total_market_cap']/1e12:.1f}T")
            print(f"   ✅ BTC Dominance: {metrics['btc_dominance']:.1f}%")
            print(f"   ✅ Fear & Greed Index: {metrics['fear_greed_index']}")
            print(f"   ✅ Active Addresses: {metrics['active_addresses']:,}")
            print(f"   ✅ Network Hash Rate: {metrics['network_hash_rate']:.1f} EH/s")
        else:
            print(f"   ❌ Market metrics failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Market metrics error: {e}")
        return False

    # Test 4: Trading signals
    print("\n4. Testing trading signals...")
    try:
        response = requests.get("http://localhost:8000/api/trading-signals")
        if response.status_code == 200:
            signals = response.json()

            if isinstance(signals, list) and len(signals) > 0:
                print(f"   ✅ Generated {len(signals)} trading signals:")
                for i, signal in enumerate(signals[:3]):  # Show first 3
                    print(
                        f"      {i+1}. {signal['indicator']}: {signal['signal']} ({signal['strength']}) - {signal['confidence']:.0f}%"
                    )
            else:
                print("   ❌ No trading signals generated")
                return False
        else:
            print(f"   ❌ Trading signals failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Trading signals error: {e}")
        return False

    # Test 5: Performance history
    print("\n5. Testing performance history...")
    try:
        response = requests.get("http://localhost:8000/api/performance-history")
        if response.status_code == 200:
            history = response.json()

            if isinstance(history, list) and len(history) > 0:
                print(f"   ✅ Generated {len(history)} historical data points")
                latest = history[-1]
                print(
                    f"      Latest: Portfolio ${latest['portfolio_value']:.2f}, P&L ${latest['pnl']:+.2f}"
                )
                print(
                    f"      BTC ${latest['btc_price']:.2f}, ETH ${latest['eth_price']:.2f}"
                )
            else:
                print("   ❌ No performance history generated")
                return False
        else:
            print(f"   ❌ Performance history failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Performance history error: {e}")
        return False

    # Test 6: Dashboard HTML
    print("\n6. Testing enhanced dashboard HTML...")
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            html = response.text

            # Check for enhanced elements
            enhanced_elements = [
                "Enhanced Trading Dashboard",
                "marketCap",
                "btcDominance",
                "fearGreed",
                "signalsContainer",
                "performanceChart",
                "btcChart",
                "ethChart",
                "fetchAllData",
                "updateMarketMetrics",
                "updateTradingSignals",
                "createPerformanceChart",
            ]

            missing = []
            for element in enhanced_elements:
                if element not in html:
                    missing.append(element)

            if not missing:
                print("   ✅ All enhanced HTML elements found")
            else:
                print(f"   ❌ Missing elements: {missing}")
                return False

        else:
            print(f"   ❌ Dashboard HTML failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Dashboard HTML error: {e}")
        return False

    print("\n" + "=" * 60)
    print("🎉 ENHANCED DASHBOARD TEST RESULTS")
    print("=" * 60)
    print("✅ All enhanced features working correctly!")

    return True


def show_enhanced_features():
    """Show what the enhanced dashboard includes."""
    print("\n🚀 Enhanced Dashboard Features")
    print("=" * 60)

    features = [
        "📊 Enhanced Portfolio Display",
        "   • Real-time BTC/ETH prices with 24h changes",
        "   • Position sizes, entry prices, current values",
        "   • P&L in both dollar amounts and percentages",
        "   • 24-hour trading volumes",
        "",
        "📈 Market Metrics Dashboard",
        "   • Total cryptocurrency market cap",
        "   • Bitcoin dominance percentage",
        "   • Fear & Greed Index",
        "   • Active wallet addresses",
        "   • Network hash rate",
        "",
        "🎯 Trading Signals Panel",
        "   • RSI, MACD, Bollinger Bands indicators",
        "   • BUY/SELL/HOLD recommendations",
        "   • Signal strength (WEAK/MODERATE/STRONG)",
        "   • Confidence percentages",
        "   • Real-time timestamps",
        "",
        "📊 Interactive Charts",
        "   • 24-hour portfolio performance chart",
        "   • BTC price trend visualization",
        "   • ETH price trend visualization",
        "   • Plotly.js interactive charts with zoom/pan",
        "",
        "🎨 Enhanced UI/UX",
        "   • Responsive design for all screen sizes",
        "   • Real-time color-coded P&L indicators",
        "   • Hover effects and smooth animations",
        "   • Professional gradient backgrounds",
        "   • Status bar with key metrics",
        "",
        "⚡ Real-time Updates",
        "   • Data refreshes every 10 seconds",
        "   • Live price updates",
        "   • Dynamic chart updates",
        "   • Auto-updating signals",
    ]

    for feature in features:
        print(feature)


def open_enhanced_dashboard():
    """Open the enhanced dashboard."""
    print("\n🌐 Opening Enhanced Trading Dashboard...")
    try:
        webbrowser.open("http://localhost:8000")
        print("✅ Enhanced dashboard opened in browser!")

        print("\n🎯 What you'll see:")
        print("   • Professional trading interface")
        print("   • Comprehensive market data")
        print("   • Interactive charts and visualizations")
        print("   • Real-time trading signals")
        print("   • Enhanced portfolio analytics")

        print("\n📊 Live Data:")
        try:
            response = requests.get("http://localhost:8000/api/portfolio")
            if response.status_code == 200:
                data = response.json()
                btc = data["BTCUSDT"]
                eth = data["ETHUSDT"]

                print(
                    f"   🔸 BTC: ${btc['current_price']:.2f} ({btc['price_change_24h']:+.2f}%)"
                )
                print(
                    f"   🔸 ETH: ${eth['current_price']:.2f} ({eth['price_change_24h']:+.2f}%)"
                )

                total_value = btc["current_value"] + eth["current_value"]
                total_pnl = btc["total_pnl"] + eth["total_pnl"]
                print(f"   🔸 Portfolio: ${total_value:.2f} (P&L: ${total_pnl:+.2f})")
        except:
            pass

    except Exception as e:
        print(f"❌ Error opening browser: {e}")
        print("💡 Please manually open: http://localhost:8000")


if __name__ == "__main__":
    success = test_enhanced_dashboard()

    if success:
        show_enhanced_features()
        open_enhanced_dashboard()

        print("\n🎉 ENHANCED DASHBOARD IS LIVE!")
        print("✅ Professional trading interface with advanced features")
        print("📊 Real-time data, charts, and trading signals")
        print("🚀 Built on the reliable foundation that works")

    else:
        print("\n❌ Issues found with enhanced dashboard")
        print("💡 Please check the logs")
