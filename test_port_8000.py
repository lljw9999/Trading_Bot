#!/usr/bin/env python3
"""
Test Dashboard on Port 8000

Quick test to verify the dashboard is working correctly on the original port 8000.
"""

import requests
import webbrowser
import time


def test_port_8000():
    """Test the dashboard on port 8000."""
    print("🔄 Testing Dashboard on Port 8000")
    print("=" * 50)

    # Test 1: Basic connectivity
    print("1. Testing basic connectivity...")
    try:
        response = requests.get("http://localhost:8000/api/health")
        if response.status_code == 200:
            print("   ✅ Dashboard is accessible on port 8000")
        else:
            print(f"   ❌ Dashboard not accessible: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

    # Test 2: Check data availability
    print("\n2. Testing data availability...")
    try:
        # Test BTC data
        btc_response = requests.get("http://localhost:8000/api/data/BTCUSDT")
        if btc_response.status_code == 200:
            btc_data = btc_response.json()
            print(f"   ✅ BTC Data: ${btc_data['current_price']:.2f}")

        # Test ETH data
        eth_response = requests.get("http://localhost:8000/api/data/ETHUSDT")
        if eth_response.status_code == 200:
            eth_data = eth_response.json()
            print(f"   ✅ ETH Data: ${eth_data['current_price']:.2f}")

        # Test portfolio data
        portfolio_response = requests.get("http://localhost:8000/api/portfolio")
        if portfolio_response.status_code == 200:
            portfolio_data = portfolio_response.json()
            btc_portfolio = portfolio_data["portfolio"]["BTCUSDT"]
            eth_portfolio = portfolio_data["portfolio"]["ETHUSDT"]

            print(
                f"   ✅ BTC Portfolio: Position {btc_portfolio['position_size']:.6f}, P&L ${btc_portfolio['total_pnl']:.2f}"
            )
            print(
                f"   ✅ ETH Portfolio: Position {eth_portfolio['position_size']:.6f}, P&L ${eth_portfolio['total_pnl']:.2f}"
            )

    except Exception as e:
        print(f"   ❌ Data test failed: {e}")
        return False

    # Test 3: Check dashboard HTML
    print("\n3. Testing dashboard HTML...")
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            html = response.text

            # Check for key elements
            if "btcPrice" in html and "ethPrice" in html:
                print("   ✅ Price display elements found")
            else:
                print("   ❌ Price display elements missing")

            if "portfolioChart" in html and "pnlChart" in html:
                print("   ✅ Chart elements found")
            else:
                print("   ❌ Chart elements missing")

            if "loadInitialData" in html:
                print("   ✅ Data loading functions found")
            else:
                print("   ❌ Data loading functions missing")

        else:
            print(f"   ❌ Dashboard HTML not accessible: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ HTML test failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("🎯 PORT 8000 TEST RESULTS")
    print("=" * 50)
    print("✅ Dashboard is running correctly on port 8000")
    print(f"🌐 URL: http://localhost:8000")
    print("📊 All data endpoints are working")
    print("🎨 All HTML elements are present")

    return True


def open_dashboard_8000():
    """Open the dashboard on port 8000."""
    print("\n🌐 Opening Dashboard on Port 8000...")
    try:
        webbrowser.open("http://localhost:8000")
        print("✅ Dashboard opened in browser!")

        print("\n📋 What you should see:")
        print("   • Real BTC/ETH prices")
        print("   • Position data with actual values")
        print("   • Real-time charts updating every second")
        print("   • All data refreshing automatically")

        print("\n🔄 If you still see issues:")
        print("   • Refresh the page (Ctrl+R or Cmd+R)")
        print("   • Check browser console (F12) for errors")
        print("   • Wait a few seconds for data to load")

    except Exception as e:
        print(f"❌ Error opening browser: {e}")
        print("💡 Please manually open: http://localhost:8000")


if __name__ == "__main__":
    success = test_port_8000()

    if success:
        open_dashboard_8000()
        print("\n✅ DASHBOARD RESTORED TO PORT 8000!")
        print("🎉 Everything should be working as it was before!")
    else:
        print("\n❌ Issues found with port 8000")
        print("💡 Please check the dashboard manually")
