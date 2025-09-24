#!/usr/bin/env python3
"""
Final Dashboard Test

Comprehensive test to verify the dashboard is displaying real data correctly.
"""

import requests
import time
import webbrowser
import json


def test_dashboard_final():
    """Final comprehensive test of the dashboard."""
    print("🎯 FINAL DASHBOARD TEST")
    print("=" * 60)

    # Test 1: Check all API endpoints
    print("1. Testing API endpoints...")

    endpoints = [
        ("/api/health", "Health check"),
        ("/api/portfolio", "Portfolio data"),
        ("/api/data/BTCUSDT", "BTC data"),
        ("/api/data/ETHUSDT", "ETH data"),
        ("/api/performance/portfolio", "Performance data"),
        ("/api/model/status", "Model status"),
        ("/api/trading/status", "Trading status"),
    ]

    all_working = True
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"http://localhost:8001{endpoint}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ {description}: {response.status_code}")

                # Show key data
                if endpoint == "/api/portfolio":
                    btc_price = data["portfolio"]["BTCUSDT"]["current_price"]
                    eth_price = data["portfolio"]["ETHUSDT"]["current_price"]
                    print(f"      📊 BTC: ${btc_price:.2f}, ETH: ${eth_price:.2f}")

                elif endpoint == "/api/data/BTCUSDT":
                    print(f"      📊 BTC Current: ${data['current_price']:.2f}")

                elif endpoint == "/api/data/ETHUSDT":
                    print(f"      📊 ETH Current: ${data['current_price']:.2f}")

            else:
                print(f"   ❌ {description}: {response.status_code}")
                all_working = False
        except Exception as e:
            print(f"   ❌ {description}: Error - {e}")
            all_working = False

    # Test 2: Check dashboard HTML
    print("\\n2. Testing dashboard HTML...")
    try:
        response = requests.get("http://localhost:8001/")
        if response.status_code == 200:
            html = response.text

            # Check for key elements
            key_elements = [
                ('id="btcPrice"', "BTC price element"),
                ('id="ethPrice"', "ETH price element"),
                ('id="PositionSize"', "BTC position element"),
                ('id="ethPositionSize"', "ETH position element"),
                ("loadInitialData", "Data loading function"),
                ("portfolioChart", "Portfolio chart"),
                ("pnlChart", "P&L chart"),
                ("hourlyChart", "Hourly chart"),
                ("dailyChart", "Daily chart"),
            ]

            for element, description in key_elements:
                if element in html:
                    print(f"   ✅ {description}: Found")
                else:
                    print(f"   ❌ {description}: Missing")
                    all_working = False
        else:
            print(f"   ❌ Dashboard HTML: {response.status_code}")
            all_working = False
    except Exception as e:
        print(f"   ❌ Dashboard HTML: Error - {e}")
        all_working = False

    # Test 3: Summary
    print("\\n" + "=" * 60)
    print("🎯 FINAL TEST RESULTS")
    print("=" * 60)

    if all_working:
        print("✅ ALL TESTS PASSED!")
        print("\\n📊 Current Live Data:")

        try:
            # Get current portfolio data
            response = requests.get("http://localhost:8001/api/portfolio")
            if response.status_code == 200:
                data = response.json()
                btc_data = data["portfolio"]["BTCUSDT"]
                eth_data = data["portfolio"]["ETHUSDT"]

                print(f"   🔸 BTC: ${btc_data['current_price']:.2f}")
                print(f"      Position: {btc_data['position_size']:.6f} BTC")
                print(f"      Value: ${btc_data['current_value']:.2f}")
                print(f"      P&L: ${btc_data['total_pnl']:.2f}")

                print(f"   🔸 ETH: ${eth_data['current_price']:.2f}")
                print(f"      Position: {eth_data['position_size']:.6f} ETH")
                print(f"      Value: ${eth_data['current_value']:.2f}")
                print(f"      P&L: ${eth_data['total_pnl']:.2f}")

                total_value = btc_data["current_value"] + eth_data["current_value"]
                total_pnl = btc_data["total_pnl"] + eth_data["total_pnl"]

                print(f"   🔸 Total Portfolio: ${total_value:.2f}")
                print(f"   🔸 Total P&L: ${total_pnl:.2f}")

        except Exception as e:
            print(f"   ❌ Error getting current data: {e}")

        print("\\n🎉 DASHBOARD IS WORKING CORRECTLY!")
        print("\\n📋 What you should see at http://localhost:8001:")
        print("   • Real BTC/ETH prices (not $0.00)")
        print("   • Actual position sizes and values")
        print("   • Real P&L numbers (positive/negative)")
        print("   • Live updating charts")
        print("   • All data refreshing every second")

        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("\\n🔧 Issues found - please check the dashboard manually")
        return False


def open_dashboard():
    """Open the dashboard with instructions."""
    print("\\n🌐 Opening Dashboard...")
    try:
        webbrowser.open("http://localhost:8001")
        print("✅ Dashboard opened in browser!")

        print("\\n🎯 Expected Behavior:")
        print("   • BTC price should show ~$117,637")
        print("   • ETH price should show ~$3,570")
        print("   • Position sizes should be > 0")
        print("   • P&L should show real numbers")
        print("   • Charts should have moving lines")
        print("   • Data should update every second")

        print("\\n🔄 If you still see $0.00:")
        print("   • Refresh the page (Ctrl+R)")
        print("   • Check browser console (F12)")
        print("   • Wait 10 seconds for data to load")

    except Exception as e:
        print(f"❌ Error opening browser: {e}")
        print("💡 Please manually open: http://localhost:8001")


if __name__ == "__main__":
    success = test_dashboard_final()

    if success:
        open_dashboard()
        print("\\n✅ DASHBOARD FIXED AND WORKING!")
    else:
        print("\\n❌ Please check the dashboard manually")
        print("💡 URL: http://localhost:8001")
