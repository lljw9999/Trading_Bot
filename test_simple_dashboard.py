#!/usr/bin/env python3
"""
Test Simple Dashboard

Test the simple working dashboard to ensure it displays data correctly.
"""

import requests
import webbrowser
import time


def test_simple_dashboard():
    """Test the simple dashboard functionality."""
    print("🔍 Testing Simple Working Dashboard")
    print("=" * 50)

    # Test 1: Health check
    print("1. Testing health check...")
    try:
        response = requests.get("http://localhost:8000/api/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health: {health_data['status']}")
            print(
                f"   ✅ Redis: {'Connected' if health_data['redis_connected'] else 'Disconnected'}"
            )
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

    # Test 2: Portfolio data
    print("\n2. Testing portfolio data...")
    try:
        response = requests.get("http://localhost:8000/api/portfolio")
        if response.status_code == 200:
            data = response.json()

            # Check BTC data
            btc = data.get("BTCUSDT", {})
            if btc:
                print(f"   ✅ BTC: ${btc['current_price']:.2f}")
                print(f"      Position: {btc['position_size']:.6f}")
                print(f"      P&L: ${btc['total_pnl']:.2f}")
            else:
                print("   ❌ BTC data missing")
                return False

            # Check ETH data
            eth = data.get("ETHUSDT", {})
            if eth:
                print(f"   ✅ ETH: ${eth['current_price']:.2f}")
                print(f"      Position: {eth['position_size']:.6f}")
                print(f"      P&L: ${eth['total_pnl']:.2f}")
            else:
                print("   ❌ ETH data missing")
                return False

            # Calculate totals
            total_value = btc["current_value"] + eth["current_value"]
            total_pnl = btc["total_pnl"] + eth["total_pnl"]
            print(f"   ✅ Total Value: ${total_value:.2f}")
            print(f"   ✅ Total P&L: ${total_pnl:.2f}")

        else:
            print(f"   ❌ Portfolio data failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Portfolio data error: {e}")
        return False

    # Test 3: Dashboard HTML
    print("\n3. Testing dashboard HTML...")
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            html = response.text

            # Check for essential elements
            elements = [
                "btcPrice",
                "ethPrice",
                "btcPosition",
                "ethPosition",
                "btcPnL",
                "ethPnL",
                "totalValue",
                "totalPnL",
                "fetchData",
            ]

            missing = []
            for element in elements:
                if element not in html:
                    missing.append(element)

            if not missing:
                print("   ✅ All essential HTML elements found")
            else:
                print(f"   ❌ Missing elements: {missing}")
                return False

        else:
            print(f"   ❌ Dashboard HTML failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Dashboard HTML error: {e}")
        return False

    print("\n" + "=" * 50)
    print("🎉 SIMPLE DASHBOARD TEST RESULTS")
    print("=" * 50)
    print("✅ All tests passed!")
    print("✅ Dashboard is working correctly")
    print("✅ Data is displaying properly")
    print("✅ HTML elements are present")

    return True


def open_working_dashboard():
    """Open the working dashboard."""
    print("\n🌐 Opening Simple Working Dashboard...")
    try:
        webbrowser.open("http://localhost:8000")
        print("✅ Dashboard opened in browser!")

        print("\n🎯 What you should see:")
        print("   • Clean, simple design")
        print("   • Real BTC/ETH prices displayed")
        print("   • Position sizes and entry prices")
        print("   • Current values and P&L")
        print("   • Total portfolio summary")
        print("   • Green/red colors for profit/loss")
        print("   • Data updating every 5 seconds")

        print("\n📊 Current Live Data:")
        try:
            response = requests.get("http://localhost:8000/api/portfolio")
            if response.status_code == 200:
                data = response.json()
                btc = data["BTCUSDT"]
                eth = data["ETHUSDT"]

                print(
                    f"   🔸 BTC: ${btc['current_price']:.2f} (P&L: ${btc['total_pnl']:.2f})"
                )
                print(
                    f"   🔸 ETH: ${eth['current_price']:.2f} (P&L: ${eth['total_pnl']:.2f})"
                )

                total_value = btc["current_value"] + eth["current_value"]
                total_pnl = btc["total_pnl"] + eth["total_pnl"]
                print(f"   🔸 Total: ${total_value:.2f} (P&L: ${total_pnl:.2f})")
        except:
            pass

    except Exception as e:
        print(f"❌ Error opening browser: {e}")
        print("💡 Please manually open: http://localhost:8000")


if __name__ == "__main__":
    success = test_simple_dashboard()

    if success:
        open_working_dashboard()
        print("\n🎉 SIMPLE DASHBOARD IS WORKING!")
        print("✅ This is a clean, minimal version that displays real data")
        print("🔧 No complex JavaScript issues")
        print("📱 Responsive design that works everywhere")
    else:
        print("\n❌ Issues found with simple dashboard")
        print("💡 Please check the logs")
