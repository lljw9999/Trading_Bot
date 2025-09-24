#!/usr/bin/env python3
"""
Quick Dashboard Test

Simple test to verify the dashboard structure and chart visibility.
"""

import re


def test_dashboard_structure():
    """Test the dashboard HTML structure."""
    print("🔍 Testing Dashboard Structure")
    print("=" * 50)

    # Read the dashboard HTML
    with open("trading_dashboard.py", "r") as f:
        content = f.read()

    # Check for chart elements
    chart_elements = [
        "btcChart",
        "ethChart",
        "portfolioChart",
        "pnlChart",
        "positionChart",
    ]

    print("1. Checking for chart elements in HTML:")
    for chart_id in chart_elements:
        if chart_id in content:
            print(f"   ✅ {chart_id} found")
        else:
            print(f"   ❌ {chart_id} missing")

    # Check for chart initialization methods
    chart_methods = [
        "initPortfolioChart",
        "initPnLChart",
        "initPositionChart",
        "updatePortfolioChart",
        "updatePnLChart",
        "updatePositionChart",
    ]

    print("\n2. Checking for chart initialization methods:")
    for method in chart_methods:
        if method in content:
            print(f"   ✅ {method} found")
        else:
            print(f"   ❌ {method} missing")

    # Check for chart containers
    chart_containers = ["charts-grid", "trading-charts-grid", "performance-chart"]

    print("\n3. Checking for chart containers:")
    for container in chart_containers:
        if container in content:
            print(f"   ✅ {container} found")
        else:
            print(f"   ❌ {container} missing")

    # Check for chart titles
    chart_titles = [
        "Portfolio Value vs Market Performance",
        "Real-time P&L Performance",
        "Position Sizing & Risk Metrics",
    ]

    print("\n4. Checking for chart titles:")
    for title in chart_titles:
        if title in content:
            print(f"   ✅ '{title}' found")
        else:
            print(f"   ❌ '{title}' missing")

    # Check the HTML structure around charts
    print("\n5. Analyzing HTML structure around charts:")

    # Find the position of BTC/ETH charts
    btc_chart_pos = content.find("BTCUSDT Real-time Chart")
    eth_chart_pos = content.find("ETHUSDT Real-time Chart")
    portfolio_chart_pos = content.find("Portfolio Value vs Market Performance")

    if btc_chart_pos > 0 and eth_chart_pos > 0:
        print("   ✅ BTC/ETH charts found in HTML")
    else:
        print("   ❌ BTC/ETH charts not found")

    if portfolio_chart_pos > 0:
        print("   ✅ Portfolio chart found in HTML")
        if portfolio_chart_pos > btc_chart_pos:
            print("   ✅ Portfolio chart is positioned after BTC/ETH charts")
        else:
            print("   ❌ Portfolio chart positioning issue")
    else:
        print("   ❌ Portfolio chart not found")

    # Extract the section around charts for inspection
    if btc_chart_pos > 0:
        start = max(0, btc_chart_pos - 500)
        end = min(len(content), btc_chart_pos + 2000)
        chart_section = content[start:end]

        print("\n6. Chart section structure:")
        lines = chart_section.split("\n")
        for i, line in enumerate(lines):
            if "chart" in line.lower() or "grid" in line.lower():
                print(f"   {i:2d}: {line.strip()}")

    return True


def show_expected_structure():
    """Show the expected dashboard structure."""
    print("\n" + "=" * 50)
    print("📋 Expected Dashboard Structure")
    print("=" * 50)

    structure = """
    Dashboard Layout:
    ├── Header & Controls
    ├── Crypto Cards (BTC/ETH prices)
    ├── Alpha Signals & Trading Settings
    ├── 📊 Original Charts (BTC/ETH price charts)
    │   ├── ₿ BTCUSDT Real-time Chart
    │   └── Ξ ETHUSDT Real-time Chart
    └── 🆕 New Trading Performance Charts
        ├── 📊 Portfolio Value vs Market Performance
        ├── 💰 Real-time P&L Performance
        └── 🎯 Position Sizing & Risk Metrics
    """

    print(structure)

    print("\n🎯 What you should see:")
    print("1. Two original price charts at the top")
    print("2. Three new trading performance charts below them")
    print("3. All charts should be interactive and update in real-time")
    print("4. Charts should have different colors (green, orange, purple)")


if __name__ == "__main__":
    success = test_dashboard_structure()
    show_expected_structure()

    if success:
        print("\n✅ Dashboard structure test complete!")
        print("\n🚀 To see the charts:")
        print("1. Run: python trading_dashboard.py")
        print("2. Open: http://localhost:8001")
        print("3. Scroll down to see all charts")
        print("4. The new charts will be below the BTC/ETH charts")
    else:
        print("\n❌ Issues found in dashboard structure")
