#!/usr/bin/env python3
"""
Debug Charts Loading

Check if charts are properly loading and receiving data.
"""

import requests
import time
from datetime import datetime


def debug_dashboard_charts():
    """Debug chart loading issues."""
    print("🔍 Debugging Dashboard Charts")
    print("=" * 50)

    # Test 1: Check if dashboard HTML contains chart elements
    print("1. Checking HTML structure...")
    try:
        response = requests.get("http://localhost:8001/")
        html_content = response.text

        # Check for chart containers
        charts_to_check = [
            ("portfolioChart", "Portfolio Value vs Market Performance"),
            ("pnlChart", "Real-time P&L Performance"),
            ("positionChart", "Position Sizing & Risk Metrics"),
            ("hourlyChart", "Hourly Performance"),
            ("fiveHourChart", "5-Hour Performance"),
            ("dailyChart", "Daily Performance"),
        ]

        for chart_id, title in charts_to_check:
            if chart_id in html_content:
                print(f"   ✅ {chart_id} container found")
            else:
                print(f"   ❌ {chart_id} container missing")

            if title in html_content:
                print(f"   ✅ '{title}' title found")
            else:
                print(f"   ❌ '{title}' title missing")

        # Check for JavaScript functions
        js_functions = [
            "initPortfolioChart",
            "initPnLChart",
            "initPositionChart",
            "initHourlyChart",
            "initFiveHourChart",
            "initDailyChart",
            "updateAllCharts",
            "startRealTimeUpdates",
        ]

        print("\n2. Checking JavaScript functions...")
        for func in js_functions:
            if func in html_content:
                print(f"   ✅ {func} found")
            else:
                print(f"   ❌ {func} missing")

        # Check if Plotly is loaded
        if "plot.ly" in html_content or "plotly" in html_content:
            print("   ✅ Plotly library loaded")
        else:
            print("   ❌ Plotly library not loaded")

    except Exception as e:
        print(f"   ❌ Error checking HTML: {e}")
        return False

    # Test 2: Check API endpoints
    print("\n3. Checking API endpoints...")
    api_endpoints = [
        "/api/performance/portfolio",
        "/api/performance/pnl",
        "/api/performance/positions",
    ]

    for endpoint in api_endpoints:
        try:
            response = requests.get(f"http://localhost:8001{endpoint}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ {endpoint} - Status: {response.status_code}")
                if "timestamps" in data:
                    print(f"      📊 Data points: {len(data['timestamps'])}")
                else:
                    print(f"      📊 Response keys: {list(data.keys())}")
            else:
                print(f"   ❌ {endpoint} - Status: {response.status_code}")
        except Exception as e:
            print(f"   ❌ {endpoint} - Error: {e}")

    # Test 3: Check if sample data is being generated
    print("\n4. Checking sample data generation...")
    try:
        # Check BTC data
        btc_response = requests.get("http://localhost:8001/api/data/BTCUSDT")
        if btc_response.status_code == 200:
            btc_data = btc_response.json()
            print(
                f"   ✅ BTC data available - Current price: ${btc_data['current_price']:.2f}"
            )
            print(f"   📊 Price history points: {len(btc_data['price_history'])}")

        # Check ETH data
        eth_response = requests.get("http://localhost:8001/api/data/ETHUSDT")
        if eth_response.status_code == 200:
            eth_data = eth_response.json()
            print(
                f"   ✅ ETH data available - Current price: ${eth_data['current_price']:.2f}"
            )
            print(f"   📊 Price history points: {len(eth_data['price_history'])}")

    except Exception as e:
        print(f"   ❌ Error checking price data: {e}")

    # Test 4: Suggest solutions
    print("\n5. Potential solutions if charts aren't showing:")
    print("   💡 Try refreshing the browser page")
    print("   💡 Check browser console for JavaScript errors (F12)")
    print("   💡 Ensure you're scrolling down to see all charts")
    print("   💡 The charts should appear below the BTC/ETH price charts")
    print("   💡 Clear browser cache and reload")

    # Test 5: Generate test HTML
    print("\n6. Creating test HTML file...")
    test_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Chart Test</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="testChart" style="width:100%;height:400px;"></div>
    <script>
        // Simple test chart
        const trace = {
            x: [1, 2, 3, 4, 5],
            y: [10, 20, 15, 25, 30],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Test Data'
        };
        
        const layout = {
            title: 'Test Chart - If you see this, Plotly is working!',
            xaxis: { title: 'X Axis' },
            yaxis: { title: 'Y Axis' }
        };
        
        Plotly.newPlot('testChart', [trace], layout);
    </script>
</body>
</html>
"""

    with open("/tmp/test_chart.html", "w") as f:
        f.write(test_html)

    print("   ✅ Test chart created at: /tmp/test_chart.html")
    print("   💡 Open this file in browser to test if Plotly works")

    return True


if __name__ == "__main__":
    debug_dashboard_charts()

    print("\n" + "=" * 50)
    print("🎯 DASHBOARD DEBUG COMPLETE")
    print("=" * 50)
    print("📋 Next steps:")
    print("1. Open http://localhost:8001 in your browser")
    print("2. Look for the charts below the BTC/ETH price displays")
    print("3. Check browser console (F12) for any JavaScript errors")
    print("4. If charts still don't show, try refreshing the page")
    print("5. The charts should be loading with sample data immediately")
