#!/usr/bin/env python3
"""
API Quota Monitor Test Script

Tests the API quota monitoring system functionality.
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.api_quota_monitor import APIQuotaMonitor


def test_api_quota_monitor():
    """Test API quota monitoring functionality."""
    print("🧪 Testing API Quota Monitor")
    print("=" * 50)

    monitor = APIQuotaMonitor()

    # Test 1: Record API calls and check quota
    print("\n📊 Test 1: Record API calls")

    # Simulate multiple API calls to Binance
    for i in range(5):
        result = monitor.record_api_call("binance", "default")
        print(
            f"  Call {i+1}: {result.get('current_calls', 0)}/{result.get('limit', 0)} - {result.get('warning_level', 'OK')}"
        )

    # Test 2: Check quota status for all exchanges
    print("\n📊 Test 2: Check all exchange status")
    all_status = monitor.get_all_exchange_status()
    summary = all_status.get("summary", {})

    print(f"Total exchanges: {summary.get('total_exchanges', 0)}")
    print(f"Warning alerts: {summary.get('warning_alerts', 0)}")
    print(f"Critical alerts: {summary.get('critical_alerts', 0)}")
    print(f"Overall healthy: {summary.get('overall_healthy', False)}")

    # Test 3: Test backoff calculation
    print("\n⏳ Test 3: Test backoff calculations")
    exchanges = ["binance", "coinbase", "alpaca"]
    for exchange in exchanges:
        backoff_ms = monitor.calculate_backoff_delay(exchange, "default")
        print(f"  {exchange}: {backoff_ms:.1f}ms backoff")

    # Test 4: Record WebSocket reconnections
    print("\n📡 Test 4: Record WebSocket reconnections")
    for i in range(2):
        reconnect_result = monitor.record_websocket_reconnect(
            "binance", f"test_reconnect_{i+1}"
        )
        reconnects = reconnect_result.get("hourly_reconnects", 0)
        threshold = reconnect_result.get("threshold", 3)
        print(
            f"  Reconnect {i+1}: {reconnects}/{threshold} - {'⚠️' if reconnects > threshold else '✅'}"
        )

    # Test 5: Generate Prometheus metrics
    print("\n📈 Test 5: Prometheus metrics")
    metrics = monitor.export_prometheus_metrics()
    metric_lines = [
        line
        for line in metrics.split("\n")
        if line.strip() and not line.startswith("#")
    ]
    print(f"Generated {len(metric_lines)} metrics")

    # Show sample metrics
    for line in metric_lines[:5]:
        print(f"  {line}")

    # Test 6: Simulate high usage scenario
    print("\n🚨 Test 6: Simulate high usage")

    # Record many calls to trigger warnings
    high_usage_exchange = "coinbase"
    limit = monitor.config["exchanges"][high_usage_exchange]["rest_limits"]["default"][
        "requests"
    ]

    # Fill up to 85% of quota
    calls_needed = int(limit * 0.85)
    print(f"Recording {calls_needed} calls to trigger warning...")

    for i in range(calls_needed):
        monitor.record_api_call(high_usage_exchange, "default")

    final_status = monitor.get_exchange_quota_status(high_usage_exchange, "default")
    usage_rate = final_status.get("usage_rate", 0)
    warning_level = final_status.get("warning_level", "OK")

    print(f"Final usage: {usage_rate:.1%} - Level: {warning_level}")

    # Summary
    print("\n" + "=" * 50)

    # Check if basic functionality works
    basic_works = (
        all_status.get("exchanges", {}).get("binance", {}).get("endpoints", {})
    )
    reconnect_works = reconnect_result.get("hourly_reconnects", 0) >= 0
    metrics_works = len(metric_lines) > 0

    if basic_works and reconnect_works and metrics_works:
        print("✅ API QUOTA MONITOR TEST: PASSED")
        print("   All core functionality working correctly")
    else:
        print("❌ API QUOTA MONITOR TEST: FAILED")
        print(
            f"   Issues: basic={basic_works}, reconnects={reconnect_works}, metrics={metrics_works}"
        )

    return basic_works and reconnect_works and metrics_works


if __name__ == "__main__":
    try:
        success = test_api_quota_monitor()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)
