#!/usr/bin/env python3
"""
Time Sync Monitor Test Script

Tests the time synchronization monitoring system.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.time_sync_monitor import TimeSyncMonitor


def test_time_sync_monitor():
    """Test time synchronization monitor."""
    print("🧪 Testing Time Sync Monitor")
    print("=" * 50)

    monitor = TimeSyncMonitor()

    # Test 1: Get system time status
    print("\n⏰ Test 1: Get system time status")
    time_status = monitor.get_system_time_status()
    print(
        f"System time: {time_status.get('system_time', {}).get('utc_time', 'Unknown')}"
    )
    print(
        f"Sync method: {time_status.get('sync_health', {}).get('sync_method', 'Unknown')}"
    )
    print(
        f"Max skew: {time_status.get('sync_health', {}).get('max_skew_ms', 'Unknown')}ms"
    )

    # Test 2: Check health status
    print("\n💊 Test 2: Check time sync health")
    health_check = monitor.check_time_sync_health()
    healthy = health_check.get("healthy", False)
    skew = health_check.get("max_skew_ms", -1)
    alert_level = health_check.get("alert_level", "Unknown")

    print(f"Healthy: {healthy}")
    print(f"Clock skew: {skew:.1f}ms")
    print(f"Alert level: {alert_level}")

    # Test 3: Generate Prometheus metrics
    print("\n📊 Test 3: Prometheus metrics export")
    metrics = monitor.export_prometheus_metrics()
    print("Prometheus metrics:")
    for line in metrics.strip().split("\n")[:5]:  # Show first 5 metrics
        print(f"  {line}")

    # Test 4: Run monitoring cycle
    print("\n🔄 Test 4: Run monitoring cycle")
    cycle_results = monitor.run_monitoring_cycle()
    cycle_health = cycle_results.get("health_check", {})

    print(f"Cycle completed: {cycle_results.get('timestamp', 'Unknown')}")
    print(f"Health status: {cycle_health.get('healthy', False)}")
    print(f"Alert level: {cycle_health.get('alert_level', 'Unknown')}")

    # Summary
    print("\n" + "=" * 50)
    overall_healthy = health_check.get("healthy", False)
    skew_acceptable = skew >= 0 and skew <= 1000  # Within 1 second is reasonable

    if overall_healthy or skew_acceptable:
        print("✅ TIME SYNC MONITOR TEST: PASSED")
        print("   Time synchronization monitoring working correctly")
    else:
        print("❌ TIME SYNC MONITOR TEST: FAILED")
        print(f"   Issues detected: skew={skew}ms, healthy={overall_healthy}")

    return overall_healthy or skew_acceptable


if __name__ == "__main__":
    try:
        success = test_time_sync_monitor()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)
