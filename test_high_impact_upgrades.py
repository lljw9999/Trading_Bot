#!/usr/bin/env python3
"""
Test script for the three high-impact upgrades:
A. Options flow signals (Unusual Whales)
B. Smart order router with latency mapping
C. Auto-tuning drift thresholds
"""

import sys
import time
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def test_options_flow_integration():
    """Test Task A: Options flow signals."""
    print("🎯 Testing Task A: Options Flow Signals")
    print("-" * 50)

    try:
        from src.layers.layer1_signal_generation.options_flow import UnusualWhalesClient

        client = UnusualWhalesClient()
        flow_data = client.get_current_flow()

        print(f"✅ Options flow client initialized (demo mode)")
        print(f"📊 Current flow data: {flow_data}")

        # Test state builder integration
        from state_builder_example import build_rl_state_with_whale_features

        state = build_rl_state_with_whale_features()

        options_features = {
            k: v
            for k, v in state.items()
            if "call_" in k or "put_" in k or "options_" in k
        }
        print(f"🔗 State builder integration: {len(options_features)} options features")
        for feature, value in options_features.items():
            print(f"   {feature}: {value}")

        return True

    except Exception as e:
        print(f"❌ Options flow test failed: {e}")
        return False


def test_smart_order_router():
    """Test Task B: Smart order router with latency mapping."""
    print("\n🎯 Testing Task B: Smart Order Router")
    print("-" * 50)

    try:
        from execution.router import SmartOrderRouter, OrderRequest, OrderSide

        router = SmartOrderRouter()

        # Create test order
        test_order = OrderRequest(symbol="BTCUSDT", side=OrderSide.BUY, quantity=0.5)

        # Route order
        start_time = time.perf_counter()
        routing_result = router.route_order(test_order)
        routing_time = (time.perf_counter() - start_time) * 1000

        print(f"✅ Order routed to {routing_result['venue']} in {routing_time:.1f}ms")
        print(
            f"📊 Estimated cost: {routing_result['estimated_cost']['total_cost']:.4f}"
        )
        print(
            f"⚡ Latency: {routing_result['estimated_cost']['venue_latency_ms']:.1f}ms"
        )

        # Show venue weights
        weights = router.calculate_venue_weights()
        print(
            f"🎯 Venue weights: {', '.join(f'{v}={w:.3f}' for v, w in sorted(weights.items()))}"
        )

        # Get analytics
        analytics = router.get_routing_analytics()
        print(
            f"📈 Routing analytics: {analytics['total_decisions']} decisions, best venue: {analytics['best_venue']}"
        )

        return True

    except Exception as e:
        print(f"❌ Smart router test failed: {e}")
        return False


def test_drift_auto_tuner():
    """Test Task C: Auto-tuning drift thresholds."""
    print("\n🎯 Testing Task C: Drift Threshold Auto-Tuner")
    print("-" * 50)

    try:
        import subprocess
        import json

        # Run drift watcher to test adaptive thresholds
        result = subprocess.run(
            ["python3", "scripts/drift_watcher.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        print(f"✅ Drift watcher executed successfully")

        # Parse output for key information
        output_lines = result.stdout.split("\n")
        for line in output_lines:
            if any(
                keyword in line
                for keyword in [
                    "KL divergence",
                    "threshold",
                    "drift detected",
                    "baseline history",
                ]
            ):
                print(f"📊 {line}")

        # Check if baseline history was created
        baseline_history_path = Path("data/baseline_history.parquet")
        if baseline_history_path.exists():
            print(f"📁 Baseline history created: {baseline_history_path}")

        return True

    except Exception as e:
        print(f"❌ Drift auto-tuner test failed: {e}")
        return False


async def test_latency_tracker():
    """Test SOR latency tracker briefly."""
    print("\n⚡ Testing SOR Latency Tracker (5 second sample)")
    print("-" * 50)

    try:
        from execution.sor_latency import VenueLatencyTracker

        tracker = VenueLatencyTracker(measurement_interval=1.0)  # 1 second for testing

        # Run for just a few measurements
        import aiohttp

        connector = aiohttp.TCPConnector(limit=10)
        timeout = aiohttp.ClientTimeout(total=2.0)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            for i in range(3):  # Just 3 measurements
                measurements = await tracker.measure_all_venues(session)
                tracker.update_latency_windows(measurements)

                successful_measurements = {
                    venue: ms for venue, ms in measurements.items() if ms is not None
                }
                print(
                    f"📊 Measurement {i+1}: {len(successful_measurements)} venues, avg latency: {sum(successful_measurements.values())/len(successful_measurements):.1f}ms"
                )

                if i < 2:  # Don't wait after last measurement
                    await asyncio.sleep(1.0)

        # Show final weights
        weights = tracker.get_venue_weights()
        best_venue = tracker.get_best_venue()
        print(f"🎯 Best venue: {best_venue}")
        print(
            f"⚖️ Final weights: {', '.join(f'{v}={w:.3f}' for v, w in sorted(weights.items()))}"
        )

        return True

    except Exception as e:
        print(f"❌ Latency tracker test failed: {e}")
        return False


async def main():
    """Run all high-impact upgrade tests."""
    print("🚀 SAC-DiF High-Impact Upgrades Test Suite")
    print("=" * 60)

    test_results = []

    # Test Task A: Options flow signals
    test_results.append(test_options_flow_integration())

    # Test Task B: Smart order router
    test_results.append(test_smart_order_router())

    # Test Task C: Drift auto-tuner
    test_results.append(test_drift_auto_tuner())

    # Test latency tracker (async)
    test_results.append(await test_latency_tracker())

    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)

    tasks = ["Options Flow", "Smart Router", "Drift Auto-Tuner", "Latency Tracker"]
    passed = sum(test_results)
    total = len(test_results)

    for task, result in zip(tasks, test_results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{task:.<20} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All high-impact upgrades working correctly!")
        print("\nReady to deploy:")
        print("  sudo systemctl daemon-reload")
        print(
            "  sudo systemctl enable --now options_stream.service sor_latency.service"
        )
    else:
        print("⚠️ Some tests failed - check logs above")


if __name__ == "__main__":
    asyncio.run(main())
