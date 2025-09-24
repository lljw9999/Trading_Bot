#!/usr/bin/env python3
"""
Test Stocks Session Components
"""

import asyncio
import logging
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import components
from src.layers.layer1_alpha_models.mean_rev import MeanReversionAlpha
from src.layers.layer3_position_sizing.kelly_sizing import KellySizing
from src.layers.layer5_risk.basic_risk_manager import BasicRiskManager


async def test_components():
    """Test individual components."""
    print("🧪 Testing Stocks Session Components...")

    # Test 1: Alpha model
    print("\n1. Testing Mean Reversion Alpha...")
    alpha = MeanReversionAlpha()

    # Feed some test data
    for i in range(15):
        price = 150.0 + i * 0.5  # Rising price
        signal = alpha.update_price("AAPL", price, f"2025-01-15T10:{i:02d}:00Z")
        if signal:
            print(
                f"   Signal: edge={signal.edge_bps:.1f}bps, conf={signal.confidence:.2f}"
            )

    # Test 2: Kelly sizing
    print("\n2. Testing Kelly Sizing...")
    kelly = KellySizing()
    pos, reason = kelly.calculate_position_size(
        "AAPL", 10.0, 0.8, Decimal("150"), Decimal("100000"), "stocks"
    )
    print(f"   Position: ${pos:.0f}")
    print(f"   Reason: {reason}")

    # Test 3: Risk manager
    print("\n3. Testing Risk Manager...")
    risk_mgr = BasicRiskManager()
    allowed, risk_reason, max_allowed = risk_mgr.check_position_risk(
        "AAPL", pos, Decimal("150"), Decimal("100000")
    )
    print(f"   Risk check: {allowed}")
    print(f"   Reason: {risk_reason}")
    print(f"   Max allowed: ${max_allowed:.0f}")

    # Test 4: Integration test
    print("\n4. Integration Test...")
    print("   L1 Alpha → L3 Kelly → L5 Risk pipeline:")

    # Simulate a strong signal
    test_signal = alpha.update_price(
        "AAPL", 148.0, "2025-01-15T10:20:00Z"
    )  # Price drop
    if test_signal:
        print(f"   📊 Alpha signal: {test_signal.edge_bps:.1f}bps")

        # Kelly sizing
        pos, reason = kelly.calculate_position_size(
            "AAPL",
            test_signal.edge_bps,
            test_signal.confidence,
            Decimal("148"),
            Decimal("100000"),
            "stocks",
        )
        print(f"   💰 Kelly position: ${pos:.0f}")

        # Risk check
        allowed, risk_reason, max_allowed = risk_mgr.check_position_risk(
            "AAPL", pos, Decimal("148"), Decimal("100000")
        )
        print(f"   🛡️  Risk check: {'✅ APPROVED' if allowed else '❌ REJECTED'}")

        if allowed:
            print(f"   🎯 Final position: ${max_allowed:.0f}")
            print("   ✅ Order would be submitted to L4 Executor")

    print("\n🎉 All component tests completed!")


if __name__ == "__main__":
    asyncio.run(test_components())
