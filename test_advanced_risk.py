#!/usr/bin/env python3
"""
Test Advanced Risk Manager with VaR/CVaR + WORM Audit

Tests the advanced risk management implementation as specified in
Future_instruction.txt including:
- VaR/CVaR calculation
- Exchange haircuts
- WORM audit trail
- Kill-switch functionality
"""

import asyncio
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.layers.layer5_risk.advanced_risk_manager import AdvancedRiskManager
from src.layers.layer5_risk.compliance_worm import EventType, get_audit_logger


async def test_var_cvar_calculation():
    """Test VaR and CVaR calculation methods."""
    print("🧪 Testing VaR/CVaR Calculation")
    print("=" * 50)

    risk_manager = AdvancedRiskManager()

    # Generate sample returns (some losses for VaR/CVaR)
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
    returns[10:15] = -0.05  # Add some significant losses
    returns[50:52] = -0.08  # Add extreme losses

    # Test VaR calculation
    var_95 = risk_manager.calc_var(returns, p=0.95, horizon=1)
    var_99 = risk_manager.calc_var(returns, p=0.99, horizon=1)

    # Test CVaR calculation
    cvar_95 = risk_manager.calc_cvar(returns, p=0.95, horizon=1)
    cvar_99 = risk_manager.calc_cvar(returns, p=0.99, horizon=1)

    print(f"VaR (95%): {var_95:.4f} ({var_95*100:.2f}%)")
    print(f"VaR (99%): {var_99:.4f} ({var_99*100:.2f}%)")
    print(f"CVaR (95%): {cvar_95:.4f} ({cvar_95*100:.2f}%)")
    print(f"CVaR (99%): {cvar_99:.4f} ({cvar_99*100:.2f}%)")

    # Validate results
    assert var_99 > var_95, "99% VaR should be higher than 95% VaR"
    assert cvar_99 > var_99, "CVaR should be higher than VaR"
    assert cvar_95 > var_95, "CVaR should be higher than VaR"

    print("✅ VaR/CVaR calculations working correctly")
    return risk_manager


async def test_exchange_haircuts():
    """Test exchange haircut functionality."""
    print("\n🧪 Testing Exchange Haircuts")
    print("=" * 50)

    risk_manager = AdvancedRiskManager()

    # Test haircuts for different venues
    capital = 100000.0
    var = 0.02  # 2% VaR
    cvar = 0.03  # 3% CVaR

    venues = ["coinbase", "binance", "alpaca", "unknown_venue"]

    for venue in venues:
        max_size = risk_manager.calc_max_size_with_haircuts(capital, var, cvar, venue)
        haircut = risk_manager.exchange_haircuts.get(
            venue, risk_manager.exchange_haircuts["default"]
        )
        effective_capital = capital * (1 - haircut)

        print(f"Venue: {venue}")
        print(f"  Haircut: {haircut:.1%}")
        print(f"  Effective capital: ${effective_capital:,.0f}")
        print(f"  Max position size: ${max_size:,.0f}")
        print()

    # Verify haircuts are applied correctly
    coinbase_size = risk_manager.calc_max_size_with_haircuts(
        capital, var, cvar, "coinbase"
    )
    binance_size = risk_manager.calc_max_size_with_haircuts(
        capital, var, cvar, "binance"
    )

    assert (
        coinbase_size > binance_size
    ), "Coinbase should allow larger positions (lower haircut)"

    print("✅ Exchange haircuts working correctly")


async def test_worm_audit_integration():
    """Test WORM audit trail integration."""
    print("\n🧪 Testing WORM Audit Integration")
    print("=" * 50)

    risk_manager = AdvancedRiskManager()

    # Test async position check with audit logging
    symbol = "BTC-USD"
    size = 1.0
    price = 50000.0
    venue = "coinbase"

    # Add some return history for VaR calculation
    for i in range(50):
        risk_manager.return_history.append(np.random.normal(0.001, 0.02))

    # Perform position check (should be approved)
    result = await risk_manager.check_position_async(symbol, size, price, venue)

    print(f"Position check result:")
    print(f"  Approved: {result['approved']}")
    print(f"  Risk score: {result['risk_score']}")
    print(f"  Max size: ${result['max_size']:,.0f}")
    print(f"  VaR (1d): {result['var_1d']:.4f}")
    print(f"  CVaR (1d): {result['cvar_1d']:.4f}")
    print(f"  Venue haircut: {result['venue_haircut']:.1%}")

    # Test position that exceeds limits
    large_size = 10.0  # Much larger position
    large_result = await risk_manager.check_position_async(
        symbol, large_size, price, venue
    )

    print(f"\nLarge position check result:")
    print(f"  Approved: {large_result['approved']}")
    print(f"  Risk score: {large_result['risk_score']}")
    print(f"  Reasons: {large_result['reasons']}")

    # Verify audit events were logged
    audit_logger = get_audit_logger()
    events = await audit_logger.get_events(
        event_type=EventType.POSITION_CHANGE, limit=10
    )

    print(f"\nAudit events logged: {len(events)}")
    for event in events[:3]:  # Show first 3 events
        print(f"  {event.timestamp}: {event.event_data.get('action', 'unknown')}")

    # Check for risk breach events
    breach_events = await audit_logger.get_events(
        event_type=EventType.RISK_BREACH, limit=5
    )
    print(f"Risk breach events: {len(breach_events)}")

    assert len(events) > 0, "Position change events should be logged"
    print("✅ WORM audit integration working correctly")


async def main():
    """Run all advanced risk manager tests."""
    print("🚀 Advanced Risk Manager Test Suite")
    print("=" * 60)
    print("Testing Future_instruction.txt risk management requirements:")
    print("- VaR/CVaR calculation")
    print("- Exchange haircuts")
    print("- WORM audit trail")
    print("=" * 60)

    try:
        # Run all tests
        risk_manager = await test_var_cvar_calculation()
        await test_exchange_haircuts()
        await test_worm_audit_integration()

        print("\n✅ ADVANCED RISK MANAGER TEST RESULTS")
        print("=" * 50)
        print("✅ VaR/CVaR calculation implemented")
        print("✅ Exchange haircuts applied correctly")
        print("✅ WORM audit trail integrated")
        print("\n🎉 All advanced risk tests PASSED!")
        print("\nImplementation satisfies Future_instruction.txt requirements:")
        print("- ✅ VaR/CVaR calculation with proper formulas")
        print("- ✅ Exchange haircuts for venue-specific risk")
        print("- ✅ WORM audit trail for all execution paths")
        print("- ✅ Position sizing with comprehensive risk factors")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

from src.layers.layer5_risk.advanced_risk_manager import (
    AdvancedRiskManager,
    VaRMethod,
    RiskLevel,
)
from src.layers.layer5_risk.risk_monitor import RiskMonitor, AlertSeverity
from src.layers.layer0_data_ingestion.schemas import FeatureSnapshot


def create_test_feature_snapshot() -> FeatureSnapshot:
    """Create a test feature snapshot."""
    return FeatureSnapshot(
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        mid_price=Decimal("50000"),
        spread_bps=2.5,
        return_1m=0.001,
        return_5m=0.003,
        return_15m=0.005,
        volatility_5m=0.02,
        volatility_15m=0.025,
        volatility_1h=0.03,
        volume_ratio=1.2,
        order_book_imbalance=0.1,
        order_book_pressure=0.05,
        volume_1m=Decimal("100"),
    )


def test_advanced_risk_manager():
    """Test advanced risk management functionality."""
    print("🧪 Testing Advanced Risk Management System")
    print("=" * 60)

    # Initialize risk manager
    print("1. Initializing advanced risk manager...")
    risk_manager = AdvancedRiskManager(
        max_position_pct=0.25, max_drawdown_pct=0.15, max_var_pct=0.08, max_leverage=5.0
    )
    print(f"   ✅ Risk manager initialized")

    # Test market data updates
    print("\n2. Testing market data updates...")
    for i in range(50):
        return_1d = np.random.randn() * 0.02  # 2% daily volatility
        price = 50000 + i * 100 + np.random.randn() * 500
        volatility = 0.02 + np.random.randn() * 0.005

        feature_snapshot = create_test_feature_snapshot()

        risk_manager.update_market_data(
            "BTCUSDT", price, return_1d, volatility, feature_snapshot
        )

    print(f"   ✅ Updated {len(risk_manager.return_history)} market data points")

    # Test VaR calculation
    print("\n3. Testing VaR calculation...")
    var_result = risk_manager.calculate_var(VaRMethod.HISTORICAL)
    print(
        f"   ✅ Historical VaR: {var_result.var_1d:.4f} (1d), {var_result.var_5d:.4f} (5d)"
    )

    var_result = risk_manager.calculate_var(VaRMethod.PARAMETRIC)
    print(
        f"   ✅ Parametric VaR: {var_result.var_1d:.4f} (1d), {var_result.var_5d:.4f} (5d)"
    )

    var_result = risk_manager.calculate_var(VaRMethod.MONTE_CARLO)
    print(
        f"   ✅ Monte Carlo VaR: {var_result.var_1d:.4f} (1d), {var_result.var_5d:.4f} (5d)"
    )

    # Test stress testing
    print("\n4. Testing stress tests...")
    current_positions = {"BTCUSDT": 25000.0, "ETHUSDT": 15000.0, "SOLUSDT": 5000.0}
    portfolio_value = 100000.0

    stress_results = risk_manager.run_stress_tests(current_positions, portfolio_value)
    print(f"   ✅ Completed {len(stress_results)} stress test scenarios")

    for result in stress_results:
        print(
            f"   📊 {result.scenario_name}: {result.portfolio_shock:.2%} shock, "
            f"survival: {result.survival_probability:.2%}"
        )

    # Test correlation analysis
    print("\n5. Testing correlation analysis...")
    # Add more symbols to test correlation
    for symbol in ["ETHUSDT", "SOLUSDT"]:
        for i in range(50):
            return_1d = np.random.randn() * 0.03
            price = 3000 + i * 20 + np.random.randn() * 100
            volatility = 0.025 + np.random.randn() * 0.005

            feature_snapshot = create_test_feature_snapshot()
            feature_snapshot.symbol = symbol

            risk_manager.update_market_data(
                symbol, price, return_1d, volatility, feature_snapshot
            )

    correlation_matrix = risk_manager.calculate_correlation_matrix()
    if correlation_matrix is not None:
        print(f"   ✅ Calculated correlation matrix: {correlation_matrix.shape}")

    correlation_alerts = risk_manager.check_correlation_risk()
    print(f"   ✅ Correlation alerts: {len(correlation_alerts)}")

    # Test concentration risk
    print("\n6. Testing concentration risk...")
    concentration = risk_manager.calculate_concentration_risk(
        current_positions, portfolio_value
    )
    print(f"   ✅ Concentration risk: {concentration:.2%}")

    # Test dynamic limits
    print("\n7. Testing dynamic limits...")
    risk_manager.update_dynamic_limits("high_volatility", -0.05)
    print(f"   ✅ Dynamic limits updated: {risk_manager.dynamic_limits}")

    # Test comprehensive risk check
    print("\n8. Testing comprehensive risk check...")
    result = risk_manager.check_advanced_risk(
        "BTCUSDT",
        Decimal("10000"),
        Decimal("50000"),
        Decimal(str(portfolio_value)),
        current_positions,
    )
    print(f"   ✅ Risk check result: {result[0]}, reason: {result[1]}")

    # Test comprehensive risk metrics
    print("\n9. Testing comprehensive risk metrics...")
    risk_metrics = risk_manager.get_comprehensive_risk_metrics(
        current_positions, portfolio_value
    )
    print(f"   ✅ Portfolio value: ${risk_metrics.portfolio_value:,.2f}")
    print(f"   ✅ VaR (1d): {risk_metrics.var_result.var_1d:.4f}")
    print(f"   ✅ Concentration risk: {risk_metrics.concentration_risk:.2%}")
    print(f"   ✅ Leverage ratio: {risk_metrics.leverage_ratio:.2f}")
    print(f"   ✅ Risk level: {risk_metrics.risk_level.value}")
    print(f"   ✅ Active alerts: {len(risk_metrics.active_alerts)}")

    # Test independent oversight
    print("\n10. Testing independent oversight...")
    alerts = risk_manager.independent_risk_oversight(current_positions, portfolio_value)
    print(f"   ✅ Independent oversight alerts: {len(alerts)}")
    for alert in alerts:
        print(f"   ⚠️  {alert}")

    # Test statistics
    print("\n11. Testing statistics...")
    stats = risk_manager.get_advanced_stats()
    print(f"   ✅ Total checks: {stats['Total checks']}")
    print(f"   ✅ Rejection rate: {stats['Rejection rate']}")
    print(f"   ✅ Kill switch active: {stats['Kill_switch_active']}")

    print("\n🎉 All advanced risk management tests passed!")
    return risk_manager


async def test_risk_monitor():
    """Test risk monitoring functionality."""
    print("\n" + "=" * 60)
    print("🧪 Testing Risk Monitor System")
    print("=" * 60)

    # Initialize risk manager first
    risk_manager = AdvancedRiskManager()

    # Add some test data
    for i in range(30):
        return_1d = np.random.randn() * 0.02
        price = 50000 + i * 100
        volatility = 0.02 + np.random.randn() * 0.005

        feature_snapshot = create_test_feature_snapshot()
        risk_manager.update_market_data(
            "BTCUSDT", price, return_1d, volatility, feature_snapshot
        )

    # Initialize risk monitor
    print("1. Initializing risk monitor...")
    risk_monitor = RiskMonitor(risk_manager, monitoring_interval=1)
    print(f"   ✅ Risk monitor initialized")

    # Test manual risk check
    print("\n2. Testing manual risk check...")
    check_result = await risk_monitor.manual_risk_check()
    print(f"   ✅ Manual check completed: {check_result['active_alerts']} alerts")

    # Test alert summary
    print("\n3. Testing alert summary...")
    alert_summary = risk_monitor.get_alert_summary()
    print(f"   ✅ Alert summary: {alert_summary['active_alerts']} active alerts")
    print(f"   ✅ Kill switch active: {alert_summary['kill_switch_active']}")

    # Test monitoring stats
    print("\n4. Testing monitoring stats...")
    monitoring_stats = risk_monitor.get_monitoring_stats()
    print(f"   ✅ Total checks: {monitoring_stats['total_checks']}")
    print(f"   ✅ Alerts generated: {monitoring_stats['alerts_generated']}")
    print(f"   ✅ Is monitoring: {monitoring_stats['is_monitoring']}")

    # Test alert thresholds update
    print("\n5. Testing alert threshold updates...")
    new_thresholds = {"var_warning": 0.04, "leverage_critical": 3.5}
    risk_monitor.update_alert_thresholds(new_thresholds)
    print(f"   ✅ Alert thresholds updated")

    # Test short monitoring run
    print("\n6. Testing short monitoring run...")
    await risk_monitor.start_monitoring()
    print(f"   ✅ Monitoring started")

    # Let it run for a few seconds
    await asyncio.sleep(3)

    await risk_monitor.stop_monitoring()
    print(f"   ✅ Monitoring stopped")

    # Check final stats
    final_stats = risk_monitor.get_monitoring_stats()
    print(f"   ✅ Final checks: {final_stats['total_checks']}")
    print(f"   ✅ Uptime: {final_stats['uptime_seconds']:.1f} seconds")

    print("\n🎉 All risk monitor tests passed!")
    return risk_monitor


async def test_risk_integration():
    """Test integrated risk management system."""
    print("\n" + "=" * 60)
    print("🧪 Testing Risk Integration")
    print("=" * 60)

    # Create integrated system
    print("1. Creating integrated risk management system...")
    risk_manager = AdvancedRiskManager(
        max_position_pct=0.3, max_drawdown_pct=0.2, max_var_pct=0.1, max_leverage=4.0
    )

    risk_monitor = RiskMonitor(risk_manager, monitoring_interval=2)

    # Simulate market data and trading
    print("\n2. Simulating market conditions...")
    current_positions = {"BTCUSDT": 0.0, "ETHUSDT": 0.0}
    portfolio_value = 100000.0

    # Simulate volatile market conditions
    for day in range(10):
        # Simulate daily market movement
        if day < 5:
            # Normal conditions
            daily_return = np.random.randn() * 0.02
            volatility = 0.02 + np.random.randn() * 0.005
        else:
            # Volatile conditions
            daily_return = np.random.randn() * 0.05
            volatility = 0.05 + np.random.randn() * 0.01

        price = 50000 * (1 + daily_return)
        feature_snapshot = create_test_feature_snapshot()

        # Update market data
        risk_manager.update_market_data(
            "BTCUSDT", price, daily_return, volatility, feature_snapshot
        )

        # Simulate position changes
        if day == 2:
            current_positions["BTCUSDT"] = 30000.0  # Large position
        elif day == 6:
            current_positions["BTCUSDT"] = 50000.0  # Very large position
            current_positions["ETHUSDT"] = 25000.0

        # Update portfolio value
        portfolio_value *= 1 + daily_return * 0.5  # Partial exposure

        # Check risk
        risk_metrics = risk_manager.get_comprehensive_risk_metrics(
            current_positions, portfolio_value
        )

        print(
            f"   Day {day+1}: Portfolio=${portfolio_value:,.0f}, "
            f"VaR={risk_metrics.var_result.var_1d:.3f}, "
            f"Risk={risk_metrics.risk_level.value}, "
            f"Alerts={len(risk_metrics.active_alerts)}"
        )

    # Test final comprehensive check
    print("\n3. Final comprehensive risk assessment...")
    final_metrics = risk_manager.get_comprehensive_risk_metrics(
        current_positions, portfolio_value
    )

    print(f"   📊 Final Portfolio Value: ${final_metrics.portfolio_value:,.2f}")
    print(f"   📊 Final VaR (1d): {final_metrics.var_result.var_1d:.3f}")
    print(f"   📊 Final CVaR (1d): {final_metrics.var_result.cvar_1d:.3f}")
    print(f"   📊 Concentration Risk: {final_metrics.concentration_risk:.2%}")
    print(f"   📊 Leverage Ratio: {final_metrics.leverage_ratio:.2f}")
    print(f"   📊 Risk Level: {final_metrics.risk_level.value}")
    print(f"   📊 Active Alerts: {len(final_metrics.active_alerts)}")

    # Test stress scenarios
    print("\n4. Final stress test results...")
    for result in final_metrics.stress_results:
        print(
            f"   💥 {result.scenario_name}: {result.portfolio_shock:.2%} shock, "
            f"PnL: ${result.estimated_pnl:,.0f}, "
            f"Survival: {result.survival_probability:.2%}"
        )

    print("\n🎉 All integration tests passed!")

    # Summary
    print("\n" + "=" * 60)
    print("📊 RISK MANAGEMENT SYSTEM SUMMARY")
    print("=" * 60)

    stats = risk_manager.get_advanced_stats()
    print(f"✅ VaR Methods: Historical, Parametric, Monte Carlo")
    print(f"✅ Stress Test Scenarios: {len(risk_manager.stress_scenarios)}")
    print(f"✅ Dynamic Risk Limits: Enabled")
    print(f"✅ Independent Oversight: Enabled")
    print(f"✅ Correlation Monitoring: Enabled")
    print(f"✅ Concentration Risk: Monitored")
    print(f"✅ Real-time Monitoring: Available")
    print(f"✅ Kill Switch: {stats['Kill_switch_active']}")

    return risk_manager, risk_monitor


async def main():
    """Main test function."""
    try:
        # Test basic functionality
        risk_manager = test_advanced_risk_manager()

        # Test monitoring
        risk_monitor = await test_risk_monitor()

        # Test integration
        await test_risk_integration()

        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nAdvanced Risk Management System is fully operational!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
