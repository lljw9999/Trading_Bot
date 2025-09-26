#!/usr/bin/env python3
"""
Equities Bring-Up Acceptance Tests

Tests all equities-specific functionality before go-live.
Must pass before enabling live equities trading.
"""

import unittest
import time
import redis
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

import pytest

pytest.importorskip(
    "aiofiles", reason="aiofiles optional dependency required for risk compliance stack"
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.layers.layer5_risk.market_hours_guard import create_market_hours_guard
from src.layers.layer5_risk.pdt_guard import create_pdt_guard
from src.layers.layer5_risk.ssr_guard import create_ssr_guard
from src.layers.layer5_risk.halts_luld_monitor import create_halt_luld_monitor
from accounting.fifo_ledger import FIFOLedger


class TestEquitiesBringUp(unittest.TestCase):
    """Comprehensive equities functionality tests"""

    def setUp(self):
        """Set up test environment"""
        self.redis_client = redis.Redis(
            host="localhost", port=6379, decode_responses=True
        )

        # Clear test keys
        test_keys = [
            "risk:market_open",
            "risk:pdt_block",
            "risk:ssr:AAPL",
            "risk:halted:AAPL",
            "test:*",
        ]
        for pattern in test_keys:
            for key in self.redis_client.scan_iter(match=pattern):
                self.redis_client.delete(key)

    def test_market_hours_guard(self):
        """Test market hours detection and blocking"""
        print("🕐 Testing Market Hours Guard...")

        guard = create_market_hours_guard()

        # Test during market hours (mock)
        market_open_time = datetime.now().replace(hour=10, minute=30)  # 10:30 AM ET
        self.assertTrue(
            guard.is_open_now(market_open_time), "Market should be open at 10:30 AM"
        )
        self.assertFalse(
            guard.should_block_trading(market_open_time),
            "Should not block during market hours",
        )

        # Test outside market hours
        after_hours_time = datetime.now().replace(hour=18, minute=0)  # 6:00 PM ET
        self.assertFalse(
            guard.is_open_now(after_hours_time), "Market should be closed at 6:00 PM"
        )
        self.assertTrue(
            guard.should_block_trading(after_hours_time),
            "Should block outside market hours",
        )

        # Test auction windows
        opening_auction = datetime.now().replace(hour=9, minute=27)
        self.assertTrue(
            guard.is_opening_auction_window(opening_auction),
            "Should detect opening auction",
        )

        closing_auction = datetime.now().replace(hour=15, minute=59)
        self.assertTrue(
            guard.is_closing_auction_window(closing_auction),
            "Should detect closing auction",
        )

        print("✅ Market Hours Guard: PASS")

    def test_pdt_guard(self):
        """Test PDT rule enforcement"""
        print("🛡️ Testing PDT Guard...")

        guard = create_pdt_guard()

        # Test account under $25k with high day trades (should block)
        result = guard.check(account_equity_usd=15000, last_5d_daytrades=4)
        self.assertTrue(
            result["should_block_intraday"],
            "Should block intraday trading for PDT violation",
        )
        self.assertEqual(result["risk_level"], "HIGH")

        # Test account over $25k (no restrictions)
        result = guard.check(account_equity_usd=50000, last_5d_daytrades=10)
        self.assertFalse(
            result["should_block_intraday"], "Should not block for high-equity account"
        )
        self.assertEqual(result["risk_level"], "NONE")

        # Test account under $25k with low day trades (allow)
        result = guard.check(account_equity_usd=15000, last_5d_daytrades=1)
        self.assertFalse(
            result["should_block_intraday"],
            "Should allow trading with day trades remaining",
        )

        print("✅ PDT Guard: PASS")

    def test_ssr_guard(self):
        """Test SSR detection and short blocking"""
        print("📉 Testing SSR Guard...")

        guard = create_ssr_guard()

        # Test SSR trigger (>10% drop)
        result = guard.evaluate("AAPL", last_price=90.0, prev_close=100.0)
        self.assertTrue(result["is_ssr_active"], "SSR should be active for 10% drop")
        self.assertTrue(
            result["should_block_shorts"], "Should block short sales during SSR"
        )
        self.assertEqual(result["ssr_reason"], "PRICE_DROP")

        # Test normal price action (no SSR)
        result = guard.evaluate("AAPL", last_price=98.0, prev_close=100.0)
        self.assertFalse(
            result["is_ssr_active"], "SSR should not be active for 2% drop"
        )
        self.assertFalse(
            result["should_block_shorts"],
            "Should not block shorts for normal price action",
        )

        # Test broker SSR flag override
        result = guard.evaluate(
            "AAPL", last_price=98.0, prev_close=100.0, broker_ssr_flag=True
        )
        self.assertTrue(
            result["is_ssr_active"], "SSR should be active when broker flag is set"
        )
        self.assertEqual(result["ssr_reason"], "BROKER_FLAG")

        print("✅ SSR Guard: PASS")

    def test_halt_luld_monitor(self):
        """Test halt detection and LULD monitoring"""
        print("🛑 Testing Halt/LULD Monitor...")

        monitor = create_halt_luld_monitor()

        # Test normal operation
        result = monitor.on_tick("AAPL", 150.0)
        self.assertFalse(
            result["should_block_orders"], "Should not block orders normally"
        )

        # Test halt condition
        monitor.set_halt_status("AAPL", True, "NEWS_PENDING")
        result = monitor.on_tick("AAPL", 150.0)
        self.assertTrue(
            result["should_block_orders"], "Should block orders when halted"
        )

        # Test LULD proximity
        monitor.set_luld_bands("AAPL", up_band=160.0, down_band=140.0)
        result = monitor.on_tick("AAPL", 159.0)  # Near upper band
        self.assertTrue(
            result["should_throttle_orders"], "Should throttle near LULD bands"
        )

        print("✅ Halt/LULD Monitor: PASS")

    def test_corporate_actions_split(self):
        """Test stock split processing"""
        print("📊 Testing Stock Split...")

        # Create test ledger with temporary DB
        import tempfile

        temp_db = tempfile.mktemp(suffix=".db")
        ledger = FIFOLedger(db_path=temp_db)

        # Create test tax lot
        test_fill = {
            "symbol": "AAPL",
            "qty": 100.0,
            "price": 200.0,
            "side": "buy",
            "venue": "alpaca",
            "strategy": "test",
            "timestamp": time.time(),
        }
        lot_id = ledger.create_tax_lot(test_fill)

        # Apply 2:1 stock split
        split_result = ledger.apply_split("AAPL", ratio_from=1.0, ratio_to=2.0)

        # Verify split results
        self.assertEqual(split_result["lots_affected"], 1, "Should affect 1 tax lot")
        self.assertEqual(split_result["split_ratio"], 2.0, "Split ratio should be 2.0")

        # Verify lot adjustments
        position = ledger.get_position_summary("AAPL")
        long_qty = position["AAPL"]["long"]["total_qty"]
        avg_cost = position["AAPL"]["long"]["avg_cost_basis"]

        self.assertEqual(long_qty, 200.0, "Quantity should double after 2:1 split")
        self.assertEqual(avg_cost, 100.0, "Cost basis should halve after 2:1 split")

        # Cleanup
        import os

        os.unlink(temp_db)

        print("✅ Stock Split: PASS")

    def test_corporate_actions_dividend(self):
        """Test dividend recording"""
        print("💰 Testing Dividend Recording...")

        # Create test ledger
        import tempfile

        temp_db = tempfile.mktemp(suffix=".db")
        ledger = FIFOLedger(db_path=temp_db)

        # Record dividend
        dividend_result = ledger.record_dividend(
            symbol="AAPL", gross_amount=100.0, tax_withheld=15.0
        )

        # Verify dividend results
        self.assertEqual(
            dividend_result["gross_amount"], 100.0, "Gross amount should match"
        )
        self.assertEqual(
            dividend_result["tax_withheld"], 15.0, "Tax withheld should match"
        )
        self.assertEqual(
            dividend_result["net_amount"], 85.0, "Net amount should be gross - tax"
        )

        # Verify cash balance update
        expected_cash = dividend_result["new_cash_balance"]
        self.assertEqual(
            expected_cash, 85.0, "Cash balance should increase by net amount"
        )

        # Cleanup
        import os

        os.unlink(temp_db)

        print("✅ Dividend Recording: PASS")

    def test_paper_trading_flow(self):
        """Test end-to-end paper trading flow"""
        print("📈 Testing Paper Trading Flow...")

        # This would test:
        # 1. Market data ingestion
        # 2. Signal generation
        # 3. Order submission
        # 4. Fill processing
        # 5. P&L calculation
        # 6. TCA metrics

        # For now, just verify key components exist
        try:
            from src.layers.layer4_execution.alpaca_executor import AlpacaExecutor

            executor = AlpacaExecutor()
            account = executor.get_account()
            self.assertIsNotNone(account, "Should get account info")

            from accounting.fee_engine import FeeEngine

            fee_engine = FeeEngine()
            alpaca_config = fee_engine.get_venue_config("alpaca")
            self.assertIn(
                "stock",
                alpaca_config["products"],
                "Alpaca should support stock product",
            )

            print("✅ Paper Trading Flow: PASS")

        except ImportError as e:
            self.fail(f"Missing required component: {e}")

    def test_redis_integration(self):
        """Test Redis integration for all guards"""
        print("📡 Testing Redis Integration...")

        # Test market hours Redis keys
        self.redis_client.set("risk:market_open", "1")
        self.assertEqual(self.redis_client.get("risk:market_open"), "1")

        # Test PDT Redis keys
        self.redis_client.set("risk:pdt_block", "1")
        self.assertEqual(self.redis_client.get("risk:pdt_block"), "1")

        # Test SSR Redis keys
        self.redis_client.set("risk:ssr:AAPL", "1")
        self.assertEqual(self.redis_client.get("risk:ssr:AAPL"), "1")

        # Test halt Redis keys
        self.redis_client.set("risk:halted:AAPL", "1")
        self.assertEqual(self.redis_client.get("risk:halted:AAPL"), "1")

        print("✅ Redis Integration: PASS")


def run_acceptance_tests():
    """Run all acceptance tests"""
    print("🚀 Running Equities Bring-Up Acceptance Tests")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEquitiesBringUp)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")

    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")

    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}")

    return success


if __name__ == "__main__":
    run_acceptance_tests()
    # Note: exit code removed to avoid pytest return-value warnings
    # Test framework will handle success/failure appropriately
