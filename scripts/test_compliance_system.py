#!/usr/bin/env python3
"""
Test script for the complete compliance system
Validates FIFO ledger, WORM archive, audit trail, and tax reporting
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from accounting.fifo_ledger import FIFOLedger
from accounting.worm_archive import WORMArchive
from accounting.fee_engine import FeeEngine
from accounting.tax_reporting import TaxReportingEngine
from audit.transaction_audit import TransactionAuditTrail, AuditEventType

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("compliance_test")


def create_mock_fills():
    """Create mock fills for testing."""
    mock_fills = []

    # Generate some realistic mock trades
    venues = ["binance", "coinbase", "ftx"]
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    strategies = ["RL", "BASIS", "MM"]

    base_time = time.time() - 30 * 24 * 3600  # 30 days ago

    for i in range(50):
        fill_time = base_time + i * 3600 * 12  # Every 12 hours

        # Alternate between buy and sell
        side = "buy" if i % 2 == 0 else "sell"
        venue = venues[i % len(venues)]
        symbol = symbols[i % len(symbols)]
        strategy = strategies[i % len(strategies)]

        # Mock prices
        base_prices = {"BTCUSDT": 97000, "ETHUSDT": 3500, "SOLUSDT": 180}
        price = base_prices[symbol] * (
            1 + (i % 10 - 5) * 0.001
        )  # Small price variation

        fill = {
            "fill_id": f"fill_{i}_{int(fill_time)}",
            "venue": venue,
            "symbol": symbol,
            "side": side,
            "qty": 0.1 + (i % 5) * 0.05,  # Varying quantities
            "price": price,
            "timestamp": fill_time,
            "strategy": strategy,
            "product": "spot",
            "maker": i % 3 == 0,  # 1/3 maker, 2/3 taker
            "funding_bps": 0.0,
        }

        mock_fills.append(fill)

    return mock_fills


def test_fifo_ledger():
    """Test FIFO ledger functionality."""
    logger.info("🧪 Testing FIFO Ledger...")

    try:
        # Initialize FIFO ledger with test database
        ledger = FIFOLedger("/tmp/test_fifo_ledger.db")

        # Create mock fills
        mock_fills = create_mock_fills()

        # Process fills
        results = []
        for fill in mock_fills[:20]:  # Process first 20 fills
            result = ledger.process_fill(fill)
            results.append(result)
            logger.debug(f"Processed fill: {result['action']}")

        # Get position summary
        position_summary = ledger.get_position_summary()
        logger.info(
            f"Position summary: {position_summary['total_symbols']} symbols, {position_summary['total_lots']} lots"
        )

        # Get realized P&L report
        start_time = time.time() - 30 * 24 * 3600
        end_time = time.time()
        pnl_report = ledger.get_realized_pnl_report(start_time, end_time)
        logger.info(
            f"Realized P&L: ${pnl_report['total_realized_pnl']:.2f} from {pnl_report['total_dispositions']} dispositions"
        )

        # Verify integrity
        integrity = ledger.verify_integrity()
        logger.info(f"FIFO ledger integrity: {integrity['checks']['integrity_status']}")

        return {
            "success": True,
            "processed_fills": len(results),
            "position_summary": position_summary,
            "pnl_report": pnl_report,
            "integrity_status": integrity["checks"]["integrity_status"],
        }

    except Exception as e:
        logger.error(f"FIFO ledger test failed: {e}")
        return {"success": False, "error": str(e)}


def test_worm_archive():
    """Test WORM archive functionality."""
    logger.info("🧪 Testing WORM Archive...")

    try:
        # Initialize WORM archive with test directory
        archive = WORMArchive("/tmp/test_worm_archive")

        # Store various types of records
        test_records = []

        # Store JSON document
        trade_record = {
            "trade_id": "trade_12345",
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 1.5,
            "price": 97500.00,
            "timestamp": time.time(),
            "venue": "binance",
        }

        record_id_1 = archive.store_record(
            trade_record,
            "application/json",
            retention_years=7,
            metadata={"type": "trade_record", "symbol": "BTCUSDT"},
        )
        test_records.append(record_id_1)

        # Store text document
        audit_log = "System started at " + datetime.now().isoformat()
        record_id_2 = archive.store_record(
            audit_log,
            "text/plain",
            retention_years=3,
            metadata={"type": "audit_log", "component": "system"},
        )
        test_records.append(record_id_2)

        # Store binary data
        binary_data = b"Mock binary compliance data"
        record_id_3 = archive.store_record(
            binary_data,
            "application/octet-stream",
            retention_years=10,
            metadata={"type": "compliance_document"},
        )
        test_records.append(record_id_3)

        # Retrieve and verify records
        retrieved_records = []
        for record_id in test_records:
            retrieved = archive.retrieve_record(record_id, verify_signature=True)
            retrieved_records.append(retrieved)
            logger.debug(f"Retrieved record {record_id}: {retrieved['content_type']}")

        # List records
        record_list = archive.list_records(limit=50)
        logger.info(f"Archive contains {len(record_list)} records")

        # Verify archive integrity
        integrity = archive.verify_archive_integrity()
        logger.info(f"Archive integrity: {integrity['integrity_status']}")

        # Get status report
        status = archive.get_status_report()

        return {
            "success": True,
            "stored_records": len(test_records),
            "retrieved_records": len(retrieved_records),
            "total_records": len(record_list),
            "integrity_status": integrity["integrity_status"],
            "compression_ratio": status["statistics"]["compression_ratio_percent"],
        }

    except Exception as e:
        logger.error(f"WORM archive test failed: {e}")
        return {"success": False, "error": str(e)}


def test_audit_trail():
    """Test transaction audit trail."""
    logger.info("🧪 Testing Audit Trail...")

    try:
        # Initialize audit trail with test database
        audit = TransactionAuditTrail("/tmp/test_audit_trail.db")

        # Log various audit events
        mock_fills = create_mock_fills()

        event_ids = []

        # Log system start
        event_id = audit.log_system_event(
            AuditEventType.SYSTEM_START,
            "compliance_test",
            {"test_session": True, "start_time": time.time()},
        )
        event_ids.append(event_id)

        # Log trade executions
        for fill in mock_fills[:10]:
            event_id = audit.log_trade_execution(fill, actor="test_trader")
            event_ids.append(event_id)

        # Log some risk checks
        for i in range(5):
            risk_check = {
                "check_id": f"risk_check_{i}",
                "check_type": "position_limit",
                "entity_type": "position",
                "risk_level": "low" if i % 2 == 0 else "medium",
            }
            event_id = audit.log_risk_check(risk_check, success=True)
            event_ids.append(event_id)

        # Log parameter changes
        old_params = {"max_position": 100, "risk_multiplier": 1.0}
        new_params = {"max_position": 150, "risk_multiplier": 1.2}

        event_id = audit.log_event(
            AuditEventType.PARAMETER_CHANGE,
            "risk_parameters",
            "global_risk_params",
            "risk_manager",
            "update_parameters",
            old_state=old_params,
            new_state=new_params,
            metadata={"reason": "increased_capital"},
        )
        event_ids.append(event_id)

        # Get audit trail
        trail = audit.get_audit_trail(limit=50)
        logger.info(f"Audit trail contains {len(trail)} events")

        # Get entity history
        history = audit.get_entity_history("risk_parameters", "global_risk_params")
        logger.info(f"Entity history: {len(history)} changes")

        # Verify integrity
        integrity = audit.verify_integrity()
        logger.info(f"Audit trail integrity: {integrity['integrity_status']}")

        return {
            "success": True,
            "logged_events": len(event_ids),
            "trail_events": len(trail),
            "entity_changes": len(history),
            "integrity_status": integrity["integrity_status"],
        }

    except Exception as e:
        logger.error(f"Audit trail test failed: {e}")
        return {"success": False, "error": str(e)}


def test_tax_reporting():
    """Test tax reporting functionality."""
    logger.info("🧪 Testing Tax Reporting...")

    try:
        # Initialize tax reporting engine
        tax_engine = TaxReportingEngine()

        # Use current year - 1 for tax year
        tax_year = datetime.now().year - 1

        # Generate 1099-B equivalent
        form_1099b = tax_engine.generate_1099b_equivalent(tax_year, "TEST_TRADER")
        logger.info(
            f"Generated 1099-B: {form_1099b['summary']['total_transactions']} transactions"
        )

        # Generate Schedule D
        schedule_d = tax_engine.generate_schedule_d(tax_year, "TEST_TRADER")
        logger.info(
            f"Generated Schedule D: ${schedule_d['summary']['total_capital_gain_loss']:.2f} total gain/loss"
        )

        # Generate tax summary
        tax_summary = tax_engine.generate_trader_tax_summary(tax_year)
        logger.info(
            f"Generated tax summary with {len(tax_summary['tax_optimization_suggestions'])} suggestions"
        )

        # Export complete tax package
        export_file = tax_engine.export_tax_package(tax_year, "json")
        logger.info(f"Exported tax package: {export_file}")

        return {
            "success": True,
            "form_1099b_transactions": form_1099b["summary"]["total_transactions"],
            "schedule_d_gain_loss": schedule_d["summary"]["total_capital_gain_loss"],
            "tax_suggestions": len(tax_summary["tax_optimization_suggestions"]),
            "export_file": export_file,
        }

    except Exception as e:
        logger.error(f"Tax reporting test failed: {e}")
        return {"success": False, "error": str(e)}


def test_fee_engine():
    """Test fee engine functionality."""
    logger.info("🧪 Testing Fee Engine...")

    try:
        # Initialize fee engine
        fee_engine = FeeEngine()

        # Create mock fills
        mock_fills = create_mock_fills()

        # Process fills to calculate costs
        cost_breakdowns = fee_engine.process_fills_batch(mock_fills[:20])

        # Calculate net P&L
        net_pnl_result = fee_engine.calculate_net_pnl(mock_fills[:20])

        # Get status report
        status = fee_engine.get_status_report()

        total_fees = sum(c.get("total_usd", 0) for c in cost_breakdowns)

        logger.info(f"Processed {len(cost_breakdowns)} fills")
        logger.info(f"Total fees: ${total_fees:.2f}")
        logger.info(f"Net P&L: ${net_pnl_result['net_pnl_usd']:.2f}")

        return {
            "success": True,
            "processed_fills": len(cost_breakdowns),
            "total_fees": total_fees,
            "net_pnl": net_pnl_result["net_pnl_usd"],
            "venue_count": len(net_pnl_result.get("venue_breakdown", {})),
        }

    except Exception as e:
        logger.error(f"Fee engine test failed: {e}")
        return {"success": False, "error": str(e)}


def run_comprehensive_test():
    """Run comprehensive compliance system test."""
    logger.info("🚀 Starting Comprehensive Compliance System Test")

    test_results = {
        "test_timestamp": time.time(),
        "test_date": datetime.now().isoformat(),
        "results": {},
    }

    # Test each component
    test_results["results"]["fifo_ledger"] = test_fifo_ledger()
    test_results["results"]["worm_archive"] = test_worm_archive()
    test_results["results"]["audit_trail"] = test_audit_trail()
    test_results["results"]["fee_engine"] = test_fee_engine()
    test_results["results"]["tax_reporting"] = test_tax_reporting()

    # Calculate overall success
    all_successful = all(
        result.get("success", False) for result in test_results["results"].values()
    )

    test_results["overall_success"] = all_successful
    test_results["successful_components"] = sum(
        1 for result in test_results["results"].values() if result.get("success", False)
    )
    test_results["total_components"] = len(test_results["results"])

    # Log summary
    logger.info("=" * 60)
    logger.info("COMPLIANCE SYSTEM TEST SUMMARY")
    logger.info("=" * 60)

    for component, result in test_results["results"].items():
        status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
        logger.info(f"{component:20} {status}")
        if not result.get("success", False):
            logger.error(f"  Error: {result.get('error', 'Unknown error')}")

    logger.info("=" * 60)
    logger.info(
        f"Overall Result: {'✅ ALL TESTS PASSED' if all_successful else '❌ SOME TESTS FAILED'}"
    )
    logger.info(
        f"Success Rate: {test_results['successful_components']}/{test_results['total_components']}"
    )
    logger.info("=" * 60)

    # Save test results
    test_report_file = f"/tmp/compliance_test_report_{int(time.time())}.json"
    with open(test_report_file, "w") as f:
        json.dump(test_results, f, indent=2, default=str)

    logger.info(f"Test report saved: {test_report_file}")

    return test_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compliance System Test")
    parser.add_argument(
        "--component",
        choices=["fifo", "worm", "audit", "tax", "fee", "all"],
        default="all",
        help="Component to test",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run specific component test or comprehensive test
    if args.component == "all":
        results = run_comprehensive_test()
    elif args.component == "fifo":
        results = test_fifo_ledger()
    elif args.component == "worm":
        results = test_worm_archive()
    elif args.component == "audit":
        results = test_audit_trail()
    elif args.component == "tax":
        results = test_tax_reporting()
    elif args.component == "fee":
        results = test_fee_engine()

    # Print results
    print(json.dumps(results, indent=2, default=str))

    # Exit with appropriate code
    if isinstance(results, dict):
        success = results.get("success", results.get("overall_success", False))
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
