#!/usr/bin/env python3
"""
Compliance End-of-Day (EOD) Procedures

Implements the daily compliance close procedures:
- FIFO ledger day-close run
- WORM append for tamper-proof records
- Fee reconciliation
- Realized/unrealized P&L roll-forward
- Tax-lot integrity check
- Archive dashboard PNG + Alpha report + TCA to S3 + IPFS CID record
"""

import argparse
import json
import logging
import hashlib
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import base64

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import boto3
    import redis
    import pandas as pd

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger("compliance_eod")


class ComplianceEODManager:
    """
    Manages end-of-day compliance procedures including ledger close,
    WORM archival, reconciliation, and regulatory record-keeping.
    """

    def __init__(self):
        """Initialize compliance EOD manager."""
        self.redis_client = None
        self.s3_client = None
        self.ipfs_available = False

        if DEPS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
                self.s3_client = boto3.client("s3")
            except Exception as e:
                logger.warning(f"Redis/AWS unavailable: {e}")

        # Compliance configuration
        self.config = {
            "s3_bucket": "trading-compliance-archive",
            "fifo_ledger_table": "fifo_ledger",
            "tax_lots_table": "tax_lots",
            "worm_archive_table": "worm_records",
            "tolerance_usd": 0.01,  # $0.01 tolerance for reconciliation
            "retention_days": 2557,  # 7 years regulatory retention
        }

        logger.info("Initialized compliance EOD manager")

    def run_fifo_ledger_close(self, trade_date: date) -> Dict[str, any]:
        """
        Run FIFO ledger day-close procedures.

        Args:
            trade_date: Date to close

        Returns:
            FIFO ledger close results
        """
        try:
            logger.info(f"📋 Running FIFO ledger close for {trade_date}")

            fifo_results = {
                "timestamp": datetime.now().isoformat(),
                "trade_date": trade_date.isoformat(),
                "procedure": "fifo_ledger_close",
                "status": "running",
            }

            # Step 1: Collect all trades for the date
            trades = self._collect_trades_for_date(trade_date)
            fifo_results["trades_processed"] = len(trades)

            # Step 2: Apply FIFO matching algorithm
            fifo_matches = self._apply_fifo_matching(trades)
            fifo_results["fifo_matches"] = len(fifo_matches)

            # Step 3: Calculate realized P&L
            realized_pnl = self._calculate_realized_pnl(fifo_matches)
            fifo_results["realized_pnl_usd"] = realized_pnl

            # Step 4: Update tax lots
            tax_lot_updates = self._update_tax_lots(fifo_matches)
            fifo_results["tax_lots_updated"] = len(tax_lot_updates)

            # Step 5: Record FIFO ledger entries
            ledger_entries = self._record_fifo_entries(fifo_matches, trade_date)
            fifo_results["ledger_entries"] = len(ledger_entries)

            # Step 6: Validate ledger integrity
            integrity_check = self._validate_ledger_integrity(trade_date)
            fifo_results["integrity_check"] = integrity_check

            fifo_results["status"] = (
                "completed" if integrity_check["passed"] else "failed"
            )

            logger.info(f"✅ FIFO ledger close completed: {fifo_results['status']}")
            return fifo_results

        except Exception as e:
            logger.error(f"Error in FIFO ledger close: {e}")
            return {"error": str(e), "status": "failed"}

    def append_worm_records(self, trade_date: date) -> Dict[str, any]:
        """
        Append Write-Once-Read-Many records for tamper-proof audit trail.

        Args:
            trade_date: Date to archive

        Returns:
            WORM append results
        """
        try:
            logger.info(f"🔐 Appending WORM records for {trade_date}")

            worm_results = {
                "timestamp": datetime.now().isoformat(),
                "trade_date": trade_date.isoformat(),
                "procedure": "worm_append",
                "records": [],
            }

            # Step 1: Create WORM records for all critical data
            critical_records = self._create_worm_records(trade_date)

            for record_type, data in critical_records.items():
                # Generate cryptographic hash
                data_hash = self._generate_record_hash(data)

                # Create WORM entry
                worm_entry = {
                    "record_id": f"{trade_date.strftime('%Y%m%d')}_{record_type}",
                    "record_type": record_type,
                    "trade_date": trade_date.isoformat(),
                    "data_hash": data_hash,
                    "data_size_bytes": len(json.dumps(data)),
                    "created_timestamp": datetime.now().isoformat(),
                    "retention_until": (
                        datetime.now() + timedelta(days=self.config["retention_days"])
                    ).isoformat(),
                }

                # Store WORM entry (immutable)
                self._store_worm_entry(worm_entry, data)

                worm_results["records"].append(
                    {
                        "record_type": record_type,
                        "record_id": worm_entry["record_id"],
                        "hash": data_hash,
                        "size_bytes": worm_entry["data_size_bytes"],
                    }
                )

            worm_results["total_records"] = len(worm_results["records"])
            worm_results["status"] = "completed"

            logger.info(
                f"✅ WORM records appended: {worm_results['total_records']} records"
            )
            return worm_results

        except Exception as e:
            logger.error(f"Error appending WORM records: {e}")
            return {"error": str(e), "status": "failed"}

    def reconcile_fees(self, trade_date: date) -> Dict[str, any]:
        """
        Reconcile trading fees across all venues.

        Args:
            trade_date: Date to reconcile

        Returns:
            Fee reconciliation results
        """
        try:
            logger.info(f"💳 Reconciling fees for {trade_date}")

            fee_results = {
                "timestamp": datetime.now().isoformat(),
                "trade_date": trade_date.isoformat(),
                "procedure": "fee_reconciliation",
                "venues": {},
            }

            # Get fees from each venue
            venues = ["binance", "coinbase", "alpaca", "deribit"]
            total_calculated_fees = 0
            total_reported_fees = 0

            for venue in venues:
                venue_fees = self._reconcile_venue_fees(venue, trade_date)
                fee_results["venues"][venue] = venue_fees

                total_calculated_fees += venue_fees["calculated_fees_usd"]
                total_reported_fees += venue_fees["reported_fees_usd"]

            # Calculate reconciliation difference
            fee_difference = abs(total_calculated_fees - total_reported_fees)
            fee_results.update(
                {
                    "total_calculated_fees_usd": total_calculated_fees,
                    "total_reported_fees_usd": total_reported_fees,
                    "difference_usd": fee_difference,
                    "within_tolerance": fee_difference <= self.config["tolerance_usd"],
                    "tolerance_usd": self.config["tolerance_usd"],
                }
            )

            fee_results["status"] = (
                "passed" if fee_results["within_tolerance"] else "failed"
            )

            if not fee_results["within_tolerance"]:
                logger.warning(
                    f"⚠️ Fee reconciliation failed: ${fee_difference:.4f} difference"
                )
            else:
                logger.info(
                    f"✅ Fee reconciliation passed: ${fee_difference:.4f} difference"
                )

            return fee_results

        except Exception as e:
            logger.error(f"Error reconciling fees: {e}")
            return {"error": str(e), "status": "failed"}

    def rollforward_pnl(self, trade_date: date) -> Dict[str, any]:
        """
        Roll-forward realized and unrealized P&L to next trading day.

        Args:
            trade_date: Date to roll forward from

        Returns:
            P&L rollforward results
        """
        try:
            logger.info(f"📊 Rolling forward P&L from {trade_date}")

            pnl_results = {
                "timestamp": datetime.now().isoformat(),
                "from_date": trade_date.isoformat(),
                "to_date": (trade_date + timedelta(days=1)).isoformat(),
                "procedure": "pnl_rollforward",
            }

            # Get current P&L positions
            positions = self._get_eod_positions(trade_date)
            pnl_results["positions_count"] = len(positions)

            # Calculate unrealized P&L at market close
            unrealized_pnl = self._calculate_unrealized_pnl(positions, trade_date)
            pnl_results["unrealized_pnl_usd"] = unrealized_pnl

            # Get realized P&L for the day
            realized_pnl = self._get_realized_pnl(trade_date)
            pnl_results["realized_pnl_usd"] = realized_pnl

            # Roll forward to next day
            rollforward_entries = self._create_rollforward_entries(
                positions, realized_pnl, unrealized_pnl, trade_date
            )
            pnl_results["rollforward_entries"] = len(rollforward_entries)

            # Validate roll-forward
            validation = self._validate_rollforward(rollforward_entries)
            pnl_results["validation"] = validation

            pnl_results["status"] = "completed" if validation["passed"] else "failed"

            logger.info(
                f"✅ P&L rollforward completed: ${realized_pnl:.2f} realized, ${unrealized_pnl:.2f} unrealized"
            )
            return pnl_results

        except Exception as e:
            logger.error(f"Error in P&L rollforward: {e}")
            return {"error": str(e), "status": "failed"}

    def check_tax_lot_integrity(self, trade_date: date) -> Dict[str, any]:
        """
        Verify tax lot integrity and consistency.

        Args:
            trade_date: Date to check

        Returns:
            Tax lot integrity results
        """
        try:
            logger.info(f"🧮 Checking tax lot integrity for {trade_date}")

            integrity_results = {
                "timestamp": datetime.now().isoformat(),
                "trade_date": trade_date.isoformat(),
                "procedure": "tax_lot_integrity_check",
                "checks": {},
            }

            # Check 1: Position quantity consistency
            position_check = self._check_position_consistency(trade_date)
            integrity_results["checks"]["position_consistency"] = position_check

            # Check 2: Tax lot chronological order
            chronology_check = self._check_tax_lot_chronology(trade_date)
            integrity_results["checks"]["chronological_order"] = chronology_check

            # Check 3: Cost basis calculations
            cost_basis_check = self._check_cost_basis_calculations(trade_date)
            integrity_results["checks"]["cost_basis"] = cost_basis_check

            # Check 4: Wash sale rules compliance
            wash_sale_check = self._check_wash_sale_compliance(trade_date)
            integrity_results["checks"]["wash_sale"] = wash_sale_check

            # Overall integrity assessment
            all_checks_passed = all(
                check.get("passed", False)
                for check in integrity_results["checks"].values()
            )

            integrity_results["overall_passed"] = all_checks_passed
            integrity_results["status"] = "passed" if all_checks_passed else "failed"

            if all_checks_passed:
                logger.info("✅ Tax lot integrity check passed")
            else:
                logger.warning("⚠️ Tax lot integrity check failed")

            return integrity_results

        except Exception as e:
            logger.error(f"Error checking tax lot integrity: {e}")
            return {"error": str(e), "status": "failed"}

    def archive_compliance_artifacts(self, trade_date: date) -> Dict[str, any]:
        """
        Archive compliance artifacts to S3 and IPFS.

        Args:
            trade_date: Date to archive

        Returns:
            Archive results
        """
        try:
            logger.info(f"📁 Archiving compliance artifacts for {trade_date}")

            archive_results = {
                "timestamp": datetime.now().isoformat(),
                "trade_date": trade_date.isoformat(),
                "procedure": "compliance_archive",
                "artifacts": {},
            }

            # Create compliance artifacts
            artifacts = self._create_compliance_artifacts(trade_date)

            for artifact_name, artifact_data in artifacts.items():
                artifact_result = {
                    "name": artifact_name,
                    "size_bytes": len(artifact_data),
                    "s3_uploaded": False,
                    "ipfs_pinned": False,
                }

                try:
                    # Upload to S3
                    s3_key = (
                        f"compliance/{trade_date.strftime('%Y/%m/%d')}/{artifact_name}"
                    )
                    if self.s3_client:
                        self.s3_client.put_object(
                            Bucket=self.config["s3_bucket"],
                            Key=s3_key,
                            Body=artifact_data,
                            ServerSideEncryption="AES256",
                        )
                        artifact_result["s3_uploaded"] = True
                        artifact_result["s3_key"] = s3_key

                    # Pin to IPFS (if available)
                    if self.ipfs_available:
                        ipfs_cid = self._pin_to_ipfs(artifact_data)
                        artifact_result["ipfs_pinned"] = True
                        artifact_result["ipfs_cid"] = ipfs_cid

                except Exception as e:
                    artifact_result["error"] = str(e)

                archive_results["artifacts"][artifact_name] = artifact_result

            # Overall status
            successful_uploads = sum(
                1
                for artifact in archive_results["artifacts"].values()
                if artifact.get("s3_uploaded", False)
            )

            archive_results["successful_uploads"] = successful_uploads
            archive_results["total_artifacts"] = len(artifacts)
            archive_results["status"] = (
                "completed" if successful_uploads == len(artifacts) else "partial"
            )

            logger.info(
                f"✅ Archived {successful_uploads}/{len(artifacts)} compliance artifacts"
            )
            return archive_results

        except Exception as e:
            logger.error(f"Error archiving compliance artifacts: {e}")
            return {"error": str(e), "status": "failed"}

    def run_full_eod_procedure(self, trade_date: date) -> Dict[str, any]:
        """
        Run complete end-of-day compliance procedure.

        Args:
            trade_date: Date to close

        Returns:
            Complete EOD procedure results
        """
        try:
            logger.info(f"🏁 Running full EOD compliance procedure for {trade_date}")

            eod_results = {
                "timestamp": datetime.now().isoformat(),
                "trade_date": trade_date.isoformat(),
                "procedure": "full_eod_compliance",
                "steps": {},
            }

            # Step 1: FIFO ledger close
            fifo_results = self.run_fifo_ledger_close(trade_date)
            eod_results["steps"]["fifo_ledger"] = fifo_results

            # Step 2: WORM append
            worm_results = self.append_worm_records(trade_date)
            eod_results["steps"]["worm_records"] = worm_results

            # Step 3: Fee reconciliation
            fee_results = self.reconcile_fees(trade_date)
            eod_results["steps"]["fee_reconciliation"] = fee_results

            # Step 4: P&L rollforward
            pnl_results = self.rollforward_pnl(trade_date)
            eod_results["steps"]["pnl_rollforward"] = pnl_results

            # Step 5: Tax lot integrity check
            integrity_results = self.check_tax_lot_integrity(trade_date)
            eod_results["steps"]["tax_lot_integrity"] = integrity_results

            # Step 6: Archive compliance artifacts
            archive_results = self.archive_compliance_artifacts(trade_date)
            eod_results["steps"]["compliance_archive"] = archive_results

            # Overall success assessment
            critical_steps = [
                "fifo_ledger",
                "worm_records",
                "fee_reconciliation",
                "tax_lot_integrity",
            ]
            critical_success = all(
                eod_results["steps"][step].get("status") in ["completed", "passed"]
                for step in critical_steps
            )

            eod_results["critical_success"] = critical_success
            eod_results["overall_status"] = "success" if critical_success else "failure"

            if critical_success:
                logger.info(
                    "✅ EOD COMPLIANCE: All critical procedures completed successfully"
                )
            else:
                logger.error("❌ EOD COMPLIANCE: Critical procedure failures detected")

            return eod_results

        except Exception as e:
            logger.error(f"Error in full EOD procedure: {e}")
            return {"error": str(e), "overall_status": "failure"}

    # Helper methods (simplified implementations for demo)

    def _collect_trades_for_date(self, trade_date: date) -> List[Dict[str, any]]:
        """Collect all trades for the specified date."""
        # Mock implementation
        return [
            {
                "id": "trade_001",
                "symbol": "BTC",
                "side": "buy",
                "qty": 0.5,
                "price": 45000,
                "timestamp": "2025-01-15T10:30:00Z",
            },
            {
                "id": "trade_002",
                "symbol": "ETH",
                "side": "sell",
                "qty": 2.0,
                "price": 3200,
                "timestamp": "2025-01-15T14:15:00Z",
            },
        ]

    def _apply_fifo_matching(
        self, trades: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """Apply FIFO matching algorithm."""
        # Mock FIFO matching
        return [
            {"buy_trade": "trade_001", "sell_trade": "trade_002", "matched_qty": 0.5}
        ]

    def _calculate_realized_pnl(self, fifo_matches: List[Dict[str, any]]) -> float:
        """Calculate realized P&L from FIFO matches."""
        return 1250.75  # Mock realized P&L

    def _update_tax_lots(
        self, fifo_matches: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """Update tax lots based on FIFO matches."""
        return [{"tax_lot_id": "lot_001", "remaining_qty": 1.5, "cost_basis": 44500}]

    def _record_fifo_entries(
        self, fifo_matches: List[Dict[str, any]], trade_date: date
    ) -> List[Dict[str, any]]:
        """Record FIFO ledger entries."""
        return [
            {
                "entry_id": "fifo_001",
                "trade_date": trade_date.isoformat(),
                "realized_pnl": 1250.75,
            }
        ]

    def _validate_ledger_integrity(self, trade_date: date) -> Dict[str, bool]:
        """Validate ledger integrity."""
        return {"passed": True, "errors": []}

    def _create_worm_records(self, trade_date: date) -> Dict[str, any]:
        """Create WORM records for critical data."""
        return {
            "trades": {"count": 25, "total_volume": 125000},
            "positions": {"symbols": ["BTC", "ETH"], "total_value": 89500},
            "pnl": {"realized": 1250.75, "unrealized": -340.25},
        }

    def _generate_record_hash(self, data: any) -> str:
        """Generate cryptographic hash for record."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _store_worm_entry(self, entry: Dict[str, any], data: any):
        """Store WORM entry (immutable)."""
        # Would store in immutable database
        pass

    def _reconcile_venue_fees(self, venue: str, trade_date: date) -> Dict[str, any]:
        """Reconcile fees for a specific venue."""
        return {
            "venue": venue,
            "calculated_fees_usd": 45.75,
            "reported_fees_usd": 45.73,
            "difference_usd": 0.02,
            "trade_count": 8,
        }

    def _get_eod_positions(self, trade_date: date) -> List[Dict[str, any]]:
        """Get end-of-day positions."""
        return [
            {"symbol": "BTC", "qty": 2.5, "avg_price": 44800},
            {"symbol": "ETH", "qty": 8.0, "avg_price": 3150},
        ]

    def _calculate_unrealized_pnl(
        self, positions: List[Dict[str, any]], trade_date: date
    ) -> float:
        """Calculate unrealized P&L."""
        return -340.25  # Mock unrealized P&L

    def _get_realized_pnl(self, trade_date: date) -> float:
        """Get realized P&L for the date."""
        return 1250.75  # Mock realized P&L

    def _create_rollforward_entries(
        self,
        positions: List[Dict[str, any]],
        realized_pnl: float,
        unrealized_pnl: float,
        trade_date: date,
    ) -> List[Dict[str, any]]:
        """Create rollforward entries."""
        return [{"entry_type": "rollforward", "date": trade_date.isoformat()}]

    def _validate_rollforward(self, entries: List[Dict[str, any]]) -> Dict[str, bool]:
        """Validate rollforward entries."""
        return {"passed": True, "errors": []}

    def _check_position_consistency(self, trade_date: date) -> Dict[str, any]:
        """Check position quantity consistency."""
        return {"passed": True, "discrepancies": []}

    def _check_tax_lot_chronology(self, trade_date: date) -> Dict[str, any]:
        """Check tax lot chronological order."""
        return {"passed": True, "out_of_order": []}

    def _check_cost_basis_calculations(self, trade_date: date) -> Dict[str, any]:
        """Check cost basis calculations."""
        return {"passed": True, "calculation_errors": []}

    def _check_wash_sale_compliance(self, trade_date: date) -> Dict[str, any]:
        """Check wash sale rules compliance."""
        return {"passed": True, "potential_violations": []}

    def _create_compliance_artifacts(self, trade_date: date) -> Dict[str, bytes]:
        """Create compliance artifacts for archival."""
        return {
            "dashboard_screenshot.png": b"mock_dashboard_image_data",
            "alpha_report.pdf": b"mock_alpha_report_data",
            "tca_analysis.json": json.dumps({"tca": "mock_data"}).encode(),
            "eod_summary.json": json.dumps({"summary": "mock_summary"}).encode(),
        }

    def _pin_to_ipfs(self, data: bytes) -> str:
        """Pin data to IPFS and return CID."""
        # Mock IPFS CID
        return "QmXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXx"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compliance EOD Manager")

    parser.add_argument(
        "--date", type=str, help="Trade date (YYYY-MM-DD), defaults to yesterday"
    )
    parser.add_argument(
        "--step",
        choices=["fifo", "worm", "fees", "pnl", "integrity", "archive", "full"],
        default="full",
        help="Specific step to run",
    )
    parser.add_argument("--output", type=str, help="Output file for results")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Default to yesterday's date
    if args.date:
        trade_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        trade_date = date.today() - timedelta(days=1)

    logger.info(f"🏁 Starting Compliance EOD for {trade_date}")

    try:
        manager = ComplianceEODManager()

        if args.step == "fifo":
            results = manager.run_fifo_ledger_close(trade_date)
        elif args.step == "worm":
            results = manager.append_worm_records(trade_date)
        elif args.step == "fees":
            results = manager.reconcile_fees(trade_date)
        elif args.step == "pnl":
            results = manager.rollforward_pnl(trade_date)
        elif args.step == "integrity":
            results = manager.check_tax_lot_integrity(trade_date)
        elif args.step == "archive":
            results = manager.archive_compliance_artifacts(trade_date)
        else:  # full
            results = manager.run_full_eod_procedure(trade_date)

        print(f"\n🏁 COMPLIANCE EOD RESULTS ({trade_date}):")
        print(json.dumps(results, indent=2))

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        # Return appropriate exit code
        if args.step == "full":
            return 0 if results.get("overall_status") == "success" else 1
        else:
            return 0

    except Exception as e:
        logger.error(f"Error in compliance EOD: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
