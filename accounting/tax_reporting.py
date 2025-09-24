#!/usr/bin/env python3
"""
Tax Reporting Utilities
Generate tax reports, forms, and analysis for compliance
"""

import os
import sys
import json
import time
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP
import calendar

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis
from accounting.fifo_ledger import FIFOLedger
from accounting.worm_archive import WORMArchive
from accounting.fee_engine import FeeEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("tax_reporting")


@dataclass
class TaxLotSale:
    """Represents a tax lot sale for reporting."""

    sale_date: str
    acquisition_date: str
    symbol: str
    quantity: Decimal
    proceeds: Decimal
    cost_basis: Decimal
    gain_loss: Decimal
    term: str  # 'short' or 'long'
    venue: str
    wash_sale: bool = False


@dataclass
class TaxSummary:
    """Tax year summary."""

    tax_year: int
    total_proceeds: Decimal
    total_cost_basis: Decimal
    total_gain_loss: Decimal
    short_term_gain_loss: Decimal
    long_term_gain_loss: Decimal
    wash_sale_adjustments: Decimal
    trading_expenses: Decimal
    net_capital_gain_loss: Decimal


class TaxReportingEngine:
    """Comprehensive tax reporting for trading activities."""

    def __init__(self):
        """Initialize tax reporting engine."""
        self.fifo_ledger = FIFOLedger()
        self.worm_archive = WORMArchive()
        self.fee_engine = FeeEngine()
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Tax constants
        self.LONG_TERM_THRESHOLD_DAYS = 365
        self.PRECISION = Decimal("0.01")  # Currency precision

        logger.info("💰 Tax Reporting Engine initialized")

    def get_tax_year_dates(self, tax_year: int) -> Tuple[float, float]:
        """Get start and end timestamps for tax year."""
        start_date = datetime(tax_year, 1, 1).timestamp()
        end_date = datetime(tax_year + 1, 1, 1).timestamp()
        return start_date, end_date

    def generate_1099b_equivalent(
        self, tax_year: int, taxpayer_id: str = "TRADER_001"
    ) -> Dict[str, Any]:
        """Generate 1099-B equivalent report."""
        try:
            start_date, end_date = self.get_tax_year_dates(tax_year)

            # Get realized P&L report from FIFO ledger
            pnl_report = self.fifo_ledger.get_realized_pnl_report(start_date, end_date)

            # Get detailed dispositions for 1099-B
            tax_lot_sales = self._get_tax_lot_sales(start_date, end_date)

            # Calculate totals
            total_proceeds = sum(sale.proceeds for sale in tax_lot_sales)
            total_cost_basis = sum(sale.cost_basis for sale in tax_lot_sales)
            total_gain_loss = sum(sale.gain_loss for sale in tax_lot_sales)

            # Separate short-term and long-term
            short_term_sales = [s for s in tax_lot_sales if s.term == "short"]
            long_term_sales = [s for s in tax_lot_sales if s.term == "long"]

            short_term_proceeds = sum(sale.proceeds for sale in short_term_sales)
            short_term_cost_basis = sum(sale.cost_basis for sale in short_term_sales)
            short_term_gain_loss = sum(sale.gain_loss for sale in short_term_sales)

            long_term_proceeds = sum(sale.proceeds for sale in long_term_sales)
            long_term_cost_basis = sum(sale.cost_basis for sale in long_term_sales)
            long_term_gain_loss = sum(sale.gain_loss for sale in long_term_sales)

            # Generate 1099-B report
            form_1099b = {
                "form_type": "1099-B_EQUIVALENT",
                "tax_year": tax_year,
                "taxpayer_id": taxpayer_id,
                "generated_date": datetime.now().isoformat(),
                "summary": {
                    "total_transactions": len(tax_lot_sales),
                    "total_proceeds": float(total_proceeds),
                    "total_cost_basis": float(total_cost_basis),
                    "total_gain_loss": float(total_gain_loss),
                },
                "short_term": {
                    "transactions": len(short_term_sales),
                    "proceeds": float(short_term_proceeds),
                    "cost_basis": float(short_term_cost_basis),
                    "gain_loss": float(short_term_gain_loss),
                },
                "long_term": {
                    "transactions": len(long_term_sales),
                    "proceeds": float(long_term_proceeds),
                    "cost_basis": float(long_term_cost_basis),
                    "gain_loss": float(long_term_gain_loss),
                },
                "transactions": [
                    {
                        "sale_date": sale.sale_date,
                        "acquisition_date": sale.acquisition_date,
                        "symbol": sale.symbol,
                        "quantity": float(sale.quantity),
                        "proceeds": float(sale.proceeds),
                        "cost_basis": float(sale.cost_basis),
                        "gain_loss": float(sale.gain_loss),
                        "term": sale.term,
                        "venue": sale.venue,
                        "wash_sale": sale.wash_sale,
                    }
                    for sale in tax_lot_sales
                ],
            }

            # Archive the report
            archive_id = self.worm_archive.store_record(
                form_1099b,
                "application/json",
                retention_years=10,
                metadata={
                    "type": "tax_form_1099b",
                    "tax_year": tax_year,
                    "taxpayer_id": taxpayer_id,
                    "generation_date": time.time(),
                },
            )

            form_1099b["archive_id"] = archive_id

            logger.info(
                f"Generated 1099-B for {tax_year}: {len(tax_lot_sales)} transactions"
            )
            return form_1099b

        except Exception as e:
            logger.error(f"Error generating 1099-B: {e}")
            raise

    def _get_tax_lot_sales(
        self, start_date: float, end_date: float
    ) -> List[TaxLotSale]:
        """Get all tax lot sales for period."""
        try:
            # Get dispositions from FIFO ledger database
            with sqlite3.connect(self.fifo_ledger.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT d.*, t.acquisition_date 
                    FROM dispositions d
                    JOIN tax_lots t ON d.lot_id = t.lot_id
                    WHERE d.disposition_date >= ? AND d.disposition_date < ?
                    ORDER BY d.disposition_date
                """,
                    (start_date, end_date),
                )

                dispositions = cursor.fetchall()

            tax_lot_sales = []

            for disp in dispositions:
                # Calculate term (short vs long)
                holding_days = disp[10]  # holding_period_days
                term = (
                    "long" if holding_days > self.LONG_TERM_THRESHOLD_DAYS else "short"
                )

                # Format dates
                sale_date = datetime.fromtimestamp(disp[5]).strftime(
                    "%Y-%m-%d"
                )  # disposition_date
                acquisition_date = datetime.fromtimestamp(disp[-1]).strftime(
                    "%Y-%m-%d"
                )  # acquisition_date from join

                sale = TaxLotSale(
                    sale_date=sale_date,
                    acquisition_date=acquisition_date,
                    symbol=disp[2],  # symbol
                    quantity=Decimal(str(disp[3])).quantize(self.PRECISION),  # quantity
                    proceeds=Decimal(str(disp[4] * disp[3])).quantize(
                        self.PRECISION
                    ),  # sale_price * quantity
                    cost_basis=Decimal(str(disp[8] * disp[3])).quantize(
                        self.PRECISION
                    ),  # cost_basis * quantity
                    gain_loss=Decimal(str(disp[9])).quantize(
                        self.PRECISION
                    ),  # realized_pnl
                    term=term,
                    venue=disp[6],  # venue
                    wash_sale=False,  # Would implement wash sale detection
                )

                tax_lot_sales.append(sale)

            return tax_lot_sales

        except Exception as e:
            logger.error(f"Error getting tax lot sales: {e}")
            return []

    def generate_schedule_d(
        self, tax_year: int, taxpayer_id: str = "TRADER_001"
    ) -> Dict[str, Any]:
        """Generate Schedule D (Capital Gains and Losses)."""
        try:
            start_date, end_date = self.get_tax_year_dates(tax_year)

            # Get tax lot sales
            tax_lot_sales = self._get_tax_lot_sales(start_date, end_date)

            # Separate by term and aggregate
            short_term_sales = [s for s in tax_lot_sales if s.term == "short"]
            long_term_sales = [s for s in tax_lot_sales if s.term == "long"]

            # Calculate totals
            short_term_proceeds = sum(sale.proceeds for sale in short_term_sales)
            short_term_cost = sum(sale.cost_basis for sale in short_term_sales)
            short_term_gain_loss = sum(sale.gain_loss for sale in short_term_sales)

            long_term_proceeds = sum(sale.proceeds for sale in long_term_sales)
            long_term_cost = sum(sale.cost_basis for sale in long_term_sales)
            long_term_gain_loss = sum(sale.gain_loss for sale in long_term_sales)

            # Net capital gain/loss calculation
            net_short_term = short_term_gain_loss
            net_long_term = long_term_gain_loss

            # Apply capital loss limitations (simplified)
            capital_loss_carryover = Decimal("0")  # Would get from previous year
            total_capital_gain_loss = (
                net_short_term + net_long_term + capital_loss_carryover
            )

            # Capital loss deduction limit ($3,000)
            capital_loss_deduction = (
                min(Decimal("3000"), abs(total_capital_gain_loss))
                if total_capital_gain_loss < 0
                else Decimal("0")
            )
            capital_loss_carryforward = (
                abs(total_capital_gain_loss) - capital_loss_deduction
                if total_capital_gain_loss < 0
                else Decimal("0")
            )

            schedule_d = {
                "form_type": "SCHEDULE_D",
                "tax_year": tax_year,
                "taxpayer_id": taxpayer_id,
                "generated_date": datetime.now().isoformat(),
                "part_i_short_term": {
                    "total_sales": len(short_term_sales),
                    "proceeds": float(short_term_proceeds),
                    "cost_basis": float(short_term_cost),
                    "adjustments": 0.0,  # Wash sales, etc.
                    "gain_loss": float(short_term_gain_loss),
                    "transactions": [
                        {
                            "description": f"{sale.quantity} {sale.symbol}",
                            "acquisition_date": sale.acquisition_date,
                            "sale_date": sale.sale_date,
                            "proceeds": float(sale.proceeds),
                            "cost_basis": float(sale.cost_basis),
                            "gain_loss": float(sale.gain_loss),
                        }
                        for sale in short_term_sales[:50]  # Limit for readability
                    ],
                },
                "part_ii_long_term": {
                    "total_sales": len(long_term_sales),
                    "proceeds": float(long_term_proceeds),
                    "cost_basis": float(long_term_cost),
                    "adjustments": 0.0,
                    "gain_loss": float(long_term_gain_loss),
                    "transactions": [
                        {
                            "description": f"{sale.quantity} {sale.symbol}",
                            "acquisition_date": sale.acquisition_date,
                            "sale_date": sale.sale_date,
                            "proceeds": float(sale.proceeds),
                            "cost_basis": float(sale.cost_basis),
                            "gain_loss": float(sale.gain_loss),
                        }
                        for sale in long_term_sales[:50]
                    ],
                },
                "summary": {
                    "net_short_term_gain_loss": float(net_short_term),
                    "net_long_term_gain_loss": float(net_long_term),
                    "capital_loss_carryover_from_prior_year": float(
                        capital_loss_carryover
                    ),
                    "total_capital_gain_loss": float(total_capital_gain_loss),
                    "capital_loss_deduction": float(capital_loss_deduction),
                    "capital_loss_carryforward": float(capital_loss_carryforward),
                },
            }

            # Archive Schedule D
            archive_id = self.worm_archive.store_record(
                schedule_d,
                "application/json",
                retention_years=10,
                metadata={
                    "type": "tax_form_schedule_d",
                    "tax_year": tax_year,
                    "taxpayer_id": taxpayer_id,
                },
            )

            schedule_d["archive_id"] = archive_id

            logger.info(f"Generated Schedule D for {tax_year}")
            return schedule_d

        except Exception as e:
            logger.error(f"Error generating Schedule D: {e}")
            raise

    def generate_trader_tax_summary(self, tax_year: int) -> Dict[str, Any]:
        """Generate comprehensive tax summary for traders."""
        try:
            start_date, end_date = self.get_tax_year_dates(tax_year)

            # Get trading data
            pnl_report = self.fifo_ledger.get_realized_pnl_report(start_date, end_date)

            # Get trading expenses (fees, commissions, etc.)
            trading_expenses = self._calculate_trading_expenses(start_date, end_date)

            # Get wash sale adjustments
            wash_sale_adjustments = self._calculate_wash_sales(start_date, end_date)

            # Calculate mark-to-market election impact (if applicable)
            mtm_analysis = self._analyze_mtm_election(tax_year)

            # Generate comprehensive summary
            tax_summary = {
                "tax_year": tax_year,
                "generated_date": datetime.now().isoformat(),
                "trading_classification": "capital_gains",  # vs 'trader_status'
                "realized_gains_losses": {
                    "total_realized_pnl": pnl_report.get("total_realized_pnl", 0),
                    "short_term_pnl": pnl_report.get("short_term_pnl", 0),
                    "long_term_pnl": pnl_report.get("long_term_pnl", 0),
                    "total_dispositions": pnl_report.get("total_dispositions", 0),
                },
                "trading_expenses": trading_expenses,
                "wash_sale_adjustments": wash_sale_adjustments,
                "mtm_election_analysis": mtm_analysis,
                "tax_optimization_suggestions": self._generate_tax_suggestions(
                    pnl_report, trading_expenses
                ),
                "quarterly_breakdown": self._get_quarterly_breakdown(tax_year),
                "venue_breakdown": pnl_report.get("by_symbol", {}),
                "strategy_breakdown": pnl_report.get("by_strategy", {}),
            }

            # Archive tax summary
            archive_id = self.worm_archive.store_record(
                tax_summary,
                "application/json",
                retention_years=10,
                metadata={"type": "trader_tax_summary", "tax_year": tax_year},
            )

            tax_summary["archive_id"] = archive_id

            logger.info(f"Generated trader tax summary for {tax_year}")
            return tax_summary

        except Exception as e:
            logger.error(f"Error generating trader tax summary: {e}")
            raise

    def _calculate_trading_expenses(
        self, start_date: float, end_date: float
    ) -> Dict[str, Any]:
        """Calculate deductible trading expenses."""
        try:
            # Get fills for period to calculate fees
            fills = self.fee_engine.get_fills_for_period(start_date, end_date)
            net_pnl_result = self.fee_engine.calculate_net_pnl(fills)

            trading_expenses = {
                "commission_fees": net_pnl_result.get("cost_breakdown", {}).get(
                    "trading_fees_usd", 0
                ),
                "exchange_fees": 0.0,  # Would extract from cost breakdown
                "data_fees": 0.0,  # Would track separately
                "software_fees": 0.0,  # Would track separately
                "interest_expense": net_pnl_result.get("cost_breakdown", {}).get(
                    "borrow_costs_usd", 0
                ),
                "other_expenses": 0.0,
                "total_expenses": net_pnl_result.get("total_fees_usd", 0),
            }

            return trading_expenses

        except Exception as e:
            logger.error(f"Error calculating trading expenses: {e}")
            return {"total_expenses": 0.0}

    def _calculate_wash_sales(
        self, start_date: float, end_date: float
    ) -> Dict[str, Any]:
        """Calculate wash sale adjustments (simplified implementation)."""
        try:
            # This is a simplified implementation
            # Full wash sale detection would require analyzing all trades
            # within 30 days before and after each loss

            wash_sale_analysis = {
                "total_wash_sales": 0,
                "disallowed_losses": 0.0,
                "adjusted_basis_increases": 0.0,
                "note": "Wash sale detection requires comprehensive analysis of all trades within 61-day windows",
            }

            return wash_sale_analysis

        except Exception as e:
            logger.error(f"Error calculating wash sales: {e}")
            return {"total_wash_sales": 0}

    def _analyze_mtm_election(self, tax_year: int) -> Dict[str, Any]:
        """Analyze mark-to-market election benefits."""
        try:
            # Mark-to-market election analysis
            mtm_analysis = {
                "eligible_for_mtm": True,  # Would analyze trading frequency/volume
                "current_year_benefit": 0.0,  # Would calculate difference
                "potential_savings": 0.0,
                "wash_sale_relief": True,  # MTM eliminates wash sale rules
                "capital_loss_relief": True,  # No $3,000 limit under MTM
                "recommendation": "Consider MTM election for active traders",
                "note": "MTM election must be made by due date of return for first year",
            }

            return mtm_analysis

        except Exception as e:
            logger.error(f"Error analyzing MTM election: {e}")
            return {"eligible_for_mtm": False}

    def _generate_tax_suggestions(
        self, pnl_report: Dict[str, Any], trading_expenses: Dict[str, Any]
    ) -> List[str]:
        """Generate tax optimization suggestions."""
        suggestions = []

        try:
            total_pnl = pnl_report.get("total_realized_pnl", 0)
            short_term_pnl = pnl_report.get("short_term_pnl", 0)
            long_term_pnl = pnl_report.get("long_term_pnl", 0)

            if total_pnl > 0:
                suggestions.append("Consider tax-loss harvesting to offset gains")

            if short_term_pnl > long_term_pnl and short_term_pnl > 0:
                suggestions.append(
                    "High short-term gains - consider holding positions longer for better tax treatment"
                )

            if trading_expenses.get("total_expenses", 0) > abs(total_pnl) * 0.1:
                suggestions.append(
                    "High trading costs relative to P&L - consider cost optimization"
                )

            if pnl_report.get("total_dispositions", 0) > 1000:
                suggestions.append(
                    "High trading volume - consider mark-to-market election"
                )

            if total_pnl < -3000:
                suggestions.append(
                    "Capital losses exceed annual deduction limit - losses will carry forward"
                )

            suggestions.append("Consult with tax professional for personalized advice")

        except Exception as e:
            logger.error(f"Error generating tax suggestions: {e}")

        return suggestions

    def _get_quarterly_breakdown(self, tax_year: int) -> Dict[str, Any]:
        """Get quarterly P&L breakdown."""
        try:
            quarterly_data = {}

            for quarter in range(1, 5):
                if quarter == 1:
                    start_month, end_month = 1, 3
                elif quarter == 2:
                    start_month, end_month = 4, 6
                elif quarter == 3:
                    start_month, end_month = 7, 9
                else:
                    start_month, end_month = 10, 12

                quarter_start = datetime(tax_year, start_month, 1).timestamp()
                quarter_end = datetime(
                    tax_year,
                    end_month,
                    calendar.monthrange(tax_year, end_month)[1],
                    23,
                    59,
                    59,
                ).timestamp()

                quarter_pnl = self.fifo_ledger.get_realized_pnl_report(
                    quarter_start, quarter_end
                )

                quarterly_data[f"Q{quarter}"] = {
                    "period": f"{tax_year}-Q{quarter}",
                    "realized_pnl": quarter_pnl.get("total_realized_pnl", 0),
                    "dispositions": quarter_pnl.get("total_dispositions", 0),
                    "short_term_pnl": quarter_pnl.get("short_term_pnl", 0),
                    "long_term_pnl": quarter_pnl.get("long_term_pnl", 0),
                }

            return quarterly_data

        except Exception as e:
            logger.error(f"Error getting quarterly breakdown: {e}")
            return {}

    def export_tax_package(self, tax_year: int, export_format: str = "json") -> str:
        """Export complete tax package for accountant."""
        try:
            # Generate all tax documents
            form_1099b = self.generate_1099b_equivalent(tax_year)
            schedule_d = self.generate_schedule_d(tax_year)
            tax_summary = self.generate_trader_tax_summary(tax_year)

            # Package all documents
            tax_package = {
                "tax_year": tax_year,
                "generated_date": datetime.now().isoformat(),
                "package_type": "complete_tax_package",
                "documents": {
                    "form_1099b": form_1099b,
                    "schedule_d": schedule_d,
                    "tax_summary": tax_summary,
                },
                "supporting_data": {
                    "total_trades": form_1099b["summary"]["total_transactions"],
                    "total_gain_loss": form_1099b["summary"]["total_gain_loss"],
                    "trading_venues": list(
                        set(t["venue"] for t in form_1099b["transactions"])
                    ),
                    "symbols_traded": list(
                        set(t["symbol"] for t in form_1099b["transactions"])
                    ),
                },
            }

            # Archive complete package
            archive_id = self.worm_archive.store_record(
                tax_package,
                "application/json",
                retention_years=10,
                metadata={
                    "type": "complete_tax_package",
                    "tax_year": tax_year,
                    "export_format": export_format,
                },
            )

            # Save to file
            export_dir = Path(f"/tmp/tax_exports/{tax_year}")
            export_dir.mkdir(parents=True, exist_ok=True)

            export_file = export_dir / f"tax_package_{tax_year}.json"
            with open(export_file, "w") as f:
                json.dump(tax_package, f, indent=2, default=str)

            logger.info(f"Exported tax package for {tax_year}: {export_file}")
            return str(export_file)

        except Exception as e:
            logger.error(f"Error exporting tax package: {e}")
            raise


def main():
    """Main entry point for tax reporting."""
    import argparse

    parser = argparse.ArgumentParser(description="Tax Reporting Utilities")
    parser.add_argument(
        "--1099b", action="store_true", help="Generate 1099-B equivalent"
    )
    parser.add_argument("--schedule-d", action="store_true", help="Generate Schedule D")
    parser.add_argument(
        "--tax-summary", action="store_true", help="Generate trader tax summary"
    )
    parser.add_argument(
        "--export-package", action="store_true", help="Export complete tax package"
    )
    parser.add_argument(
        "--tax-year",
        type=int,
        default=datetime.now().year - 1,
        help="Tax year to process",
    )
    parser.add_argument(
        "--taxpayer-id", type=str, default="TRADER_001", help="Taxpayer ID"
    )

    args = parser.parse_args()

    # Create tax reporting engine
    tax_engine = TaxReportingEngine()

    if args.export_package:
        export_file = tax_engine.export_tax_package(args.tax_year)
        print(f"Tax package exported to: {export_file}")
        return

    if args.form_1099b:
        form_1099b = tax_engine.generate_1099b_equivalent(
            args.tax_year, args.taxpayer_id
        )
        print(json.dumps(form_1099b, indent=2, default=str))
        return

    if args.schedule_d:
        schedule_d = tax_engine.generate_schedule_d(args.tax_year, args.taxpayer_id)
        print(json.dumps(schedule_d, indent=2, default=str))
        return

    if args.tax_summary:
        tax_summary = tax_engine.generate_trader_tax_summary(args.tax_year)
        print(json.dumps(tax_summary, indent=2, default=str))
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
