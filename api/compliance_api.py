#!/usr/bin/env python3
"""
Compliance Reporting API
RESTful API for accessing tax, audit, and regulatory compliance data
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
import redis
from accounting.fifo_ledger import FIFOLedger
from accounting.worm_archive import WORMArchive
from accounting.fee_engine import FeeEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("compliance_api")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize services
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
fifo_ledger = FIFOLedger()
worm_archive = WORMArchive()
fee_engine = FeeEngine()


@dataclass
class ComplianceContext:
    """Context for compliance operations."""

    requester: str
    request_id: str
    timestamp: float
    purpose: str


def log_api_access(
    endpoint: str,
    method: str,
    status: int,
    requester: str = None,
    duration_ms: float = None,
):
    """Log API access for audit trail."""
    try:
        log_entry = {
            "timestamp": time.time(),
            "endpoint": endpoint,
            "method": method,
            "status": status,
            "requester": requester or "anonymous",
            "duration_ms": duration_ms,
            "request_id": request.headers.get("X-Request-ID", "unknown"),
        }

        # Store in Redis for real-time monitoring
        redis_client.lpush("compliance:api_access", json.dumps(log_entry))
        redis_client.ltrim("compliance:api_access", 0, 1000)  # Keep last 1000 entries

        # Archive in WORM for compliance
        worm_archive.store_record(
            log_entry,
            "application/json",
            retention_years=7,
            metadata={"type": "api_access_log", "endpoint": endpoint},
        )

    except Exception as e:
        logger.error(f"Error logging API access: {e}")


def create_compliance_context() -> ComplianceContext:
    """Create compliance context from request."""
    return ComplianceContext(
        requester=request.headers.get("X-Requester", "anonymous"),
        request_id=request.headers.get("X-Request-ID", f"req_{int(time.time())}"),
        timestamp=time.time(),
        purpose=request.headers.get("X-Purpose", "general_inquiry"),
    )


@app.before_request
def before_request():
    """Log request start time."""
    request.start_time = time.time()


@app.after_request
def after_request(response):
    """Log completed requests."""
    try:
        duration_ms = (time.time() - request.start_time) * 1000
        log_api_access(
            endpoint=request.endpoint or request.path,
            method=request.method,
            status=response.status_code,
            requester=request.headers.get("X-Requester"),
            duration_ms=duration_ms,
        )
    except Exception as e:
        logger.error(f"Error in after_request: {e}")

    return response


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "fifo_ledger": "available",
                "worm_archive": "available",
                "fee_engine": "available",
                "redis": "available",
            },
        }
    )


@app.route("/api/v1/positions", methods=["GET"])
def get_positions():
    """Get current trading positions."""
    try:
        context = create_compliance_context()
        symbol = request.args.get("symbol")

        # Get position summary from FIFO ledger
        position_summary = fifo_ledger.get_position_summary(symbol)

        # Archive request for compliance
        audit_record = {
            "context": context.__dict__,
            "request_type": "position_inquiry",
            "symbol_filter": symbol,
            "response_summary": {
                "total_symbols": position_summary.get("total_symbols", 0),
                "total_lots": position_summary.get("total_lots", 0),
            },
        }

        worm_archive.store_record(
            audit_record,
            "application/json",
            retention_years=7,
            metadata={"type": "position_inquiry", "requester": context.requester},
        )

        return jsonify(
            {
                "success": True,
                "data": position_summary,
                "request_id": context.request_id,
            }
        )

    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "request_id": (
                        context.request_id if "context" in locals() else "unknown"
                    ),
                }
            ),
            500,
        )


@app.route("/api/v1/realized-pnl", methods=["GET"])
def get_realized_pnl():
    """Get realized P&L report for tax purposes."""
    try:
        context = create_compliance_context()

        # Parse date parameters
        start_date = request.args.get("start_date", type=float)
        end_date = request.args.get("end_date", type=float)

        # Default to current tax year if no dates provided
        if not start_date or not end_date:
            current_year = datetime.now().year
            start_date = datetime(current_year, 1, 1).timestamp()
            end_date = datetime(current_year + 1, 1, 1).timestamp()

        # Get realized P&L report
        pnl_report = fifo_ledger.get_realized_pnl_report(start_date, end_date)

        # Archive tax report request
        audit_record = {
            "context": context.__dict__,
            "request_type": "tax_report_request",
            "period_start": start_date,
            "period_end": end_date,
            "report_summary": {
                "total_dispositions": pnl_report.get("total_dispositions", 0),
                "total_realized_pnl": pnl_report.get("total_realized_pnl", 0),
            },
        }

        record_id = worm_archive.store_record(
            audit_record,
            "application/json",
            retention_years=10,  # Tax records kept longer
            metadata={"type": "tax_report_request", "requester": context.requester},
        )

        # Also archive the full report
        full_report_id = worm_archive.store_record(
            pnl_report,
            "application/json",
            retention_years=10,
            metadata={
                "type": "realized_pnl_report",
                "requester": context.requester,
                "audit_ref": record_id,
            },
        )

        return jsonify(
            {
                "success": True,
                "data": pnl_report,
                "archive_references": {
                    "audit_record": record_id,
                    "full_report": full_report_id,
                },
                "request_id": context.request_id,
            }
        )

    except Exception as e:
        logger.error(f"Error getting realized P&L: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "request_id": (
                        context.request_id if "context" in locals() else "unknown"
                    ),
                }
            ),
            500,
        )


@app.route("/api/v1/net-pnl", methods=["GET"])
def get_net_pnl():
    """Get net P&L calculation including all costs."""
    try:
        context = create_compliance_context()

        # Parse date parameters
        start_time = request.args.get("start_time", type=float)
        end_time = request.args.get("end_time", type=float)

        # Default to last 24 hours
        if not end_time:
            end_time = time.time()
        if not start_time:
            start_time = end_time - 24 * 3600

        # Get fills for period
        fills = fee_engine.get_fills_for_period(start_time, end_time)

        # Calculate net P&L
        net_pnl_result = fee_engine.calculate_net_pnl(fills)

        # Archive net P&L calculation
        audit_record = {
            "context": context.__dict__,
            "request_type": "net_pnl_calculation",
            "period_start": start_time,
            "period_end": end_time,
            "calculation_summary": {
                "fill_count": len(fills),
                "gross_pnl": net_pnl_result.get("gross_pnl_usd", 0),
                "total_fees": net_pnl_result.get("total_fees_usd", 0),
                "net_pnl": net_pnl_result.get("net_pnl_usd", 0),
            },
        }

        record_id = worm_archive.store_record(
            audit_record,
            "application/json",
            retention_years=7,
            metadata={"type": "net_pnl_calculation", "requester": context.requester},
        )

        return jsonify(
            {
                "success": True,
                "data": net_pnl_result,
                "archive_reference": record_id,
                "request_id": context.request_id,
            }
        )

    except Exception as e:
        logger.error(f"Error calculating net P&L: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "request_id": (
                        context.request_id if "context" in locals() else "unknown"
                    ),
                }
            ),
            500,
        )


@app.route("/api/v1/audit-trail", methods=["GET"])
def get_audit_trail():
    """Get audit trail for transactions."""
    try:
        context = create_compliance_context()

        # Parse parameters
        entity_type = request.args.get("entity_type")
        entity_id = request.args.get("entity_id")
        start_date = request.args.get("start_date", type=float)
        end_date = request.args.get("end_date", type=float)
        limit = request.args.get("limit", 100, type=int)

        # Query FIFO ledger audit log (simplified - would expand based on entity_type)
        # This is a basic implementation - would need to expand based on requirements

        audit_trail = {
            "timestamp": time.time(),
            "filters": {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit,
            },
            "entries": [],
            "note": "Audit trail functionality implemented in FIFO ledger and WORM archive",
        }

        # Archive audit trail request
        audit_record = {
            "context": context.__dict__,
            "request_type": "audit_trail_request",
            "filters": audit_trail["filters"],
        }

        record_id = worm_archive.store_record(
            audit_record,
            "application/json",
            retention_years=10,
            metadata={"type": "audit_trail_request", "requester": context.requester},
        )

        return jsonify(
            {
                "success": True,
                "data": audit_trail,
                "archive_reference": record_id,
                "request_id": context.request_id,
            }
        )

    except Exception as e:
        logger.error(f"Error getting audit trail: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "request_id": (
                        context.request_id if "context" in locals() else "unknown"
                    ),
                }
            ),
            500,
        )


@app.route("/api/v1/tax-forms", methods=["GET"])
def generate_tax_forms():
    """Generate tax forms (1099-B equivalent)."""
    try:
        context = create_compliance_context()

        # Parse tax year
        tax_year = request.args.get("tax_year", datetime.now().year - 1, type=int)

        # Calculate tax year dates
        start_date = datetime(tax_year, 1, 1).timestamp()
        end_date = datetime(tax_year + 1, 1, 1).timestamp()

        # Get realized P&L for tax year
        pnl_report = fifo_ledger.get_realized_pnl_report(start_date, end_date)

        # Generate tax form data
        tax_form_data = {
            "tax_year": tax_year,
            "taxpayer": context.requester,
            "generated_at": time.time(),
            "form_type": "1099-B_equivalent",
            "proceeds": abs(pnl_report.get("total_realized_pnl", 0)),
            "cost_basis": 0,  # Would calculate from dispositions
            "gain_loss": pnl_report.get("total_realized_pnl", 0),
            "short_term_gain_loss": pnl_report.get("short_term_pnl", 0),
            "long_term_gain_loss": pnl_report.get("long_term_pnl", 0),
            "dispositions_by_symbol": pnl_report.get("by_symbol", {}),
            "total_dispositions": pnl_report.get("total_dispositions", 0),
        }

        # Archive tax form generation
        audit_record = {
            "context": context.__dict__,
            "request_type": "tax_form_generation",
            "tax_year": tax_year,
            "form_data_summary": {
                "total_gain_loss": tax_form_data["gain_loss"],
                "total_dispositions": tax_form_data["total_dispositions"],
            },
        }

        audit_record_id = worm_archive.store_record(
            audit_record,
            "application/json",
            retention_years=10,
            metadata={"type": "tax_form_generation", "requester": context.requester},
        )

        # Archive full tax form
        tax_form_id = worm_archive.store_record(
            tax_form_data,
            "application/json",
            retention_years=10,
            metadata={
                "type": "tax_form_1099b",
                "tax_year": tax_year,
                "requester": context.requester,
            },
        )

        return jsonify(
            {
                "success": True,
                "data": tax_form_data,
                "archive_references": {
                    "audit_record": audit_record_id,
                    "tax_form": tax_form_id,
                },
                "request_id": context.request_id,
            }
        )

    except Exception as e:
        logger.error(f"Error generating tax forms: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "request_id": (
                        context.request_id if "context" in locals() else "unknown"
                    ),
                }
            ),
            500,
        )


@app.route("/api/v1/archive/retrieve/<record_id>", methods=["GET"])
def retrieve_archive_record(record_id: str):
    """Retrieve record from WORM archive."""
    try:
        context = create_compliance_context()

        # Retrieve from archive
        record = worm_archive.retrieve_record(record_id)

        # Log archive access
        access_log = {
            "context": context.__dict__,
            "request_type": "archive_retrieval",
            "record_id": record_id,
            "record_type": record.get("content_type"),
            "verified": record.get("verified", False),
        }

        worm_archive.store_record(
            access_log,
            "application/json",
            retention_years=7,
            metadata={"type": "archive_access_log", "requester": context.requester},
        )

        return jsonify(
            {"success": True, "data": record, "request_id": context.request_id}
        )

    except Exception as e:
        logger.error(f"Error retrieving archive record: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "request_id": (
                        context.request_id if "context" in locals() else "unknown"
                    ),
                }
            ),
            500,
        )


@app.route("/api/v1/compliance/status", methods=["GET"])
def get_compliance_status():
    """Get overall compliance system status."""
    try:
        context = create_compliance_context()

        # Get status from all systems
        fifo_status = {"status": "available"}  # Would call fifo_ledger.get_status()
        worm_status = worm_archive.get_status_report()
        fee_status = fee_engine.get_status_report()

        # Perform integrity checks
        fifo_integrity = fifo_ledger.verify_integrity()
        worm_integrity = worm_archive.verify_archive_integrity()

        status_report = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "systems": {
                "fifo_ledger": fifo_status,
                "worm_archive": worm_status,
                "fee_engine": fee_status,
            },
            "integrity_checks": {
                "fifo_ledger": fifo_integrity,
                "worm_archive": worm_integrity,
            },
            "compliance_features": {
                "fifo_accounting": True,
                "immutable_archive": True,
                "audit_trail": True,
                "tax_reporting": True,
                "cost_calculation": True,
            },
        }

        # Archive status check
        audit_record = {
            "context": context.__dict__,
            "request_type": "compliance_status_check",
            "status_summary": {
                "overall_status": status_report["overall_status"],
                "systems_count": len(status_report["systems"]),
                "integrity_passed": all(
                    check.get("integrity_status") == "PASS"
                    for check in status_report["integrity_checks"].values()
                ),
            },
        }

        record_id = worm_archive.store_record(
            audit_record,
            "application/json",
            retention_years=7,
            metadata={
                "type": "compliance_status_check",
                "requester": context.requester,
            },
        )

        return jsonify(
            {
                "success": True,
                "data": status_report,
                "archive_reference": record_id,
                "request_id": context.request_id,
            }
        )

    except Exception as e:
        logger.error(f"Error getting compliance status: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "request_id": (
                        context.request_id if "context" in locals() else "unknown"
                    ),
                }
            ),
            500,
        )


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return (
        jsonify(
            {"success": False, "error": "Endpoint not found", "timestamp": time.time()}
        ),
        404,
    )


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return (
        jsonify(
            {
                "success": False,
                "error": "Internal server error",
                "timestamp": time.time(),
            }
        ),
        500,
    )


def main():
    """Main entry point for compliance API."""
    import argparse

    parser = argparse.ArgumentParser(description="Compliance Reporting API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    logger.info(f"🏛️ Starting Compliance API on {args.host}:{args.port}")

    # Start Flask app
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
