#!/usr/bin/env python3
"""
FIFO Tax-Lot Ledger for Compliance & Reporting
Implements First-In-First-Out tax accounting with full audit trail
"""

import os
import sys
import json
import time
import logging
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP
import threading

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("fifo_ledger")


@dataclass
class TaxLot:
    """Represents a tax lot for FIFO accounting."""

    lot_id: str
    symbol: str
    quantity: Decimal
    cost_basis: Decimal  # Per unit cost
    acquisition_date: float
    venue: str
    strategy: str
    lot_type: str  # 'long' or 'short'
    remaining_qty: Decimal
    original_qty: Decimal

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "lot_id": self.lot_id,
            "symbol": self.symbol,
            "quantity": float(self.quantity),
            "cost_basis": float(self.cost_basis),
            "acquisition_date": self.acquisition_date,
            "venue": self.venue,
            "strategy": self.strategy,
            "lot_type": self.lot_type,
            "remaining_qty": float(self.remaining_qty),
            "original_qty": float(self.original_qty),
        }


@dataclass
class Disposition:
    """Represents a disposition (sale/close) of a tax lot."""

    disposition_id: str
    lot_id: str
    symbol: str
    quantity: Decimal
    sale_price: Decimal  # Per unit price
    disposition_date: float
    venue: str
    strategy: str
    cost_basis: Decimal  # Per unit cost from lot
    realized_pnl: Decimal
    holding_period_days: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "disposition_id": self.disposition_id,
            "lot_id": self.lot_id,
            "symbol": self.symbol,
            "quantity": float(self.quantity),
            "sale_price": float(self.sale_price),
            "disposition_date": self.disposition_date,
            "venue": self.venue,
            "strategy": self.strategy,
            "cost_basis": float(self.cost_basis),
            "realized_pnl": float(self.realized_pnl),
            "holding_period_days": self.holding_period_days,
        }


class FIFOLedger:
    """FIFO tax-lot ledger with full audit trail and compliance features."""

    def __init__(self, db_path: str = None):
        """Initialize FIFO ledger."""
        self.db_path = db_path or "/tmp/fifo_ledger.db"
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self._redis_prefix = (
            f"fifoledger:{hashlib.sha1(self.db_path.encode()).hexdigest()}"
        )
        self.lock = threading.RLock()

        # Precision for decimal calculations
        self.precision = Decimal("0.00000001")  # 8 decimal places

        # Initialize database
        self._init_database()

        logger.info(f"💼 FIFO Ledger initialized: {self.db_path}")

    def _redis_key(self, *parts: str) -> str:
        return ":".join([self._redis_prefix, *parts])

    def _connect(self) -> sqlite3.Connection:
        """Return a SQLite connection with a busy timeout for concurrent tests."""
        return sqlite3.connect(self.db_path, timeout=5.0)

    def _init_database(self):
        """Initialize SQLite database with tables."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tax_lots (
                        lot_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        quantity DECIMAL(20,8) NOT NULL,
                        cost_basis DECIMAL(20,8) NOT NULL,
                        acquisition_date REAL NOT NULL,
                        venue TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        lot_type TEXT NOT NULL,
                        remaining_qty DECIMAL(20,8) NOT NULL,
                        original_qty DECIMAL(20,8) NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        hash_signature TEXT NOT NULL
                    )
                """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS dispositions (
                        disposition_id TEXT PRIMARY KEY,
                        lot_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        quantity DECIMAL(20,8) NOT NULL,
                        sale_price DECIMAL(20,8) NOT NULL,
                        disposition_date REAL NOT NULL,
                        venue TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        cost_basis DECIMAL(20,8) NOT NULL,
                        realized_pnl DECIMAL(20,8) NOT NULL,
                        holding_period_days INTEGER NOT NULL,
                        created_at REAL NOT NULL,
                        hash_signature TEXT NOT NULL,
                        FOREIGN KEY (lot_id) REFERENCES tax_lots (lot_id)
                    )
                """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_log (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        old_data TEXT,
                        new_data TEXT,
                        timestamp REAL NOT NULL,
                        hash_signature TEXT NOT NULL
                    )
                """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS corporate_actions (
                        action_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        effective_date REAL NOT NULL,
                        ratio_from DECIMAL(20,8),
                        ratio_to DECIMAL(20,8),
                        dividend_amount DECIMAL(20,8),
                        tax_withheld DECIMAL(20,8),
                        processed_date REAL NOT NULL,
                        hash_signature TEXT NOT NULL
                    )
                """
                )

                # Create indexes for performance
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tax_lots_symbol ON tax_lots (symbol)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tax_lots_acquisition ON tax_lots (acquisition_date)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_dispositions_symbol ON dispositions (symbol)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_dispositions_date ON dispositions (disposition_date)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_log_entity ON audit_log (entity_type, entity_id)"
                )

                conn.commit()

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for data integrity."""
        try:
            # Sort keys for consistent hashing
            sorted_data = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(sorted_data.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash: {e}")
            return "hash_error"

    def _log_audit_event(
        self,
        operation: str,
        entity_type: str,
        entity_id: str,
        old_data: Dict[str, Any] = None,
        new_data: Dict[str, Any] = None,
    ):
        """Log audit event with tamper-proof hash."""
        try:
            audit_record = {
                "operation": operation,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "old_data": json.dumps(old_data, default=str) if old_data else None,
                "new_data": json.dumps(new_data, default=str) if new_data else None,
                "timestamp": time.time(),
            }

            hash_signature = self._calculate_hash(audit_record)

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO audit_log 
                    (operation, entity_type, entity_id, old_data, new_data, timestamp, hash_signature)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        operation,
                        entity_type,
                        entity_id,
                        audit_record["old_data"],
                        audit_record["new_data"],
                        audit_record["timestamp"],
                        hash_signature,
                    ),
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Error logging audit event: {e}")

    def create_tax_lot(self, fill: Dict[str, Any]) -> str:
        """Create a new tax lot from a fill."""
        try:
            with self.lock:
                # Generate unique lot ID
                lot_id = f"lot_{int(time.time() * 1000)}_{hash(json.dumps(fill, sort_keys=True)) % 100000000:08d}"

                # Determine lot type based on fill side
                lot_type = "long" if fill["side"].lower() == "buy" else "short"

                # Create tax lot
                tax_lot = TaxLot(
                    lot_id=lot_id,
                    symbol=fill["symbol"],
                    quantity=Decimal(str(fill["qty"])).quantize(self.precision),
                    cost_basis=Decimal(str(fill["price"])).quantize(self.precision),
                    acquisition_date=fill["timestamp"],
                    venue=fill["venue"],
                    strategy=fill.get("strategy", "unknown"),
                    lot_type=lot_type,
                    remaining_qty=Decimal(str(fill["qty"])).quantize(self.precision),
                    original_qty=Decimal(str(fill["qty"])).quantize(self.precision),
                )

                # Store in database
                lot_data = tax_lot.to_dict()
                lot_data["created_at"] = time.time()
                lot_data["updated_at"] = time.time()
                lot_data["hash_signature"] = self._calculate_hash(lot_data)

                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO tax_lots 
                        (lot_id, symbol, quantity, cost_basis, acquisition_date, venue, 
                         strategy, lot_type, remaining_qty, original_qty, created_at, updated_at, hash_signature)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            tax_lot.lot_id,
                            tax_lot.symbol,
                            float(tax_lot.quantity),
                            float(tax_lot.cost_basis),
                            tax_lot.acquisition_date,
                            tax_lot.venue,
                            tax_lot.strategy,
                            tax_lot.lot_type,
                            float(tax_lot.remaining_qty),
                            float(tax_lot.original_qty),
                            lot_data["created_at"],
                            lot_data["updated_at"],
                            lot_data["hash_signature"],
                        ),
                    )
                    conn.commit()

                # Log audit event
                self._log_audit_event(
                    "CREATE", "tax_lot", lot_id, None, tax_lot.to_dict()
                )

                logger.debug(f"Created tax lot {lot_id} for {fill['symbol']}")
                return lot_id

        except Exception as e:
            logger.error(f"Error creating tax lot: {e}")
            raise

    def dispose_fifo(self, fill: Dict[str, Any]) -> List[Disposition]:
        """Dispose of tax lots using FIFO methodology."""
        try:
            with self.lock:
                dispositions = []
                remaining_qty = Decimal(str(fill["qty"])).quantize(self.precision)

                # Determine opposite lot type to dispose
                target_lot_type = "long" if fill["side"].lower() == "sell" else "short"

                # Get available lots in FIFO order (oldest first)
                with self._connect() as conn:
                    cursor = conn.execute(
                        """
                        SELECT * FROM tax_lots 
                        WHERE symbol = ? AND lot_type = ? AND remaining_qty > 0
                        ORDER BY acquisition_date ASC
                    """,
                        (fill["symbol"], target_lot_type),
                    )

                    available_lots = cursor.fetchall()

                if not available_lots:
                    logger.warning(
                        f"No available {target_lot_type} lots for {fill['symbol']}"
                    )
                    return dispositions

                # Dispose lots in FIFO order
                for lot_row in available_lots:
                    if remaining_qty <= 0:
                        break

                    lot_remaining = Decimal(str(lot_row[8]))  # remaining_qty column
                    dispose_qty = min(remaining_qty, lot_remaining)

                    # Create disposition record
                    disposition_id = f"disp_{int(time.time() * 1000)}_{hash(str(lot_row[0]) + str(time.time())) % 100000000:08d}"

                    holding_period = (fill["timestamp"] - lot_row[4]) / (
                        24 * 3600
                    )  # Days

                    disposition = Disposition(
                        disposition_id=disposition_id,
                        lot_id=lot_row[0],  # lot_id
                        symbol=fill["symbol"],
                        quantity=dispose_qty,
                        sale_price=Decimal(str(fill["price"])).quantize(self.precision),
                        disposition_date=fill["timestamp"],
                        venue=fill["venue"],
                        strategy=fill.get("strategy", "unknown"),
                        cost_basis=Decimal(str(lot_row[3])).quantize(
                            self.precision
                        ),  # cost_basis
                        realized_pnl=(
                            Decimal(str(fill["price"])) - Decimal(str(lot_row[3]))
                        )
                        * dispose_qty,
                        holding_period_days=int(holding_period),
                    )

                    # Update lot remaining quantity
                    new_remaining = lot_remaining - dispose_qty

                    with self._connect() as conn:
                        # Insert disposition
                        disp_data = disposition.to_dict()
                        disp_data["created_at"] = time.time()
                        disp_data["hash_signature"] = self._calculate_hash(disp_data)

                        conn.execute(
                            """
                            INSERT INTO dispositions 
                            (disposition_id, lot_id, symbol, quantity, sale_price, disposition_date,
                             venue, strategy, cost_basis, realized_pnl, holding_period_days, 
                             created_at, hash_signature)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                disposition.disposition_id,
                                disposition.lot_id,
                                disposition.symbol,
                                float(disposition.quantity),
                                float(disposition.sale_price),
                                disposition.disposition_date,
                                disposition.venue,
                                disposition.strategy,
                                float(disposition.cost_basis),
                                float(disposition.realized_pnl),
                                disposition.holding_period_days,
                                disp_data["created_at"],
                                disp_data["hash_signature"],
                            ),
                        )

                        # Update tax lot remaining quantity
                        conn.execute(
                            """
                            UPDATE tax_lots 
                            SET remaining_qty = ?, updated_at = ?
                            WHERE lot_id = ?
                        """,
                            (float(new_remaining), time.time(), lot_row[0]),
                        )

                        conn.commit()

                    # Log audit events
                    self._log_audit_event(
                        "CREATE",
                        "disposition",
                        disposition_id,
                        None,
                        disposition.to_dict(),
                    )
                    self._log_audit_event(
                        "UPDATE",
                        "tax_lot",
                        lot_row[0],
                        {"remaining_qty": float(lot_remaining)},
                        {"remaining_qty": float(new_remaining)},
                    )

                    dispositions.append(disposition)
                    remaining_qty -= dispose_qty

                    logger.debug(f"Disposed {dispose_qty} from lot {lot_row[0]}")

                if remaining_qty > 0:
                    logger.warning(
                        f"Could not dispose full quantity: {remaining_qty} remaining"
                    )

                return dispositions

        except Exception as e:
            logger.error(f"Error disposing FIFO: {e}")
            raise

    def process_fill(self, fill: Dict[str, Any]) -> Dict[str, Any]:
        """Process a fill and update tax lots accordingly."""
        try:
            result = {
                "fill_id": fill.get("fill_id", "unknown"),
                "symbol": fill["symbol"],
                "side": fill["side"],
                "quantity": fill["qty"],
                "price": fill["price"],
                "timestamp": fill["timestamp"],
                "action": None,
                "tax_lots_created": [],
                "dispositions_created": [],
                "realized_pnl": Decimal("0"),
                "unrealized_pnl": Decimal("0"),
            }

            if fill["side"].lower() in ["buy", "long"]:
                # Opening position - create tax lot
                lot_id = self.create_tax_lot(fill)
                result["action"] = "create_lot"
                result["tax_lots_created"].append(lot_id)

            elif fill["side"].lower() in ["sell", "short"]:
                # Closing position - dispose using FIFO
                dispositions = self.dispose_fifo(fill)
                result["action"] = "dispose_fifo"
                result["dispositions_created"] = [
                    d.disposition_id for d in dispositions
                ]
                result["realized_pnl"] = sum(d.realized_pnl for d in dispositions)

                # If remaining quantity, create short tax lot
                disposed_qty = sum(d.quantity for d in dispositions)
                remaining_qty = Decimal(str(fill["qty"])) - disposed_qty

                if remaining_qty > 0:
                    # Create short lot for remaining quantity
                    short_fill = fill.copy()
                    short_fill["qty"] = float(remaining_qty)
                    short_fill["side"] = "short"
                    lot_id = self.create_tax_lot(short_fill)
                    result["tax_lots_created"].append(lot_id)

                    # Store processing result in Redis for audit trail
                    self.redis.hset(
                        self._redis_key("fill", fill.get("fill_id", int(time.time()))),
                        mapping={
                            k: json.dumps(v, default=str) for k, v in result.items()
                        },
                    )

            logger.info(f"Processed fill {fill.get('fill_id')}: {result['action']}")
            return result

        except Exception as e:
            logger.error(f"Error processing fill: {e}")
            raise

    def get_position_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get current position summary with unrealized P&L."""
        try:
            query = "SELECT * FROM tax_lots WHERE remaining_qty > 0"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            query += " ORDER BY symbol, acquisition_date"

            with self._connect() as conn:
                cursor = conn.execute(query, params)
                active_lots = cursor.fetchall()

            positions: Dict[str, Dict[str, Dict[str, float]]] = {}

            for lot in active_lots:
                sym = lot[1]
                lot_type = lot[7]

                buckets = positions.setdefault(
                    sym,
                    {
                        "long": {
                            "total_qty": Decimal("0"),
                            "sum_cost": Decimal("0"),
                            "lots": 0,
                        },
                        "short": {
                            "total_qty": Decimal("0"),
                            "sum_cost": Decimal("0"),
                            "lots": 0,
                        },
                    },
                )

                bucket = buckets["long" if lot_type == "long" else "short"]
                remaining_qty = Decimal(str(lot[8]))
                cost_basis = Decimal(str(lot[3]))

                bucket["total_qty"] += remaining_qty
                bucket["sum_cost"] += remaining_qty * cost_basis
                bucket["lots"] += 1

            # Finalize averages and convert to floats
            for sym, buckets in positions.items():
                for side in ("long", "short"):
                    bucket = buckets[side]
                    qty = bucket["total_qty"]
                    sum_cost = bucket["sum_cost"]

                    avg_cost = (sum_cost / qty) if qty > 0 else Decimal("0")
                    bucket["avg_cost_basis"] = float(avg_cost)
                    bucket["total_qty"] = float(qty)
                    bucket["lots"] = int(bucket["lots"])
                    del bucket["sum_cost"]

            summary = {
                "timestamp": time.time(),
                "positions": positions,
                "total_symbols": len(positions),
                "total_lots": len(active_lots),
            }

            if symbol:
                return positions

            return summary

        except Exception as e:
            logger.error(f"Error getting position summary: {e}")
            return {"error": str(e), "timestamp": time.time()}

    def get_realized_pnl_report(
        self, start_date: float = None, end_date: float = None
    ) -> Dict[str, Any]:
        """Generate realized P&L report for tax purposes."""
        try:
            query = "SELECT * FROM dispositions"
            params = []

            if start_date or end_date:
                query += " WHERE"
                conditions = []

                if start_date:
                    conditions.append(" disposition_date >= ?")
                    params.append(start_date)

                if end_date:
                    conditions.append(" disposition_date <= ?")
                    params.append(end_date)

                query += " AND".join(conditions)

            query += " ORDER BY disposition_date"

            with self._connect() as conn:
                cursor = conn.execute(query, params)
                dispositions = cursor.fetchall()

            # Calculate summary metrics
            total_realized_pnl = Decimal("0")
            short_term_pnl = Decimal("0")
            long_term_pnl = Decimal("0")

            by_symbol = {}
            by_strategy = {}

            for disp in dispositions:
                symbol = disp[2]
                realized_pnl = Decimal(str(disp[9]))
                holding_days = disp[10]
                strategy = disp[7]

                total_realized_pnl += realized_pnl

                # Short-term vs long-term (365 days threshold)
                if holding_days <= 365:
                    short_term_pnl += realized_pnl
                else:
                    long_term_pnl += realized_pnl

                # By symbol
                if symbol not in by_symbol:
                    by_symbol[symbol] = {"realized_pnl": Decimal("0"), "count": 0}
                by_symbol[symbol]["realized_pnl"] += realized_pnl
                by_symbol[symbol]["count"] += 1

                # By strategy
                if strategy not in by_strategy:
                    by_strategy[strategy] = {"realized_pnl": Decimal("0"), "count": 0}
                by_strategy[strategy]["realized_pnl"] += realized_pnl
                by_strategy[strategy]["count"] += 1

            # Convert to float for JSON
            for sym_data in by_symbol.values():
                sym_data["realized_pnl"] = float(sym_data["realized_pnl"])

            for strat_data in by_strategy.values():
                strat_data["realized_pnl"] = float(strat_data["realized_pnl"])

            report = {
                "timestamp": time.time(),
                "period_start": start_date,
                "period_end": end_date,
                "total_dispositions": len(dispositions),
                "total_realized_pnl": float(total_realized_pnl),
                "short_term_pnl": float(short_term_pnl),
                "long_term_pnl": float(long_term_pnl),
                "by_symbol": by_symbol,
                "by_strategy": by_strategy,
            }

            return report

        except Exception as e:
            logger.error(f"Error generating realized P&L report: {e}")
            return {"error": str(e), "timestamp": time.time()}

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify database integrity and audit trail."""
        try:
            integrity_report = {
                "timestamp": time.time(),
                "checks": {},
                "errors": [],
                "warnings": [],
            }

            with self._connect() as conn:
                # Check hash signatures
                cursor = conn.execute("SELECT COUNT(*) FROM tax_lots")
                total_lots = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM dispositions")
                total_dispositions = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM audit_log")
                total_audit_logs = cursor.fetchone()[0]

                integrity_report["checks"]["total_lots"] = total_lots
                integrity_report["checks"]["total_dispositions"] = total_dispositions
                integrity_report["checks"]["total_audit_logs"] = total_audit_logs

                # Verify hash signatures for tax lots
                cursor = conn.execute("SELECT * FROM tax_lots LIMIT 100")
                for lot in cursor.fetchall():
                    # Reconstruct data and verify hash
                    lot_data = {
                        "lot_id": lot[0],
                        "symbol": lot[1],
                        "quantity": lot[2],
                        "cost_basis": lot[3],
                        "acquisition_date": lot[4],
                        "venue": lot[5],
                        "strategy": lot[6],
                        "lot_type": lot[7],
                        "remaining_qty": lot[8],
                        "original_qty": lot[9],
                        "created_at": lot[10],
                        "updated_at": lot[11],
                    }
                    expected_hash = self._calculate_hash(lot_data)
                    actual_hash = lot[12]

                    if expected_hash != actual_hash:
                        integrity_report["errors"].append(
                            f"Hash mismatch for tax lot {lot[0]}: expected {expected_hash}, got {actual_hash}"
                        )

                # Check for orphaned dispositions
                cursor = conn.execute(
                    """
                    SELECT d.disposition_id 
                    FROM dispositions d 
                    LEFT JOIN tax_lots t ON d.lot_id = t.lot_id 
                    WHERE t.lot_id IS NULL
                """
                )
                orphaned_dispositions = cursor.fetchall()

                if orphaned_dispositions:
                    integrity_report["warnings"].append(
                        f"Found {len(orphaned_dispositions)} orphaned dispositions"
                    )

                integrity_report["checks"]["integrity_status"] = (
                    "PASS" if not integrity_report["errors"] else "FAIL"
                )

            return integrity_report

        except Exception as e:
            logger.error(f"Error verifying integrity: {e}")
            return {
                "timestamp": time.time(),
                "checks": {"integrity_status": "ERROR"},
                "errors": [str(e)],
            }

    def apply_split(
        self,
        symbol: str,
        ratio_from: float,
        ratio_to: float,
        effective_date: float = None,
    ) -> Dict[str, Any]:
        """
        Apply stock split to all tax lots for a symbol.

        Args:
            symbol: Stock symbol
            ratio_from: Original shares (e.g., 1 for 1:2 split)
            ratio_to: New shares (e.g., 2 for 1:2 split)
            effective_date: Unix timestamp of split (defaults to now)

        Returns:
            Dict with split processing results
        """
        if effective_date is None:
            effective_date = time.time()

        split_ratio = Decimal(str(ratio_to)) / Decimal(str(ratio_from))

        try:
            with self.lock:
                with self._connect() as conn:
                    # Get all active lots for symbol
                    cursor = conn.execute(
                        "SELECT * FROM tax_lots WHERE symbol = ? AND remaining_qty > 0 ORDER BY acquisition_date",
                        (symbol,),
                    )
                    active_lots = cursor.fetchall()

                    if not active_lots:
                        logger.warning(f"No active lots found for split: {symbol}")
                        return {"lots_affected": 0, "split_ratio": float(split_ratio)}

                    # Record corporate action
                    action_id = f"split_{symbol}_{int(effective_date)}"
                    corp_action = {
                        "action_id": action_id,
                        "symbol": symbol,
                        "action_type": "STOCK_SPLIT",
                        "effective_date": effective_date,
                        "ratio_from": float(ratio_from),
                        "ratio_to": float(ratio_to),
                        "processed_date": time.time(),
                        "hash_signature": "",
                    }
                    corp_action["hash_signature"] = self._calculate_hash(corp_action)

                    conn.execute(
                        """
                        INSERT INTO corporate_actions 
                        (action_id, symbol, action_type, effective_date, ratio_from, ratio_to, processed_date, hash_signature)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            action_id,
                            symbol,
                            "STOCK_SPLIT",
                            effective_date,
                            ratio_from,
                            ratio_to,
                            corp_action["processed_date"],
                            corp_action["hash_signature"],
                        ),
                    )

                    lots_updated = 0
                    for lot_row in active_lots:
                        old_lot = {
                            "lot_id": lot_row[0],
                            "quantity": lot_row[2],
                            "cost_basis": lot_row[3],
                            "remaining_qty": lot_row[8],
                        }

                        # Apply split: quantity increases, cost basis decreases proportionally
                        new_qty = Decimal(str(lot_row[2])) * split_ratio
                        new_cost_basis = Decimal(str(lot_row[3])) / split_ratio
                        new_remaining_qty = Decimal(str(lot_row[8])) * split_ratio

                        # Update the lot
                        conn.execute(
                            """
                            UPDATE tax_lots 
                            SET quantity = ?, cost_basis = ?, remaining_qty = ?, updated_at = ?
                            WHERE lot_id = ?
                        """,
                            (
                                float(new_qty),
                                float(new_cost_basis),
                                float(new_remaining_qty),
                                time.time(),
                                lot_row[0],
                            ),
                        )

                        new_lot = {
                            "lot_id": lot_row[0],
                            "quantity": float(new_qty),
                            "cost_basis": float(new_cost_basis),
                            "remaining_qty": float(new_remaining_qty),
                        }

                        # Log audit event
                        self._log_audit_event(
                            "STOCK_SPLIT", "tax_lot", lot_row[0], old_lot, new_lot
                        )
                        lots_updated += 1

                    conn.commit()

                    # Update Redis metrics
                    self.redis.hset(
                        self._redis_key("corporate_actions", symbol),
                        "last_split",
                        effective_date,
                    )
                    self.redis.incr(
                        self._redis_key(
                            "metrics", "corporate_actions", "splits_processed"
                        )
                    )

                    result = {
                        "action_id": action_id,
                        "symbol": symbol,
                        "split_ratio": float(split_ratio),
                        "lots_affected": lots_updated,
                        "effective_date": effective_date,
                        "processed_at": time.time(),
                    }

                    logger.info(
                        f"✅ Stock split processed: {symbol} {ratio_from}:{ratio_to}, {lots_updated} lots updated"
                    )
                    return result

        except Exception as e:
            logger.error(f"Error applying stock split for {symbol}: {e}")
            raise

    def record_dividend(
        self,
        symbol: str,
        gross_amount: float,
        tax_withheld: float = 0.0,
        record_date: float = None,
    ) -> Dict[str, Any]:
        """
        Record dividend payment.

        Args:
            symbol: Stock symbol
            gross_amount: Gross dividend amount
            tax_withheld: Tax withheld amount
            record_date: Record date (defaults to now)

        Returns:
            Dict with dividend recording results
        """
        if record_date is None:
            record_date = time.time()

        net_amount = gross_amount - tax_withheld

        try:
            with self.lock:
                with self._connect() as conn:
                    # Record corporate action
                    action_id = f"dividend_{symbol}_{int(record_date)}"
                    corp_action = {
                        "action_id": action_id,
                        "symbol": symbol,
                        "action_type": "DIVIDEND",
                        "effective_date": record_date,
                        "dividend_amount": gross_amount,
                        "tax_withheld": tax_withheld,
                        "processed_date": time.time(),
                        "hash_signature": "",
                    }
                    corp_action["hash_signature"] = self._calculate_hash(corp_action)

                    conn.execute(
                        """
                        INSERT INTO corporate_actions 
                        (action_id, symbol, action_type, effective_date, dividend_amount, tax_withheld, processed_date, hash_signature)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            action_id,
                            symbol,
                            "DIVIDEND",
                            record_date,
                            gross_amount,
                            tax_withheld,
                            corp_action["processed_date"],
                            corp_action["hash_signature"],
                        ),
                    )

                    conn.commit()

                    # Update cash balance in Redis
                    current_cash = float(
                        self.redis.get(self._redis_key("portfolio", "cash_usd")) or 0
                    )
                    new_cash = current_cash + net_amount
                    self.redis.set(self._redis_key("portfolio", "cash_usd"), new_cash)

                    # Update metrics
                    self.redis.hincrbyfloat(
                        self._redis_key(
                            "metrics", "corporate_actions", "total_dividends"
                        ),
                        gross_amount,
                    )
                    self.redis.hincrbyfloat(
                        self._redis_key(
                            "metrics", "corporate_actions", "total_tax_withheld"
                        ),
                        tax_withheld,
                    )
                    self.redis.incr(
                        self._redis_key(
                            "metrics", "corporate_actions", "dividends_processed"
                        )
                    )

                    # Log to WORM archive
                    worm_record = {
                        "type": "DIVIDEND",
                        "symbol": symbol,
                        "gross_amount": gross_amount,
                        "tax_withheld": tax_withheld,
                        "net_amount": net_amount,
                        "record_date": record_date,
                        "action_id": action_id,
                    }
                    self.redis.xadd(
                        self._redis_key("worm", "corporate_actions"), worm_record
                    )

                    result = {
                        "action_id": action_id,
                        "symbol": symbol,
                        "gross_amount": gross_amount,
                        "tax_withheld": tax_withheld,
                        "net_amount": net_amount,
                        "record_date": record_date,
                        "new_cash_balance": new_cash,
                    }

                    logger.info(
                        f"✅ Dividend recorded: {symbol} ${gross_amount:.2f} gross, ${net_amount:.2f} net"
                    )
                    return result

        except Exception as e:
            logger.error(f"Error recording dividend for {symbol}: {e}")
            raise

    def get_corporate_actions(
        self, symbol: str = None, start_date: float = None, end_date: float = None
    ) -> List[Dict[str, Any]]:
        """
        Get corporate actions history.

        Args:
            symbol: Filter by symbol (optional)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            List of corporate actions
        """
        try:
            query = "SELECT * FROM corporate_actions WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if start_date:
                query += " AND effective_date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND effective_date <= ?"
                params.append(end_date)

            query += " ORDER BY effective_date DESC"

            with self._connect() as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

            actions = []
            for row in rows:
                action = {
                    "action_id": row[0],
                    "symbol": row[1],
                    "action_type": row[2],
                    "effective_date": row[3],
                    "ratio_from": row[4],
                    "ratio_to": row[5],
                    "dividend_amount": row[6],
                    "tax_withheld": row[7],
                    "processed_date": row[8],
                    "hash_signature": row[9],
                }
                actions.append(action)

            return actions

        except Exception as e:
            logger.error(f"Error getting corporate actions: {e}")
            return []


def main():
    """Main entry point for FIFO ledger."""
    import argparse

    parser = argparse.ArgumentParser(description="FIFO Tax-Lot Ledger")
    parser.add_argument(
        "--positions", action="store_true", help="Show current positions"
    )
    parser.add_argument(
        "--realized-pnl", action="store_true", help="Show realized P&L report"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify database integrity"
    )
    parser.add_argument("--symbol", type=str, help="Filter by symbol")
    parser.add_argument("--start-date", type=float, help="Start date (Unix timestamp)")
    parser.add_argument("--end-date", type=float, help="End date (Unix timestamp)")

    args = parser.parse_args()

    # Create FIFO ledger
    ledger = FIFOLedger()

    if args.positions:
        summary = ledger.get_position_summary(args.symbol)
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.realized_pnl:
        report = ledger.get_realized_pnl_report(args.start_date, args.end_date)
        print(json.dumps(report, indent=2, default=str))
        return

    if args.verify:
        integrity = ledger.verify_integrity()
        print(json.dumps(integrity, indent=2, default=str))
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
