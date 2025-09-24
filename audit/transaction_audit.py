#!/usr/bin/env python3
"""
Transaction Audit Trail System
Comprehensive audit logging for all trading system transactions
"""

import os
import sys
import json
import time
import logging
import sqlite3
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("transaction_audit")


class AuditEventType(Enum):
    """Types of audit events."""

    TRADE_EXECUTION = "trade_execution"
    ORDER_SUBMISSION = "order_submission"
    ORDER_CANCELLATION = "order_cancellation"
    POSITION_UPDATE = "position_update"
    RISK_CHECK = "risk_check"
    PARAMETER_CHANGE = "parameter_change"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    ERROR_EVENT = "error_event"
    COMPLIANCE_CHECK = "compliance_check"
    USER_ACCESS = "user_access"
    DATA_MODIFICATION = "data_modification"


@dataclass
class AuditEvent:
    """Represents an audit event."""

    event_id: str
    event_type: AuditEventType
    timestamp: float
    entity_type: str
    entity_id: str
    actor: str
    action: str
    old_state: Optional[Dict[str, Any]]
    new_state: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str]
    correlation_id: Optional[str]
    session_id: Optional[str]
    checksum: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "actor": self.actor,
            "action": self.action,
            "old_state": self.old_state,
            "new_state": self.new_state,
            "metadata": self.metadata,
            "success": self.success,
            "error_message": self.error_message,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "checksum": self.checksum,
        }


class TransactionAuditTrail:
    """Comprehensive audit trail system for all transactions."""

    def __init__(self, db_path: str = None):
        """Initialize audit trail system."""
        self.db_path = db_path or "/tmp/transaction_audit.db"
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.lock = threading.RLock()

        # Session tracking
        self.current_session = str(uuid.uuid4())

        # Initialize database
        self._init_database()

        # Start background processes
        self._start_background_tasks()

        logger.info(f"🔍 Transaction Audit Trail initialized: {self.db_path}")
        logger.info(f"   Session ID: {self.current_session}")

    def _init_database(self):
        """Initialize SQLite database with audit tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Main audit events table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        entity_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        action TEXT NOT NULL,
                        old_state TEXT,
                        new_state TEXT,
                        metadata TEXT NOT NULL,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        correlation_id TEXT,
                        session_id TEXT,
                        checksum TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                """
                )

                # Audit trail integrity table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_integrity (
                        block_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        start_event_id TEXT NOT NULL,
                        end_event_id TEXT NOT NULL,
                        event_count INTEGER NOT NULL,
                        block_hash TEXT NOT NULL,
                        previous_block_hash TEXT,
                        timestamp REAL NOT NULL
                    )
                """
                )

                # Session tracking table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_sessions (
                        session_id TEXT PRIMARY KEY,
                        start_time REAL NOT NULL,
                        end_time REAL,
                        actor TEXT NOT NULL,
                        session_type TEXT NOT NULL,
                        metadata TEXT,
                        event_count INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'active'
                    )
                """
                )

                # Entity change history
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS entity_history (
                        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        change_timestamp REAL NOT NULL,
                        event_id TEXT NOT NULL,
                        field_name TEXT NOT NULL,
                        old_value TEXT,
                        new_value TEXT,
                        FOREIGN KEY (event_id) REFERENCES audit_events (event_id)
                    )
                """
                )

                # Create indexes for performance
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events (timestamp)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_events (entity_type, entity_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_events (actor)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events (event_type)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_correlation ON audit_events (correlation_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_events (session_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_entity_history ON entity_history (entity_type, entity_id)"
                )

                conn.commit()

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _calculate_checksum(self, event_data: Dict[str, Any]) -> str:
        """Calculate checksum for event integrity."""
        try:
            # Create deterministic string representation
            checksum_data = {
                "event_type": event_data["event_type"],
                "timestamp": event_data["timestamp"],
                "entity_type": event_data["entity_type"],
                "entity_id": event_data["entity_id"],
                "actor": event_data["actor"],
                "action": event_data["action"],
                "old_state": event_data.get("old_state"),
                "new_state": event_data.get("new_state"),
            }

            checksum_string = json.dumps(checksum_data, sort_keys=True, default=str)
            return hashlib.sha256(checksum_string.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            return "checksum_error"

    def _start_background_tasks(self):
        """Start background tasks for audit maintenance."""
        try:
            # Register current session
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO audit_sessions 
                    (session_id, start_time, actor, session_type, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        self.current_session,
                        time.time(),
                        "system",
                        "trading_session",
                        json.dumps({"component": "transaction_audit"}),
                    ),
                )
                conn.commit()

            # Start integrity block creation timer
            threading.Timer(
                300.0, self._create_integrity_block
            ).start()  # Every 5 minutes

        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")

    def _create_integrity_block(self):
        """Create integrity block for audit trail tamper detection."""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Get latest block
                    cursor = conn.execute(
                        """
                        SELECT block_hash FROM audit_integrity 
                        ORDER BY block_id DESC LIMIT 1
                    """
                    )
                    previous_block = cursor.fetchone()
                    previous_hash = previous_block[0] if previous_block else "genesis"

                    # Get events since last block
                    cursor = conn.execute(
                        """
                        SELECT event_id FROM audit_events 
                        WHERE event_id NOT IN (
                            SELECT event_id FROM audit_events e
                            JOIN audit_integrity i ON e.timestamp BETWEEN 
                                (SELECT timestamp FROM audit_events WHERE event_id = i.start_event_id) AND
                                (SELECT timestamp FROM audit_events WHERE event_id = i.end_event_id)
                        )
                        ORDER BY timestamp
                    """
                    )
                    new_events = cursor.fetchall()

                    if len(new_events) < 10:  # Wait for at least 10 events
                        threading.Timer(300.0, self._create_integrity_block).start()
                        return

                    # Create block hash
                    start_event = new_events[0][0]
                    end_event = new_events[-1][0]
                    event_count = len(new_events)

                    block_data = {
                        "start_event_id": start_event,
                        "end_event_id": end_event,
                        "event_count": event_count,
                        "previous_block_hash": previous_hash,
                        "timestamp": time.time(),
                    }

                    block_hash = hashlib.sha256(
                        json.dumps(block_data, sort_keys=True).encode()
                    ).hexdigest()

                    # Insert integrity block
                    conn.execute(
                        """
                        INSERT INTO audit_integrity 
                        (start_event_id, end_event_id, event_count, block_hash, 
                         previous_block_hash, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            start_event,
                            end_event,
                            event_count,
                            block_hash,
                            previous_hash,
                            block_data["timestamp"],
                        ),
                    )
                    conn.commit()

                    logger.debug(
                        f"Created integrity block: {block_hash[:16]}... ({event_count} events)"
                    )

            # Schedule next block creation
            threading.Timer(300.0, self._create_integrity_block).start()

        except Exception as e:
            logger.error(f"Error creating integrity block: {e}")
            threading.Timer(300.0, self._create_integrity_block).start()

    def log_event(
        self,
        event_type: AuditEventType,
        entity_type: str,
        entity_id: str,
        actor: str,
        action: str,
        old_state: Optional[Dict[str, Any]] = None,
        new_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Log an audit event."""
        try:
            with self.lock:
                # Generate event ID
                event_id = str(uuid.uuid4())
                timestamp = time.time()

                # Create event data
                event_data = {
                    "event_id": event_id,
                    "event_type": (
                        event_type.value
                        if hasattr(event_type, "value")
                        else str(event_type)
                    ),
                    "timestamp": timestamp,
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "actor": actor,
                    "action": action,
                    "old_state": old_state,
                    "new_state": new_state,
                    "metadata": metadata or {},
                    "success": success,
                    "error_message": error_message,
                    "correlation_id": correlation_id,
                    "session_id": self.current_session,
                }

                # Calculate checksum
                checksum = self._calculate_checksum(event_data)
                event_data["checksum"] = checksum

                # Create audit event
                audit_event = AuditEvent(**event_data)

                # Store in database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO audit_events 
                        (event_id, event_type, timestamp, entity_type, entity_id, actor, action,
                         old_state, new_state, metadata, success, error_message, 
                         correlation_id, session_id, checksum, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            event_id,
                            (
                                event_type.value
                                if hasattr(event_type, "value")
                                else str(event_type)
                            ),
                            timestamp,
                            entity_type,
                            entity_id,
                            actor,
                            action,
                            json.dumps(old_state, default=str) if old_state else None,
                            json.dumps(new_state, default=str) if new_state else None,
                            json.dumps(metadata or {}, default=str),
                            success,
                            error_message,
                            correlation_id,
                            self.current_session,
                            checksum,
                            time.time(),
                        ),
                    )

                    # Update session event count
                    conn.execute(
                        """
                        UPDATE audit_sessions 
                        SET event_count = event_count + 1 
                        WHERE session_id = ?
                    """,
                        (self.current_session,),
                    )

                    # Track entity field changes
                    if old_state and new_state:
                        self._track_entity_changes(
                            conn,
                            event_id,
                            entity_type,
                            entity_id,
                            old_state,
                            new_state,
                            timestamp,
                        )

                    conn.commit()

                # Store in Redis for real-time monitoring
                event_type_str = (
                    event_type.value
                    if hasattr(event_type, "value")
                    else str(event_type)
                )
                redis_key = f"audit:events:{event_type_str}"
                self.redis.lpush(
                    redis_key, json.dumps(audit_event.to_dict(), default=str)
                )
                self.redis.ltrim(redis_key, 0, 1000)  # Keep last 1000 events per type

                # Update metrics
                self.redis.incr("audit:total_events")
                self.redis.incr(f"audit:events_by_type:{event_type_str}")
                self.redis.incr(f"audit:events_by_actor:{actor}")

                logger.debug(f"Logged audit event: {event_id} ({event_type_str})")
                return event_id

        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
            raise

    def _track_entity_changes(
        self,
        conn,
        event_id: str,
        entity_type: str,
        entity_id: str,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        timestamp: float,
    ):
        """Track individual field changes for entities."""
        try:
            # Compare old and new states
            all_fields = set(old_state.keys()) | set(new_state.keys())

            for field in all_fields:
                old_value = old_state.get(field)
                new_value = new_state.get(field)

                if old_value != new_value:
                    conn.execute(
                        """
                        INSERT INTO entity_history 
                        (entity_type, entity_id, change_timestamp, event_id, 
                         field_name, old_value, new_value)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            entity_type,
                            entity_id,
                            timestamp,
                            event_id,
                            field,
                            (
                                json.dumps(old_value, default=str)
                                if old_value is not None
                                else None
                            ),
                            (
                                json.dumps(new_value, default=str)
                                if new_value is not None
                                else None
                            ),
                        ),
                    )

        except Exception as e:
            logger.error(f"Error tracking entity changes: {e}")

    def log_trade_execution(
        self, fill: Dict[str, Any], actor: str = "trading_system"
    ) -> str:
        """Log trade execution event."""
        return self.log_event(
            event_type=AuditEventType.TRADE_EXECUTION,
            entity_type="trade",
            entity_id=fill.get("fill_id", "unknown"),
            actor=actor,
            action="execute_trade",
            new_state=fill,
            metadata={
                "venue": fill.get("venue"),
                "symbol": fill.get("symbol"),
                "side": fill.get("side"),
                "quantity": fill.get("qty"),
                "price": fill.get("price"),
                "strategy": fill.get("strategy", "unknown"),
            },
        )

    def log_order_event(
        self, order: Dict[str, Any], action: str, actor: str = "trading_system"
    ) -> str:
        """Log order-related event."""
        event_type_map = {
            "submit": AuditEventType.ORDER_SUBMISSION,
            "cancel": AuditEventType.ORDER_CANCELLATION,
            "modify": AuditEventType.DATA_MODIFICATION,
        }

        return self.log_event(
            event_type=event_type_map.get(action, AuditEventType.DATA_MODIFICATION),
            entity_type="order",
            entity_id=order.get("order_id", "unknown"),
            actor=actor,
            action=action,
            new_state=order,
            metadata={
                "venue": order.get("venue"),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "order_type": order.get("order_type"),
            },
        )

    def log_position_update(
        self,
        position: Dict[str, Any],
        old_position: Dict[str, Any] = None,
        actor: str = "position_manager",
    ) -> str:
        """Log position update event."""
        return self.log_event(
            event_type=AuditEventType.POSITION_UPDATE,
            entity_type="position",
            entity_id=f"{position.get('symbol', 'unknown')}_{position.get('venue', 'unknown')}",
            actor=actor,
            action="update_position",
            old_state=old_position,
            new_state=position,
            metadata={
                "symbol": position.get("symbol"),
                "venue": position.get("venue"),
                "quantity_change": (
                    position.get("quantity", 0) - old_position.get("quantity", 0)
                    if old_position
                    else position.get("quantity", 0)
                ),
            },
        )

    def log_risk_check(
        self,
        risk_check: Dict[str, Any],
        success: bool = True,
        error_message: str = None,
        actor: str = "risk_manager",
    ) -> str:
        """Log risk check event."""
        return self.log_event(
            event_type=AuditEventType.RISK_CHECK,
            entity_type="risk_check",
            entity_id=risk_check.get("check_id", str(uuid.uuid4())),
            actor=actor,
            action="perform_risk_check",
            new_state=risk_check,
            success=success,
            error_message=error_message,
            metadata={
                "check_type": risk_check.get("check_type"),
                "entity_checked": risk_check.get("entity_type"),
                "risk_level": risk_check.get("risk_level"),
            },
        )

    def log_system_event(
        self,
        event_type: AuditEventType,
        component: str,
        details: Dict[str, Any],
        actor: str = "system",
    ) -> str:
        """Log system-level event."""
        return self.log_event(
            event_type=event_type,
            entity_type="system",
            entity_id=component,
            actor=actor,
            action=f"system_{event_type.value}",
            new_state=details,
            metadata={"component": component, "timestamp": time.time()},
        )

    def get_audit_trail(
        self,
        entity_type: str = None,
        entity_id: str = None,
        start_time: float = None,
        end_time: float = None,
        actor: str = None,
        event_types: List[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Retrieve audit trail with filtering."""
        try:
            query = "SELECT * FROM audit_events WHERE 1=1"
            params = []

            if entity_type:
                query += " AND entity_type = ?"
                params.append(entity_type)

            if entity_id:
                query += " AND entity_id = ?"
                params.append(entity_id)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if actor:
                query += " AND actor = ?"
                params.append(actor)

            if event_types:
                placeholders = ",".join("?" * len(event_types))
                query += f" AND event_type IN ({placeholders})"
                params.extend(event_types)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                events = cursor.fetchall()

            # Convert to dictionaries
            columns = [desc[0] for desc in cursor.description]
            result = []

            for event in events:
                event_dict = dict(zip(columns, event))
                # Parse JSON fields
                for field in ["old_state", "new_state", "metadata"]:
                    if event_dict[field]:
                        try:
                            event_dict[field] = json.loads(event_dict[field])
                        except json.JSONDecodeError:
                            event_dict[field] = None

                result.append(event_dict)

            return result

        except Exception as e:
            logger.error(f"Error retrieving audit trail: {e}")
            return []

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify audit trail integrity."""
        try:
            integrity_report = {
                "timestamp": time.time(),
                "total_events": 0,
                "verified_events": 0,
                "checksum_failures": 0,
                "block_integrity": True,
                "errors": [],
            }

            with sqlite3.connect(self.db_path) as conn:
                # Check total events
                cursor = conn.execute("SELECT COUNT(*) FROM audit_events")
                integrity_report["total_events"] = cursor.fetchone()[0]

                # Verify event checksums
                cursor = conn.execute(
                    """
                    SELECT event_id, event_type, timestamp, entity_type, entity_id, 
                           actor, action, old_state, new_state, checksum 
                    FROM audit_events 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                """
                )

                for event in cursor.fetchall():
                    try:
                        # Reconstruct event data for checksum verification
                        event_data = {
                            "event_type": event[1],
                            "timestamp": event[2],
                            "entity_type": event[3],
                            "entity_id": event[4],
                            "actor": event[5],
                            "action": event[6],
                            "old_state": json.loads(event[7]) if event[7] else None,
                            "new_state": json.loads(event[8]) if event[8] else None,
                        }

                        calculated_checksum = self._calculate_checksum(event_data)
                        stored_checksum = event[9]

                        if calculated_checksum == stored_checksum:
                            integrity_report["verified_events"] += 1
                        else:
                            integrity_report["checksum_failures"] += 1
                            integrity_report["errors"].append(
                                f"Checksum mismatch for event {event[0]}"
                            )

                    except Exception as e:
                        integrity_report["errors"].append(
                            f"Error verifying event {event[0]}: {str(e)}"
                        )

                # Verify integrity blocks
                cursor = conn.execute(
                    """
                    SELECT * FROM audit_integrity 
                    ORDER BY block_id DESC 
                    LIMIT 10
                """
                )

                blocks = cursor.fetchall()
                previous_hash = None

                for block in reversed(blocks):
                    block_data = {
                        "start_event_id": block[1],
                        "end_event_id": block[2],
                        "event_count": block[3],
                        "previous_block_hash": block[5],
                        "timestamp": block[6],
                    }

                    calculated_hash = hashlib.sha256(
                        json.dumps(block_data, sort_keys=True).encode()
                    ).hexdigest()

                    if calculated_hash != block[4]:  # block_hash
                        integrity_report["block_integrity"] = False
                        integrity_report["errors"].append(
                            f"Block hash mismatch for block {block[0]}"
                        )

                    if previous_hash and block[5] != previous_hash:
                        integrity_report["block_integrity"] = False
                        integrity_report["errors"].append(
                            f"Block chain break at block {block[0]}"
                        )

                    previous_hash = block[4]

            integrity_report["integrity_status"] = (
                "PASS" if len(integrity_report["errors"]) == 0 else "FAIL"
            )

            return integrity_report

        except Exception as e:
            logger.error(f"Error verifying integrity: {e}")
            return {
                "timestamp": time.time(),
                "integrity_status": "ERROR",
                "error": str(e),
            }

    def get_entity_history(
        self, entity_type: str, entity_id: str
    ) -> List[Dict[str, Any]]:
        """Get complete change history for an entity."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT h.*, e.actor, e.action, e.timestamp as event_timestamp
                    FROM entity_history h
                    JOIN audit_events e ON h.event_id = e.event_id
                    WHERE h.entity_type = ? AND h.entity_id = ?
                    ORDER BY h.change_timestamp
                """,
                    (entity_type, entity_id),
                )

                changes = cursor.fetchall()

            # Convert to list of dictionaries
            result = []
            for change in changes:
                change_dict = {
                    "history_id": change[0],
                    "entity_type": change[1],
                    "entity_id": change[2],
                    "change_timestamp": change[3],
                    "event_id": change[4],
                    "field_name": change[5],
                    "old_value": json.loads(change[6]) if change[6] else None,
                    "new_value": json.loads(change[7]) if change[7] else None,
                    "actor": change[8],
                    "action": change[9],
                    "event_timestamp": change[10],
                }
                result.append(change_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting entity history: {e}")
            return []

    def close_session(self):
        """Close current audit session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE audit_sessions 
                    SET end_time = ?, status = 'closed'
                    WHERE session_id = ?
                """,
                    (time.time(), self.current_session),
                )
                conn.commit()

            logger.info(f"Closed audit session: {self.current_session}")

        except Exception as e:
            logger.error(f"Error closing session: {e}")


def main():
    """Main entry point for transaction audit."""
    import argparse

    parser = argparse.ArgumentParser(description="Transaction Audit Trail")
    parser.add_argument("--trail", action="store_true", help="Show audit trail")
    parser.add_argument("--verify", action="store_true", help="Verify audit integrity")
    parser.add_argument("--entity-type", type=str, help="Filter by entity type")
    parser.add_argument("--entity-id", type=str, help="Filter by entity ID")
    parser.add_argument("--start-time", type=float, help="Start time (Unix timestamp)")
    parser.add_argument("--end-time", type=float, help="End time (Unix timestamp)")
    parser.add_argument(
        "--limit", type=int, default=100, help="Limit number of results"
    )

    args = parser.parse_args()

    # Create audit trail
    audit = TransactionAuditTrail()

    if args.trail:
        trail = audit.get_audit_trail(
            entity_type=args.entity_type,
            entity_id=args.entity_id,
            start_time=args.start_time,
            end_time=args.end_time,
            limit=args.limit,
        )
        print(json.dumps(trail, indent=2, default=str))
        return

    if args.verify:
        integrity = audit.verify_integrity()
        print(json.dumps(integrity, indent=2, default=str))
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
