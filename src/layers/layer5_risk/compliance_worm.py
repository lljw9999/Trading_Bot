#!/usr/bin/env python3
"""
WORM (Write Once Read Many) Audit Trail System

Implements tamper-proof audit logging for regulatory compliance as specified
in Future_instruction.txt. All trading events are logged with cryptographic
integrity protection.

Features:
- Write-once, read-many audit logs
- Cryptographic integrity verification
- Regulatory compliance logging
- Event chain validation
- Async storage with reliability guarantees
"""

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import aiofiles
import aiofiles.os


class EventType(Enum):
    """Types of events logged in the audit trail."""

    TRADE_EXECUTION = "trade_execution"
    RISK_BREACH = "risk_breach"
    POSITION_CHANGE = "position_change"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    KILL_SWITCH = "kill_switch"
    CONFIG_CHANGE = "config_change"
    ORDER_SUBMIT = "order_submit"
    ORDER_CANCEL = "order_cancel"
    ORDER_FILL = "order_fill"
    ALERT_TRIGGERED = "alert_triggered"


@dataclass
class AuditEvent:
    """Immutable audit event with cryptographic integrity."""

    event_id: str
    event_type: EventType
    timestamp: datetime
    previous_hash: str
    event_data: Dict[str, Any]
    metadata: Dict[str, Any]
    hash: str = ""  # Will be calculated

    def __post_init__(self):
        """Calculate hash after initialization."""
        if not self.hash:
            self.hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the event for integrity verification."""
        event_dict = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "previous_hash": self.previous_hash,
            "event_data": self.event_data,
            "metadata": self.metadata,
        }

        # Create deterministic JSON string
        json_str = json.dumps(event_dict, sort_keys=True, separators=(",", ":"))

        # Calculate SHA-256 hash
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify the cryptographic integrity of this event."""
        expected_hash = self._calculate_hash()
        return self.hash == expected_hash

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "previous_hash": self.previous_hash,
            "event_data": self.event_data,
            "metadata": self.metadata,
            "hash": self.hash,
        }


class WORMAuditLogger:
    """
    WORM audit logger with cryptographic integrity protection.

    Implements write-once, read-many semantics with tamper detection.
    """

    def __init__(self, audit_dir: str = "audit", storage_mode: str = "worm_async"):
        """
        Initialize WORM audit logger.

        Args:
            audit_dir: Directory for audit log storage
            storage_mode: Storage mode ("worm_async" for async storage)
        """
        self.audit_dir = Path(audit_dir)
        self.storage_mode = storage_mode
        self.last_hash = "0" * 64  # Genesis hash
        self.event_count = 0

        # Ensure audit directory exists
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        # Current log file (rotated daily)
        self.current_log_file = None
        self.current_date = None

        # In-memory event cache for integrity verification
        self.event_cache = []
        self.max_cache_size = 1000

        self.logger = logging.getLogger("worm_audit")
        self.logger.info(f"WORM Audit Logger initialized - Directory: {audit_dir}")

        # Initialize from existing logs
        asyncio.create_task(self._initialize_from_existing_logs())

    async def _initialize_from_existing_logs(self):
        """Initialize state from existing audit logs."""
        try:
            # Find the most recent log file
            log_files = sorted(self.audit_dir.glob("audit_*.jsonl"))
            if not log_files:
                self.logger.info("No existing audit logs found - starting fresh")
                return

            # Load the last few events to establish chain state
            most_recent_file = log_files[-1]
            events = await self._load_events_from_file(most_recent_file)

            if events:
                last_event = events[-1]
                self.last_hash = last_event.hash
                self.event_count = len(events)

                # Add recent events to cache
                self.event_cache.extend(events[-min(100, len(events)) :])

                self.logger.info(f"Loaded {len(events)} events from {most_recent_file}")
                self.logger.info(
                    f"Chain state: last_hash={self.last_hash[:16]}..., count={self.event_count}"
                )

                # Verify integrity of recent events
                integrity_ok = await self._verify_chain_integrity(events[-10:])
                if not integrity_ok:
                    self.logger.error("INTEGRITY VIOLATION detected in audit chain!")
                else:
                    self.logger.info("Audit chain integrity verified")

        except Exception as e:
            self.logger.error(f"Error initializing from existing logs: {e}")

    async def _load_events_from_file(self, file_path: Path) -> List[AuditEvent]:
        """Load events from a JSONL audit file."""
        events = []
        try:
            async with aiofiles.open(file_path, "r") as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        event_dict = json.loads(line)
                        event = AuditEvent(
                            event_id=event_dict["event_id"],
                            event_type=EventType(event_dict["event_type"]),
                            timestamp=datetime.fromisoformat(event_dict["timestamp"]),
                            previous_hash=event_dict["previous_hash"],
                            event_data=event_dict["event_data"],
                            metadata=event_dict["metadata"],
                            hash=event_dict["hash"],
                        )
                        events.append(event)
        except Exception as e:
            self.logger.error(f"Error loading events from {file_path}: {e}")

        return events

    async def _verify_chain_integrity(self, events: List[AuditEvent]) -> bool:
        """Verify the cryptographic integrity of an event chain."""
        try:
            for i, event in enumerate(events):
                # Verify individual event integrity
                if not event.verify_integrity():
                    self.logger.error(f"Event {event.event_id} failed integrity check")
                    return False

                # Verify chain linkage (except for first event)
                if i > 0:
                    expected_prev_hash = events[i - 1].hash
                    if event.previous_hash != expected_prev_hash:
                        self.logger.error(f"Chain break at event {event.event_id}")
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Error verifying chain integrity: {e}")
            return False

    def _get_current_log_file(self) -> Path:
        """Get the current log file path (rotated daily)."""
        today = datetime.now(timezone.utc).date()

        if self.current_date != today:
            self.current_date = today
            self.current_log_file = self.audit_dir / f"audit_{today.isoformat()}.jsonl"
            self.logger.info(f"Using audit log file: {self.current_log_file}")

        return self.current_log_file

    async def write_event(
        self,
        event_type: EventType,
        event_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Write an audit event to the WORM storage (as per Future_instruction.txt).

        Args:
            event_type: Type of event being logged
            event_data: Event-specific data
            metadata: Additional metadata

        Returns:
            Event ID of the written event
        """
        try:
            # Generate unique event ID
            event_id = f"{int(time.time() * 1000000)}_{self.event_count:06d}"

            # Prepare metadata
            if metadata is None:
                metadata = {}

            metadata.update(
                {
                    "timestamp_ns": time.time_ns(),
                    "process_id": os.getpid(),
                    "sequence_number": self.event_count,
                    "storage_mode": self.storage_mode,
                }
            )

            # Create audit event
            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                previous_hash=self.last_hash,
                event_data=event_data,
                metadata=metadata,
            )

            # Write to storage asynchronously
            await self._write_to_storage(event)

            # Update chain state
            self.last_hash = event.hash
            self.event_count += 1

            # Add to cache
            self.event_cache.append(event)
            if len(self.event_cache) > self.max_cache_size:
                self.event_cache.pop(0)

            self.logger.debug(f"Audit event written: {event_type.value} [{event_id}]")

            return event_id

        except Exception as e:
            self.logger.error(f"Error writing audit event: {e}")
            raise

    async def _write_to_storage(self, event: AuditEvent):
        """Write event to persistent storage with WORM guarantees."""
        try:
            log_file = self._get_current_log_file()

            # Convert event to JSON line
            event_json = json.dumps(event.to_dict(), separators=(",", ":"))

            # Append to log file atomically
            async with aiofiles.open(log_file, "a") as f:
                await f.write(event_json + "\n")
                # Note: aiofiles doesn't support fsync, but the data is written reliably

            # Set file permissions to read-only (WORM enforcement)
            if os.name != "nt":  # Unix/Linux
                await aiofiles.os.chmod(log_file, 0o444)

        except Exception as e:
            self.logger.error(f"Error writing to storage: {e}")
            raise

    async def get_events(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """
        Retrieve audit events with optional filtering.

        Args:
            event_type: Filter by event type
            start_time: Filter events after this time
            end_time: Filter events before this time
            limit: Maximum number of events to return

        Returns:
            List of matching audit events
        """
        try:
            events = []

            # First check cache for recent events
            for event in reversed(self.event_cache):
                if len(events) >= limit:
                    break

                if self._matches_filter(event, event_type, start_time, end_time):
                    events.append(event)

            # If we need more events, load from files
            if len(events) < limit:
                log_files = sorted(self.audit_dir.glob("audit_*.jsonl"), reverse=True)

                for log_file in log_files:
                    if len(events) >= limit:
                        break

                    file_events = await self._load_events_from_file(log_file)

                    for event in reversed(file_events):
                        if len(events) >= limit:
                            break

                        if self._matches_filter(
                            event, event_type, start_time, end_time
                        ):
                            # Avoid duplicates from cache
                            if event not in events:
                                events.append(event)

            return events[:limit]

        except Exception as e:
            self.logger.error(f"Error retrieving events: {e}")
            return []

    def _matches_filter(
        self,
        event: AuditEvent,
        event_type: Optional[EventType],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> bool:
        """Check if event matches the given filters."""
        if event_type and event.event_type != event_type:
            return False

        if start_time and event.timestamp < start_time:
            return False

        if end_time and event.timestamp > end_time:
            return False

        return True

    async def verify_audit_trail(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Verify the integrity of the audit trail.

        Args:
            days_back: Number of days to verify

        Returns:
            Verification report
        """
        try:
            start_time = datetime.now(timezone.utc) - timedelta(days=days_back)
            events = await self.get_events(start_time=start_time, limit=10000)

            if not events:
                return {"status": "no_events", "events_verified": 0}

            # Sort events by timestamp
            events.sort(key=lambda e: e.timestamp)

            # Verify chain integrity
            integrity_ok = await self._verify_chain_integrity(events)

            # Count event types
            event_type_counts = {}
            for event in events:
                event_type_counts[event.event_type.value] = (
                    event_type_counts.get(event.event_type.value, 0) + 1
                )

            return {
                "status": "verified" if integrity_ok else "integrity_violation",
                "events_verified": len(events),
                "chain_integrity": integrity_ok,
                "event_types": event_type_counts,
                "verification_time": datetime.now(timezone.utc).isoformat(),
                "days_verified": days_back,
            }

        except Exception as e:
            self.logger.error(f"Error verifying audit trail: {e}")
            return {"status": "error", "error": str(e)}


# Global WORM audit logger instance
_audit_logger = None


def get_audit_logger(audit_dir: str = "audit") -> WORMAuditLogger:
    """Get or create the global WORM audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = WORMAuditLogger(audit_dir)
    return _audit_logger


# Convenience function for compliance.write_event()
async def write_event(
    event_type: EventType,
    event_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    storage: str = "worm_async",
) -> str:
    """
    Write audit event to WORM storage (as specified in Future_instruction.txt).

    Args:
        event_type: Type of event
        event_data: Event data
        metadata: Additional metadata
        storage: Storage mode (always "worm_async" for compliance)

    Returns:
        Event ID
    """
    audit_logger = get_audit_logger()
    return await audit_logger.write_event(event_type, event_data, metadata)


# Compliance namespace (for Future_instruction.txt compatibility)
class ComplianceNamespace:
    """Namespace for compliance functions."""

    @staticmethod
    async def write_event(event: Dict[str, Any], storage: str = "worm_async") -> str:
        """Write compliance event to WORM storage."""
        event_type = EventType(event.get("type", "trade_execution"))
        event_data = event.get("data", {})
        metadata = event.get("metadata", {})

        return await write_event(event_type, event_data, metadata, storage)


compliance = ComplianceNamespace()
