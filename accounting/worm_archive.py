#!/usr/bin/env python3
"""
WORM (Write Once Read Many) Archive System
Implements immutable compliance archive with cryptographic integrity
"""

import os
import sys
import json
import time
import logging
import hashlib
import sqlite3
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import threading
import gzip
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.padding import PSS, MGF1
from cryptography.hazmat.backends import default_backend

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("worm_archive")


@dataclass
class ArchiveRecord:
    """Represents an immutable archive record."""

    record_id: str
    content_type: str
    content_hash: str
    compressed_size: int
    original_size: int
    timestamp: float
    retention_years: int
    signature: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "record_id": self.record_id,
            "content_type": self.content_type,
            "content_hash": self.content_hash,
            "compressed_size": self.compressed_size,
            "original_size": self.original_size,
            "timestamp": self.timestamp,
            "retention_years": self.retention_years,
            "signature": self.signature,
            "metadata": self.metadata,
        }


class WORMArchive:
    """Write Once Read Many archive system for compliance data."""

    def __init__(self, archive_path: str = None, key_path: str = None):
        """Initialize WORM archive system."""
        self.archive_path = Path(archive_path or "/tmp/worm_archive")
        self.key_path = Path(key_path or "/tmp/worm_keys")
        self.db_path = self.archive_path / "archive_index.db"
        self.content_path = self.archive_path / "content"

        # Create directories
        self.archive_path.mkdir(parents=True, exist_ok=True)
        self.key_path.mkdir(parents=True, exist_ok=True)
        self.content_path.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self.lock = threading.RLock()

        # Redis connection
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Initialize crypto keys and database
        self._init_crypto_keys()
        self._init_database()

        logger.info(f"📦 WORM Archive initialized: {self.archive_path}")

    def _init_crypto_keys(self):
        """Initialize or load cryptographic keys for signing."""
        try:
            private_key_path = self.key_path / "archive_private.pem"
            public_key_path = self.key_path / "archive_public.pem"

            if private_key_path.exists() and public_key_path.exists():
                # Load existing keys
                with open(private_key_path, "rb") as f:
                    self.private_key = serialization.load_pem_private_key(
                        f.read(), password=None, backend=default_backend()
                    )

                with open(public_key_path, "rb") as f:
                    self.public_key = serialization.load_pem_public_key(
                        f.read(), backend=default_backend()
                    )

                logger.info("Loaded existing crypto keys")
            else:
                # Generate new keys
                self.private_key = rsa.generate_private_key(
                    public_exponent=65537, key_size=2048, backend=default_backend()
                )
                self.public_key = self.private_key.public_key()

                # Save keys
                with open(private_key_path, "wb") as f:
                    f.write(
                        self.private_key.private_bytes(
                            encoding=serialization.Encoding.PEM,
                            format=serialization.PrivateFormat.PKCS8,
                            encryption_algorithm=serialization.NoEncryption(),
                        )
                    )

                with open(public_key_path, "wb") as f:
                    f.write(
                        self.public_key.public_bytes(
                            encoding=serialization.Encoding.PEM,
                            format=serialization.PublicFormat.SubjectPublicKeyInfo,
                        )
                    )

                # Set restrictive permissions
                os.chmod(private_key_path, 0o600)
                os.chmod(public_key_path, 0o644)

                logger.info("Generated new crypto keys")

        except Exception as e:
            logger.error(f"Error initializing crypto keys: {e}")
            raise

    def _init_database(self):
        """Initialize SQLite database for archive index."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS archive_records (
                        record_id TEXT PRIMARY KEY,
                        content_type TEXT NOT NULL,
                        content_hash TEXT NOT NULL UNIQUE,
                        compressed_size INTEGER NOT NULL,
                        original_size INTEGER NOT NULL,
                        timestamp REAL NOT NULL,
                        retention_years INTEGER NOT NULL,
                        signature TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS access_log (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        record_id TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        requester TEXT,
                        timestamp REAL NOT NULL,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        FOREIGN KEY (record_id) REFERENCES archive_records (record_id)
                    )
                """
                )

                # Create indexes
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_records_timestamp ON archive_records (timestamp)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_records_type ON archive_records (content_type)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_records_hash ON archive_records (content_hash)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_access_log_record ON access_log (record_id)"
                )

                conn.commit()

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _calculate_content_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()

    def _sign_content(self, content: bytes) -> str:
        """Create digital signature for content."""
        try:
            signature = self.private_key.sign(
                content,
                PSS(mgf=MGF1(hashes.SHA256()), salt_length=PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            return base64.b64encode(signature).decode("utf-8")

        except Exception as e:
            logger.error(f"Error signing content: {e}")
            raise

    def _verify_signature(self, content: bytes, signature: str) -> bool:
        """Verify digital signature of content."""
        try:
            signature_bytes = base64.b64decode(signature.encode("utf-8"))
            self.public_key.verify(
                signature_bytes,
                content,
                PSS(mgf=MGF1(hashes.SHA256()), salt_length=PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            return True

        except Exception as e:
            logger.debug(f"Signature verification failed: {e}")
            return False

    def _log_access(
        self,
        record_id: str,
        operation: str,
        requester: str = None,
        success: bool = True,
        error_message: str = None,
    ):
        """Log access to archive record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO access_log 
                    (record_id, operation, requester, timestamp, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        record_id,
                        operation,
                        requester,
                        time.time(),
                        success,
                        error_message,
                    ),
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Error logging access: {e}")

    def store_record(
        self,
        content: Union[str, bytes, Dict[str, Any]],
        content_type: str,
        retention_years: int = 7,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Store content in WORM archive."""
        try:
            with self.lock:
                # Convert content to bytes
                if isinstance(content, dict):
                    content_bytes = json.dumps(
                        content, sort_keys=True, default=str
                    ).encode("utf-8")
                elif isinstance(content, str):
                    content_bytes = content.encode("utf-8")
                else:
                    content_bytes = content

                # Calculate hash and check for duplicates
                content_hash = self._calculate_content_hash(content_bytes)

                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT record_id FROM archive_records WHERE content_hash = ?",
                        (content_hash,),
                    )
                    existing = cursor.fetchone()

                    if existing:
                        logger.info(f"Content already archived: {existing[0]}")
                        return existing[0]

                # Compress content
                compressed_content = gzip.compress(content_bytes)

                # Generate record ID
                record_id = f"worm_{int(time.time() * 1000)}_{content_hash[:16]}"

                # Create digital signature
                signature = self._sign_content(content_bytes)

                # Create archive record
                archive_record = ArchiveRecord(
                    record_id=record_id,
                    content_type=content_type,
                    content_hash=content_hash,
                    compressed_size=len(compressed_content),
                    original_size=len(content_bytes),
                    timestamp=time.time(),
                    retention_years=retention_years,
                    signature=signature,
                    metadata=metadata or {},
                )

                # Store compressed content to file
                file_path = self.content_path / f"{record_id}.gz"
                with open(file_path, "wb") as f:
                    f.write(compressed_content)

                # Make file immutable (read-only)
                os.chmod(file_path, 0o444)

                # Store record in database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO archive_records 
                        (record_id, content_type, content_hash, compressed_size, original_size,
                         timestamp, retention_years, signature, metadata, file_path, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            record_id,
                            content_type,
                            content_hash,
                            archive_record.compressed_size,
                            archive_record.original_size,
                            archive_record.timestamp,
                            retention_years,
                            signature,
                            json.dumps(metadata or {}),
                            str(file_path),
                            time.time(),
                        ),
                    )
                    conn.commit()

                # Log access
                self._log_access(record_id, "STORE", success=True)

                # Update Redis metrics
                self.redis.incr("worm:records_stored")
                self.redis.incr(f"worm:content_type:{content_type}")

                logger.info(f"Stored record {record_id} ({content_type})")
                return record_id

        except Exception as e:
            logger.error(f"Error storing record: {e}")
            if "record_id" in locals():
                self._log_access(
                    record_id, "STORE", success=False, error_message=str(e)
                )
            raise

    def retrieve_record(
        self, record_id: str, verify_signature: bool = True
    ) -> Dict[str, Any]:
        """Retrieve content from WORM archive."""
        try:
            # Get record metadata
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM archive_records WHERE record_id = ?", (record_id,)
                )
                record_row = cursor.fetchone()

                if not record_row:
                    raise ValueError(f"Record {record_id} not found")

            # Read compressed content
            file_path = Path(record_row[9])  # file_path column

            if not file_path.exists():
                raise FileNotFoundError(f"Archive file not found: {file_path}")

            with open(file_path, "rb") as f:
                compressed_content = f.read()

            # Decompress content
            content_bytes = gzip.decompress(compressed_content)

            # Verify integrity
            calculated_hash = self._calculate_content_hash(content_bytes)
            stored_hash = record_row[2]  # content_hash column

            if calculated_hash != stored_hash:
                error_msg = f"Content hash mismatch for {record_id}"
                self._log_access(
                    record_id, "RETRIEVE", success=False, error_message=error_msg
                )
                raise ValueError(error_msg)

            # Verify signature if requested
            if verify_signature:
                signature = record_row[7]  # signature column
                if not self._verify_signature(content_bytes, signature):
                    error_msg = f"Signature verification failed for {record_id}"
                    self._log_access(
                        record_id, "RETRIEVE", success=False, error_message=error_msg
                    )
                    raise ValueError(error_msg)

            # Parse content based on type
            content_type = record_row[1]
            if content_type.startswith("application/json"):
                content = json.loads(content_bytes.decode("utf-8"))
            elif content_type.startswith("text/"):
                content = content_bytes.decode("utf-8")
            else:
                content = content_bytes

            # Create result
            result = {
                "record_id": record_id,
                "content_type": content_type,
                "content": content,
                "metadata": json.loads(record_row[8]),  # metadata column
                "timestamp": record_row[5],  # timestamp column
                "retention_years": record_row[6],  # retention_years column
                "verified": True,
                "original_size": record_row[4],  # original_size column
                "compressed_size": record_row[3],  # compressed_size column
            }

            # Log successful access
            self._log_access(record_id, "RETRIEVE", success=True)

            logger.debug(f"Retrieved record {record_id}")
            return result

        except Exception as e:
            logger.error(f"Error retrieving record {record_id}: {e}")
            self._log_access(record_id, "RETRIEVE", success=False, error_message=str(e))
            raise

    def list_records(
        self,
        content_type: str = None,
        start_date: float = None,
        end_date: float = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List archived records with optional filtering."""
        try:
            query = "SELECT * FROM archive_records WHERE 1=1"
            params = []

            if content_type:
                query += " AND content_type = ?"
                params.append(content_type)

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                records = cursor.fetchall()

            # Convert to list of dictionaries
            result = []
            for record in records:
                result.append(
                    {
                        "record_id": record[0],
                        "content_type": record[1],
                        "content_hash": record[2],
                        "compressed_size": record[3],
                        "original_size": record[4],
                        "timestamp": record[5],
                        "retention_years": record[6],
                        "metadata": json.loads(record[8]),
                        "created_at": record[10],
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error listing records: {e}")
            return []

    def verify_archive_integrity(self) -> Dict[str, Any]:
        """Verify integrity of entire archive."""
        try:
            integrity_report = {
                "timestamp": time.time(),
                "total_records": 0,
                "verified_records": 0,
                "failed_records": 0,
                "missing_files": 0,
                "signature_failures": 0,
                "hash_failures": 0,
                "errors": [],
            }

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM archive_records")
                records = cursor.fetchall()

            integrity_report["total_records"] = len(records)

            for record in records:
                record_id = record[0]
                content_hash = record[2]
                signature = record[7]
                file_path = Path(record[9])

                try:
                    # Check if file exists
                    if not file_path.exists():
                        integrity_report["missing_files"] += 1
                        integrity_report["errors"].append(
                            f"Missing file for {record_id}"
                        )
                        continue

                    # Read and decompress content
                    with open(file_path, "rb") as f:
                        compressed_content = f.read()

                    content_bytes = gzip.decompress(compressed_content)

                    # Verify hash
                    calculated_hash = self._calculate_content_hash(content_bytes)
                    if calculated_hash != content_hash:
                        integrity_report["hash_failures"] += 1
                        integrity_report["errors"].append(
                            f"Hash mismatch for {record_id}"
                        )
                        continue

                    # Verify signature
                    if not self._verify_signature(content_bytes, signature):
                        integrity_report["signature_failures"] += 1
                        integrity_report["errors"].append(
                            f"Signature failure for {record_id}"
                        )
                        continue

                    integrity_report["verified_records"] += 1

                except Exception as e:
                    integrity_report["failed_records"] += 1
                    integrity_report["errors"].append(
                        f"Error verifying {record_id}: {str(e)}"
                    )

            integrity_report["integrity_status"] = (
                "PASS"
                if integrity_report["failed_records"] == 0
                and integrity_report["missing_files"] == 0
                and integrity_report["hash_failures"] == 0
                and integrity_report["signature_failures"] == 0
                else "FAIL"
            )

            return integrity_report

        except Exception as e:
            logger.error(f"Error verifying archive integrity: {e}")
            return {
                "timestamp": time.time(),
                "integrity_status": "ERROR",
                "error": str(e),
            }

    def get_retention_report(self) -> Dict[str, Any]:
        """Generate retention policy compliance report."""
        try:
            current_time = time.time()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT retention_years, COUNT(*) as count, 
                           MIN(timestamp) as oldest, MAX(timestamp) as newest
                    FROM archive_records 
                    GROUP BY retention_years
                    ORDER BY retention_years
                """
                )
                retention_groups = cursor.fetchall()

                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM archive_records 
                    WHERE timestamp + (retention_years * 365 * 24 * 3600) < ?
                """,
                    (current_time,),
                )
                expired_count = cursor.fetchone()[0]

            report = {
                "timestamp": current_time,
                "retention_groups": [],
                "expired_records": expired_count,
                "total_records": sum(group[1] for group in retention_groups),
            }

            for group in retention_groups:
                retention_years = group[0]
                count = group[1]
                oldest = group[2]
                newest = group[3]

                # Calculate expiration date for oldest record in group
                oldest_expiry = oldest + (retention_years * 365 * 24 * 3600)
                days_to_expiry = (oldest_expiry - current_time) / (24 * 3600)

                report["retention_groups"].append(
                    {
                        "retention_years": retention_years,
                        "record_count": count,
                        "oldest_record": oldest,
                        "newest_record": newest,
                        "days_to_first_expiry": max(0, int(days_to_expiry)),
                    }
                )

            return report

        except Exception as e:
            logger.error(f"Error generating retention report: {e}")
            return {"error": str(e), "timestamp": time.time()}

    def get_status_report(self) -> Dict[str, Any]:
        """Get WORM archive status report."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM archive_records")
                total_records = cursor.fetchone()[0]

                cursor = conn.execute("SELECT SUM(original_size) FROM archive_records")
                total_size = cursor.fetchone()[0] or 0

                cursor = conn.execute(
                    "SELECT SUM(compressed_size) FROM archive_records"
                )
                compressed_size = cursor.fetchone()[0] or 0

                cursor = conn.execute(
                    """
                    SELECT content_type, COUNT(*) 
                    FROM archive_records 
                    GROUP BY content_type
                """
                )
                content_types = dict(cursor.fetchall())

                cursor = conn.execute(
                    "SELECT COUNT(*) FROM access_log WHERE success = 1"
                )
                successful_accesses = cursor.fetchone()[0]

                cursor = conn.execute(
                    "SELECT COUNT(*) FROM access_log WHERE success = 0"
                )
                failed_accesses = cursor.fetchone()[0]

            compression_ratio = (
                (total_size - compressed_size) / total_size * 100
                if total_size > 0
                else 0
            )

            status = {
                "service": "worm_archive",
                "timestamp": time.time(),
                "archive_path": str(self.archive_path),
                "statistics": {
                    "total_records": total_records,
                    "total_size_bytes": total_size,
                    "compressed_size_bytes": compressed_size,
                    "compression_ratio_percent": round(compression_ratio, 2),
                    "successful_accesses": successful_accesses,
                    "failed_accesses": failed_accesses,
                },
                "content_types": content_types,
                "status": "healthy",
            }

            return status

        except Exception as e:
            logger.error(f"Error getting status report: {e}")
            return {
                "service": "worm_archive",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }


def main():
    """Main entry point for WORM archive."""
    import argparse

    parser = argparse.ArgumentParser(description="WORM Archive System")
    parser.add_argument("--store", type=str, help="Store content from file")
    parser.add_argument("--retrieve", type=str, help="Retrieve record by ID")
    parser.add_argument("--list", action="store_true", help="List archived records")
    parser.add_argument(
        "--verify", action="store_true", help="Verify archive integrity"
    )
    parser.add_argument(
        "--retention-report", action="store_true", help="Generate retention report"
    )
    parser.add_argument("--status", action="store_true", help="Show status report")
    parser.add_argument(
        "--content-type",
        type=str,
        default="application/octet-stream",
        help="Content type for storage",
    )
    parser.add_argument(
        "--retention-years", type=int, default=7, help="Retention period in years"
    )

    args = parser.parse_args()

    # Create WORM archive
    archive = WORMArchive()

    if args.store:
        with open(args.store, "rb") as f:
            content = f.read()

        record_id = archive.store_record(
            content,
            args.content_type,
            args.retention_years,
            metadata={"source_file": args.store},
        )
        print(f"Stored as record: {record_id}")
        return

    if args.retrieve:
        record = archive.retrieve_record(args.retrieve)
        print(json.dumps(record, indent=2, default=str))
        return

    if args.list:
        records = archive.list_records()
        print(json.dumps(records, indent=2, default=str))
        return

    if args.verify:
        integrity = archive.verify_archive_integrity()
        print(json.dumps(integrity, indent=2, default=str))
        return

    if args.retention_report:
        report = archive.get_retention_report()
        print(json.dumps(report, indent=2, default=str))
        return

    if args.status:
        status = archive.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
