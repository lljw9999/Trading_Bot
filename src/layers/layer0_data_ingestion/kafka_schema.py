#!/usr/bin/env python3
"""
Kafka Schema Registry for Market Data

Provides Avro schema validation and serialization for market data messages
following the Future_instruction.txt L2 depth schema specification.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

try:
    import avro.schema
    import avro.io
    import io

    HAVE_AVRO = True
except ImportError:
    HAVE_AVRO = False
    avro = None

logger = logging.getLogger(__name__)


class KafkaSchemaRegistry:
    """Schema registry for validating and serializing Kafka messages."""

    def __init__(self, schema_dir: str = "schemas/kafka"):
        """
        Initialize schema registry.

        Args:
            schema_dir: Directory containing Avro schema files
        """
        self.schema_dir = Path(schema_dir)
        self.schemas = {}

        if not HAVE_AVRO:
            logger.warning("Avro not available - schema validation disabled")
            return

        self._load_schemas()

    def _load_schemas(self):
        """Load all Avro schemas from the schema directory."""
        if not HAVE_AVRO:
            return

        try:
            if not self.schema_dir.exists():
                logger.warning(f"Schema directory {self.schema_dir} does not exist")
                return

            # Load depth snapshot schema
            depth_schema_path = self.schema_dir / "depth_snapshot.avsc"
            if depth_schema_path.exists():
                with open(depth_schema_path, "r") as f:
                    schema_dict = json.load(f)
                    self.schemas["depth_snapshot"] = avro.schema.parse(
                        json.dumps(schema_dict)
                    )
                logger.info("Loaded depth snapshot schema")

        except Exception as e:
            logger.error(f"Error loading schemas: {e}")

    def validate_depth_snapshot(self, data: Dict[str, Any]) -> bool:
        """
        Validate depth snapshot data against Avro schema.

        Args:
            data: Depth snapshot data to validate

        Returns:
            True if valid, False otherwise
        """
        if not HAVE_AVRO or "depth_snapshot" not in self.schemas:
            return True  # Skip validation if Avro not available

        try:
            schema = self.schemas["depth_snapshot"]

            # Convert timestamp to long if needed
            if "ts" in data and isinstance(data["ts"], float):
                data["ts"] = int(data["ts"] * 1000)  # Convert to milliseconds

            # Validate data structure
            writer = avro.io.DatumWriter(schema)
            bytes_writer = io.BytesIO()
            encoder = avro.io.BinaryEncoder(bytes_writer)
            writer.write(data, encoder)

            return True

        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False

    def serialize_depth_snapshot(self, data: Dict[str, Any]) -> Optional[bytes]:
        """
        Serialize depth snapshot data using Avro.

        Args:
            data: Depth snapshot data

        Returns:
            Serialized bytes or None if serialization fails
        """
        if not HAVE_AVRO or "depth_snapshot" not in self.schemas:
            # Fallback to JSON serialization
            return json.dumps(data).encode("utf-8")

        try:
            schema = self.schemas["depth_snapshot"]

            # Prepare data
            if "ts" in data and isinstance(data["ts"], float):
                data["ts"] = int(data["ts"] * 1000)

            # Add defaults
            data.setdefault("exchange", "coinbase")
            data.setdefault("sequence", None)

            writer = avro.io.DatumWriter(schema)
            bytes_writer = io.BytesIO()
            encoder = avro.io.BinaryEncoder(bytes_writer)
            writer.write(data, encoder)

            return bytes_writer.getvalue()

        except Exception as e:
            logger.error(f"Avro serialization failed: {e}")
            # Fallback to JSON
            return json.dumps(data).encode("utf-8")

    def deserialize_depth_snapshot(self, data: bytes) -> Optional[Dict[str, Any]]:
        """
        Deserialize depth snapshot data from Avro.

        Args:
            data: Serialized bytes

        Returns:
            Deserialized data or None if deserialization fails
        """
        if not HAVE_AVRO or "depth_snapshot" not in self.schemas:
            # Fallback to JSON deserialization
            try:
                return json.loads(data.decode("utf-8"))
            except Exception as e:
                logger.error(f"JSON deserialization failed: {e}")
                return None

        try:
            schema = self.schemas["depth_snapshot"]

            reader = avro.io.DatumReader(schema)
            bytes_reader = io.BytesIO(data)
            decoder = avro.io.BinaryDecoder(bytes_reader)
            result = reader.read(decoder)

            return result

        except Exception as e:
            logger.error(f"Avro deserialization failed: {e}")
            # Fallback to JSON
            try:
                return json.loads(data.decode("utf-8"))
            except Exception as json_e:
                logger.error(f"JSON fallback also failed: {json_e}")
                return None

    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about loaded schemas."""
        return {
            "avro_available": HAVE_AVRO,
            "schema_dir": str(self.schema_dir),
            "loaded_schemas": list(self.schemas.keys()) if HAVE_AVRO else [],
            "total_schemas": len(self.schemas) if HAVE_AVRO else 0,
        }


# Global schema registry instance
_schema_registry = None


def get_schema_registry(schema_dir: str = "schemas/kafka") -> KafkaSchemaRegistry:
    """Get or create the global schema registry instance."""
    global _schema_registry
    if _schema_registry is None:
        _schema_registry = KafkaSchemaRegistry(schema_dir)
    return _schema_registry


# Convenience functions
def validate_depth_snapshot(data: Dict[str, Any]) -> bool:
    """Validate depth snapshot data."""
    return get_schema_registry().validate_depth_snapshot(data)


def serialize_depth_snapshot(data: Dict[str, Any]) -> Optional[bytes]:
    """Serialize depth snapshot data."""
    return get_schema_registry().serialize_depth_snapshot(data)


def deserialize_depth_snapshot(data: bytes) -> Optional[Dict[str, Any]]:
    """Deserialize depth snapshot data."""
    return get_schema_registry().deserialize_depth_snapshot(data)
