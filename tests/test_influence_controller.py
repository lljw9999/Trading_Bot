#!/usr/bin/env python3
"""
Unit tests for InfluenceController
Tests safety features, TTL behavior, and error handling
"""
import pytest
import redis
import time
import os
from unittest.mock import patch, MagicMock
from src.rl.influence_controller import (
    InfluenceController,
    get_current_influence,
    set_influence,
)


class TestInfluenceController:

    def setup_method(self):
        """Set up test Redis connection."""
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.r = redis.Redis.from_url(self.redis_url, decode_responses=True)

        # Clean up any existing test keys
        self.r.delete("policy:allowed_influence_pct")

        # Initialize controller with short TTL for testing
        self.ic = InfluenceController(ttl_sec=5)

    def teardown_method(self):
        """Clean up after tests."""
        self.r.delete("policy:allowed_influence_pct")

    def test_default_weight_when_key_missing(self):
        """Test that weight defaults to 0.0 when Redis key is missing."""
        # Ensure key doesn't exist
        self.r.delete("policy:allowed_influence_pct")

        weight = self.ic.get_weight()
        assert weight == 0.0, "Should default to 0% when key missing"

    def test_valid_weight_retrieval(self):
        """Test retrieving valid weight values."""
        # Test various valid percentages
        test_cases = [0, 10, 50, 100]

        for pct in test_cases:
            self.r.set("policy:allowed_influence_pct", pct)
            weight = self.ic.get_weight()
            expected_weight = pct / 100.0
            assert (
                weight == expected_weight
            ), f"Weight should be {expected_weight} for {pct}%"

    def test_invalid_weight_handling(self):
        """Test handling of invalid weight values."""
        invalid_values = ["invalid", "NaN", "", "12.5.7", "null"]

        for invalid_val in invalid_values:
            self.r.set("policy:allowed_influence_pct", invalid_val)
            weight = self.ic.get_weight()
            assert (
                weight == 0.0
            ), f"Should default to 0% for invalid value: {invalid_val}"

    def test_weight_clamping_on_set(self):
        """Test that set_weight clamps values to valid range."""
        test_cases = [
            (-10, 0),  # Negative clamped to 0
            (150, 100),  # Over 100 clamped to 100
            (50, 50),  # Valid value unchanged
            (0, 0),  # Zero unchanged
            (100, 100),  # Max unchanged
        ]

        for input_pct, expected_pct in test_cases:
            actual_pct = self.ic.set_weight(input_pct)
            assert (
                actual_pct == expected_pct
            ), f"Input {input_pct} should clamp to {expected_pct}"

            # Verify in Redis
            stored_val = self.r.get("policy:allowed_influence_pct")
            assert int(stored_val) == expected_pct

    def test_ttl_behavior(self):
        """Test TTL expiration behavior."""
        # Set weight with short TTL
        self.ic.set_weight(50)

        # Check initial weight
        assert self.ic.get_weight() == 0.5

        # Check TTL is set
        ttl = self.r.ttl("policy:allowed_influence_pct")
        assert 0 < ttl <= 5, f"TTL should be between 0-5 seconds, got {ttl}"

        # Wait for expiration (plus buffer)
        time.sleep(6)

        # Should default to 0 after expiration
        weight = self.ic.get_weight()
        assert weight == 0.0, "Weight should default to 0% after TTL expiration"

    def test_no_ttl_mode(self):
        """Test operation without TTL."""
        ic_no_ttl = InfluenceController(ttl_sec=0)

        ic_no_ttl.set_weight(30)

        # Check no TTL is set
        ttl = self.r.ttl("policy:allowed_influence_pct")
        assert ttl == -1, "No TTL should be set when ttl_sec=0"

        # Value should persist
        assert ic_no_ttl.get_weight() == 0.3

    def test_redis_connection_error_handling(self):
        """Test behavior when Redis is unavailable."""
        # Mock Redis to raise connection error
        with patch.object(self.ic, "r") as mock_redis:
            mock_redis.get.side_effect = redis.ConnectionError("Connection failed")

            # Should default to 0% on connection error
            weight = self.ic.get_weight()
            assert weight == 0.0, "Should default to 0% on Redis connection error"

    def test_emergency_stop(self):
        """Test emergency stop functionality."""
        # Set non-zero influence
        self.ic.set_weight(75)
        assert self.ic.get_weight() == 0.75

        # Execute emergency stop
        success = self.ic.emergency_stop()
        assert success, "Emergency stop should succeed"

        # Should be back to 0%
        assert self.ic.get_weight() == 0.0, "Emergency stop should set weight to 0%"

        # Check Redis value directly
        stored_val = self.r.get("policy:allowed_influence_pct")
        assert stored_val == "0", "Redis should contain '0' after emergency stop"

    def test_status_information(self):
        """Test status information retrieval."""
        # Set known state
        self.ic.set_weight(25)

        status = self.ic.get_status()

        # Check required fields
        required_fields = [
            "weight",
            "percentage",
            "key_exists",
            "ttl_seconds",
            "redis_key",
        ]
        for field in required_fields:
            assert field in status, f"Status should include {field}"

        # Check values
        assert status["weight"] == 0.25
        assert status["percentage"] == 25
        assert status["key_exists"] is True
        assert status["redis_key"] == "policy:allowed_influence_pct"

    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        # Clean state
        self.r.delete("policy:allowed_influence_pct")

        # Test get_current_influence
        assert get_current_influence() == 0.0

        # Test set_influence
        result = set_influence(40, "test_reason")
        assert result == 40

        # Verify via get function
        assert get_current_influence() == 0.4

    def test_cli_interface(self):
        """Test CLI interface behavior."""
        import subprocess

        # Test status display (no args)
        result = subprocess.run(
            ["python", "src/rl/influence_controller.py"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0
        assert "Current influence:" in result.stdout

        # Test setting percentage
        result = subprocess.run(
            ["python", "src/rl/influence_controller.py", "25", "test"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0
        assert "Set influence to 25%" in result.stdout

        # Test emergency stop
        result = subprocess.run(
            ["python", "src/rl/influence_controller.py", "stop"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0


def test_module_imports():
    """Test that all required modules can be imported."""
    from src.rl.influence_controller import (
        InfluenceController,
        get_current_influence,
        set_influence,
        emergency_stop,
    )

    # Test basic instantiation
    ic = InfluenceController()
    assert ic is not None


def test_safety_defaults():
    """Test that all safety defaults are properly configured."""
    from src.rl.influence_controller import FALLBACK, KEY

    # Safety constants
    assert FALLBACK == 0, "Fallback should be 0% for safety"
    assert KEY == "policy:allowed_influence_pct", "Key should match expected Redis key"


if __name__ == "__main__":
    # Run individual test methods
    test_influence = TestInfluenceController()
    test_influence.setup_method()

    try:
        test_influence.test_default_weight_when_key_missing()
        test_influence.test_valid_weight_retrieval()
        test_influence.test_invalid_weight_handling()
        test_influence.test_weight_clamping_on_set()
        test_influence.test_emergency_stop()
        test_influence.test_status_information()
        test_influence.test_convenience_functions()
        test_module_imports()
        test_safety_defaults()
        print("✅ All influence controller tests passed!")
    finally:
        test_influence.teardown_method()
