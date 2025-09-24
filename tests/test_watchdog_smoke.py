#!/usr/bin/env python3
"""
Smoke tests for RL staleness watchdog
"""
import subprocess
import redis
import time
import os
import json
import tempfile
import pytest


class TestRLWatchdog:

    def setup_method(self):
        """Set up test Redis connection."""
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.r = redis.Redis.from_url(self.redis_url, decode_responses=True)

        # Clean up any existing test keys
        self.r.delete("policy:last_update_ts")

    def teardown_method(self):
        """Clean up after tests."""
        self.r.delete("policy:last_update_ts")

    def test_watchdog_healthy_heartbeat(self):
        """Test watchdog with recent heartbeat (should be OK)."""
        # Set current timestamp
        current_ts = time.time()
        self.r.set("policy:last_update_ts", current_ts)

        # Run watchdog in dry-run mode
        result = subprocess.run(
            [
                "python",
                "scripts/rl_staleness_watchdog.py",
                "--dry-run",
                "--threshold-sec=3600",  # 1 hour threshold
                "--redis-url",
                self.redis_url,
            ],
            capture_output=True,
            text=True,
            cwd=".",
        )

        # Should exit successfully
        assert result.returncode == 0, f"Watchdog failed: {result.stderr}"

        # Should report OK status
        assert "OK" in result.stdout, f"Expected OK status, got: {result.stdout}"
        assert "Heartbeat healthy" in result.stdout

    def test_watchdog_stale_heartbeat(self):
        """Test watchdog with very old heartbeat (should detect staleness)."""
        # Set very old timestamp (25 hours ago)
        old_ts = time.time() - (25 * 3600)
        self.r.set("policy:last_update_ts", old_ts)

        # Run watchdog in dry-run mode
        result = subprocess.run(
            [
                "python",
                "scripts/rl_staleness_watchdog.py",
                "--dry-run",
                "--threshold-sec=86400",  # 24 hour threshold
                "--redis-url",
                self.redis_url,
            ],
            capture_output=True,
            text=True,
            cwd=".",
        )

        # Should still exit 0 (watchdog doesn't fail CI)
        assert result.returncode == 0, f"Watchdog should not fail CI: {result.stderr}"

        # Should detect staleness
        assert "STALE" in result.stdout, f"Expected STALE status, got: {result.stdout}"
        assert "restart:stale" in result.stdout

    def test_watchdog_missing_heartbeat(self):
        """Test watchdog with missing heartbeat key."""
        # Ensure key doesn't exist
        self.r.delete("policy:last_update_ts")

        # Run watchdog in dry-run mode
        result = subprocess.run(
            [
                "python",
                "scripts/rl_staleness_watchdog.py",
                "--dry-run",
                "--redis-url",
                self.redis_url,
            ],
            capture_output=True,
            text=True,
            cwd=".",
        )

        # Should still exit 0
        assert result.returncode == 0, f"Watchdog should not fail CI: {result.stderr}"

        # Should detect missing heartbeat
        assert "restart:missing_heartbeat" in result.stdout
        assert "Missing heartbeat" in result.stdout

    def test_watchdog_json_output(self):
        """Test watchdog JSON output format."""
        # Set stale timestamp
        old_ts = time.time() - (25 * 3600)
        self.r.set("policy:last_update_ts", old_ts)

        # Create temp file for output
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_file = f.name

        try:
            # Run watchdog with JSON output
            result = subprocess.run(
                [
                    "python",
                    "scripts/rl_staleness_watchdog.py",
                    "--dry-run",
                    "--threshold-sec=86400",
                    "--out",
                    output_file,
                    "--redis-url",
                    self.redis_url,
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            # Should create output file
            assert os.path.exists(output_file), "Output file should be created"

            # Parse JSON output
            with open(output_file, "r") as f:
                result_data = json.load(f)

            # Validate JSON structure
            required_fields = [
                "timestamp",
                "now",
                "last_update_ts",
                "age_sec",
                "threshold_sec",
                "action",
                "service",
                "dry_run",
                "ok",
            ]
            for field in required_fields:
                assert field in result_data, f"Missing field: {field}"

            # Should indicate staleness
            assert not result_data["ok"], "Should detect staleness"
            assert result_data["action"] == "restart:stale"
            assert result_data["age_sec"] > 86400  # Older than 24h

        finally:
            # Clean up temp file
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_watchdog_redis_connection_failure(self):
        """Test watchdog behavior with Redis connection failure."""
        # Run watchdog with invalid Redis URL
        result = subprocess.run(
            [
                "python",
                "scripts/rl_staleness_watchdog.py",
                "--dry-run",
                "--redis-url",
                "redis://localhost:9999/0",  # Invalid port
            ],
            capture_output=True,
            text=True,
            cwd=".",
        )

        # Should fail with connection error
        assert result.returncode == 1, "Should fail with connection error"
        assert "Redis connection failed" in result.stderr


def test_watchdog_integration():
    """Integration test running watchdog as external script."""
    result = subprocess.run(
        ["python", "scripts/rl_staleness_watchdog.py", "--help"],
        capture_output=True,
        text=True,
        cwd=".",
    )

    assert result.returncode == 0, "Watchdog script should show help"
    assert "staleness watchdog" in result.stdout.lower()


if __name__ == "__main__":
    # Can run individual test methods
    test_watchdog = TestRLWatchdog()
    test_watchdog.setup_method()

    try:
        test_watchdog.test_watchdog_healthy_heartbeat()
        test_watchdog.test_watchdog_stale_heartbeat()
        test_watchdog.test_watchdog_missing_heartbeat()
        test_watchdog.test_watchdog_json_output()
        test_watchdog.test_watchdog_redis_connection_failure()
        test_watchdog_integration()
        print("✅ All watchdog tests passed!")
    finally:
        test_watchdog.teardown_method()
