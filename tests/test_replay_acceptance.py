#!/usr/bin/env python3
"""
PyTest tests for replay acceptance testing
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.replay_acceptance import ReplayAcceptanceTest


class TestReplayAcceptance:
    """Test cases for replay acceptance testing."""

    def setup_method(self):
        """Set up test environment."""
        self.tester = ReplayAcceptanceTest()

    def test_synthetic_data_generation(self):
        """Test synthetic market data generation."""
        df = self.tester._generate_synthetic_market_data()

        # Check data structure
        assert len(df) == 24 * 3600  # 24 hours of 1-second data
        assert "timestamp" in df.columns
        assert "price" in df.columns
        assert "volume" in df.columns
        assert "symbol" in df.columns

        # Check data validity
        assert df["price"].min() > 0
        assert df["volume"].min() > 0
        assert all(df["symbol"] == "BTCUSDT")

        # Check timestamp ordering
        assert df["timestamp"].is_monotonic_increasing

    def test_tolerance_validation(self):
        """Test tolerance validation logic."""
        # Mock baseline and actual metrics
        baseline = {
            "pnl_final": 1000.0,
            "slippage_avg": 10.0,
            "trades_count": 100,
            "hit_rate": 0.55,
        }

        # Test passing case
        actual_pass = {
            "pnl_final": 1001.0,  # 0.1% diff - should pass
            "slippage_avg": 12.0,  # 2bps diff - should pass
            "trades_count": 105,  # 5% diff - should pass
            "hit_rate": 0.56,
        }

        result_pass = self.tester.compare_results("test-day", actual_pass, baseline)
        assert result_pass["pass"] is True
        assert result_pass["summary"]["pass"] is True

        # Test failing case
        actual_fail = {
            "pnl_final": 1020.0,  # 2% diff - should fail (>0.15%)
            "slippage_avg": 20.0,  # 10bps diff - should fail (>5bps)
            "trades_count": 150,  # 50% diff - should fail (>10%)
            "hit_rate": 0.60,
        }

        result_fail = self.tester.compare_results("test-day", actual_fail, baseline)
        assert result_fail["pass"] is False
        assert result_fail["summary"]["pass"] is False

    def test_baseline_creation(self):
        """Test baseline creation and retrieval."""
        test_day = "test-2025-07-30"
        test_metrics = {
            "pnl_final": 500.0,
            "slippage_avg": 8.5,
            "trades_count": 50,
            "hit_rate": 0.52,
        }

        # Clean up any existing baseline
        for metric, key_template in self.tester.baseline_keys.items():
            key = key_template.format(test_day)
            self.tester.redis.delete(key)

        # Create baseline
        baseline = self.tester.get_or_create_baseline(test_day, test_metrics)

        # Verify baseline matches test metrics
        assert baseline["pnl_final"] == test_metrics["pnl_final"]
        assert baseline["slippage_avg"] == test_metrics["slippage_avg"]
        assert baseline["trades_count"] == test_metrics["trades_count"]
        assert baseline["hit_rate"] == test_metrics["hit_rate"]

        # Test retrieval of existing baseline
        retrieved_baseline = self.tester.get_or_create_baseline(test_day, {})
        assert retrieved_baseline == baseline

        # Clean up
        for metric, key_template in self.tester.baseline_keys.items():
            key = key_template.format(test_day)
            self.tester.redis.delete(key)

    def test_test_environment_setup(self):
        """Test test environment setup and cleanup."""
        # Setup
        self.tester.setup_test_environment()

        # Verify test mode flags
        assert self.tester.redis.get("mode") == "test"
        assert self.tester.redis.get("paper_mode") == "1"

        # Cleanup
        self.tester.cleanup_test_environment()

        # Verify cleanup
        assert self.tester.redis.get("mode") is None
        assert self.tester.redis.get("paper_mode") is None

    def test_full_replay_with_synthetic_data(self):
        """Test full replay process with synthetic data."""
        # Create temporary file path (doesn't need to exist for synthetic data)
        test_path = "/tmp/test-2025-07-30.parquet"

        # Run replay test
        result = self.tester.replay(test_path)

        # Verify result structure
        assert "pass" in result
        assert "summary" in result
        assert "actual_metrics" in result
        assert "baseline_metrics" in result
        assert "test_duration_seconds" in result

        # First run should pass (creates baseline)
        assert result["pass"] is True

        # Verify metrics are reasonable
        actual = result["actual_metrics"]
        assert "pnl_final" in actual
        assert "slippage_avg" in actual
        assert "trades_count" in actual
        assert "hit_rate" in actual

        assert isinstance(actual["pnl_final"], (int, float))
        assert isinstance(actual["slippage_avg"], (int, float))
        assert isinstance(actual["trades_count"], int)
        assert isinstance(actual["hit_rate"], (int, float))

        # Clean up baseline
        day = Path(test_path).stem.split(".")[0]
        for metric, key_template in self.tester.baseline_keys.items():
            key = key_template.format(day)
            self.tester.redis.delete(key)


def test_replay_acceptance_cli():
    """Test replay acceptance command line interface."""
    # This test verifies the CLI can be imported and basic functions work
    from scripts.replay_acceptance import ReplayAcceptanceTest

    tester = ReplayAcceptanceTest()

    # Test synthetic data generation
    df = tester._generate_synthetic_market_data()
    assert len(df) > 0

    # Test tolerance settings
    assert tester.tolerances["pnl_pct"] == 0.15
    assert tester.tolerances["slippage_bps"] == 5.0
    assert tester.tolerances["trades_pct"] == 10.0


# Integration test for specific day (if data file exists)
@pytest.mark.integration
def test_replay_specific_day():
    """Integration test for specific day (requires real data)."""
    data_path = Path("data/replays/2025-07-30.parquet")

    if not data_path.exists():
        pytest.skip("Real market data not available for integration test")

    from scripts.replay_acceptance import main
    import subprocess
    import sys

    # Run as subprocess to test CLI
    cmd = [sys.executable, "scripts/replay_acceptance.py", str(data_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

    # Parse JSON output
    output_lines = result.stdout.strip().split("\n")
    json_output = None
    for line in output_lines:
        try:
            json_output = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    assert json_output is not None
    assert "pass" in json_output
    assert isinstance(json_output["pass"], bool)


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
