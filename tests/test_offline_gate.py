#!/usr/bin/env python3
"""
Tests for offline gate system
"""

import json
import subprocess
import tempfile
import os
import pytest
import yaml
from pathlib import Path


def create_eval_fixture(return_mean, entropy_mean, has_nan=False, episodes=32):
    """Create a test evaluation JSON fixture."""
    return {
        "timestamp": "2025-08-13T06:30:00Z",
        "ckpt_path": "test_checkpoint.pt",
        "episodes": episodes,
        "return_mean": return_mean,
        "return_std": 0.045,
        "entropy_mean": entropy_mean,
        "entropy_p05": max(0.1, entropy_mean - 0.3),
        "entropy_p95": min(2.0, entropy_mean + 0.3),
        "q_spread_mean": 35.5,
        "grad_norm_p95": 0.85,
        "has_nan": has_nan,
        "steps_total": episodes * 400,
    }


def create_gate_fixture(entropy_range=[1.0, 2.0], episodes_min=24):
    """Create a test gate YAML fixture."""
    return {
        "name": "test_gate",
        "entropy_range": entropy_range,
        "min_return_vs_last_good": -0.05,
        "max_grad_norm_p95": 1.25,
        "require_no_nans": True,
        "q_spread_max_ratio": 2.0,
        "baseline_json": "nonexistent.json",  # No baseline for these tests
        "episodes_min": episodes_min,
    }


class TestOfflineGate:

    def test_gate_pass_scenario(self):
        """Test gate checker with a passing scenario."""
        # Create passing evaluation
        eval_data = create_eval_fixture(
            return_mean=0.01,  # Positive return
            entropy_mean=1.5,  # Good entropy
            has_nan=False,  # No NaNs
            episodes=32,  # Sufficient episodes
        )

        gate_config = create_gate_fixture()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write fixtures
            eval_path = os.path.join(tmpdir, "eval.json")
            gate_path = os.path.join(tmpdir, "gate.yaml")
            report_path = os.path.join(tmpdir, "report.md")

            with open(eval_path, "w") as f:
                json.dump(eval_data, f)
            with open(gate_path, "w") as f:
                yaml.dump(gate_config, f)

            # Run gate checker
            result = subprocess.run(
                [
                    "python",
                    "tools/check_eval_gate.py",
                    "--eval",
                    eval_path,
                    "--gate",
                    gate_path,
                    "--out-md",
                    report_path,
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            # Should pass (exit code 0)
            assert (
                result.returncode == 0
            ), f"Gate should pass but failed: {result.stderr}"
            assert "GATE_PASS" in result.stdout

            # Check report was created
            assert os.path.exists(report_path)
            with open(report_path, "r") as f:
                report_content = f.read()
                assert "PASS" in report_content
                assert "✅" in report_content

    def test_gate_fail_entropy(self):
        """Test gate checker failing due to entropy out of range."""
        # Create failing evaluation (entropy too low)
        eval_data = create_eval_fixture(
            return_mean=0.01,
            entropy_mean=0.5,  # Too low entropy
            has_nan=False,
            episodes=32,
        )

        gate_config = create_gate_fixture()

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_path = os.path.join(tmpdir, "eval.json")
            gate_path = os.path.join(tmpdir, "gate.yaml")
            report_path = os.path.join(tmpdir, "report.md")

            with open(eval_path, "w") as f:
                json.dump(eval_data, f)
            with open(gate_path, "w") as f:
                yaml.dump(gate_config, f)

            # Run gate checker
            result = subprocess.run(
                [
                    "python",
                    "tools/check_eval_gate.py",
                    "--eval",
                    eval_path,
                    "--gate",
                    gate_path,
                    "--out-md",
                    report_path,
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            # Should fail (exit code 1)
            assert result.returncode == 1, "Gate should fail due to low entropy"
            assert "GATE_FAIL" in result.stdout
            assert "entropy" in result.stdout.lower()

            # Check failure is documented in report
            assert os.path.exists(report_path)
            with open(report_path, "r") as f:
                report_content = f.read()
                assert "Failures" in report_content

    def test_gate_fail_nans(self):
        """Test gate checker failing due to NaNs detected."""
        eval_data = create_eval_fixture(
            return_mean=0.01,
            entropy_mean=1.5,
            has_nan=True,  # NaNs detected!
            episodes=32,
        )

        gate_config = create_gate_fixture()

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_path = os.path.join(tmpdir, "eval.json")
            gate_path = os.path.join(tmpdir, "gate.yaml")

            with open(eval_path, "w") as f:
                json.dump(eval_data, f)
            with open(gate_path, "w") as f:
                yaml.dump(gate_config, f)

            # Run gate checker
            result = subprocess.run(
                [
                    "python",
                    "tools/check_eval_gate.py",
                    "--eval",
                    eval_path,
                    "--gate",
                    gate_path,
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            # Should fail due to NaNs
            assert result.returncode == 1, "Gate should fail due to NaNs"
            assert "GATE_FAIL" in result.stdout
            assert "nan" in result.stdout.lower()

    def test_gate_fail_insufficient_episodes(self):
        """Test gate checker failing due to insufficient episodes."""
        eval_data = create_eval_fixture(
            return_mean=0.01,
            entropy_mean=1.5,
            has_nan=False,
            episodes=10,  # Too few episodes
        )

        gate_config = create_gate_fixture(episodes_min=24)

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_path = os.path.join(tmpdir, "eval.json")
            gate_path = os.path.join(tmpdir, "gate.yaml")

            with open(eval_path, "w") as f:
                json.dump(eval_data, f)
            with open(gate_path, "w") as f:
                yaml.dump(gate_config, f)

            # Run gate checker
            result = subprocess.run(
                [
                    "python",
                    "tools/check_eval_gate.py",
                    "--eval",
                    eval_path,
                    "--gate",
                    gate_path,
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            # Should fail due to insufficient episodes
            assert (
                result.returncode == 1
            ), "Gate should fail due to insufficient episodes"
            assert "episodes_min" in result.stdout

    def test_file_not_found_handling(self):
        """Test gate checker handles missing files gracefully."""
        result = subprocess.run(
            [
                "python",
                "tools/check_eval_gate.py",
                "--eval",
                "nonexistent_eval.json",
                "--gate",
                "nonexistent_gate.yaml",
            ],
            capture_output=True,
            text=True,
            cwd=".",
        )

        # Should fail with file not found
        assert result.returncode == 1
        assert "GATE_FAIL" in result.stdout
        assert "not found" in result.stdout.lower()


def test_eval_offline_basic():
    """Test basic functionality of eval_offline.py"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock checkpoint
        ckpt_path = os.path.join(tmpdir, "test_checkpoint.pt")
        with open(ckpt_path, "w") as f:
            f.write("mock checkpoint")

        eval_path = os.path.join(tmpdir, "eval.json")
        md_path = os.path.join(tmpdir, "eval.md")

        # Run eval_offline.py
        result = subprocess.run(
            [
                "python",
                "tools/eval_offline.py",
                "--ckpt",
                ckpt_path,
                "--episodes",
                "4",  # Small number for fast test
                "--out",
                eval_path,
                "--md-out",
                md_path,
            ],
            capture_output=True,
            text=True,
            cwd=".",
        )

        # Should succeed
        assert result.returncode == 0, f"eval_offline failed: {result.stderr}"

        # Check outputs exist and have expected structure
        assert os.path.exists(eval_path)
        assert os.path.exists(md_path)

        # Validate JSON structure
        with open(eval_path, "r") as f:
            eval_data = json.load(f)

        required_fields = [
            "timestamp",
            "ckpt_path",
            "episodes",
            "return_mean",
            "entropy_mean",
            "has_nan",
            "steps_total",
        ]
        for field in required_fields:
            assert field in eval_data, f"Missing field: {field}"

        assert eval_data["episodes"] == 4
        assert eval_data["ckpt_path"] == ckpt_path


if __name__ == "__main__":
    # Can run individual test methods
    test_gate = TestOfflineGate()
    test_gate.test_gate_pass_scenario()
    test_gate.test_gate_fail_entropy()
    test_gate.test_gate_fail_nans()
    test_gate.test_gate_fail_insufficient_episodes()
    test_gate.test_file_not_found_handling()
    test_eval_offline_basic()
    print("✅ All tests passed!")
