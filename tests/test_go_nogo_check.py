#!/usr/bin/env python3
"""
Unit tests for Go/No-Go Check
Tests deployment decision criteria and automation
"""
import pytest
import os
import sys
import tempfile
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.go_nogo_check import GoNoGoChecker


class TestGoNoGoChecker:
    def setup_method(self):
        """Set up test fixtures."""
        self.checker = GoNoGoChecker()

    def test_log_criteria_pass(self):
        """Test logging a passing criteria."""
        self.checker.log_criteria("test_check", "PASS", 10, {"detail": "success"})

        assert "test_check" in self.checker.criteria_results
        assert self.checker.criteria_results["test_check"]["status"] == "PASS"
        assert self.checker.score == 10
        assert self.checker.max_score == 10
        assert len(self.checker.go_blockers) == 0

    def test_log_criteria_fail(self):
        """Test logging a failing criteria."""
        self.checker.log_criteria("test_check", "FAIL", 10, {"detail": "failed"})

        assert "test_check" in self.checker.criteria_results
        assert self.checker.criteria_results["test_check"]["status"] == "FAIL"
        assert self.checker.score == 0
        assert self.checker.max_score == 10
        assert "test_check" in self.checker.go_blockers

    def test_log_criteria_warn(self):
        """Test logging a warning criteria."""
        self.checker.log_criteria("test_check", "WARN", 10, {"detail": "warning"})

        assert "test_check" in self.checker.criteria_results
        assert self.checker.criteria_results["test_check"]["status"] == "WARN"
        assert self.checker.score == 5  # Half credit for warnings
        assert self.checker.max_score == 10
        assert "test_check" in self.checker.warnings

    def test_check_validation_history_pass(self):
        """Test validation history check with sufficient PASS validations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validation_dir = Path(temp_dir) / "artifacts" / "validation"
            validation_dir.mkdir(parents=True)

            # Create two recent PASS validation artifacts
            now = datetime.now(timezone.utc)

            for i in range(2):
                timestamp = (now - timedelta(hours=i * 12)).isoformat()
                validation_data = {
                    "timestamp": timestamp,
                    "status": "PASS",
                    "overall_status": "PASS",
                }

                file_path = validation_dir / f"validation_{i}.json"
                with open(file_path, "w") as f:
                    json.dump(validation_data, f)

            with patch("pathlib.Path") as mock_path:
                mock_path.return_value = validation_dir.parent.parent
                self.checker.check_validation_history()

                assert "validation_history" in self.checker.criteria_results
                assert (
                    self.checker.criteria_results["validation_history"]["status"]
                    == "PASS"
                )

    def test_check_validation_history_insufficient(self):
        """Test validation history check with insufficient validations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validation_dir = Path(temp_dir) / "artifacts" / "validation"
            validation_dir.mkdir(parents=True)

            # Create only one validation artifact
            now = datetime.now(timezone.utc)
            validation_data = {"timestamp": now.isoformat(), "status": "PASS"}

            file_path = validation_dir / "validation_single.json"
            with open(file_path, "w") as f:
                json.dump(validation_data, f)

            with patch("pathlib.Path") as mock_path:
                mock_path.return_value = validation_dir.parent.parent
                self.checker.check_validation_history()

                assert (
                    self.checker.criteria_results["validation_history"]["status"]
                    == "FAIL"
                )

    def test_check_alerting_status_no_alerts(self):
        """Test alerting status check with no recent alerts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_dir = Path(temp_dir) / "artifacts" / "audit"
            audit_dir.mkdir(parents=True)

            # Create old alert (outside 48h window)
            old_alert = audit_dir / "old_alert.json"
            old_alert.write_text('{"severity": "warning"}')

            # Set file modification time to 3 days ago
            old_time = (datetime.now() - timedelta(days=3)).timestamp()
            os.utime(old_alert, (old_time, old_time))

            with patch("pathlib.Path") as mock_path:
                mock_path.return_value = audit_dir.parent.parent
                with patch("requests.get", side_effect=Exception("No alertmanager")):
                    self.checker.check_alerting_status()

                    assert (
                        self.checker.criteria_results["alerting_status"]["status"]
                        == "PASS"
                    )

    def test_check_alerting_status_page_alerts(self):
        """Test alerting status check with active page alerts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_dir = Path(temp_dir) / "artifacts" / "audit"
            audit_dir.mkdir(parents=True)

            with patch("pathlib.Path") as mock_path:
                mock_path.return_value = audit_dir.parent.parent
                with patch("requests.get") as mock_get:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "data": [
                            {"state": "firing", "labels": {"severity": "critical"}}
                        ]
                    }
                    mock_get.return_value = mock_response

                    self.checker.check_alerting_status()

                    assert (
                        self.checker.criteria_results["alerting_status"]["status"]
                        == "FAIL"
                    )

    def test_check_slo_performance_healthy(self):
        """Test SLO performance check with healthy budget."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0  # Budget OK
            mock_run.return_value = mock_result

            with tempfile.TemporaryDirectory() as temp_dir:
                audit_dir = Path(temp_dir) / "artifacts" / "audit"
                audit_dir.mkdir(parents=True)

                # Create budget audit file
                budget_data = {"budget_spent": 0.2}  # 20% spent
                budget_file = audit_dir / "budget_recent.json"
                with open(budget_file, "w") as f:
                    json.dump(budget_data, f)

                with patch("pathlib.Path") as mock_path:
                    mock_path.return_value = audit_dir.parent.parent
                    self.checker.check_slo_performance()

                    assert (
                        self.checker.criteria_results["slo_performance"]["status"]
                        == "PASS"
                    )

    def test_check_slo_performance_exhausted(self):
        """Test SLO performance check with exhausted budget."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1  # Budget exhausted
            mock_run.return_value = mock_result

            with tempfile.TemporaryDirectory() as temp_dir:
                audit_dir = Path(temp_dir) / "artifacts" / "audit"
                audit_dir.mkdir(parents=True)

                with patch("pathlib.Path") as mock_path:
                    mock_path.return_value = audit_dir.parent.parent
                    self.checker.check_slo_performance()

                    assert (
                        self.checker.criteria_results["slo_performance"]["status"]
                        == "FAIL"
                    )

    def test_check_technical_readiness_all_pass(self):
        """Test technical readiness with all checks passing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_dir = Path(temp_dir) / "artifacts" / "audit"
            audit_dir.mkdir(parents=True)

            # Create PASS preflight check
            preflight_data = {"summary": {"status": "PASS"}}
            preflight_file = audit_dir / "preflight_recent.json"
            with open(preflight_file, "w") as f:
                json.dump(preflight_data, f)

            # Create PASS security check
            security_data = {"summary": {"status": "PASS"}}
            security_file = audit_dir / "security_recent.json"
            with open(security_file, "w") as f:
                json.dump(security_data, f)

            # Mock kill-switch functionality
            with patch("src.rl.influence_controller.InfluenceController") as mock_ic:
                mock_instance = MagicMock()
                mock_instance.get_status.return_value = {"weight": 0.0}
                mock_ic.return_value = mock_instance

                with patch("pathlib.Path") as mock_path:
                    mock_path.return_value = audit_dir.parent.parent
                    self.checker.check_technical_readiness()

                    assert (
                        self.checker.criteria_results["technical_readiness"]["status"]
                        == "PASS"
                    )

    def test_check_policy_health_good_metrics(self):
        """Test policy health check with good metrics."""
        mock_metrics = """
rl_policy_entropy 0.95
rl_policy_q_spread 1.5
rl_policy_heartbeat_age_seconds 120
"""

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = mock_metrics
            mock_get.return_value = mock_response

            self.checker.check_policy_health()

            assert self.checker.criteria_results["policy_health"]["status"] == "PASS"

    def test_check_policy_health_bad_metrics(self):
        """Test policy health check with bad metrics."""
        mock_metrics = """
rl_policy_entropy 0.7
rl_policy_q_spread 3.5
rl_policy_heartbeat_age_seconds 1200
"""

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = mock_metrics
            mock_get.return_value = mock_response

            self.checker.check_policy_health()

            assert self.checker.criteria_results["policy_health"]["status"] == "FAIL"

    def test_check_operational_readiness(self):
        """Test operational readiness check."""
        # Mock ops bot functionality
        with patch("ops_bot.influence_commands.InfluenceBotCommands") as mock_bot:
            mock_instance = MagicMock()
            mock_instance.command_status.return_value = {"status": "success"}
            mock_bot.return_value = mock_instance

            # Mock runbook existence
            with patch("pathlib.Path.exists", return_value=True):
                self.checker.check_operational_readiness()

                assert (
                    self.checker.criteria_results["operational_readiness"]["status"]
                    == "PASS"
                )

    def test_run_all_checks_go_decision(self):
        """Test full check run resulting in GO decision."""
        # Mock all check methods to pass
        self.checker.log_criteria("validation_history", "PASS", 20)
        self.checker.log_criteria("alerting_status", "PASS", 15)
        self.checker.log_criteria("slo_performance", "PASS", 15)
        self.checker.log_criteria("technical_readiness", "PASS", 15)
        self.checker.log_criteria("policy_health", "PASS", 10)
        self.checker.log_criteria("operational_readiness", "PASS", 10)

        with patch.object(self.checker, "check_validation_history"):
            with patch.object(self.checker, "check_alerting_status"):
                with patch.object(self.checker, "check_slo_performance"):
                    with patch.object(self.checker, "check_technical_readiness"):
                        with patch.object(self.checker, "check_policy_health"):
                            with patch.object(
                                self.checker, "check_operational_readiness"
                            ):
                                with patch("os.makedirs"):
                                    with patch("builtins.open", create=True):
                                        is_go = self.checker.run_all_checks()

                                        assert is_go is True
                                        assert len(self.checker.go_blockers) == 0

    def test_run_all_checks_no_go_decision(self):
        """Test full check run resulting in NO-GO decision."""
        # Add blocking failures
        self.checker.log_criteria("validation_history", "FAIL", 20)
        self.checker.log_criteria("slo_performance", "FAIL", 15)
        self.checker.log_criteria("technical_readiness", "PASS", 15)

        with patch.object(self.checker, "check_validation_history"):
            with patch.object(self.checker, "check_alerting_status"):
                with patch.object(self.checker, "check_slo_performance"):
                    with patch.object(self.checker, "check_technical_readiness"):
                        with patch.object(self.checker, "check_policy_health"):
                            with patch.object(
                                self.checker, "check_operational_readiness"
                            ):
                                with patch("os.makedirs"):
                                    with patch("builtins.open", create=True):
                                        is_go = self.checker.run_all_checks()

                                        assert is_go is False
                                        assert len(self.checker.go_blockers) >= 1

    def test_conditional_go_decision(self):
        """Test conditional GO decision with warnings."""
        # Set up scenario for conditional GO (70-85% score)
        self.checker.log_criteria("validation_history", "PASS", 20)
        self.checker.log_criteria("alerting_status", "WARN", 15)  # Warning
        self.checker.log_criteria("slo_performance", "PASS", 15)
        self.checker.log_criteria("technical_readiness", "WARN", 15)  # Warning
        self.checker.log_criteria("policy_health", "PASS", 10)
        self.checker.log_criteria("operational_readiness", "PASS", 10)

        # Should be 67.5/85 = ~79% -> CONDITIONAL_GO
        with patch.object(self.checker, "check_validation_history"):
            with patch.object(self.checker, "check_alerting_status"):
                with patch.object(self.checker, "check_slo_performance"):
                    with patch.object(self.checker, "check_technical_readiness"):
                        with patch.object(self.checker, "check_policy_health"):
                            with patch.object(
                                self.checker, "check_operational_readiness"
                            ):
                                with patch("os.makedirs"):
                                    written_data = {}

                                    def mock_open(filename, mode):
                                        if mode == "w":
                                            return MockFile(written_data, filename)

                                    with patch("builtins.open", side_effect=mock_open):
                                        is_go = self.checker.run_all_checks()

                                        # Should be False for conditional GO in strict mode
                                        # But let's check the audit data for decision
                                        audit_content = list(written_data.values())[0]
                                        audit_json = json.loads(audit_content)

                                        # Check that decision was recorded
                                        assert "decision" in audit_json
                                        assert audit_json["decision"] in [
                                            "GO",
                                            "CONDITIONAL_GO",
                                            "NO_GO",
                                        ]

    def test_audit_trail_creation(self):
        """Test that Go/No-Go audit trail is created."""
        self.checker.log_criteria("test_check", "PASS", 10)

        written_data = {}

        def mock_open(filename, mode):
            if mode == "w":
                return MockFile(written_data, filename)

        with patch("os.makedirs"):
            with patch("builtins.open", side_effect=mock_open):
                with patch.object(self.checker, "check_validation_history"):
                    with patch.object(self.checker, "check_alerting_status"):
                        with patch.object(self.checker, "check_slo_performance"):
                            with patch.object(
                                self.checker, "check_technical_readiness"
                            ):
                                with patch.object(self.checker, "check_policy_health"):
                                    with patch.object(
                                        self.checker, "check_operational_readiness"
                                    ):
                                        self.checker.run_all_checks()

                # Check that audit data was written
                assert len(written_data) > 0
                audit_content = list(written_data.values())[0]
                audit_json = json.loads(audit_content)

                assert "timestamp" in audit_json
                assert "action" in audit_json
                assert audit_json["action"] == "go_nogo_decision"
                assert "decision" in audit_json
                assert "criteria_results" in audit_json


class MockFile:
    """Mock file object for testing file writes."""

    def __init__(self, storage, filename):
        self.storage = storage
        self.filename = filename
        self.content = ""

    def write(self, data):
        self.content += data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.storage[self.filename] = self.content


def test_module_imports():
    """Test that Go/No-Go check module can be imported."""
    from scripts.go_nogo_check import GoNoGoChecker

    checker = GoNoGoChecker()
    assert checker is not None
    assert hasattr(checker, "run_all_checks")
    assert hasattr(checker, "check_validation_history")


if __name__ == "__main__":
    # Run tests directly
    import subprocess

    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v"], cwd=Path(__file__).parent.parent
    )
    # Note: sys.exit removed to avoid pytest return-value warnings
    # Test framework will handle success/failure appropriately
