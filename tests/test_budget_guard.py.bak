#!/usr/bin/env python3
"""
Unit tests for Error Budget Guard
Tests budget calculation, exhaustion handling, and auto-remediation
"""
import pytest
import os
import sys
import tempfile
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.error_budget_guard import ErrorBudgetGuard


class TestErrorBudgetGuard:
    def setup_method(self):
        """Set up test fixtures."""
        self.guard = ErrorBudgetGuard()

    def test_slo_heartbeat_fresh_pass(self):
        """Test heartbeat freshness SLO check - pass case."""
        with patch.object(self.guard, "get_prometheus_metric", return_value=0.996):
            result = self.guard.check_slo_heartbeat_fresh()
            assert result == 0.996

    def test_slo_heartbeat_fresh_redis_fallback(self):
        """Test heartbeat freshness with Redis fallback."""
        with patch.object(self.guard, "get_prometheus_metric", return_value=None):
            with patch("redis.Redis") as mock_redis:
                mock_client = MagicMock()
                mock_redis.from_url.return_value = mock_client
                mock_client.get.return_value = "1234567890"

                with patch("time.time", return_value=1234567950):  # 60 seconds later
                    result = self.guard.check_slo_heartbeat_fresh()
                    assert result == 1.0  # Fresh heartbeat

    def test_slo_heartbeat_fresh_stale(self):
        """Test heartbeat freshness with stale heartbeat."""
        with patch.object(self.guard, "get_prometheus_metric", return_value=None):
            with patch("redis.Redis") as mock_redis:
                mock_client = MagicMock()
                mock_redis.from_url.return_value = mock_client
                mock_client.get.return_value = "1234567890"

                with patch("time.time", return_value=1234568490):  # 10 minutes later
                    result = self.guard.check_slo_heartbeat_fresh()
                    assert result == 0.0  # Stale heartbeat

    def test_slo_exporter_uptime_pass(self):
        """Test exporter uptime SLO check - pass case."""
        with patch.object(self.guard, "get_prometheus_metric", return_value=1.0):
            result = self.guard.check_slo_exporter_uptime()
            assert result == 1.0

    def test_slo_exporter_uptime_http_fallback(self):
        """Test exporter uptime with HTTP fallback."""
        with patch.object(self.guard, "get_prometheus_metric", return_value=None):
            with patch("requests.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_get.return_value = mock_response

                result = self.guard.check_slo_exporter_uptime()
                assert result == 1.0

    def test_slo_exporter_uptime_fail(self):
        """Test exporter uptime failure."""
        with patch.object(self.guard, "get_prometheus_metric", return_value=None):
            with patch("requests.get", side_effect=Exception("Connection failed")):
                result = self.guard.check_slo_exporter_uptime()
                assert result == 0.0

    def test_slo_validation_cadence_pass(self):
        """Test validation cadence SLO - pass case."""
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts" / "validation"
            artifacts_dir.mkdir(parents=True)

            # Create validation artifacts for recent days
            from datetime import datetime, timedelta

            now = datetime.now()

            for i in range(30):  # All 30 days have artifacts
                date = now - timedelta(days=i)
                date_str = date.strftime("%Y%m%d")
                artifact_file = artifacts_dir / f"validation_{date_str}_test.json"
                artifact_file.write_text('{"status": "PASS"}')

            with patch("pathlib.Path") as mock_path:
                mock_path.return_value = artifacts_dir.parent.parent
                result = self.guard.check_slo_validation_cadence()
                assert result == 1.0  # 100% of days have artifacts

    def test_slo_validation_cadence_partial(self):
        """Test validation cadence SLO - partial coverage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts" / "validation"
            artifacts_dir.mkdir(parents=True)

            # Create artifacts for only 50% of days
            from datetime import datetime, timedelta

            now = datetime.now()

            for i in range(15):  # Only 15 out of 30 days
                date = now - timedelta(days=i * 2)
                date_str = date.strftime("%Y%m%d")
                artifact_file = artifacts_dir / f"validation_{date_str}_test.json"
                artifact_file.write_text('{"status": "PASS"}')

            with patch("pathlib.Path") as mock_path:
                mock_path.return_value = artifacts_dir.parent.parent
                result = self.guard.check_slo_validation_cadence()
                assert result == 0.5  # 50% coverage

    def test_compute_error_budget_burn_healthy(self):
        """Test error budget computation - healthy case."""
        with patch.object(self.guard, "check_slo_heartbeat_fresh", return_value=0.996):
            with patch.object(
                self.guard, "check_slo_exporter_uptime", return_value=0.999
            ):
                with patch.object(
                    self.guard, "check_slo_validation_cadence", return_value=0.98
                ):
                    burn_rate = self.guard.compute_error_budget_burn()
                    assert burn_rate < 0.1  # Low burn rate

    def test_compute_error_budget_burn_degraded(self):
        """Test error budget computation - degraded case."""
        with patch.object(self.guard, "check_slo_heartbeat_fresh", return_value=0.85):
            with patch.object(
                self.guard, "check_slo_exporter_uptime", return_value=0.9
            ):
                with patch.object(
                    self.guard, "check_slo_validation_cadence", return_value=0.8
                ):
                    burn_rate = self.guard.compute_error_budget_burn()
                    assert burn_rate > 0.5  # High burn rate

    def test_execute_remediation_exhausted(self):
        """Test remediation execution when budget exhausted."""
        with patch(
            "src.rl.influence_controller.emergency_stop", return_value=True
        ) as mock_stop:
            success = self.guard.execute_remediation(1.0)  # 100% budget spent
            assert success is True
            mock_stop.assert_called_once()

    def test_execute_remediation_warning(self):
        """Test remediation execution on warning threshold."""
        with patch("src.rl.influence_controller.emergency_stop") as mock_stop:
            success = self.guard.execute_remediation(0.8)  # 80% budget spent
            assert success is True
            mock_stop.assert_not_called()  # No emergency action for warning

    def test_execute_remediation_emergency_stop_fail(self):
        """Test remediation when emergency stop fails."""
        with patch("src.rl.influence_controller.emergency_stop", return_value=False):
            with patch("redis.Redis") as mock_redis:
                mock_client = MagicMock()
                mock_redis.from_url.return_value = mock_client

                success = self.guard.execute_remediation(1.0)
                assert success is True  # Should succeed via Redis fallback
                mock_client.set.assert_called_with("policy:allowed_influence_pct", 0)

    def test_run_guard_budget_ok(self):
        """Test guard run with healthy budget."""
        with patch.object(self.guard, "compute_error_budget_burn", return_value=0.1):
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch("os.makedirs"):
                    with patch("builtins.open", create=True) as mock_open:
                        exit_code = self.guard.run_guard(dry_run=True)
                        assert exit_code == 0  # Budget OK

    def test_run_guard_budget_warning(self):
        """Test guard run with budget warning."""
        with patch.object(self.guard, "compute_error_budget_burn", return_value=0.8):
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch("os.makedirs"):
                    with patch("builtins.open", create=True) as mock_open:
                        exit_code = self.guard.run_guard(dry_run=True)
                        assert exit_code == 2  # Budget warning

    def test_run_guard_budget_exhausted(self):
        """Test guard run with exhausted budget."""
        with patch.object(self.guard, "compute_error_budget_burn", return_value=1.2):
            with patch.object(self.guard, "execute_remediation", return_value=True):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with patch("os.makedirs"):
                        with patch("builtins.open", create=True) as mock_open:
                            exit_code = self.guard.run_guard(dry_run=False)
                            assert exit_code == 1  # Budget exhausted

    def test_audit_trail_creation(self):
        """Test that audit trail is properly created."""
        with patch.object(self.guard, "compute_error_budget_burn", return_value=0.5):
            with tempfile.TemporaryDirectory() as temp_dir:
                audit_dir = Path(temp_dir) / "artifacts" / "audit"
                audit_dir.mkdir(parents=True)

                with patch("os.makedirs"):
                    written_data = {}

                    def mock_open(filename, mode):
                        if mode == "w":
                            return MockFile(written_data, filename)

                    with patch("builtins.open", side_effect=mock_open):
                        self.guard.run_guard(dry_run=True)

                        # Check that audit data was written
                        assert len(written_data) > 0
                        audit_content = list(written_data.values())[0]
                        audit_json = json.loads(audit_content)

                        assert "ts" in audit_json
                        assert "action" in audit_json
                        assert audit_json["action"] == "error_budget_guard"
                        assert "budget_spent" in audit_json
                        assert audit_json["budget_spent"] == 0.5


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
    """Test that error budget guard module can be imported."""
    from scripts.error_budget_guard import ErrorBudgetGuard

    guard = ErrorBudgetGuard()
    assert guard is not None
    assert hasattr(guard, "run_guard")
    assert hasattr(guard, "compute_error_budget_burn")


if __name__ == "__main__":
    # Run tests directly
    import subprocess

    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v"], cwd=Path(__file__).parent.parent
    )
    sys.exit(result.returncode)
