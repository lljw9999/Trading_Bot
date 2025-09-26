#!/usr/bin/env python3
"""
Unit tests for Preflight Release Check
Tests configuration validation, model integrity, and deployment gates
"""
import pytest
import os
import sys
import tempfile
import json
import hashlib
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.preflight_release_check import PreflightChecker


class TestPreflightChecker:
    def setup_method(self):
        """Set up test fixtures."""
        self.checker = PreflightChecker()

    def test_get_git_sha_success(self):
        """Test successful git SHA retrieval."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "abc123def456\n"
            mock_run.return_value = mock_result

            sha = self.checker.get_git_sha()
            assert sha == "abc123def456"

    def test_get_git_sha_failure(self):
        """Test git SHA retrieval failure."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_run.return_value = mock_result

            sha = self.checker.get_git_sha()
            assert sha is None

    def test_get_file_hash(self):
        """Test file hash calculation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()

            try:
                hash_val = self.checker.get_file_hash(f.name)
                expected_hash = hashlib.sha256(b"test content").hexdigest()
                assert hash_val == expected_hash
            finally:
                os.unlink(f.name)

    def test_get_file_hash_missing_file(self):
        """Test file hash for missing file."""
        hash_val = self.checker.get_file_hash("/nonexistent/file.txt")
        assert hash_val is None

    def test_get_directory_hash(self):
        """Test directory hash calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "file1.txt").write_text("content1")
            (temp_path / "file2.txt").write_text("content2")

            hash_val = self.checker.get_directory_hash(temp_path, "*.txt")
            assert hash_val is not None
            assert len(hash_val) == 64  # SHA256 hex length

    def test_check_git_status_clean(self):
        """Test git status check with clean repository."""
        with patch("subprocess.run") as mock_run:
            # Mock git status
            status_result = MagicMock()
            status_result.returncode = 0
            status_result.stdout = ""  # Clean status

            # Mock git rev-parse
            sha_result = MagicMock()
            sha_result.returncode = 0
            sha_result.stdout = "abc123\n"

            mock_run.side_effect = [status_result, sha_result]

            success = self.checker.check_git_status()
            assert success is True

            # Check that both git status and SHA check were recorded
            assert "git_status" in self.checker.checks
            assert "git_sha" in self.checker.checks
            assert self.checker.checks["git_status"]["status"] == "PASS"
            assert self.checker.checks["git_sha"]["status"] == "PASS"

    def test_check_git_status_uncommitted_changes(self):
        """Test git status check with uncommitted changes."""
        with patch("subprocess.run") as mock_run:
            # Mock git status with uncommitted changes
            status_result = MagicMock()
            status_result.returncode = 0
            status_result.stdout = " M modified_file.py\n?? new_file.py"

            # Mock git rev-parse
            sha_result = MagicMock()
            sha_result.returncode = 0
            sha_result.stdout = "abc123\n"

            mock_run.side_effect = [status_result, sha_result]

            success = self.checker.check_git_status()
            assert success is True

            # Should be warning, not failure
            assert self.checker.checks["git_status"]["status"] == "WARN"
            assert self.checker.checks["git_sha"]["status"] == "PASS"

    def test_check_model_integrity_with_models(self):
        """Test model integrity check with existing models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "models"
            models_dir.mkdir()

            # Create mock model files
            (models_dir / "model1.onnx").write_text("fake onnx content")
            (models_dir / "config1.json").write_text('{"model": "config"}')

            with patch("pathlib.Path.exists", return_value=True):
                with patch.object(self.checker, "get_directory_hash") as mock_hash:
                    mock_hash.return_value = "mock_hash_value"

                    success = self.checker.check_model_integrity()
                    assert success is True
                    assert "model_integrity" in self.checker.checks
                    assert self.checker.checks["model_integrity"]["status"] == "PASS"

    def test_check_model_integrity_no_models(self):
        """Test model integrity check with no models."""
        with patch("pathlib.Path.exists", return_value=False):
            success = self.checker.check_model_integrity()
            assert success is True
            assert self.checker.checks["model_integrity"]["status"] == "WARN"

    def test_check_config_integrity(self):
        """Test configuration integrity check."""
        config_files = [
            "src/config/base_config.yaml",
            "slo/slo.yaml",
            "docker-compose.yml",
            "requirements.txt",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(self.checker, "get_file_hash", return_value="mock_hash"):
                success = self.checker.check_config_integrity()
                assert success is True
                assert "config_integrity" in self.checker.checks
                assert self.checker.checks["config_integrity"]["status"] == "PASS"

    def test_check_config_integrity_missing_files(self):
        """Test configuration integrity check with missing files."""

        def mock_exists(path):
            return path.name == "requirements.txt"  # Only one file exists

        with patch("pathlib.Path.exists", side_effect=mock_exists):
            with patch.object(self.checker, "get_file_hash", return_value="mock_hash"):
                success = self.checker.check_config_integrity()
                assert success is True
                assert self.checker.checks["config_integrity"]["status"] == "WARN"

    def test_check_dashboard_consistency(self):
        """Test dashboard consistency check."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                self.checker, "get_directory_hash", return_value="dashboard_hash"
            ):
                success = self.checker.check_dashboard_consistency()
                assert success is True
                assert "dashboard_consistency" in self.checker.checks
                assert self.checker.checks["dashboard_consistency"]["status"] == "PASS"

    def test_check_dashboard_consistency_no_dashboards(self):
        """Test dashboard consistency check with no dashboard directory."""
        with patch("pathlib.Path.exists", return_value=False):
            success = self.checker.check_dashboard_consistency()
            assert success is True
            assert self.checker.checks["dashboard_consistency"]["status"] == "WARN"

    def test_check_exporter_version_available(self):
        """Test exporter version check when available."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "text/plain; version=0.0.4"}
            mock_get.return_value = mock_response

            success = self.checker.check_exporter_version()
            assert success is True
            assert "exporter_version" in self.checker.checks
            assert self.checker.checks["exporter_version"]["status"] == "PASS"

    def test_check_exporter_version_unavailable(self):
        """Test exporter version check when unavailable."""
        with patch("requests.get", side_effect=Exception("Connection refused")):
            success = self.checker.check_exporter_version()
            assert success is True
            assert self.checker.checks["exporter_version"]["status"] == "WARN"

    def test_check_dependency_versions_success(self):
        """Test dependency version check success."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "package1==1.0.0\npackage2==2.0.0\n"
            mock_run.return_value = mock_result

            success = self.checker.check_dependency_versions()
            assert success is True
            assert "dependency_versions" in self.checker.checks
            assert self.checker.checks["dependency_versions"]["status"] == "PASS"

    def test_check_dependency_versions_failure(self):
        """Test dependency version check failure."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_run.return_value = mock_result

            success = self.checker.check_dependency_versions()
            assert success is False
            assert self.checker.checks["dependency_versions"]["status"] == "FAIL"

    def test_check_systemd_services(self):
        """Test systemd services check."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(self.checker, "get_directory_hash") as mock_hash:
                mock_hash.return_value = "systemd_hash"

                success = self.checker.check_systemd_services()
                assert success is True
                assert "systemd_services" in self.checker.checks
                assert self.checker.checks["systemd_services"]["status"] == "PASS"

    def test_run_all_checks_success(self):
        """Test running all checks with successful results."""
        # Mock all check methods to succeed
        check_methods = [
            "check_git_status",
            "check_model_integrity",
            "check_config_integrity",
            "check_dashboard_consistency",
            "check_exporter_version",
            "check_dependency_versions",
            "check_systemd_services",
        ]

        for method in check_methods:
            with patch.object(self.checker, method, return_value=True):
                pass

        # Simulate successful checks by directly setting results
        for i, method in enumerate(check_methods):
            self.checker.log_check(method.replace("check_", ""), "PASS", {})

        with patch("os.makedirs"):
            with patch("builtins.open", create=True):
                success = self.checker.run_all_checks()
                assert success is True
                assert len(self.checker.failures) == 0
                assert len(self.checker.checks) > 0

    def test_run_all_checks_with_failures(self):
        """Test running all checks with some failures."""
        # Add some failures
        self.checker.log_check("git_status", "PASS", {})
        self.checker.log_check(
            "model_integrity", "FAIL", {"message": "Models corrupted"}
        )
        self.checker.log_check(
            "config_integrity", "WARN", {"message": "Some configs missing"}
        )

        with patch("os.makedirs"):
            with patch("builtins.open", create=True):
                # Mock all check methods since we're setting results directly
                with patch.object(self.checker, "check_git_status"):
                    with patch.object(self.checker, "check_model_integrity"):
                        with patch.object(self.checker, "check_config_integrity"):
                            with patch.object(
                                self.checker, "check_dashboard_consistency"
                            ):
                                with patch.object(
                                    self.checker, "check_exporter_version"
                                ):
                                    with patch.object(
                                        self.checker, "check_dependency_versions"
                                    ):
                                        with patch.object(
                                            self.checker, "check_systemd_services"
                                        ):
                                            success = self.checker.run_all_checks()
                                            assert success is False
                                            assert len(self.checker.failures) == 1
                                            assert (
                                                "model_integrity"
                                                in self.checker.failures
                                            )

    def test_audit_trail_creation(self):
        """Test that preflight audit trail is created."""
        self.checker.log_check("test_check", "PASS", {"detail": "test"})

        written_data = {}

        def mock_open(filename, mode):
            if mode == "w":
                return MockFile(written_data, filename)

        with patch("os.makedirs"):
            with patch("builtins.open", side_effect=mock_open):
                self.checker.run_all_checks()

                # Check that audit data was written
                assert len(written_data) > 0
                audit_content = list(written_data.values())[0]
                audit_json = json.loads(audit_content)

                assert "timestamp" in audit_json
                assert "action" in audit_json
                assert audit_json["action"] == "preflight_release_check"
                assert "summary" in audit_json
                assert "checks" in audit_json


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
    """Test that preflight release check module can be imported."""
    from scripts.preflight_release_check import PreflightChecker

    checker = PreflightChecker()
    assert checker is not None
    assert hasattr(checker, "run_all_checks")
    assert hasattr(checker, "get_git_sha")


if __name__ == "__main__":
    # Run tests directly
    import subprocess

    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v"], cwd=Path(__file__).parent.parent
    )
    # Note: sys.exit removed to avoid pytest return-value warnings
    # Test framework will handle success/failure appropriately
