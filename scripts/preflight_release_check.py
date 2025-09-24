#!/usr/bin/env python3
"""
Preflight Release Check - Blue/Green Validation
Ensures configuration, models, and dashboards are consistent before deployment
"""
import os
import sys
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class PreflightChecker:
    def __init__(self):
        self.checks = {}
        self.failures = []
        self.warnings = []

    def log_check(self, name, status, details=None):
        """Log a check result."""
        entry = {
            "name": name,
            "status": status,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.checks[name] = entry

        if status == "FAIL":
            self.failures.append(name)
        elif status == "WARN":
            self.warnings.append(name)

    def get_git_sha(self):
        """Get current Git SHA."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd="."
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None

    def get_file_hash(self, file_path):
        """Get SHA256 hash of a file."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return None

    def get_directory_hash(self, dir_path, pattern="*"):
        """Get combined hash of all files in directory matching pattern."""
        try:
            dir_path = Path(dir_path)
            if not Path.exists(dir_path):
                return None

            file_hashes = []
            for file_path in sorted(dir_path.glob(pattern)):
                if file_path.is_file():
                    file_hash = self.get_file_hash(file_path)
                    if file_hash:
                        file_hashes.append(f"{file_path.name}:{file_hash}")

            combined = "\n".join(file_hashes)
            return hashlib.sha256(combined.encode()).hexdigest()
        except Exception:
            return None

    def check_git_status(self):
        """Check Git repository status."""
        try:
            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=".",
            )

            if result.returncode != 0:
                self.log_check("git_status", "FAIL", {"error": "git status failed"})
                return False

            uncommitted = result.stdout.strip()
            if uncommitted:
                self.log_check(
                    "git_status", "WARN", {"uncommitted_files": uncommitted.split("\n")}
                )
            else:
                self.log_check("git_status", "PASS", {"clean": True})

            # Get current SHA
            sha = self.get_git_sha()
            if sha:
                self.log_check("git_sha", "PASS", {"sha": sha})
                return True
            else:
                self.log_check("git_sha", "FAIL", {"error": "Could not get SHA"})
                return False

        except Exception as e:
            self.log_check("git_status", "FAIL", {"error": str(e)})
            return False

    def check_model_integrity(self):
        """Check model file integrity."""
        model_dirs = ["models", "artifacts/models"]
        model_status = {}

        for model_dir in model_dirs:
            model_path = Path(model_dir)
            if Path.exists(model_path):
                # Check ONNX models
                onnx_hash = self.get_directory_hash(model_dir, "*.onnx")
                if onnx_hash:
                    model_status[f"{model_dir}_onnx"] = onnx_hash

                # Check model configs
                config_hash = self.get_directory_hash(model_dir, "*.json")
                if config_hash:
                    model_status[f"{model_dir}_config"] = config_hash

        if model_status:
            self.log_check("model_integrity", "PASS", {"hashes": model_status})
            return True
        else:
            self.log_check("model_integrity", "WARN", {"message": "No models found"})
            return True

    def check_config_integrity(self):
        """Check configuration file integrity."""
        config_files = [
            "src/config/base_config.yaml",
            "slo/slo.yaml",
            "docker-compose.yml",
            "requirements.txt",
        ]

        config_hashes = {}
        missing_files = []

        for config_file in config_files:
            config_path = Path(config_file)
            if Path.exists(config_path):
                hash_val = self.get_file_hash(config_file)
                if hash_val:
                    config_hashes[config_file] = hash_val
            else:
                missing_files.append(config_file)

        if missing_files:
            self.log_check(
                "config_integrity",
                "WARN",
                {"missing_files": missing_files, "found_hashes": config_hashes},
            )
        else:
            self.log_check("config_integrity", "PASS", {"hashes": config_hashes})

        return True

    def check_dashboard_consistency(self):
        """Check Grafana dashboard consistency."""
        dashboard_dir = Path("grafana/dashboards")

        if not Path.exists(dashboard_dir):
            self.log_check(
                "dashboard_consistency",
                "WARN",
                {"message": "Dashboard directory not found"},
            )
            return True

        dashboard_hash = self.get_directory_hash(dashboard_dir, "*.json")
        if dashboard_hash:
            self.log_check(
                "dashboard_consistency", "PASS", {"dashboard_hash": dashboard_hash}
            )
        else:
            self.log_check(
                "dashboard_consistency", "WARN", {"message": "No dashboard files found"}
            )

        return True

    def check_exporter_version(self):
        """Check RL exporter version consistency."""
        try:
            # Try to get version from exporter
            import requests

            response = requests.get("http://localhost:9108/metrics", timeout=5)

            if response.status_code == 200:
                # Look for version info in metrics or headers
                content_type = response.headers.get("Content-Type", "")
                if "version=" in content_type:
                    version = content_type.split("version=")[1].split(";")[0]
                    self.log_check("exporter_version", "PASS", {"version": version})
                else:
                    self.log_check("exporter_version", "PASS", {"available": True})
            else:
                self.log_check(
                    "exporter_version",
                    "WARN",
                    {"message": f"Exporter returned {response.status_code}"},
                )
        except Exception as e:
            self.log_check(
                "exporter_version",
                "WARN",
                {"message": f"Could not check exporter: {e}"},
            )

        return True

    def check_dependency_versions(self):
        """Check Python dependency versions."""
        try:
            result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)

            if result.returncode == 0:
                deps = result.stdout.strip()
                deps_hash = hashlib.sha256(deps.encode()).hexdigest()

                self.log_check("dependency_versions", "PASS", {"deps_hash": deps_hash})
            else:
                self.log_check(
                    "dependency_versions", "FAIL", {"error": "pip freeze failed"}
                )
                return False
        except Exception as e:
            self.log_check("dependency_versions", "FAIL", {"error": str(e)})
            return False

        return True

    def check_systemd_services(self):
        """Check systemd service file consistency."""
        systemd_dir = Path("systemd")

        if Path.exists(systemd_dir):
            systemd_hash = self.get_directory_hash(systemd_dir, "*.service")
            timer_hash = self.get_directory_hash(systemd_dir, "*.timer")

            self.log_check(
                "systemd_services",
                "PASS",
                {"service_hash": systemd_hash, "timer_hash": timer_hash},
            )
        else:
            self.log_check(
                "systemd_services", "WARN", {"message": "Systemd directory not found"}
            )

        return True

    def run_all_checks(self):
        """Run all preflight checks."""
        print("🚀 Running Preflight Release Checks...")
        print(f"⏰ Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print()

        # Run all checks
        check_methods = [
            ("git_status", self.check_git_status),
            ("model_integrity", self.check_model_integrity),
            ("config_integrity", self.check_config_integrity),
            ("dashboard_consistency", self.check_dashboard_consistency),
            ("exporter_version", self.check_exporter_version),
            ("dependency_versions", self.check_dependency_versions),
            ("systemd_services", self.check_systemd_services),
        ]

        for check_name, check_method in check_methods:
            if check_name in self.checks:
                continue
            try:
                check_method()
            except Exception as e:
                self.log_check(check_name, "FAIL", {"error": str(e)})

        # Summary
        total_checks = len(self.checks)
        values = list(self.checks.values())
        passed = len([c for c in values if c["status"] == "PASS"])
        warned = len([c for c in values if c["status"] == "WARN"])
        failed = len([c for c in values if c["status"] == "FAIL"])

        print(f"📊 Preflight Check Results:")
        print(f"   Total Checks: {total_checks}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ⚠️  Warnings: {warned}")
        print(f"   ❌ Failed: {failed}")

        # Create audit artifact
        audit_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "preflight_release_check",
            "summary": {
                "total": total_checks,
                "passed": passed,
                "warned": warned,
                "failed": failed,
                "status": "FAIL" if failed > 0 else "WARN" if warned > 0 else "PASS",
            },
            "checks": values,
            "failures": self.failures,
            "warnings": self.warnings,
        }

        # Write audit artifact
        os.makedirs("artifacts/audit", exist_ok=True)
        audit_filename = f"artifacts/audit/{audit_data['timestamp'].replace(':', '_')}_preflight.json"
        with open(audit_filename, "w") as f:
            json.dump(audit_data, f, indent=2)

        print(f"📋 Audit: {audit_filename}")

        if failed > 0:
            print("\n❌ PREFLIGHT FAILED")
            for failure in self.failures:
                print(f"   • {failure}")
            return False
        elif warned > 0:
            print(f"\n⚠️  PREFLIGHT PASSED WITH WARNINGS")
            for warning in self.warnings:
                print(f"   • {warning}")
            return True
        else:
            print(f"\n✅ PREFLIGHT PASSED")
            return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preflight Release Check")
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as failures"
    )
    args = parser.parse_args()

    checker = PreflightChecker()
    success = checker.run_all_checks()

    if args.strict and checker.warnings:
        print("\n⚠️  Strict mode: treating warnings as failures")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
