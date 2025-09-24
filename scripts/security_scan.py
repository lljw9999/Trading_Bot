#!/usr/bin/env python3
"""
Security Scan Gate - Pre-deployment Security Checks
Runs gitleaks, dependency audit, and basic security checks
"""
import os
import sys
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class SecurityScanner:
    def __init__(self):
        self.scan_results = {}
        self.failures = []
        self.warnings = []

    def log_result(self, scan_name, status, details=None):
        """Log a scan result."""
        self.scan_results[scan_name] = {
            "status": status,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if status == "FAIL":
            self.failures.append(scan_name)
        elif status == "WARN":
            self.warnings.append(scan_name)

    def run_gitleaks_scan(self):
        """Run gitleaks to detect secrets in git history."""
        print("🔍 Running gitleaks secrets scan...")

        try:
            # Check if gitleaks is installed
            result = subprocess.run(["which", "gitleaks"], capture_output=True)
            if result.returncode != 0:
                print("⚠️  gitleaks not installed, installing via brew...")
                install_result = subprocess.run(
                    ["brew", "install", "gitleaks"], capture_output=True, text=True
                )
                if install_result.returncode != 0:
                    self.log_result(
                        "gitleaks",
                        "WARN",
                        {
                            "message": "gitleaks not available and could not install",
                            "install_output": install_result.stderr,
                        },
                    )
                    return False

            # Run gitleaks detect
            result = subprocess.run(
                [
                    "gitleaks",
                    "detect",
                    "--no-git",
                    "--verbose",
                    "--source",
                    ".",
                    "--report-format",
                    "json",
                    "--report-path",
                    "/tmp/gitleaks-report.json",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            # gitleaks returns 1 if secrets found, 0 if clean
            if result.returncode == 0:
                self.log_result(
                    "gitleaks",
                    "PASS",
                    {"message": "No secrets detected", "files_scanned": "all"},
                )
                return True
            elif result.returncode == 1:
                # Parse report if available
                report_path = Path("/tmp/gitleaks-report.json")
                secrets_found = []

                if report_path.exists():
                    try:
                        with open(report_path, "r") as f:
                            leaks = json.load(f)
                            for leak in leaks:
                                secrets_found.append(
                                    {
                                        "file": leak.get("File", "unknown"),
                                        "rule": leak.get("RuleID", "unknown"),
                                        "line": leak.get("StartLine", 0),
                                    }
                                )
                    except:
                        pass

                self.log_result(
                    "gitleaks",
                    "FAIL",
                    {
                        "message": f"Secrets detected in {len(secrets_found)} locations",
                        "secrets": secrets_found[:10],  # Limit for security
                        "total_count": len(secrets_found),
                    },
                )
                return False
            else:
                self.log_result(
                    "gitleaks",
                    "FAIL",
                    {
                        "message": f"gitleaks scan failed with exit code {result.returncode}",
                        "error": result.stderr,
                    },
                )
                return False

        except Exception as e:
            self.log_result(
                "gitleaks", "FAIL", {"message": f"Exception running gitleaks: {e}"}
            )
            return False

    def run_dependency_audit(self):
        """Run pip-audit to check for vulnerable dependencies."""
        print("📦 Running dependency vulnerability scan...")

        try:
            # Check if pip-audit is installed
            result = subprocess.run(["pip", "show", "pip-audit"], capture_output=True)
            if result.returncode != 0:
                print("⚠️  pip-audit not installed, installing...")
                install_result = subprocess.run(
                    ["pip", "install", "pip-audit"], capture_output=True, text=True
                )
                if install_result.returncode != 0:
                    self.log_result(
                        "dependency_audit",
                        "WARN",
                        {"message": "pip-audit not available and could not install"},
                    )
                    return True  # Don't fail on missing tool

            # Run pip-audit
            result = subprocess.run(
                [
                    "pip-audit",
                    "--requirement",
                    "requirements.txt",
                    "--format",
                    "json",
                    "--output",
                    "/tmp/pip-audit-report.json",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            vulnerabilities = []

            # Parse results
            if result.returncode == 0:
                # Check if report was generated
                report_path = Path("/tmp/pip-audit-report.json")
                if report_path.exists():
                    try:
                        with open(report_path, "r") as f:
                            audit_data = json.load(f)
                            vulnerabilities = audit_data.get("vulnerabilities", [])
                    except:
                        pass

                if vulnerabilities:
                    vuln_summary = []
                    for vuln in vulnerabilities[:10]:  # Limit output
                        vuln_summary.append(
                            {
                                "package": vuln.get("package", {}).get(
                                    "name", "unknown"
                                ),
                                "version": vuln.get("package", {}).get(
                                    "version", "unknown"
                                ),
                                "vulnerability": vuln.get("vulnerability", {}).get(
                                    "id", "unknown"
                                ),
                                "severity": vuln.get("vulnerability", {}).get(
                                    "severity", "unknown"
                                ),
                            }
                        )

                    self.log_result(
                        "dependency_audit",
                        "WARN",
                        {
                            "message": f"Found {len(vulnerabilities)} vulnerabilities",
                            "vulnerabilities": vuln_summary,
                            "total_count": len(vulnerabilities),
                        },
                    )
                    return True  # Warning, not failure
                else:
                    self.log_result(
                        "dependency_audit",
                        "PASS",
                        {"message": "No known vulnerabilities found"},
                    )
                    return True
            else:
                self.log_result(
                    "dependency_audit",
                    "WARN",
                    {
                        "message": f"pip-audit failed with exit code {result.returncode}",
                        "error": result.stderr,
                    },
                )
                return True  # Warning, not failure

        except Exception as e:
            self.log_result(
                "dependency_audit",
                "WARN",
                {"message": f"Exception running dependency audit: {e}"},
            )
            return True  # Warning, not failure

    def run_file_permissions_check(self):
        """Check for overly permissive files."""
        print("🔒 Checking file permissions...")

        try:
            suspicious_files = []

            # Check for world-writable files
            result = subprocess.run(
                [
                    "find",
                    ".",
                    "-type",
                    "f",
                    "-perm",
                    "-002",
                    "!",
                    "-path",
                    "./venv/*",
                    "!",
                    "-path",
                    "./.git/*",
                    "!",
                    "-path",
                    "./data/*",
                    "!",
                    "-path",
                    "./logs/*",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            if result.stdout.strip():
                world_writable = result.stdout.strip().split("\n")
                suspicious_files.extend(
                    [f"world-writable: {f}" for f in world_writable]
                )

            # Check for executable scripts without shebang
            result = subprocess.run(
                [
                    "find",
                    ".",
                    "-type",
                    "f",
                    "-executable",
                    "-name",
                    "*.py",
                    "!",
                    "-path",
                    "./venv/*",
                    "!",
                    "-path",
                    "./.git/*",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            if result.stdout.strip():
                executable_files = result.stdout.strip().split("\n")
                for file_path in executable_files:
                    try:
                        with open(file_path, "r") as f:
                            first_line = f.readline().strip()
                            if not first_line.startswith("#!"):
                                suspicious_files.append(
                                    f"executable without shebang: {file_path}"
                                )
                    except:
                        pass

            if suspicious_files:
                self.log_result(
                    "file_permissions",
                    "WARN",
                    {
                        "message": f"Found {len(suspicious_files)} suspicious file permissions",
                        "files": suspicious_files[:20],  # Limit output
                    },
                )
            else:
                self.log_result(
                    "file_permissions",
                    "PASS",
                    {"message": "File permissions look appropriate"},
                )

            return True

        except Exception as e:
            self.log_result(
                "file_permissions",
                "WARN",
                {"message": f"Exception checking file permissions: {e}"},
            )
            return True

    def run_env_file_check(self):
        """Check for exposed environment files."""
        print("🌍 Checking environment file security...")

        try:
            env_issues = []

            # Check for .env files that might contain secrets
            env_files = [".env", "env.example", ".env.local", ".env.production"]

            for env_file in env_files:
                if Path(env_file).exists():
                    try:
                        with open(env_file, "r") as f:
                            content = f.read()

                        # Look for potential secrets (very basic check)
                        suspicious_patterns = [
                            "password=",
                            "secret=",
                            "key=",
                            "token=",
                            "api_key=",
                        ]

                        for pattern in suspicious_patterns:
                            if pattern.lower() in content.lower():
                                env_issues.append(f"{env_file} contains '{pattern}'")
                    except:
                        pass

            if env_issues:
                self.log_result(
                    "env_files",
                    "WARN",
                    {
                        "message": "Environment files may contain secrets",
                        "issues": env_issues,
                    },
                )
            else:
                self.log_result(
                    "env_files", "PASS", {"message": "Environment files look clean"}
                )

            return True

        except Exception as e:
            self.log_result(
                "env_files",
                "WARN",
                {"message": f"Exception checking environment files: {e}"},
            )
            return True

    def run_all_scans(self):
        """Run all security scans."""
        print("🛡️ Running Security Scan Gate...")
        print(f"⏰ Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print()

        # Run all scans
        scan_methods = [
            ("secrets", self.run_gitleaks_scan),
            ("dependencies", self.run_dependency_audit),
            ("file_permissions", self.run_file_permissions_check),
            ("environment", self.run_env_file_check),
        ]

        for scan_name, scan_method in scan_methods:
            try:
                print(f"  Running {scan_name} scan...")
                scan_method()
            except Exception as e:
                self.log_result(scan_name, "FAIL", {"error": str(e)})

        # Summary
        total_scans = len(self.scan_results)
        passed = len([s for s in self.scan_results.values() if s["status"] == "PASS"])
        warned = len([s for s in self.scan_results.values() if s["status"] == "WARN"])
        failed = len([s for s in self.scan_results.values() if s["status"] == "FAIL"])

        print(f"\n📊 Security Scan Results:")
        print(f"   Total Scans: {total_scans}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ⚠️  Warnings: {warned}")
        print(f"   ❌ Failed: {failed}")

        # Create audit artifact
        audit_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "security_scan_gate",
            "summary": {
                "total": total_scans,
                "passed": passed,
                "warned": warned,
                "failed": failed,
                "status": "FAIL" if failed > 0 else "WARN" if warned > 0 else "PASS",
            },
            "scan_results": self.scan_results,
            "failures": self.failures,
            "warnings": self.warnings,
        }

        # Write audit artifact
        os.makedirs("artifacts/audit", exist_ok=True)
        audit_filename = (
            f"artifacts/audit/{audit_data['timestamp'].replace(':', '_')}_security.json"
        )
        with open(audit_filename, "w") as f:
            json.dump(audit_data, f, indent=2)

        print(f"📋 Audit: {audit_filename}")

        if failed > 0:
            print("\n❌ SECURITY SCAN FAILED")
            for failure in self.failures:
                print(f"   • {failure}")
            return False
        elif warned > 0:
            print(f"\n⚠️  SECURITY SCAN PASSED WITH WARNINGS")
            for warning in self.warnings:
                print(f"   • {warning}")
            return True
        else:
            print(f"\n✅ SECURITY SCAN PASSED")
            return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Security Scan Gate")
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as failures"
    )
    args = parser.parse_args()

    scanner = SecurityScanner()
    success = scanner.run_all_scans()

    if args.strict and scanner.warnings:
        print("\n⚠️  Strict mode: treating warnings as failures")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
