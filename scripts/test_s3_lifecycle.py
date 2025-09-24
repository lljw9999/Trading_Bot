#!/usr/bin/env python3
"""
S3 Lifecycle Manager Test Script

Tests S3 lifecycle and WORM retention functionality.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.s3_lifecycle_manager import S3LifecycleManager


def test_s3_lifecycle_manager():
    """Test S3 lifecycle and WORM manager functionality."""
    print("🧪 Testing S3 Lifecycle Manager")
    print("=" * 50)

    manager = S3LifecycleManager()

    # Test 1: Verify compliance (without AWS)
    print("\n🔍 Test 1: Compliance verification")
    compliance_results = manager.verify_compliance()

    overall_compliant = compliance_results.get("overall_compliant", False)
    buckets_checked = len(compliance_results.get("bucket_compliance", []))
    issues_found = len(compliance_results.get("issues_found", []))

    print(f"Buckets checked: {buckets_checked}")
    print(f"Overall compliant: {overall_compliant}")
    print(f"Issues found: {issues_found}")

    # Show bucket compliance status
    for bucket in compliance_results.get("bucket_compliance", []):
        bucket_name = bucket.get("bucket", "unknown")
        exists = bucket.get("exists", False)
        lifecycle_ok = bucket.get("lifecycle_compliant", False)
        versioning_ok = bucket.get("versioning_compliant", False)
        worm_ok = bucket.get("worm_compliant", False)

        status_icon = "✅" if (lifecycle_ok and versioning_ok and worm_ok) else "❌"
        print(
            f"  {status_icon} {bucket_name}: exists={exists}, lifecycle={lifecycle_ok}, versioning={versioning_ok}, worm={worm_ok}"
        )

    # Test 2: Generate compliance report
    print("\n📊 Test 2: Generate compliance report")
    report = manager.generate_compliance_report()

    summary = report.get("summary", {})
    total_buckets = summary.get("total_buckets", 0)
    compliant_buckets = summary.get("compliant_buckets", 0)
    total_issues = summary.get("total_issues", 0)

    print(f"Total buckets: {total_buckets}")
    print(f"Compliant buckets: {compliant_buckets}")
    print(f"Total issues: {total_issues}")

    # Show WORM retention policies
    worm_policies = report.get("worm_retention_policies", {})
    print(f"WORM retention policies:")
    for bucket, years in worm_policies.items():
        print(f"  {bucket}: {years} years")

    # Test 3: Lifecycle configuration validation
    print("\n🗂️ Test 3: Lifecycle configuration validation")
    lifecycle_config = manager.config["lifecycle_transitions"]
    print(f"Configured transitions: {len(lifecycle_config)}")

    for transition in lifecycle_config:
        days = transition["days"]
        storage_class = transition["storage_class"]
        print(f"  {days} days → {storage_class}")

    # Test 4: Configuration validation
    print("\n⚙️ Test 4: Configuration validation")
    bucket_configs = manager.config["buckets"]
    print(f"Configured buckets: {len(bucket_configs)}")

    for bucket_name, config in bucket_configs.items():
        retention_years = config.get("worm_retention_years", 0)
        lifecycle_enabled = config.get("lifecycle_enabled", False)
        versioning = config.get("versioning", False)
        print(
            f"  {bucket_name}: {retention_years}y retention, lifecycle={lifecycle_enabled}, versioning={versioning}"
        )

    # Summary
    print("\n" + "=" * 50)

    # Check if basic functionality works
    config_valid = len(bucket_configs) > 0 and len(lifecycle_config) > 0
    compliance_working = buckets_checked > 0
    report_working = "summary" in report

    if config_valid and compliance_working and report_working:
        print("✅ S3 LIFECYCLE MANAGER TEST: PASSED")
        print("   All core functionality working correctly")
    else:
        print("❌ S3 LIFECYCLE MANAGER TEST: FAILED")
        print(
            f"   Issues: config={config_valid}, compliance={compliance_working}, report={report_working}"
        )

    # Show compliance status
    if overall_compliant:
        print("🔒 S3 Compliance Status: COMPLIANT")
    else:
        print("⚠️ S3 Compliance Status: ISSUES FOUND")
        for issue in compliance_results.get("issues_found", [])[
            :3
        ]:  # Show first 3 issues
            print(f"   - {issue}")

    return config_valid and compliance_working and report_working


if __name__ == "__main__":
    try:
        success = test_s3_lifecycle_manager()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)
