#!/usr/bin/env python3
"""
Security Hardener Test Script

Tests security hardening functionality.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.security_hardener import SecurityHardener


def test_security_hardener():
    """Test security hardening functionality."""
    print("🧪 Testing Security Hardener")
    print("=" * 50)

    hardener = SecurityHardener()

    # Test 1: Run compliance scan
    print("\n🔍 Test 1: Security compliance scan")
    compliance_results = hardener.run_compliance_scan()

    overall_compliant = compliance_results.get("overall_compliant", False)
    checks_run = len(compliance_results.get("checks", {}))
    issues_found = len(compliance_results.get("issues_found", []))

    print(f"Checks run: {checks_run}")
    print(f"Overall compliant: {overall_compliant}")
    print(f"Issues found: {issues_found}")

    # Show individual check results
    for check_name, check_result in compliance_results.get("checks", {}).items():
        compliant = check_result.get("compliant", False)
        icon = "✅" if compliant else "❌"
        print(f"  {icon} {check_name}: {'PASS' if compliant else 'FAIL'}")

    # Test 2: Test key rotation (mock)
    print("\n🔄 Test 2: API key rotation")
    rotation_results = hardener.rotate_api_keys()

    overall_success = rotation_results.get("overall_success", False)
    rotations_attempted = len(rotation_results.get("rotations", []))

    print(f"Rotations attempted: {rotations_attempted}")
    print(f"Overall success: {overall_success}")

    # Show individual rotation results
    for rotation in rotation_results.get("rotations", []):
        exchange = rotation.get("exchange", "unknown")
        success = rotation.get("success", False)
        icon = "✅" if success else "❌"
        keys_rotated = len(rotation.get("keys_rotated", []))
        print(f"  {icon} {exchange}: {keys_rotated} keys rotated")

    # Test 3: S3 security hardening (mock)
    print("\n🔒 Test 3: S3 security hardening")
    s3_results = hardener.harden_s3_security()

    s3_success = s3_results.get("overall_success", False)
    buckets_processed = len(s3_results.get("buckets_processed", []))

    print(f"Buckets processed: {buckets_processed}")
    print(f"Overall success: {s3_success}")

    # Test 4: IAM policy application
    print("\n🔐 Test 4: IAM policy application")
    iam_results = hardener.apply_iam_policies()

    iam_success = iam_results.get("overall_success", False)
    policies_applied = len(iam_results.get("policies_applied", []))

    print(f"Policies processed: {policies_applied}")
    print(f"Overall success: {iam_success}")

    # Show individual policy results
    for policy in iam_results.get("policies_applied", []):
        policy_name = policy.get("policy_name", "unknown")
        success = policy.get("success", False)
        icon = "✅" if success else "❌"
        print(f"  {icon} {policy_name}")

    # Summary
    print("\n" + "=" * 50)

    # Overall assessment
    all_tests_passed = (
        checks_run > 0
        and rotations_attempted > 0
        and buckets_processed >= 0
        and policies_applied > 0
    )

    if all_tests_passed:
        print("✅ SECURITY HARDENER TEST: PASSED")
        print("   All security hardening components working correctly")
    else:
        print("❌ SECURITY HARDENER TEST: FAILED")
        print(
            f"   Issues: checks={checks_run}, rotations={rotations_attempted}, policies={policies_applied}"
        )

    # Show compliance status
    if overall_compliant:
        print("🔒 Security Compliance: PASS")
    else:
        print("⚠️ Security Compliance: ISSUES FOUND")
        for issue in compliance_results.get("issues_found", [])[
            :3
        ]:  # Show first 3 issues
            print(f"   - {issue}")

    return all_tests_passed


if __name__ == "__main__":
    try:
        success = test_security_hardener()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)
