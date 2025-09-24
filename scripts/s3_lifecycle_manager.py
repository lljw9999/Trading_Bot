#!/usr/bin/env python3
"""
S3 Lifecycle and WORM Retention Manager

Implements S3 data lifecycle and retention policies:
- Lifecycle rules: warm 30d → infrequent access 90d → glacier deep archive 1y
- Enforce retention lock matching WORM policy (3-10 years)
- Compliance monitoring and audit trails
- Automated policy application across all buckets
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import boto3
    import redis

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger("s3_lifecycle_manager")


class S3LifecycleManager:
    """
    Manages S3 lifecycle policies and WORM retention compliance.
    Ensures proper data archival and regulatory compliance.
    """

    def __init__(self):
        """Initialize S3 lifecycle manager."""
        self.redis_client = None
        if DEPS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        # AWS clients
        self.aws_clients = {}
        if DEPS_AVAILABLE:
            try:
                self.aws_clients = {
                    "s3": boto3.client("s3"),
                    "s3control": boto3.client("s3control"),
                    "sts": boto3.client("sts"),
                }
            except Exception as e:
                logger.warning(f"AWS clients unavailable: {e}")

        # Lifecycle configuration
        self.config = {
            "buckets": {
                "trading-system-data": {
                    "worm_retention_years": 7,  # Trading data: 7 years
                    "lifecycle_enabled": True,
                    "versioning": True,
                },
                "trading-system-models": {
                    "worm_retention_years": 3,  # Model artifacts: 3 years
                    "lifecycle_enabled": True,
                    "versioning": True,
                },
                "trading-system-backups": {
                    "worm_retention_years": 5,  # Backup data: 5 years
                    "lifecycle_enabled": True,
                    "versioning": True,
                },
                "trading-system-compliance": {
                    "worm_retention_years": 10,  # Compliance data: 10 years
                    "lifecycle_enabled": True,
                    "versioning": True,
                },
            },
            "lifecycle_transitions": [
                {
                    "days": 30,
                    "storage_class": "STANDARD_IA",
                    "description": "Transition to Infrequent Access after 30 days",
                },
                {
                    "days": 90,
                    "storage_class": "GLACIER",
                    "description": "Transition to Glacier after 90 days",
                },
                {
                    "days": 365,
                    "storage_class": "DEEP_ARCHIVE",
                    "description": "Transition to Deep Archive after 1 year",
                },
            ],
            "object_lock_configuration": {
                "ObjectLockEnabled": "Enabled",
                "Rule": {
                    "DefaultRetention": {
                        "Mode": "COMPLIANCE",  # Cannot be overridden
                        "Days": None,  # Set per bucket based on WORM policy
                    }
                },
            },
        }

        logger.info("Initialized S3 lifecycle manager")

    def apply_lifecycle_policies(self) -> Dict[str, any]:
        """
        Apply lifecycle policies to all configured buckets.

        Returns:
            Lifecycle policy application results
        """
        try:
            logger.info("🗂️ Applying S3 lifecycle policies")

            results = {
                "timestamp": datetime.now().isoformat(),
                "buckets_processed": [],
                "overall_success": True,
            }

            if not self.aws_clients.get("s3"):
                return {"error": "S3 client unavailable", "overall_success": False}

            s3_client = self.aws_clients["s3"]

            for bucket_name, bucket_config in self.config["buckets"].items():
                bucket_result = {
                    "bucket": bucket_name,
                    "lifecycle_applied": False,
                    "versioning_enabled": False,
                    "object_lock_enabled": False,
                    "errors": [],
                }

                try:
                    # Check if bucket exists
                    try:
                        s3_client.head_bucket(Bucket=bucket_name)
                    except:
                        logger.info(f"Bucket {bucket_name} does not exist, creating...")
                        self._create_bucket_with_worm(bucket_name, bucket_config)

                    # Enable versioning (required for object lock)
                    if bucket_config.get("versioning", True):
                        versioning_result = self._enable_bucket_versioning(bucket_name)
                        bucket_result["versioning_enabled"] = versioning_result.get(
                            "success", False
                        )
                        if not versioning_result.get("success", False):
                            bucket_result["errors"].append(
                                f"Versioning failed: {versioning_result.get('error', 'Unknown')}"
                            )

                    # Apply lifecycle policy
                    if bucket_config.get("lifecycle_enabled", True):
                        lifecycle_result = self._apply_bucket_lifecycle_policy(
                            bucket_name
                        )
                        bucket_result["lifecycle_applied"] = lifecycle_result.get(
                            "success", False
                        )
                        if not lifecycle_result.get("success", False):
                            bucket_result["errors"].append(
                                f"Lifecycle failed: {lifecycle_result.get('error', 'Unknown')}"
                            )

                    # Apply WORM retention (Object Lock)
                    worm_result = self._apply_worm_retention(bucket_name, bucket_config)
                    bucket_result["object_lock_enabled"] = worm_result.get(
                        "success", False
                    )
                    if not worm_result.get("success", False):
                        bucket_result["errors"].append(
                            f"WORM failed: {worm_result.get('error', 'Unknown')}"
                        )

                    if bucket_result["errors"]:
                        results["overall_success"] = False

                except Exception as e:
                    bucket_result["errors"].append(str(e))
                    results["overall_success"] = False

                results["buckets_processed"].append(bucket_result)

            # Store results in Redis
            if self.redis_client:
                self.redis_client.set("s3:last_lifecycle_update", json.dumps(results))
                self.redis_client.expire(
                    "s3:last_lifecycle_update", 30 * 86400
                )  # 30 days

            logger.info(
                f"✅ Lifecycle policies applied to {len(results['buckets_processed'])} buckets"
            )
            return results

        except Exception as e:
            logger.error(f"Error applying lifecycle policies: {e}")
            return {"error": str(e), "overall_success": False}

    def _create_bucket_with_worm(
        self, bucket_name: str, bucket_config: Dict
    ) -> Dict[str, any]:
        """Create bucket with WORM (Object Lock) enabled."""
        try:
            s3_client = self.aws_clients["s3"]

            # Create bucket with Object Lock enabled
            create_params = {"Bucket": bucket_name, "ObjectLockEnabledForBucket": True}

            # Add region if needed (for regions other than us-east-1)
            try:
                region = s3_client.meta.region_name
                if region and region != "us-east-1":
                    create_params["CreateBucketConfiguration"] = {
                        "LocationConstraint": region
                    }
            except:
                pass

            s3_client.create_bucket(**create_params)
            logger.info(f"✅ Created bucket {bucket_name} with Object Lock enabled")

            return {"success": True}

        except Exception as e:
            logger.error(f"Error creating bucket {bucket_name}: {e}")
            return {"success": False, "error": str(e)}

    def _enable_bucket_versioning(self, bucket_name: str) -> Dict[str, any]:
        """Enable versioning on bucket."""
        try:
            s3_client = self.aws_clients["s3"]

            s3_client.put_bucket_versioning(
                Bucket=bucket_name, VersioningConfiguration={"Status": "Enabled"}
            )

            logger.info(f"✅ Enabled versioning for bucket {bucket_name}")
            return {"success": True}

        except Exception as e:
            logger.error(f"Error enabling versioning for {bucket_name}: {e}")
            return {"success": False, "error": str(e)}

    def _apply_bucket_lifecycle_policy(self, bucket_name: str) -> Dict[str, any]:
        """Apply lifecycle policy to bucket."""
        try:
            s3_client = self.aws_clients["s3"]

            # Build lifecycle rules
            lifecycle_rules = []

            # Current version transitions
            transitions = []
            for transition in self.config["lifecycle_transitions"]:
                transitions.append(
                    {
                        "Days": transition["days"],
                        "StorageClass": transition["storage_class"],
                    }
                )

            lifecycle_rules.append(
                {
                    "ID": f"{bucket_name}-lifecycle-policy",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},  # Apply to all objects
                    "Transitions": transitions,
                }
            )

            # Non-current version transitions (for versioned objects)
            noncurrent_transitions = []
            for transition in self.config["lifecycle_transitions"]:
                noncurrent_transitions.append(
                    {
                        "NoncurrentDays": transition["days"]
                        + 30,  # Keep non-current versions longer
                        "StorageClass": transition["storage_class"],
                    }
                )

            lifecycle_rules.append(
                {
                    "ID": f"{bucket_name}-noncurrent-versions",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},
                    "NoncurrentVersionTransitions": noncurrent_transitions,
                    "NoncurrentVersionExpiration": {
                        "NoncurrentDays": 2555  # ~7 years for non-current versions
                    },
                }
            )

            # Apply incomplete multipart upload cleanup
            lifecycle_rules.append(
                {
                    "ID": f"{bucket_name}-incomplete-uploads",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},
                    "AbortIncompleteMultipartUpload": {
                        "DaysAfterInitiation": 7  # Clean up after 7 days
                    },
                }
            )

            # Apply lifecycle configuration
            s3_client.put_bucket_lifecycle_configuration(
                Bucket=bucket_name, LifecycleConfiguration={"Rules": lifecycle_rules}
            )

            logger.info(f"✅ Applied lifecycle policy to bucket {bucket_name}")
            return {"success": True, "rules_applied": len(lifecycle_rules)}

        except Exception as e:
            logger.error(f"Error applying lifecycle policy to {bucket_name}: {e}")
            return {"success": False, "error": str(e)}

    def _apply_worm_retention(
        self, bucket_name: str, bucket_config: Dict
    ) -> Dict[str, any]:
        """Apply WORM retention policy to bucket."""
        try:
            s3_client = self.aws_clients["s3"]

            retention_years = bucket_config.get("worm_retention_years", 7)
            retention_days = retention_years * 365

            # Configure default retention
            object_lock_config = {
                "ObjectLockEnabled": "Enabled",
                "Rule": {
                    "DefaultRetention": {
                        "Mode": "COMPLIANCE",  # COMPLIANCE mode cannot be overridden
                        "Days": retention_days,
                    }
                },
            }

            s3_client.put_object_lock_configuration(
                Bucket=bucket_name, ObjectLockConfiguration=object_lock_config
            )

            logger.info(
                f"✅ Applied WORM retention ({retention_years} years) to bucket {bucket_name}"
            )
            return {"success": True, "retention_years": retention_years}

        except Exception as e:
            logger.error(f"Error applying WORM retention to {bucket_name}: {e}")
            return {"success": False, "error": str(e)}

    def verify_compliance(self) -> Dict[str, any]:
        """
        Verify S3 lifecycle and WORM compliance.

        Returns:
            Compliance verification results
        """
        try:
            logger.info("🔍 Verifying S3 lifecycle and WORM compliance")

            compliance_results = {
                "timestamp": datetime.now().isoformat(),
                "bucket_compliance": [],
                "overall_compliant": True,
                "issues_found": [],
            }

            if not self.aws_clients.get("s3"):
                return {"error": "S3 client unavailable", "overall_compliant": False}

            s3_client = self.aws_clients["s3"]

            for bucket_name, bucket_config in self.config["buckets"].items():
                bucket_compliance = {
                    "bucket": bucket_name,
                    "exists": False,
                    "lifecycle_compliant": False,
                    "versioning_compliant": False,
                    "worm_compliant": False,
                    "issues": [],
                }

                try:
                    # Check bucket existence
                    try:
                        s3_client.head_bucket(Bucket=bucket_name)
                        bucket_compliance["exists"] = True
                    except:
                        bucket_compliance["issues"].append("Bucket does not exist")
                        compliance_results["issues_found"].append(
                            f"{bucket_name}: Bucket does not exist"
                        )
                        compliance_results["overall_compliant"] = False
                        compliance_results["bucket_compliance"].append(
                            bucket_compliance
                        )
                        continue

                    # Check lifecycle policy
                    lifecycle_compliant = self._verify_bucket_lifecycle(bucket_name)
                    bucket_compliance["lifecycle_compliant"] = lifecycle_compliant.get(
                        "compliant", False
                    )
                    if not lifecycle_compliant.get("compliant", False):
                        issues = lifecycle_compliant.get(
                            "issues", ["Lifecycle policy issues"]
                        )
                        bucket_compliance["issues"].extend(issues)
                        compliance_results["issues_found"].extend(
                            [f"{bucket_name}: {issue}" for issue in issues]
                        )
                        compliance_results["overall_compliant"] = False

                    # Check versioning
                    versioning_compliant = self._verify_bucket_versioning(bucket_name)
                    bucket_compliance["versioning_compliant"] = (
                        versioning_compliant.get("compliant", False)
                    )
                    if not versioning_compliant.get("compliant", False):
                        issue = versioning_compliant.get(
                            "issue", "Versioning not enabled"
                        )
                        bucket_compliance["issues"].append(issue)
                        compliance_results["issues_found"].append(
                            f"{bucket_name}: {issue}"
                        )
                        compliance_results["overall_compliant"] = False

                    # Check WORM/Object Lock
                    worm_compliant = self._verify_bucket_worm(
                        bucket_name, bucket_config
                    )
                    bucket_compliance["worm_compliant"] = worm_compliant.get(
                        "compliant", False
                    )
                    if not worm_compliant.get("compliant", False):
                        issues = worm_compliant.get("issues", ["WORM retention issues"])
                        bucket_compliance["issues"].extend(issues)
                        compliance_results["issues_found"].extend(
                            [f"{bucket_name}: {issue}" for issue in issues]
                        )
                        compliance_results["overall_compliant"] = False

                except Exception as e:
                    bucket_compliance["issues"].append(f"Verification error: {e}")
                    compliance_results["issues_found"].append(
                        f"{bucket_name}: Verification error: {e}"
                    )
                    compliance_results["overall_compliant"] = False

                compliance_results["bucket_compliance"].append(bucket_compliance)

            # Store compliance results
            if self.redis_client:
                self.redis_client.set(
                    "s3:last_compliance_check", json.dumps(compliance_results)
                )
                self.redis_client.expire(
                    "s3:last_compliance_check", 7 * 86400
                )  # 7 days

            status = (
                "✅ COMPLIANT"
                if compliance_results["overall_compliant"]
                else "❌ NON-COMPLIANT"
            )
            logger.info(f"Compliance verification complete: {status}")

            return compliance_results

        except Exception as e:
            logger.error(f"Error verifying compliance: {e}")
            return {"error": str(e), "overall_compliant": False}

    def _verify_bucket_lifecycle(self, bucket_name: str) -> Dict[str, any]:
        """Verify bucket lifecycle policy compliance."""
        try:
            s3_client = self.aws_clients["s3"]

            # Get lifecycle configuration
            try:
                response = s3_client.get_bucket_lifecycle_configuration(
                    Bucket=bucket_name
                )
                rules = response.get("Rules", [])
            except s3_client.exceptions.NoSuchLifecycleConfiguration:
                return {
                    "compliant": False,
                    "issues": ["No lifecycle configuration found"],
                }

            # Verify required transitions exist
            transitions_found = []
            for rule in rules:
                for transition in rule.get("Transitions", []):
                    transitions_found.append(
                        {
                            "days": transition.get("Days"),
                            "storage_class": transition.get("StorageClass"),
                        }
                    )

            issues = []
            for required_transition in self.config["lifecycle_transitions"]:
                found = any(
                    t["days"] == required_transition["days"]
                    and t["storage_class"] == required_transition["storage_class"]
                    for t in transitions_found
                )
                if not found:
                    issues.append(
                        f"Missing transition: {required_transition['days']} days to {required_transition['storage_class']}"
                    )

            return {"compliant": len(issues) == 0, "issues": issues}

        except Exception as e:
            return {"compliant": False, "issues": [str(e)]}

    def _verify_bucket_versioning(self, bucket_name: str) -> Dict[str, any]:
        """Verify bucket versioning compliance."""
        try:
            s3_client = self.aws_clients["s3"]

            response = s3_client.get_bucket_versioning(Bucket=bucket_name)
            status = response.get("Status", "")

            if status == "Enabled":
                return {"compliant": True}
            else:
                return {
                    "compliant": False,
                    "issue": f"Versioning status: {status} (should be Enabled)",
                }

        except Exception as e:
            return {"compliant": False, "issue": str(e)}

    def _verify_bucket_worm(
        self, bucket_name: str, bucket_config: Dict
    ) -> Dict[str, any]:
        """Verify bucket WORM/Object Lock compliance."""
        try:
            s3_client = self.aws_clients["s3"]

            # Check Object Lock configuration
            try:
                response = s3_client.get_object_lock_configuration(Bucket=bucket_name)
                object_lock_config = response.get("ObjectLockConfiguration", {})
            except s3_client.exceptions.NoSuchObjectLockConfiguration:
                return {
                    "compliant": False,
                    "issues": ["No Object Lock configuration found"],
                }

            issues = []

            # Verify Object Lock is enabled
            if object_lock_config.get("ObjectLockEnabled") != "Enabled":
                issues.append("Object Lock not enabled")

            # Verify default retention
            rule = object_lock_config.get("Rule", {})
            default_retention = rule.get("DefaultRetention", {})

            expected_retention_days = bucket_config.get("worm_retention_years", 7) * 365
            actual_retention_days = default_retention.get("Days")

            if default_retention.get("Mode") != "COMPLIANCE":
                issues.append(
                    f"Retention mode: {default_retention.get('Mode')} (should be COMPLIANCE)"
                )

            if actual_retention_days != expected_retention_days:
                issues.append(
                    f"Retention days: {actual_retention_days} (should be {expected_retention_days})"
                )

            return {"compliant": len(issues) == 0, "issues": issues}

        except Exception as e:
            return {"compliant": False, "issues": [str(e)]}

    def generate_compliance_report(self) -> Dict[str, any]:
        """Generate comprehensive compliance report."""
        try:
            logger.info("📊 Generating S3 compliance report")

            # Run compliance verification
            compliance_results = self.verify_compliance()

            # Add additional metrics
            report = {
                "timestamp": datetime.now().isoformat(),
                "report_type": "s3_lifecycle_worm_compliance",
                "summary": {
                    "total_buckets": len(self.config["buckets"]),
                    "compliant_buckets": 0,
                    "non_compliant_buckets": 0,
                    "total_issues": len(compliance_results.get("issues_found", [])),
                    "overall_compliant": compliance_results.get(
                        "overall_compliant", False
                    ),
                },
                "bucket_details": compliance_results.get("bucket_compliance", []),
                "issues_found": compliance_results.get("issues_found", []),
                "lifecycle_policy": self.config["lifecycle_transitions"],
                "worm_retention_policies": {
                    bucket: config.get("worm_retention_years", 7)
                    for bucket, config in self.config["buckets"].items()
                },
            }

            # Calculate summary statistics
            for bucket_compliance in report["bucket_details"]:
                if (
                    bucket_compliance.get("lifecycle_compliant", False)
                    and bucket_compliance.get("versioning_compliant", False)
                    and bucket_compliance.get("worm_compliant", False)
                ):
                    report["summary"]["compliant_buckets"] += 1
                else:
                    report["summary"]["non_compliant_buckets"] += 1

            return report

        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {"error": str(e)}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="S3 Lifecycle and WORM Manager")

    parser.add_argument(
        "--action",
        choices=["apply-policies", "verify-compliance", "generate-report"],
        default="verify-compliance",
        help="Action to perform",
    )
    parser.add_argument("--output", type=str, help="Output file for results")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info("🗂️ Starting S3 Lifecycle and WORM Manager")

    try:
        manager = S3LifecycleManager()

        if args.action == "apply-policies":
            results = manager.apply_lifecycle_policies()
            print(f"\n🗂️ LIFECYCLE POLICY APPLICATION:")
            print(json.dumps(results, indent=2))

        elif args.action == "verify-compliance":
            results = manager.verify_compliance()
            print(f"\n🔍 COMPLIANCE VERIFICATION:")
            print(json.dumps(results, indent=2))

        elif args.action == "generate-report":
            results = manager.generate_compliance_report()
            print(f"\n📊 COMPLIANCE REPORT:")
            print(json.dumps(results, indent=2))

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        # Return appropriate exit code
        success_key = (
            "overall_success" if "overall_success" in results else "overall_compliant"
        )
        return 0 if results.get(success_key, False) else 1

    except Exception as e:
        logger.error(f"Error in S3 lifecycle manager: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
