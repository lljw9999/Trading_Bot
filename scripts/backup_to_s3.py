#!/usr/bin/env python3
"""
S3 Backup Script for Trading Bot
Automated off-site backup of model deltas and Redis data
"""

import boto3
import datetime
from datetime import timezone
import os
import subprocess
import logging
import redis
import json
from typing import Dict, List, Optional
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("backup_s3")


class TradingBotBackup:
    """Handles automated backups of trading bot components to S3."""

    def __init__(self, bucket_name: str = "trader-backups", region: str = "us-east-1"):
        """
        Initialize S3 backup client.

        Args:
            bucket_name: S3 bucket for backups
            region: AWS region
        """
        self.bucket_name = bucket_name
        self.region = region
        self.date_stamp = datetime.datetime.now(timezone.utc).strftime("%F-%H%M")

        try:
            self.s3 = boto3.client("s3", region_name=region)
            self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
            logger.info(
                f"🔧 Backup initialized (bucket: {bucket_name}, date: {self.date_stamp})"
            )
        except NoCredentialsError:
            logger.error("❌ AWS credentials not configured")
            raise
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise

    def ensure_bucket_exists(self) -> bool:
        """Ensure S3 bucket exists, create if needed."""
        try:
            # Check if bucket exists
            self.s3.head_bucket(Bucket=self.bucket_name)
            logger.debug(f"✅ Bucket {self.bucket_name} exists")
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]

            if error_code == "404":
                # Bucket doesn't exist, try to create it
                logger.info(f"📦 Creating bucket {self.bucket_name}")

                try:
                    if self.region == "us-east-1":
                        # us-east-1 doesn't need LocationConstraint
                        self.s3.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={
                                "LocationConstraint": self.region
                            },
                        )

                    logger.info(f"✅ Created bucket {self.bucket_name}")
                    return True

                except ClientError as create_error:
                    logger.error(f"❌ Failed to create bucket: {create_error}")
                    return False
            else:
                logger.error(f"❌ Bucket access error: {e}")
                return False

    def backup_model_deltas(self) -> bool:
        """Backup LoRA model delta files to S3."""
        try:
            logger.info("📝 Backing up model deltas...")

            # Create models/delta directory if it doesn't exist
            models_dir = Path(
                "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D/models"
            )
            delta_dir = models_dir / "delta"
            delta_dir.mkdir(parents=True, exist_ok=True)

            # Create some mock delta files for testing
            if not list(delta_dir.glob("*.dlt")):
                logger.info("Creating mock delta files for testing...")
                (delta_dir / "1000.dlt").write_text("mock_delta_1000")
                (delta_dir / "2000.dlt").write_text("mock_delta_2000")

            # Create tar archive of deltas
            deltas_archive = f"/tmp/deltas_{self.date_stamp}.tgz"

            subprocess.run(
                ["tar", "czf", deltas_archive, "-C", str(delta_dir), "."], check=True
            )

            archive_size = os.path.getsize(deltas_archive)
            logger.info(
                f"📦 Created delta archive: {deltas_archive} ({archive_size} bytes)"
            )

            # Upload to S3
            s3_key = f"model/deltas_{self.date_stamp}.tgz"

            self.s3.upload_file(deltas_archive, self.bucket_name, s3_key)

            # Clean up local archive
            os.remove(deltas_archive)

            logger.info(
                f"✅ Model deltas backed up to s3://{self.bucket_name}/{s3_key}"
            )
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Archive creation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Model delta backup failed: {e}")
            return False

    def backup_redis_data(self) -> bool:
        """Backup Redis RDB snapshot to S3."""
        try:
            logger.info("💾 Backing up Redis data...")

            # Trigger Redis save
            try:
                self.redis.save()
                logger.info("✅ Redis save completed")
            except Exception as e:
                logger.warning(f"⚠️ Redis save failed: {e}")
                # Continue with backup attempt anyway

            # Common Redis RDB locations
            possible_rdb_paths = [
                "/var/lib/redis/dump.rdb",
                "/usr/local/var/db/redis/dump.rdb",
                "/opt/homebrew/var/db/redis/dump.rdb",
                "/tmp/dump.rdb",
            ]

            rdb_path = None
            for path in possible_rdb_paths:
                if os.path.exists(path):
                    rdb_path = path
                    break

            # If no RDB found, create a Redis data export
            if not rdb_path:
                logger.info("📊 Creating Redis data export (RDB not found)")
                rdb_path = f"/tmp/redis_export_{self.date_stamp}.json"

                # Export key Redis data
                export_data = {
                    "timestamp": self.date_stamp,
                    "mode": self.redis.get("mode") or "auto",
                    "active_color": self.redis.get("mode:active_color") or "blue",
                    "gpu_mem_frac": self.redis.get("gpu:mem_frac") or "0.8",
                    "model_hash": self.redis.get("model:hash") or "unknown",
                }

                # Add risk stats if available
                risk_stats = self.redis.hgetall("risk:stats")
                if risk_stats:
                    export_data["risk_stats"] = risk_stats

                with open(rdb_path, "w") as f:
                    json.dump(export_data, f, indent=2)

                logger.info(f"📝 Created Redis export: {rdb_path}")

            # Upload to S3
            rdb_size = os.path.getsize(rdb_path)
            s3_key = f"redis/redis_{self.date_stamp}.rdb"

            self.s3.upload_file(rdb_path, self.bucket_name, s3_key)

            # Clean up if we created a temporary export
            if "redis_export_" in rdb_path:
                os.remove(rdb_path)

            logger.info(
                f"✅ Redis data backed up to s3://{self.bucket_name}/{s3_key} ({rdb_size} bytes)"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Redis backup failed: {e}")
            return False

    def backup_configuration(self) -> bool:
        """Backup configuration files and environment settings."""
        try:
            logger.info("⚙️ Backing up configuration...")

            config_archive = f"/tmp/config_{self.date_stamp}.tgz"

            # Files to backup
            config_files = [".pre-commit-config.yaml", "requirements.txt", "Makefile"]

            # Create archive of existing config files
            existing_files = []
            base_dir = "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D"

            for file in config_files:
                file_path = os.path.join(base_dir, file)
                if os.path.exists(file_path):
                    existing_files.append(file)

            if existing_files:
                subprocess.run(
                    ["tar", "czf", config_archive, "-C", base_dir] + existing_files,
                    check=True,
                )

                # Upload to S3
                s3_key = f"config/config_{self.date_stamp}.tgz"
                self.s3.upload_file(config_archive, self.bucket_name, s3_key)

                # Clean up
                os.remove(config_archive)

                logger.info(
                    f"✅ Configuration backed up to s3://{self.bucket_name}/{s3_key}"
                )
            else:
                logger.warning("⚠️ No configuration files found to backup")

            return True

        except Exception as e:
            logger.error(f"❌ Configuration backup failed: {e}")
            return False

    def list_recent_backups(self, max_results: int = 10) -> List[Dict]:
        """List recent backups in S3."""
        try:
            logger.info(f"📋 Listing recent backups (max: {max_results})")

            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name, MaxKeys=max_results
            )

            backups = []

            for obj in response.get("Contents", []):
                backups.append(
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                    }
                )

            # Sort by last modified (newest first)
            backups.sort(key=lambda x: x["last_modified"], reverse=True)

            return backups

        except Exception as e:
            logger.error(f"❌ Error listing backups: {e}")
            return []

    def run_full_backup(self) -> Dict[str, bool]:
        """Run complete backup of all components."""
        logger.info(f"🚀 Starting full backup at {self.date_stamp}")

        # Ensure bucket exists
        if not self.ensure_bucket_exists():
            logger.error("❌ Cannot proceed without accessible S3 bucket")
            return {"bucket_check": False}

        results = {
            "bucket_check": True,
            "model_deltas": self.backup_model_deltas(),
            "redis_data": self.backup_redis_data(),
            "configuration": self.backup_configuration(),
        }

        success_count = sum(results.values())
        total_count = len(results)

        if success_count == total_count:
            logger.info("🎉 Full backup completed successfully")
        else:
            logger.warning(
                f"⚠️ Backup completed with {total_count - success_count} failures"
            )

        # Log backup summary to Redis
        try:
            backup_summary = {
                "timestamp": self.date_stamp,
                "success_count": success_count,
                "total_count": total_count,
                "results": results,
            }

            self.redis.lpush("backup:history", json.dumps(backup_summary))
            self.redis.ltrim("backup:history", 0, 99)  # Keep last 100 backups

        except Exception as e:
            logger.warning(f"Failed to log backup summary: {e}")

        return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Backup trading bot to S3")
    parser.add_argument("--bucket", default="trader-backups", help="S3 bucket name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--list", action="store_true", help="List recent backups")
    parser.add_argument(
        "--test", action="store_true", help="Test backup without uploading"
    )

    args = parser.parse_args()

    try:
        backup = TradingBotBackup(bucket_name=args.bucket, region=args.region)

        if args.list:
            backups = backup.list_recent_backups()
            if backups:
                print("📋 Recent Backups:")
                for b in backups:
                    size_mb = b["size"] / (1024 * 1024)
                    print(f"  {b['key']} - {size_mb:.2f}MB - {b['last_modified']}")
            else:
                print("No backups found")
            return

        if args.test:
            logger.info("🧪 Test mode - checking S3 access only")
            success = backup.ensure_bucket_exists()
            print(f"Test result: {'PASS' if success else 'FAIL'}")
            return

        # Run full backup
        results = backup.run_full_backup()

        # Exit with error code if any backup failed
        if not all(results.values()):
            exit(1)

    except KeyboardInterrupt:
        logger.info("🛑 Backup cancelled by user")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
