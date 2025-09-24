#!/usr/bin/env python3
"""
Setup automated backup cron job
Configures nightly S3 backups using python-crontab
"""

import os
import logging
from crontab import CronTab

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("backup_cron")


def setup_backup_cron():
    """Setup automated backup cron job."""
    try:
        # Get current user's crontab
        cron = CronTab(user=True)

        # Path to backup script
        script_path = (
            "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D/scripts/backup_to_s3.py"
        )
        python_path = (
            "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D/venv/bin/python"
        )

        # Create backup job command
        command = f"{python_path} {script_path}"

        # Check if job already exists
        existing_jobs = list(cron.find_command(command))

        if existing_jobs:
            logger.info("⚠️ Backup cron job already exists")
            for job in existing_jobs:
                logger.info(f"   Schedule: {job}")
            return True

        # Create new cron job - daily at 3:00 AM
        job = cron.new(command=command, comment="Trading Bot S3 Backup")
        job.setall("0 3 * * *")  # Every day at 3:00 AM

        # Write crontab
        cron.write()

        logger.info("✅ Backup cron job created successfully")
        logger.info(f"   Command: {command}")
        logger.info("   Schedule: Daily at 3:00 AM")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to setup cron job: {e}")
        return False


def list_backup_cron_jobs():
    """List all backup-related cron jobs."""
    try:
        cron = CronTab(user=True)

        backup_jobs = []
        for job in cron:
            if "backup_to_s3.py" in job.command or "Trading Bot S3 Backup" in str(
                job.comment
            ):
                backup_jobs.append(job)

        if backup_jobs:
            logger.info(f"📋 Found {len(backup_jobs)} backup cron jobs:")
            for job in backup_jobs:
                logger.info(f"   {job}")
        else:
            logger.info("No backup cron jobs found")

        return backup_jobs

    except Exception as e:
        logger.error(f"❌ Error listing cron jobs: {e}")
        return []


def remove_backup_cron_jobs():
    """Remove backup cron jobs."""
    try:
        cron = CronTab(user=True)

        removed_count = 0
        for job in list(cron):
            if "backup_to_s3.py" in job.command or "Trading Bot S3 Backup" in str(
                job.comment
            ):
                cron.remove(job)
                removed_count += 1

        if removed_count > 0:
            cron.write()
            logger.info(f"✅ Removed {removed_count} backup cron jobs")
        else:
            logger.info("No backup cron jobs to remove")

        return removed_count > 0

    except Exception as e:
        logger.error(f"❌ Error removing cron jobs: {e}")
        return False


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage backup cron jobs")
    parser.add_argument("--setup", action="store_true", help="Setup backup cron job")
    parser.add_argument("--list", action="store_true", help="List backup cron jobs")
    parser.add_argument("--remove", action="store_true", help="Remove backup cron jobs")

    args = parser.parse_args()

    if args.setup:
        success = setup_backup_cron()
        exit(0 if success else 1)
    elif args.list:
        list_backup_cron_jobs()
    elif args.remove:
        success = remove_backup_cron_jobs()
        exit(0 if success else 1)
    else:
        # Default: setup cron job
        logger.info("Setting up backup cron job...")
        success = setup_backup_cron()
        exit(0 if success else 1)


if __name__ == "__main__":
    main()
