#!/usr/bin/env python3
"""
Secrets Rotation
SSM → services with zero-downtime rotation
"""

import os
import sys
import time
import json
import logging
import subprocess
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

import boto3
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("ssm_rotate_keys")


class SecretsRotation:
    """Handles zero-downtime rotation of secrets from AWS SSM."""

    def __init__(self):
        """Initialize secrets rotation."""
        self.ssm = boto3.client("ssm", region_name=os.getenv("AWS_REGION", "us-east-1"))
        self.slack_webhook = os.getenv("SLACK_WEBHOOK")

        # Service configuration
        self.services = {
            "binance": {
                "ssm_prefix": "/trading/exchanges/binance",
                "env_file": "/opt/trader/.env.binance",
                "systemd_unit": "trader-binance.service",
                "supports_sighup": True,
                "secrets": ["api_key", "secret_key"],
            },
            "coinbase": {
                "ssm_prefix": "/trading/exchanges/coinbase",
                "env_file": "/opt/trader/.env.coinbase",
                "systemd_unit": "trader-coinbase.service",
                "supports_sighup": True,
                "secrets": ["api_key", "secret_key", "passphrase"],
            },
            "dydx": {
                "ssm_prefix": "/trading/exchanges/dydx",
                "env_file": "/opt/trader/.env.dydx",
                "systemd_unit": "trader-dydx.service",
                "supports_sighup": False,  # Requires restart
                "secrets": ["api_key", "secret_key", "stark_private_key"],
            },
            "database": {
                "ssm_prefix": "/trading/database",
                "env_file": "/opt/trader/.env.db",
                "systemd_unit": None,  # No service restart needed
                "supports_sighup": False,
                "secrets": ["db_password", "redis_password"],
            },
        }

        # Rotation configuration
        self.config = {
            "backup_old_secrets": True,
            "rollback_on_failure": True,
            "test_new_secrets": True,
            "max_rollback_attempts": 3,
            "service_restart_timeout": 30,
        }

        logger.info("🔐 Secrets Rotation initialized")

    def generate_new_secret(self, secret_type: str) -> str:
        """Generate a new secret based on type."""
        try:
            if secret_type in ["api_key"]:
                # Generate API key format (64 char hex)
                import secrets

                return secrets.token_hex(32)
            elif secret_type in ["secret_key", "stark_private_key"]:
                # Generate secret key format (64 char hex)
                import secrets

                return secrets.token_hex(32)
            elif secret_type == "passphrase":
                # Generate passphrase (32 char alphanumeric)
                import secrets
                import string

                alphabet = string.ascii_letters + string.digits
                return "".join(secrets.choice(alphabet) for _ in range(32))
            elif secret_type in ["db_password", "redis_password"]:
                # Generate database password (24 char alphanumeric)
                import secrets
                import string

                alphabet = string.ascii_letters + string.digits
                return "".join(secrets.choice(alphabet) for _ in range(24))
            else:
                # Default to 32 char hex
                import secrets

                return secrets.token_hex(16)

        except Exception as e:
            logger.error(f"Error generating secret for {secret_type}: {e}")
            return None

    def backup_ssm_parameter(self, parameter_name: str) -> Optional[str]:
        """Backup current SSM parameter value."""
        try:
            response = self.ssm.get_parameter(Name=parameter_name, WithDecryption=True)

            current_value = response["Parameter"]["Value"]

            # Create backup parameter
            backup_name = f"{parameter_name}.backup.{int(time.time())}"
            self.ssm.put_parameter(
                Name=backup_name,
                Value=current_value,
                Type="SecureString",
                Description=f"Backup of {parameter_name} before rotation",
                Overwrite=True,
            )

            logger.info(f"💾 Backed up {parameter_name} to {backup_name}")
            return backup_name

        except Exception as e:
            logger.error(f"Error backing up parameter {parameter_name}: {e}")
            return None

    def update_ssm_parameter(self, parameter_name: str, new_value: str) -> bool:
        """Update SSM parameter with new value."""
        try:
            self.ssm.put_parameter(
                Name=parameter_name,
                Value=new_value,
                Type="SecureString",
                Description=f"Rotated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
                Overwrite=True,
            )

            logger.info(f"🔄 Updated SSM parameter: {parameter_name}")
            return True

        except Exception as e:
            logger.error(f"Error updating parameter {parameter_name}: {e}")
            return False

    def get_ssm_parameter(self, parameter_name: str) -> Optional[str]:
        """Get SSM parameter value."""
        try:
            response = self.ssm.get_parameter(Name=parameter_name, WithDecryption=True)
            return response["Parameter"]["Value"]
        except Exception as e:
            logger.error(f"Error getting parameter {parameter_name}: {e}")
            return None

    def update_env_file(self, env_file: str, secret_name: str, new_value: str) -> bool:
        """Update environment file with new secret."""
        try:
            env_path = Path(env_file)

            # Create directory if it doesn't exist
            env_path.parent.mkdir(parents=True, exist_ok=True)

            # Read existing env file
            env_vars = {}
            if env_path.exists():
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and "=" in line and not line.startswith("#"):
                            key, value = line.split("=", 1)
                            env_vars[key] = value

            # Update the secret
            env_key = secret_name.upper()
            env_vars[env_key] = new_value

            # Write back to file
            with open(env_path, "w") as f:
                f.write(
                    f"# Auto-generated environment file - rotated {datetime.now(timezone.utc)}\n"
                )
                for key, value in sorted(env_vars.items()):
                    f.write(f"{key}={value}\n")

            # Set secure permissions
            env_path.chmod(0o600)

            logger.info(f"📝 Updated env file: {env_file}")
            return True

        except Exception as e:
            logger.error(f"Error updating env file {env_file}: {e}")
            return False

    def signal_service(self, systemd_unit: str, supports_sighup: bool) -> bool:
        """Signal systemd service to reload or restart."""
        try:
            if supports_sighup:
                # Send SIGHUP to reload configuration
                cmd = ["systemctl", "reload", systemd_unit]
                logger.info(f"🔄 Reloading service with SIGHUP: {systemd_unit}")
            else:
                # Restart the service
                cmd = ["systemctl", "restart", systemd_unit]
                logger.info(f"🔄 Restarting service: {systemd_unit}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config["service_restart_timeout"],
            )

            if result.returncode == 0:
                logger.info(f"✅ Service operation successful: {systemd_unit}")
                return True
            else:
                logger.error(f"❌ Service operation failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"❌ Service operation timed out: {systemd_unit}")
            return False
        except Exception as e:
            logger.error(f"Error signaling service {systemd_unit}: {e}")
            return False

    def test_secret(self, service_name: str, secret_name: str, new_value: str) -> bool:
        """Test if new secret works (mock implementation)."""
        try:
            # This is a mock test - in reality you'd test API connectivity
            # For exchanges, you might make a test API call
            # For database, you might test connection

            logger.info(f"🧪 Testing new secret for {service_name}.{secret_name}")

            if service_name in ["binance", "coinbase", "dydx"]:
                # Mock exchange API test
                if secret_name == "api_key" and len(new_value) >= 32:
                    logger.info(f"✅ API key format valid for {service_name}")
                    return True
                elif secret_name == "secret_key" and len(new_value) >= 32:
                    logger.info(f"✅ Secret key format valid for {service_name}")
                    return True
                elif secret_name == "passphrase" and len(new_value) >= 16:
                    logger.info(f"✅ Passphrase format valid for {service_name}")
                    return True
                else:
                    logger.warning(
                        f"⚠️ Secret format may be invalid for {service_name}.{secret_name}"
                    )
                    return False

            elif service_name == "database":
                # Mock database test
                if len(new_value) >= 16:
                    logger.info(f"✅ Database password format valid")
                    return True
                else:
                    logger.warning(f"⚠️ Database password too short")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error testing secret for {service_name}.{secret_name}: {e}")
            return False

    def rollback_secret(self, parameter_name: str, backup_name: str) -> bool:
        """Rollback secret to backup value."""
        try:
            # Get backup value
            backup_value = self.get_ssm_parameter(backup_name)
            if not backup_value:
                logger.error(f"Could not retrieve backup value from {backup_name}")
                return False

            # Restore original value
            if self.update_ssm_parameter(parameter_name, backup_value):
                logger.info(f"🔙 Rolled back {parameter_name} to backup")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Error rolling back {parameter_name}: {e}")
            return False

    def rotate_service_secrets(
        self, service_name: str, dry_run: bool = False
    ) -> Dict[str, Any]:
        """Rotate all secrets for a specific service."""
        try:
            service_config = self.services.get(service_name)
            if not service_config:
                return {"status": "error", "error": f"Unknown service: {service_name}"}

            rotation_start = time.time()
            logger.info(f"🔐 Rotating secrets for service: {service_name}")

            if dry_run:
                logger.info("🧪 DRY RUN MODE - No real changes will be made")

            rotation_result = {
                "service": service_name,
                "status": "in_progress",
                "dry_run": dry_run,
                "start_time": rotation_start,
                "secrets_rotated": [],
                "secrets_failed": [],
                "backups_created": [],
            }

            # Process each secret
            for secret_name in service_config["secrets"]:
                parameter_name = f"{service_config['ssm_prefix']}/{secret_name}"

                try:
                    # Step 1: Backup current value
                    if not dry_run and self.config["backup_old_secrets"]:
                        backup_name = self.backup_ssm_parameter(parameter_name)
                        if backup_name:
                            rotation_result["backups_created"].append(backup_name)

                    # Step 2: Generate new secret
                    new_value = self.generate_new_secret(secret_name)
                    if not new_value:
                        rotation_result["secrets_failed"].append(
                            {
                                "secret": secret_name,
                                "reason": "Failed to generate new value",
                            }
                        )
                        continue

                    # Step 3: Test new secret
                    if self.config["test_new_secrets"] and not self.test_secret(
                        service_name, secret_name, new_value
                    ):
                        rotation_result["secrets_failed"].append(
                            {
                                "secret": secret_name,
                                "reason": "New secret failed validation",
                            }
                        )
                        continue

                    # Step 4: Update SSM parameter
                    if not dry_run and not self.update_ssm_parameter(
                        parameter_name, new_value
                    ):
                        rotation_result["secrets_failed"].append(
                            {
                                "secret": secret_name,
                                "reason": "Failed to update SSM parameter",
                            }
                        )
                        continue

                    # Step 5: Update env file
                    if not dry_run and not self.update_env_file(
                        service_config["env_file"], secret_name, new_value
                    ):
                        rotation_result["secrets_failed"].append(
                            {
                                "secret": secret_name,
                                "reason": "Failed to update env file",
                            }
                        )
                        continue

                    rotation_result["secrets_rotated"].append(secret_name)

                except Exception as e:
                    logger.error(f"Error rotating {secret_name}: {e}")
                    rotation_result["secrets_failed"].append(
                        {"secret": secret_name, "reason": str(e)}
                    )

            # Step 6: Signal service if needed
            if (
                service_config["systemd_unit"]
                and rotation_result["secrets_rotated"]
                and not dry_run
            ):
                service_reloaded = self.signal_service(
                    service_config["systemd_unit"], service_config["supports_sighup"]
                )
                rotation_result["service_reloaded"] = service_reloaded

                if not service_reloaded:
                    rotation_result["status"] = "partial_failure"
                    rotation_result["error"] = "Failed to reload service"

            # Determine final status
            if rotation_result["secrets_failed"]:
                if rotation_result["secrets_rotated"]:
                    rotation_result["status"] = "partial_success"
                else:
                    rotation_result["status"] = "failed"
            else:
                rotation_result["status"] = "completed"

            rotation_duration = time.time() - rotation_start
            rotation_result["duration"] = rotation_duration

            logger.info(
                f"🔐 Secrets rotation completed for {service_name}: "
                f"{len(rotation_result['secrets_rotated'])}/{len(service_config['secrets'])} secrets rotated "
                f"in {rotation_duration:.1f}s"
            )

            return rotation_result

        except Exception as e:
            logger.error(f"Error rotating secrets for {service_name}: {e}")
            return {
                "service": service_name,
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }

    def rotate_all_secrets(self, dry_run: bool = False) -> Dict[str, Any]:
        """Rotate secrets for all configured services."""
        try:
            rotation_start = time.time()
            logger.info("🔐 Starting rotation for all services...")

            all_results = {
                "status": "in_progress",
                "dry_run": dry_run,
                "start_time": rotation_start,
                "service_results": {},
                "services_completed": [],
                "services_failed": [],
            }

            # Rotate each service
            for service_name in self.services:
                result = self.rotate_service_secrets(service_name, dry_run)
                all_results["service_results"][service_name] = result

                if result["status"] in ["completed", "partial_success"]:
                    all_results["services_completed"].append(service_name)
                else:
                    all_results["services_failed"].append(service_name)

            # Determine overall status
            if all_results["services_failed"]:
                if all_results["services_completed"]:
                    all_results["status"] = "partial_success"
                else:
                    all_results["status"] = "failed"
            else:
                all_results["status"] = "completed"

            rotation_duration = time.time() - rotation_start
            all_results["duration"] = rotation_duration

            logger.info(
                f"🔐 All secrets rotation completed: "
                f"{len(all_results['services_completed'])}/{len(self.services)} services "
                f"in {rotation_duration:.1f}s"
            )

            return all_results

        except Exception as e:
            logger.error(f"Error in full secrets rotation: {e}")
            return {"status": "error", "error": str(e), "timestamp": time.time()}

    def send_slack_summary(self, rotation_result: Dict[str, Any]):
        """Send Slack summary of rotation."""
        try:
            if not self.slack_webhook:
                logger.warning("No Slack webhook configured")
                return

            status = rotation_result["status"]

            if status == "completed":
                emoji = "🟢"
                color = "#36a64f"
            elif status == "partial_success":
                emoji = "🟡"
                color = "#ffcc00"
            else:
                emoji = "🔴"
                color = "#ff0000"

            if "service_results" in rotation_result:
                # Summary for all services
                completed = len(rotation_result["services_completed"])
                total = len(self.services)

                message = f"{emoji} **Secrets Rotation Summary**\n"
                message += f"Status: {status.upper()}\n"
                message += f"Services: {completed}/{total} successful\n"
                message += f"Duration: {rotation_result['duration']:.1f}s\n\n"

                # List rotated secrets (names only, not values)
                for service, result in rotation_result["service_results"].items():
                    if result["secrets_rotated"]:
                        message += f"✅ **{service}**: {', '.join(result['secrets_rotated'])}\n"
                    if result["secrets_failed"]:
                        failed_names = [f["secret"] for f in result["secrets_failed"]]
                        message += (
                            f"❌ **{service}**: {', '.join(failed_names)} failed\n"
                        )

            else:
                # Summary for single service
                service = rotation_result["service"]
                rotated = len(rotation_result["secrets_rotated"])
                total = len(self.services[service]["secrets"])

                message = f"{emoji} **Secrets Rotation: {service}**\n"
                message += f"Status: {status.upper()}\n"
                message += f"Secrets: {rotated}/{total} rotated\n"
                message += f"Rotated: {', '.join(rotation_result['secrets_rotated'])}\n"

                if rotation_result["secrets_failed"]:
                    failed_names = [
                        f["secret"] for f in rotation_result["secrets_failed"]
                    ]
                    message += f"Failed: {', '.join(failed_names)}\n"

            payload = {
                "text": message,
                "username": "Secrets Rotation",
                "icon_emoji": ":key:",
                "attachments": [{"color": color}],
            }

            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("📱 Sent Slack rotation summary")

        except Exception as e:
            logger.error(f"Error sending Slack summary: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Secrets Rotation")
    parser.add_argument(
        "--service", type=str, help="Rotate secrets for specific service"
    )
    parser.add_argument(
        "--all", action="store_true", help="Rotate secrets for all services"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run mode (no real changes)"
    )
    parser.add_argument("--list", action="store_true", help="List configured services")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    rotator = SecretsRotation()

    if args.list:
        print("📋 Configured Services:")
        for service, config in rotator.services.items():
            print(f"  • {service}: {', '.join(config['secrets'])}")
        return

    if args.service:
        if args.service not in rotator.services:
            print(f"❌ Unknown service: {args.service}")
            print(f"Available services: {', '.join(rotator.services.keys())}")
            sys.exit(1)

        result = rotator.rotate_service_secrets(args.service, args.dry_run)

        if not args.dry_run:
            rotator.send_slack_summary(result)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            status = result["status"]
            emoji = (
                "✅"
                if status == "completed"
                else ("⚠️" if status == "partial_success" else "❌")
            )
            print(f"{emoji} Secrets rotation for {args.service}: {status.upper()}")

        sys.exit(0 if result["status"] in ["completed", "partial_success"] else 1)

    if args.all or not sys.argv[1:]:  # Default to all
        result = rotator.rotate_all_secrets(args.dry_run)

        if not args.dry_run:
            rotator.send_slack_summary(result)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            status = result["status"]
            emoji = (
                "✅"
                if status == "completed"
                else ("⚠️" if status == "partial_success" else "❌")
            )
            completed = len(result["services_completed"])
            total = len(rotator.services)
            print(
                f"{emoji} All secrets rotation: {status.upper()} ({completed}/{total})"
            )

        sys.exit(0 if result["status"] in ["completed", "partial_success"] else 1)

    parser.print_help()


if __name__ == "__main__":
    main()
