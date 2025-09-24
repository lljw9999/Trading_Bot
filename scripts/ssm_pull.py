#!/usr/bin/env python3
"""
AWS Systems Manager Parameter Store Puller
Securely fetches encrypted configuration parameters from AWS SSM
"""

import boto3
import os
import json
import logging
from typing import List, Dict, Optional
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("ssm_pull")


class SSMParameterPuller:
    """Pulls encrypted parameters from AWS Systems Manager Parameter Store."""

    def __init__(self, region_name: str = "us-east-1"):
        """
        Initialize SSM client.

        Args:
            region_name: AWS region for SSM client
        """
        try:
            self.ssm = boto3.client("ssm", region_name=region_name)
            self.region = region_name
            logger.info(f"🔧 SSM Parameter Puller initialized (region: {region_name})")
        except NoCredentialsError:
            logger.error("❌ AWS credentials not configured")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize SSM client: {e}")
            raise

    def get_parameter(self, parameter_name: str, decrypt: bool = True) -> Optional[str]:
        """
        Get a single parameter from SSM.

        Args:
            parameter_name: Name of the parameter (e.g., '/trader/WHALE_ALERT_KEY')
            decrypt: Whether to decrypt SecureString parameters

        Returns:
            Parameter value or None if not found
        """
        try:
            logger.debug(f"Fetching parameter: {parameter_name}")

            response = self.ssm.get_parameter(
                Name=parameter_name, WithDecryption=decrypt
            )

            value = response["Parameter"]["Value"]
            param_type = response["Parameter"]["Type"]

            logger.info(
                f"✅ Retrieved parameter: {parameter_name} (type: {param_type})"
            )
            return value

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ParameterNotFound":
                logger.warning(f"⚠️ Parameter not found: {parameter_name}")
            else:
                logger.error(f"❌ AWS error getting parameter {parameter_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error getting parameter {parameter_name}: {e}")
            return None

    def get_parameters_by_path(self, path: str, decrypt: bool = True) -> Dict[str, str]:
        """
        Get all parameters under a specific path.

        Args:
            path: Parameter path prefix (e.g., '/trader/')
            decrypt: Whether to decrypt SecureString parameters

        Returns:
            Dictionary of parameter names to values
        """
        try:
            logger.info(f"Fetching parameters by path: {path}")

            parameters = {}
            paginator = self.ssm.get_paginator("get_parameters_by_path")

            for page in paginator.paginate(
                Path=path, Recursive=True, WithDecryption=decrypt
            ):
                for param in page["Parameters"]:
                    name = param["Name"]
                    value = param["Value"]
                    param_type = param["Type"]

                    # Extract key name from full path
                    key_name = name.replace(path, "").lstrip("/")
                    parameters[key_name] = value

                    logger.info(
                        f"✅ Retrieved: {name} → {key_name} (type: {param_type})"
                    )

            logger.info(f"📊 Total parameters retrieved: {len(parameters)}")
            return parameters

        except ClientError as e:
            logger.error(f"❌ AWS error getting parameters by path: {e}")
            return {}
        except Exception as e:
            logger.error(f"❌ Unexpected error getting parameters by path: {e}")
            return {}

    def save_to_env_files(
        self, parameters: Dict[str, str], env_dir: str = ".env"
    ) -> bool:
        """
        Save parameters to individual environment files.

        Args:
            parameters: Dictionary of parameter names to values
            env_dir: Directory to save environment files

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create environment directory
            os.makedirs(env_dir, exist_ok=True)
            logger.info(f"📁 Environment directory: {env_dir}")

            for key, value in parameters.items():
                env_file_path = os.path.join(env_dir, key)

                # Write parameter value to individual file
                with open(env_file_path, "w") as f:
                    f.write(value)

                # Set restrictive permissions (readable only by owner)
                os.chmod(env_file_path, 0o600)

                logger.info(f"💾 Saved: {env_file_path}")

            logger.info(f"✅ All parameters saved to {env_dir}")
            return True

        except Exception as e:
            logger.error(f"❌ Error saving parameters to files: {e}")
            return False

    def save_to_dotenv(
        self, parameters: Dict[str, str], dotenv_path: str = ".env"
    ) -> bool:
        """
        Save parameters to a single .env file.

        Args:
            parameters: Dictionary of parameter names to values
            dotenv_path: Path to .env file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(dotenv_path, "w") as f:
                f.write("# Generated by SSM Parameter Puller\n")
                f.write(f"# Timestamp: {os.popen('date').read().strip()}\n\n")

                for key, value in parameters.items():
                    # Escape quotes and newlines in values
                    escaped_value = value.replace('"', '\\"').replace("\n", "\\n")
                    f.write(f'{key}="{escaped_value}"\n')

            # Set restrictive permissions
            os.chmod(dotenv_path, 0o600)

            logger.info(f"💾 Parameters saved to {dotenv_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Error saving to .env file: {e}")
            return False

    def test_connection(self) -> bool:
        """Test SSM connection and permissions."""
        try:
            logger.info("🧪 Testing SSM connection...")

            # Try to describe parameters (minimal permissions needed)
            response = self.ssm.describe_parameters(MaxResults=1)

            logger.info("✅ SSM connection successful")
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logger.error(f"❌ SSM connection failed: {error_code}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error testing connection: {e}")
            return False


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pull parameters from AWS SSM Parameter Store"
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--path", default="/trader/", help="Parameter path prefix")
    parser.add_argument("--keys", nargs="+", help="Specific parameter keys to fetch")
    parser.add_argument(
        "--env-dir", default=".env", help="Directory for environment files"
    )
    parser.add_argument(
        "--dotenv", help="Save to single .env file instead of individual files"
    )
    parser.add_argument("--test", action="store_true", help="Test connection only")

    args = parser.parse_args()

    try:
        # Initialize SSM puller
        puller = SSMParameterPuller(region_name=args.region)

        # Test connection if requested
        if args.test:
            success = puller.test_connection()
            exit(0 if success else 1)

        # Determine what parameters to fetch
        if args.keys:
            # Fetch specific keys
            logger.info(f"Fetching specific keys: {args.keys}")
            parameters = {}

            for key in args.keys:
                # Add path prefix if not present
                param_name = (
                    key if key.startswith("/") else f"{args.path.rstrip('/')}/{key}"
                )
                value = puller.get_parameter(param_name)

                if value is not None:
                    parameters[key] = value
        else:
            # Fetch all parameters by path
            parameters = puller.get_parameters_by_path(args.path)

        if not parameters:
            logger.warning("⚠️ No parameters retrieved")
            exit(1)

        # Save parameters
        if args.dotenv:
            success = puller.save_to_dotenv(parameters, args.dotenv)
        else:
            success = puller.save_to_env_files(parameters, args.env_dir)

        if success:
            logger.info("🎉 Parameter pulling completed successfully")
            exit(0)
        else:
            logger.error("❌ Failed to save parameters")
            exit(1)

    except KeyboardInterrupt:
        logger.info("🛑 Operation cancelled by user")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    # Demo usage when run directly
    if len(os.sys.argv) == 1:
        print("🔧 SSM Parameter Puller Demo")

        # Default configuration for trading bot
        keys = ["WHALE_ALERT_KEY", "SLACK_WEBHOOK", "DB_URL"]

        try:
            puller = SSMParameterPuller(region_name="us-east-1")

            # Test connection first
            if not puller.test_connection():
                print("❌ SSM connection failed - check AWS credentials")
                exit(1)

            print(f"Fetching trading bot parameters: {keys}")

            parameters = {}
            for key in keys:
                param_name = f"/trader/{key}"
                value = puller.get_parameter(param_name)

                if value:
                    parameters[key] = value
                else:
                    # Create mock parameter for demo
                    mock_value = f"mock_{key.lower()}_value"
                    parameters[key] = mock_value
                    print(f"⚠️ Using mock value for {key}")

            # Save to .env directory
            success = puller.save_to_env_files(parameters, ".env")

            if success:
                print("✅ Demo completed - check .env/ directory")
            else:
                print("❌ Demo failed")

        except Exception as e:
            print(f"❌ Demo error: {e}")
            print("💡 Run with --help for usage options")
    else:
        main()
