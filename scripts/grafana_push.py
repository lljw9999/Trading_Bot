#!/usr/bin/env python3
"""
Grafana Dashboard Provisioner
Reads JSON dashboard files and pushes to Grafana HTTP API
"""

import os
import sys
import json
import glob
import logging
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("grafana_push")


class GrafanaProvisioner:
    """Grafana dashboard provisioning service."""

    def __init__(self):
        """Initialize Grafana provisioner."""
        self.grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
        self.grafana_token = os.getenv("GRAFANA_TOKEN", "")
        self.grafana_user = os.getenv("GRAFANA_USER", "admin")
        self.grafana_password = os.getenv("GRAFANA_PASSWORD", "admin")

        # Dashboard directory
        self.dashboard_dir = project_root / "grafana" / "dashboards"

        # Headers for API requests
        if self.grafana_token:
            self.headers = {
                "Authorization": f"Bearer {self.grafana_token}",
                "Content-Type": "application/json",
            }
            self.auth = None
        else:
            self.headers = {"Content-Type": "application/json"}
            self.auth = (self.grafana_user, self.grafana_password)

        logger.info("📊 Grafana Provisioner initialized")
        logger.info(f"   URL: {self.grafana_url}")
        logger.info(f"   Auth: {'Token' if self.grafana_token else 'Basic'}")
        logger.info(f"   Dashboard dir: {self.dashboard_dir}")

    def test_connection(self) -> bool:
        """Test connection to Grafana."""
        try:
            response = requests.get(
                f"{self.grafana_url}/api/health",
                headers=self.headers,
                auth=self.auth,
                timeout=10,
            )

            if response.status_code == 200:
                logger.info("✅ Grafana connection successful")
                return True
            else:
                logger.error(
                    f"❌ Grafana connection failed: HTTP {response.status_code}"
                )
                return False

        except Exception as e:
            logger.error(f"❌ Grafana connection error: {e}")
            return False

    def get_existing_dashboards(self) -> dict:
        """Get list of existing dashboards."""
        try:
            response = requests.get(
                f"{self.grafana_url}/api/search?type=dash-db",
                headers=self.headers,
                auth=self.auth,
                timeout=10,
            )

            response.raise_for_status()

            dashboards = {}
            for dashboard in response.json():
                dashboards[dashboard["uid"]] = {
                    "id": dashboard["id"],
                    "title": dashboard["title"],
                    "url": dashboard["url"],
                }

            logger.info(f"📋 Found {len(dashboards)} existing dashboards")
            return dashboards

        except Exception as e:
            logger.error(f"Error getting existing dashboards: {e}")
            return {}

    def load_dashboard_json(self, file_path: Path) -> dict:
        """Load and validate dashboard JSON file."""
        try:
            with open(file_path, "r") as f:
                dashboard_data = json.load(f)

            # Validate required fields
            if "title" not in dashboard_data:
                raise ValueError("Dashboard missing 'title' field")

            if "uid" not in dashboard_data:
                raise ValueError("Dashboard missing 'uid' field")

            return dashboard_data

        except Exception as e:
            logger.error(f"Error loading dashboard {file_path}: {e}")
            return {}

    def push_dashboard(self, dashboard_data: dict, overwrite: bool = True) -> bool:
        """Push dashboard to Grafana."""
        try:
            dashboard_title = dashboard_data.get("title", "Unknown")
            dashboard_uid = dashboard_data.get("uid", "unknown")

            # Prepare payload
            payload = {
                "dashboard": dashboard_data,
                "overwrite": overwrite,
                "message": "Updated via grafana_push.py",
            }

            # Push to Grafana
            response = requests.post(
                f"{self.grafana_url}/api/dashboards/db",
                headers=self.headers,
                auth=self.auth,
                json=payload,
                timeout=30,
            )

            if response.status_code in [200, 201]:
                result = response.json()
                action = "Updated" if result.get("version", 1) > 1 else "Created"
                logger.info(
                    f"✅ {action} dashboard: {dashboard_title} (UID: {dashboard_uid})"
                )
                return True
            else:
                logger.error(
                    f"❌ Failed to push dashboard {dashboard_title}: "
                    f"HTTP {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Error pushing dashboard: {e}")
            return False

    def push_all_dashboards(self, overwrite: bool = True) -> dict:
        """Push all dashboard JSON files to Grafana."""
        logger.info("🚀 Starting dashboard provisioning...")

        results = {"success": [], "failed": [], "total": 0}

        try:
            # Find all dashboard JSON files
            dashboard_pattern = str(self.dashboard_dir / "*.json")
            dashboard_files = glob.glob(dashboard_pattern)

            if not dashboard_files:
                logger.warning(f"No dashboard files found in {self.dashboard_dir}")
                return results

            logger.info(f"📁 Found {len(dashboard_files)} dashboard files")

            # Push each dashboard
            for file_path in dashboard_files:
                file_path = Path(file_path)
                results["total"] += 1

                logger.info(f"📤 Processing: {file_path.name}")

                # Load dashboard
                dashboard_data = self.load_dashboard_json(file_path)
                if not dashboard_data:
                    results["failed"].append(
                        {"file": file_path.name, "error": "Failed to load JSON"}
                    )
                    continue

                # Push dashboard
                if self.push_dashboard(dashboard_data, overwrite):
                    results["success"].append(
                        {
                            "file": file_path.name,
                            "title": dashboard_data.get("title"),
                            "uid": dashboard_data.get("uid"),
                        }
                    )
                else:
                    results["failed"].append(
                        {
                            "file": file_path.name,
                            "title": dashboard_data.get("title"),
                            "error": "Push failed",
                        }
                    )

            # Summary
            logger.info(
                f"📊 Provisioning complete: "
                f"{len(results['success'])}/{results['total']} dashboards successful"
            )

            if results["failed"]:
                logger.warning(f"❌ {len(results['failed'])} dashboards failed")
                for failure in results["failed"]:
                    logger.warning(
                        f"   - {failure['file']}: {failure.get('error', 'Unknown error')}"
                    )

            return results

        except Exception as e:
            logger.error(f"Error in dashboard provisioning: {e}")
            return results

    def get_status_report(self) -> dict:
        """Get status report of Grafana provisioner."""
        try:
            # Test connection
            connection_ok = self.test_connection()

            # Get existing dashboards
            existing_dashboards = (
                self.get_existing_dashboards() if connection_ok else {}
            )

            # Find local dashboard files
            dashboard_pattern = str(self.dashboard_dir / "*.json")
            local_files = glob.glob(dashboard_pattern)

            return {
                "service": "grafana_provisioner",
                "grafana_url": self.grafana_url,
                "connection_status": "connected" if connection_ok else "disconnected",
                "auth_method": "token" if self.grafana_token else "basic",
                "local_dashboards": len(local_files),
                "remote_dashboards": len(existing_dashboards),
                "dashboard_files": [Path(f).name for f in local_files],
                "existing_dashboards": list(existing_dashboards.values()),
            }

        except Exception as e:
            return {
                "service": "grafana_provisioner",
                "status": "error",
                "error": str(e),
            }


def main():
    """Main entry point for Grafana provisioner."""
    import argparse

    parser = argparse.ArgumentParser(description="Grafana Dashboard Provisioner")
    parser.add_argument(
        "--push", action="store_true", help="Push all dashboards to Grafana"
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Don't overwrite existing dashboards",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test connection to Grafana"
    )
    parser.add_argument("--status", action="store_true", help="Show status report")
    parser.add_argument("--list", action="store_true", help="List existing dashboards")

    args = parser.parse_args()

    # Create provisioner
    provisioner = GrafanaProvisioner()

    if args.test:
        # Test connection
        success = provisioner.test_connection()
        sys.exit(0 if success else 1)

    if args.status:
        # Show status report
        status = provisioner.get_status_report()
        print(json.dumps(status, indent=2))
        return

    if args.list:
        # List existing dashboards
        dashboards = provisioner.get_existing_dashboards()
        print(json.dumps(dashboards, indent=2))
        return

    if args.push:
        # Push dashboards
        overwrite = not args.no_overwrite
        results = provisioner.push_all_dashboards(overwrite)

        # Print results
        print(json.dumps(results, indent=2))

        # Exit with appropriate code
        if results["failed"]:
            sys.exit(1)
        else:
            sys.exit(0)
    else:
        # Default action - push dashboards
        logger.info("No action specified, use --push to provision dashboards")
        parser.print_help()


if __name__ == "__main__":
    main()
