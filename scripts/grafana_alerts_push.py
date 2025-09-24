#!/usr/bin/env python3
"""
Grafana Alerts-as-Code Push Script
Upserts alert rules via Grafana HTTP API for version-controlled monitoring
"""

import os
import sys
import json
import glob
import time
import logging
import requests
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("grafana_alerts_push")


class GrafanaAlertsManager:
    """Manager for Grafana alerts-as-code."""

    def __init__(self, grafana_url: str = None, api_token: str = None):
        """Initialize Grafana alerts manager."""
        self.grafana_url = grafana_url or os.getenv(
            "GRAFANA_URL", "http://localhost:3000"
        )
        self.api_token = (
            api_token or os.getenv("GRAFANA_TOKEN") or os.getenv("GRAFANA_API_KEY")
        )

        if not self.api_token:
            logger.warning("No Grafana API token provided - some operations may fail")
            self.api_token = "dummy_token"

        # HTTP session with auth headers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

        # Configuration
        self.config = {
            "alerts_dir": project_root / "grafana" / "alerts",
            "default_folder_uid": "trading-alerts",
            "default_folder_title": "Trading Alerts",
            "timeout": 30,
            "verify_ssl": True,
        }

        logger.info("🚨 Grafana Alerts Manager initialized")
        logger.info(f"   Grafana URL: {self.grafana_url}")
        logger.info(f"   Alerts directory: {self.config['alerts_dir']}")
        logger.info(
            f"   API token configured: {'Yes' if self.api_token != 'dummy_token' else 'No'}"
        )

    def test_connection(self) -> bool:
        """Test connection to Grafana API."""
        try:
            url = urljoin(self.grafana_url, "/api/health")
            response = self.session.get(url, timeout=self.config["timeout"])

            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ Grafana connection successful: {health_data}")
                return True
            else:
                logger.error(f"❌ Grafana health check failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Grafana connection error: {e}")
            return False

    def ensure_alert_folder(self) -> Dict[str, Any]:
        """Ensure alert folder exists."""
        try:
            folder_uid = self.config["default_folder_uid"]
            folder_title = self.config["default_folder_title"]

            # Check if folder exists
            url = urljoin(self.grafana_url, f"/api/folders/{folder_uid}")
            response = self.session.get(url, timeout=self.config["timeout"])

            if response.status_code == 200:
                folder_data = response.json()
                logger.info(f"📁 Alert folder exists: {folder_data['title']}")
                return folder_data

            elif response.status_code == 404:
                # Create folder
                logger.info(f"📁 Creating alert folder: {folder_title}")
                create_url = urljoin(self.grafana_url, "/api/folders")

                folder_payload = {
                    "uid": folder_uid,
                    "title": folder_title,
                    "tags": ["trading", "alerts", "slo"],
                }

                create_response = self.session.post(
                    create_url, json=folder_payload, timeout=self.config["timeout"]
                )

                if create_response.status_code in [200, 201]:
                    folder_data = create_response.json()
                    logger.info(f"✅ Created alert folder: {folder_data['title']}")
                    return folder_data
                else:
                    logger.error(f"❌ Failed to create folder: {create_response.text}")
                    return {"uid": folder_uid, "title": folder_title}

            else:
                logger.error(
                    f"❌ Unexpected response checking folder: {response.status_code}"
                )
                return {"uid": folder_uid, "title": folder_title}

        except Exception as e:
            logger.error(f"Error ensuring alert folder: {e}")
            return {
                "uid": self.config["default_folder_uid"],
                "title": self.config["default_folder_title"],
            }

    def load_alert_files(self) -> List[Dict[str, Any]]:
        """Load all alert definition files."""
        try:
            alerts_dir = self.config["alerts_dir"]
            if not alerts_dir.exists():
                logger.error(f"Alerts directory does not exist: {alerts_dir}")
                return []

            alert_files = list(alerts_dir.glob("*.json"))
            logger.info(f"📋 Found {len(alert_files)} alert files")

            alerts = []
            for alert_file in alert_files:
                try:
                    with open(alert_file, "r") as f:
                        alert_data = json.load(f)

                    # Add filename for reference
                    alert_data["_filename"] = alert_file.name
                    alerts.append(alert_data)

                    logger.debug(f"Loaded alert: {alert_data.get('title', 'Unknown')}")

                except Exception as e:
                    logger.error(f"Error loading alert file {alert_file}: {e}")

            return alerts

        except Exception as e:
            logger.error(f"Error loading alert files: {e}")
            return []

    def validate_alert(self, alert: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate alert definition."""
        errors = []

        # Required fields
        required_fields = ["uid", "title", "condition", "data"]
        for field in required_fields:
            if field not in alert:
                errors.append(f"Missing required field: {field}")

        # Validate UID format
        if "uid" in alert:
            uid = alert["uid"]
            if not isinstance(uid, str) or len(uid) == 0:
                errors.append("UID must be a non-empty string")
            elif not uid.replace("_", "").replace("-", "").isalnum():
                errors.append(
                    "UID should contain only alphanumeric characters, hyphens, and underscores"
                )

        # Validate data array
        if "data" in alert:
            if not isinstance(alert["data"], list) or len(alert["data"]) == 0:
                errors.append("Data must be a non-empty array")

        # Validate condition reference
        if "condition" in alert and "data" in alert:
            condition_ref = alert["condition"]
            data_refs = [item.get("refId") for item in alert["data"]]
            if condition_ref not in data_refs:
                errors.append(
                    f"Condition reference '{condition_ref}' not found in data queries"
                )

        # Validate labels
        if "labels" in alert:
            labels = alert["labels"]
            if not isinstance(labels, dict):
                errors.append("Labels must be a dictionary")
            else:
                required_labels = ["team", "severity", "component", "slo"]
                for label in required_labels:
                    if label not in labels:
                        errors.append(f"Missing recommended label: {label}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def get_existing_alert(self, uid: str) -> Optional[Dict[str, Any]]:
        """Get existing alert rule by UID."""
        try:
            url = urljoin(self.grafana_url, f"/api/v1/provisioning/alert-rules/{uid}")
            response = self.session.get(url, timeout=self.config["timeout"])

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                logger.warning(
                    f"Unexpected response getting alert {uid}: {response.status_code}"
                )
                return None

        except Exception as e:
            logger.error(f"Error getting existing alert {uid}: {e}")
            return None

    def upsert_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update an alert rule."""
        try:
            uid = alert["uid"]
            title = alert.get("title", uid)

            # Validate alert
            is_valid, errors = self.validate_alert(alert)
            if not is_valid:
                return {
                    "uid": uid,
                    "status": "error",
                    "error": "Validation failed",
                    "errors": errors,
                }

            # Check if alert exists
            existing_alert = self.get_existing_alert(uid)
            is_update = existing_alert is not None

            # Prepare alert payload
            alert_payload = {
                "uid": uid,
                "title": title,
                "condition": alert["condition"],
                "data": alert["data"],
                "intervalSeconds": alert.get("intervalSeconds", 60),
                "noDataState": alert.get("noDataState", "NoData"),
                "execErrState": alert.get("execErrState", "Alerting"),
                "for": alert.get("for", "5m"),
                "annotations": alert.get("annotations", {}),
                "labels": alert.get("labels", {}),
                "folderUID": alert.get("folderUID", self.config["default_folder_uid"]),
            }

            # Choose endpoint and method
            if is_update:
                url = urljoin(
                    self.grafana_url, f"/api/v1/provisioning/alert-rules/{uid}"
                )
                response = self.session.put(
                    url, json=alert_payload, timeout=self.config["timeout"]
                )
                action = "updated"
            else:
                url = urljoin(self.grafana_url, "/api/v1/provisioning/alert-rules")
                response = self.session.post(
                    url, json=alert_payload, timeout=self.config["timeout"]
                )
                action = "created"

            if response.status_code in [200, 201]:
                result_data = response.json()
                logger.info(f"✅ {action.title()} alert: {title} ({uid})")

                return {
                    "uid": uid,
                    "title": title,
                    "status": "success",
                    "action": action,
                    "data": result_data,
                }
            else:
                logger.error(
                    f"❌ Failed to {action.replace('d', '')} alert {title}: {response.text}"
                )
                return {
                    "uid": uid,
                    "title": title,
                    "status": "error",
                    "action": action,
                    "error": response.text,
                    "status_code": response.status_code,
                }

        except Exception as e:
            logger.error(f"Error upserting alert {alert.get('uid', 'unknown')}: {e}")
            return {
                "uid": alert.get("uid", "unknown"),
                "status": "error",
                "error": str(e),
            }

    def push_all_alerts(self) -> Dict[str, Any]:
        """Push all alerts from files to Grafana."""
        try:
            push_start = time.time()
            logger.info("🚀 Starting alerts push...")

            # Test connection
            if not self.test_connection():
                return {"status": "error", "error": "Failed to connect to Grafana"}

            # Ensure alert folder exists
            folder_info = self.ensure_alert_folder()

            # Load alert files
            alerts = self.load_alert_files()
            if not alerts:
                return {"status": "error", "error": "No alert files found"}

            # Process each alert
            results = {
                "total": len(alerts),
                "successful": 0,
                "failed": 0,
                "created": 0,
                "updated": 0,
                "errors": [],
                "details": [],
            }

            for alert in alerts:
                try:
                    result = self.upsert_alert(alert)
                    results["details"].append(result)

                    if result["status"] == "success":
                        results["successful"] += 1
                        if result["action"] == "created":
                            results["created"] += 1
                        else:
                            results["updated"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(
                            {
                                "uid": result["uid"],
                                "error": result.get("error", "Unknown error"),
                            }
                        )

                except Exception as e:
                    logger.error(f"Error processing alert: {e}")
                    results["failed"] += 1
                    results["errors"].append({"uid": "unknown", "error": str(e)})

            push_duration = time.time() - push_start

            # Summary
            logger.info(
                f"📊 Alerts push completed: {results['successful']}/{results['total']} successful "
                f"({results['created']} created, {results['updated']} updated) in {push_duration:.1f}s"
            )

            if results["failed"] > 0:
                logger.warning(f"⚠️ {results['failed']} alerts failed to push")
                for error in results["errors"]:
                    logger.warning(f"   {error['uid']}: {error['error']}")

            results["duration"] = push_duration
            results["timestamp"] = push_start
            results["status"] = "completed"

            return results

        except Exception as e:
            logger.error(f"Error in push_all_alerts: {e}")
            return {"status": "error", "error": str(e), "timestamp": time.time()}

    def list_existing_alerts(self) -> List[Dict[str, Any]]:
        """List existing alert rules in Grafana."""
        try:
            url = urljoin(self.grafana_url, "/api/v1/provisioning/alert-rules")
            response = self.session.get(url, timeout=self.config["timeout"])

            if response.status_code == 200:
                alerts = response.json()
                logger.info(f"📋 Found {len(alerts)} existing alerts in Grafana")

                # Filter to trading alerts
                trading_alerts = [
                    alert
                    for alert in alerts
                    if alert.get("folderUID") == self.config["default_folder_uid"]
                ]

                logger.info(f"📋 Trading alerts: {len(trading_alerts)}")
                return trading_alerts
            else:
                logger.error(f"Failed to list alerts: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error listing alerts: {e}")
            return []

    def validate_all_alerts(self) -> Dict[str, Any]:
        """Validate all alert files."""
        try:
            alerts = self.load_alert_files()

            validation_results = {
                "total": len(alerts),
                "valid": 0,
                "invalid": 0,
                "errors": [],
                "details": [],
            }

            for alert in alerts:
                uid = alert.get("uid", "unknown")
                title = alert.get("title", uid)
                filename = alert.get("_filename", "unknown")

                is_valid, errors = self.validate_alert(alert)

                result = {
                    "uid": uid,
                    "title": title,
                    "filename": filename,
                    "valid": is_valid,
                    "errors": errors,
                }

                validation_results["details"].append(result)

                if is_valid:
                    validation_results["valid"] += 1
                    logger.info(f"✅ Valid alert: {title} ({filename})")
                else:
                    validation_results["invalid"] += 1
                    validation_results["errors"].extend(errors)
                    logger.error(f"❌ Invalid alert: {title} ({filename})")
                    for error in errors:
                        logger.error(f"   - {error}")

            return validation_results

        except Exception as e:
            logger.error(f"Error validating alerts: {e}")
            return {"status": "error", "error": str(e)}


def main():
    """Main entry point for Grafana alerts push."""
    import time

    parser = argparse.ArgumentParser(description="Grafana Alerts-as-Code Manager")
    parser.add_argument(
        "--push", action="store_true", help="Push all alerts to Grafana"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate alert definitions"
    )
    parser.add_argument(
        "--list", action="store_true", help="List existing alerts in Grafana"
    )
    parser.add_argument("--test", action="store_true", help="Test Grafana connection")
    parser.add_argument(
        "--grafana-url", help="Grafana URL (default: $GRAFANA_URL or localhost:3000)"
    )
    parser.add_argument(
        "--api-token", help="Grafana API token (default: $GRAFANA_TOKEN)"
    )

    args = parser.parse_args()

    # Create alerts manager
    manager = GrafanaAlertsManager(
        grafana_url=args.grafana_url, api_token=args.api_token
    )

    if args.test:
        # Test connection
        success = manager.test_connection()
        if success:
            print("✅ Grafana connection successful")
        else:
            print("❌ Grafana connection failed")
        return

    if args.validate:
        # Validate alerts
        results = manager.validate_all_alerts()
        print(json.dumps(results, indent=2))

        if results.get("invalid", 0) > 0:
            sys.exit(1)
        return

    if args.list:
        # List existing alerts
        alerts = manager.list_existing_alerts()

        alert_summary = [
            {
                "uid": alert["uid"],
                "title": alert["title"],
                "folder": alert.get("folderUID"),
                "state": alert.get("noDataState"),
                "for": alert.get("for"),
            }
            for alert in alerts
        ]

        print(json.dumps(alert_summary, indent=2))
        return

    if args.push:
        # Push all alerts
        results = manager.push_all_alerts()
        print(json.dumps(results, indent=2, default=str))

        if results.get("status") == "error" or results.get("failed", 0) > 0:
            sys.exit(1)
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
