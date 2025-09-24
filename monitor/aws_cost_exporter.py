#!/usr/bin/env python3
"""
AWS Cost Exporter for Prometheus
Monitors daily AWS costs and quotas for trading bot infrastructure
"""

import boto3
import time
import datetime
import logging
from typing import Dict, Optional, List
from prometheus_client import Gauge, start_http_server
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("aws_cost_exporter")


class AWSCostExporter:
    """Exports AWS cost and quota metrics to Prometheus."""

    def __init__(self, region: str = "us-east-1", port: int = 8005):
        """
        Initialize AWS cost exporter.

        Args:
            region: AWS region for cost explorer
            port: Prometheus metrics server port
        """
        self.region = region
        self.port = port

        # Initialize Prometheus gauges
        self.daily_cost_gauge = Gauge("aws_daily_cost_usd", "AWS cost today in USD")
        self.monthly_cost_gauge = Gauge(
            "aws_monthly_cost_usd", "AWS cost this month in USD"
        )
        self.service_cost_gauge = Gauge(
            "aws_service_cost_usd", "AWS cost by service in USD", ["service"]
        )
        self.quota_usage_gauge = Gauge(
            "aws_quota_usage_pct",
            "AWS service quota usage percentage",
            ["service", "quota"],
        )

        try:
            # Initialize AWS clients
            self.ce_client = boto3.client("ce", region_name=region)  # Cost Explorer
            self.service_quotas_client = boto3.client(
                "service-quotas", region_name=region
            )

            logger.info(
                f"💰 AWS Cost Exporter initialized (region: {region}, port: {port})"
            )

        except NoCredentialsError:
            logger.error("❌ AWS credentials not configured")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize AWS clients: {e}")
            raise

    def get_cost_and_usage(
        self, start_date: datetime.date, end_date: datetime.date
    ) -> Dict[str, any]:
        """
        Get cost and usage data from AWS Cost Explorer.

        Args:
            start_date: Start date for cost query
            end_date: End date for cost query

        Returns:
            Cost and usage data
        """
        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-%d"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Granularity="DAILY",
                Metrics=["BlendedCost", "UnblendedCost", "UsageQuantity"],
                GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
            )

            return response

        except ClientError as e:
            logger.error(f"❌ AWS Cost Explorer error: {e}")
            return {}
        except Exception as e:
            logger.error(f"❌ Unexpected error getting cost data: {e}")
            return {}

    def get_daily_cost(self) -> float:
        """Get today's AWS costs."""
        try:
            today = datetime.date.today()
            yesterday = today - datetime.timedelta(days=1)

            # Cost Explorer data is typically 1 day behind, so query yesterday
            response = self.get_cost_and_usage(yesterday, today)

            if not response or "ResultsByTime" not in response:
                logger.warning("⚠️ No cost data available")
                return 0.0

            total_cost = 0.0

            for result in response["ResultsByTime"]:
                if result.get("Total", {}).get("BlendedCost", {}).get("Amount"):
                    cost = float(result["Total"]["BlendedCost"]["Amount"])
                    total_cost += cost

            logger.info(f"💰 Daily cost: ${total_cost:.2f}")
            return total_cost

        except Exception as e:
            logger.error(f"❌ Error getting daily cost: {e}")
            return 0.0

    def get_monthly_cost(self) -> float:
        """Get this month's AWS costs."""
        try:
            today = datetime.date.today()
            month_start = today.replace(day=1)

            response = self.get_cost_and_usage(month_start, today)

            if not response or "ResultsByTime" not in response:
                logger.warning("⚠️ No monthly cost data available")
                return 0.0

            total_cost = 0.0

            for result in response["ResultsByTime"]:
                if result.get("Total", {}).get("BlendedCost", {}).get("Amount"):
                    cost = float(result["Total"]["BlendedCost"]["Amount"])
                    total_cost += cost

            logger.info(f"💰 Monthly cost: ${total_cost:.2f}")
            return total_cost

        except Exception as e:
            logger.error(f"❌ Error getting monthly cost: {e}")
            return 0.0

    def get_service_costs(self) -> Dict[str, float]:
        """Get costs broken down by AWS service."""
        try:
            today = datetime.date.today()
            yesterday = today - datetime.timedelta(days=1)

            response = self.get_cost_and_usage(yesterday, today)

            if not response or "ResultsByTime" not in response:
                return {}

            service_costs = {}

            for result in response["ResultsByTime"]:
                for group in result.get("Groups", []):
                    service = group["Keys"][0] if group.get("Keys") else "Unknown"
                    cost_data = group.get("Metrics", {}).get("BlendedCost", {})

                    if cost_data.get("Amount"):
                        cost = float(cost_data["Amount"])
                        service_costs[service] = service_costs.get(service, 0.0) + cost

            logger.debug(f"📊 Service costs: {service_costs}")
            return service_costs

        except Exception as e:
            logger.error(f"❌ Error getting service costs: {e}")
            return {}

    def get_quota_usage(self) -> Dict[str, Dict[str, float]]:
        """Get AWS service quota usage percentages."""
        try:
            # Common services to monitor for trading bot
            services = ["ec2", "s3", "lambda", "apigateway", "rds"]
            quota_usage = {}

            for service in services:
                try:
                    response = self.service_quotas_client.list_service_quotas(
                        ServiceCode=service, MaxResults=10
                    )

                    for quota in response.get("Quotas", []):
                        quota_name = quota.get("QuotaName", "Unknown")
                        quota_value = quota.get("Value", 0)

                        # For demo purposes, generate mock usage
                        # In production, you'd get actual usage from CloudWatch or other APIs
                        usage_pct = min(100.0, (quota_value * 0.3))  # Mock 30% usage

                        if service not in quota_usage:
                            quota_usage[service] = {}

                        quota_usage[service][quota_name] = usage_pct

                except ClientError as e:
                    logger.warning(f"⚠️ Cannot get quotas for {service}: {e}")
                    continue

            return quota_usage

        except Exception as e:
            logger.error(f"❌ Error getting quota usage: {e}")
            return {}

    def update_metrics(self):
        """Update all Prometheus metrics."""
        try:
            logger.info("📊 Updating AWS cost metrics...")

            # Update daily cost
            daily_cost = self.get_daily_cost()
            self.daily_cost_gauge.set(daily_cost)

            # Update monthly cost
            monthly_cost = self.get_monthly_cost()
            self.monthly_cost_gauge.set(monthly_cost)

            # Update service costs
            service_costs = self.get_service_costs()
            for service, cost in service_costs.items():
                # Clean service name for Prometheus label
                clean_service = service.replace(" ", "_").lower()
                self.service_cost_gauge.labels(service=clean_service).set(cost)

            # Update quota usage
            quota_usage = self.get_quota_usage()
            for service, quotas in quota_usage.items():
                for quota_name, usage_pct in quotas.items():
                    clean_quota = quota_name.replace(" ", "_").lower()
                    self.quota_usage_gauge.labels(
                        service=service, quota=clean_quota
                    ).set(usage_pct)

            logger.info("✅ Metrics updated successfully")

        except Exception as e:
            logger.error(f"❌ Error updating metrics: {e}")

    def run_forever(self, update_interval: int = 3600):
        """
        Run cost exporter continuously.

        Args:
            update_interval: Metrics update interval in seconds (default: 1 hour)
        """
        logger.info(f"🚀 Starting AWS Cost Exporter on port {self.port}")

        # Start Prometheus HTTP server
        start_http_server(self.port)
        logger.info(
            f"📈 Prometheus metrics available at http://localhost:{self.port}/metrics"
        )

        try:
            while True:
                self.update_metrics()

                logger.info(f"😴 Sleeping for {update_interval/3600:.1f} hours")
                time.sleep(update_interval)

        except KeyboardInterrupt:
            logger.info("🛑 Cost exporter stopped by user")
        except Exception as e:
            logger.error(f"❌ Unexpected error in main loop: {e}")
            raise


def create_mock_cost_data():
    """Create mock cost data for testing without AWS credentials."""
    return {
        "daily_cost": 15.75,
        "monthly_cost": 420.50,
        "service_costs": {
            "ec2": 8.50,
            "s3": 2.25,
            "lambda": 1.75,
            "cloudwatch": 1.50,
            "apigateway": 1.75,
        },
    }


class MockCostExporter:
    """Mock cost exporter for testing without AWS credentials."""

    def __init__(self, port: int = 8005):
        self.port = port
        self.daily_cost_gauge = Gauge("aws_daily_cost_usd", "AWS cost today in USD")
        self.monthly_cost_gauge = Gauge(
            "aws_monthly_cost_usd", "AWS cost this month in USD"
        )
        self.service_cost_gauge = Gauge(
            "aws_service_cost_usd", "AWS cost by service in USD", ["service"]
        )

        logger.info(f"🧪 Mock AWS Cost Exporter initialized (port: {port})")

    def update_metrics(self):
        """Update metrics with mock data."""
        mock_data = create_mock_cost_data()

        self.daily_cost_gauge.set(mock_data["daily_cost"])
        self.monthly_cost_gauge.set(mock_data["monthly_cost"])

        for service, cost in mock_data["service_costs"].items():
            self.service_cost_gauge.labels(service=service).set(cost)

        logger.info(
            f"📊 Mock metrics updated: ${mock_data['daily_cost']:.2f} daily, ${mock_data['monthly_cost']:.2f} monthly"
        )

    def run_forever(self, update_interval: int = 3600):
        """Run mock exporter continuously."""
        logger.info(f"🧪 Starting Mock AWS Cost Exporter on port {self.port}")

        start_http_server(self.port)
        logger.info(
            f"📈 Mock metrics available at http://localhost:{self.port}/metrics"
        )

        try:
            while True:
                self.update_metrics()
                logger.info(f"😴 Sleeping for {update_interval/60:.0f} minutes")
                time.sleep(update_interval)

        except KeyboardInterrupt:
            logger.info("🛑 Mock cost exporter stopped by user")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AWS Cost Exporter for Prometheus")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--port", type=int, default=8005, help="Prometheus server port")
    parser.add_argument(
        "--interval", type=int, default=3600, help="Update interval in seconds"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Use mock data (no AWS credentials needed)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test AWS connection and exit"
    )

    args = parser.parse_args()

    try:
        if args.mock:
            exporter = MockCostExporter(port=args.port)
        else:
            exporter = AWSCostExporter(region=args.region, port=args.port)

        if args.test:
            logger.info("🧪 Testing AWS cost data retrieval...")
            if hasattr(exporter, "update_metrics"):
                exporter.update_metrics()
                logger.info("✅ Test completed successfully")
            return

        # Run continuously
        exporter.run_forever(update_interval=args.interval)

    except KeyboardInterrupt:
        logger.info("🛑 Exporter stopped by user")
    except Exception as e:
        logger.error(f"❌ Exporter failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
