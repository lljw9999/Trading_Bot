#!/usr/bin/env python3
"""
Trading System PnL Reporting Script

12-hour PnL snapshot for paper trading validation.
Outputs OK/FAIL to stdout for PagerDuty integration.

Usage: python scripts/report_pnl.py
Exit codes: 0=OK, 1=FAIL
"""

import sys
import json
import redis
import datetime
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class PnLReporter:
    """Generates PnL snapshots and validates against thresholds."""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        """Initialize PnL reporter with Redis connection."""
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )
        self.report_time = datetime.datetime.now()

        # PnL validation thresholds
        self.max_drawdown_pct = 2.0  # Max 2% drawdown
        self.max_daily_loss_usd = 1000  # Max $1000 daily loss
        self.min_sharpe_ratio = 0.5  # Min Sharpe ratio
        self.max_var_breach_count = 5  # Max VaR breaches per 12h

    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Retrieve current portfolio metrics from Redis."""
        try:
            metrics = {}

            # Get position sizes for all symbols
            position_keys = self.redis_client.keys("position_size_usd:*")
            total_position_value = 0.0

            for key in position_keys:
                symbol = key.split(":")[1]
                position_size = float(self.redis_client.get(key) or 0.0)
                metrics[f"position_{symbol}"] = position_size
                total_position_value += abs(position_size)

            metrics["total_position_value"] = total_position_value

            # Get VaR metrics
            var_keys = self.redis_client.keys("var_pct:*")
            var_values = []
            for key in var_keys:
                var_value = float(self.redis_client.get(key) or 0.0)
                var_values.append(var_value)

            metrics["current_var_pct"] = max(var_values) if var_values else 0.0
            metrics["avg_var_pct"] = (
                sum(var_values) / len(var_values) if var_values else 0.0
            )

            # Get edge metrics
            edge_keys = self.redis_client.keys("edge_blended_bps:*")
            edge_values = []
            for key in edge_keys:
                edge_value = float(self.redis_client.get(key) or 0.0)
                edge_values.append(edge_value)

            metrics["avg_edge_bps"] = (
                sum(edge_values) / len(edge_values) if edge_values else 0.0
            )
            metrics["max_edge_bps"] = max(edge_values) if edge_values else 0.0

            return metrics

        except Exception as e:
            logger.error(f"Failed to retrieve portfolio metrics: {e}")
            return {}

    def calculate_pnl_estimates(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate estimated PnL based on current metrics."""
        pnl_data = {
            "estimated_daily_pnl_usd": 0.0,
            "estimated_12h_pnl_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_estimate": 0.0,
            "var_breach_count": 0,
        }

        try:
            # Estimate daily PnL based on edge and position size
            total_position = metrics.get("total_position_value", 0.0)
            avg_edge_bps = metrics.get("avg_edge_bps", 0.0)

            # Simple estimate: PnL = position_size * edge_bps / 10000 * trading_frequency
            # Assuming ~50 trades per day on average
            estimated_daily_pnl = total_position * (avg_edge_bps / 10000) * 50
            pnl_data["estimated_daily_pnl_usd"] = estimated_daily_pnl
            pnl_data["estimated_12h_pnl_usd"] = estimated_daily_pnl / 2

            # Estimate drawdown based on VaR
            current_var = metrics.get("current_var_pct", 0.0)
            pnl_data["max_drawdown_pct"] = min(
                current_var * 1.5, 5.0
            )  # Conservative estimate

            # Estimate Sharpe ratio (simplified)
            if avg_edge_bps > 0:
                pnl_data["sharpe_estimate"] = min(
                    avg_edge_bps / 10.0, 3.0
                )  # Rough approximation

            # Count VaR breaches (mock data for now)
            # In production, this would check historical VaR breach events
            pnl_data["var_breach_count"] = int(
                current_var / 20
            )  # Estimate based on current VaR

        except Exception as e:
            logger.error(f"Failed to calculate PnL estimates: {e}")

        return pnl_data

    def validate_pnl_thresholds(
        self, pnl_data: Dict[str, float]
    ) -> tuple[bool, list[str]]:
        """Validate PnL data against defined thresholds."""
        is_healthy = True
        issues = []

        # Check daily loss limit
        if pnl_data["estimated_daily_pnl_usd"] < -self.max_daily_loss_usd:
            is_healthy = False
            issues.append(
                f"Daily loss exceeds limit: ${pnl_data['estimated_daily_pnl_usd']:.2f} "
                f"< -${self.max_daily_loss_usd}"
            )

        # Check drawdown limit
        if pnl_data["max_drawdown_pct"] > self.max_drawdown_pct:
            is_healthy = False
            issues.append(
                f"Drawdown exceeds limit: {pnl_data['max_drawdown_pct']:.2f}% "
                f"> {self.max_drawdown_pct}%"
            )

        # Check Sharpe ratio
        if pnl_data["sharpe_estimate"] < self.min_sharpe_ratio:
            is_healthy = False
            issues.append(
                f"Sharpe ratio below minimum: {pnl_data['sharpe_estimate']:.2f} "
                f"< {self.min_sharpe_ratio}"
            )

        # Check VaR breach count
        if pnl_data["var_breach_count"] > self.max_var_breach_count:
            is_healthy = False
            issues.append(
                f"VaR breaches exceed limit: {pnl_data['var_breach_count']} "
                f"> {self.max_var_breach_count}"
            )

        return is_healthy, issues

    def generate_report(self) -> Dict[str, Any]:
        """Generate complete PnL report."""
        logger.info("🔍 Generating 12-hour PnL snapshot...")

        # Get current metrics
        portfolio_metrics = self.get_portfolio_metrics()
        if not portfolio_metrics:
            logger.error("❌ Failed to retrieve portfolio metrics")
            return {"status": "FAIL", "error": "No portfolio data available"}

        # Calculate PnL estimates
        pnl_data = self.calculate_pnl_estimates(portfolio_metrics)

        # Validate against thresholds
        is_healthy, issues = self.validate_pnl_thresholds(pnl_data)

        # Build report
        report = {
            "timestamp": self.report_time.isoformat(),
            "status": "OK" if is_healthy else "FAIL",
            "portfolio_metrics": portfolio_metrics,
            "pnl_estimates": pnl_data,
            "validation": {
                "is_healthy": is_healthy,
                "issues": issues,
                "thresholds": {
                    "max_drawdown_pct": self.max_drawdown_pct,
                    "max_daily_loss_usd": self.max_daily_loss_usd,
                    "min_sharpe_ratio": self.min_sharpe_ratio,
                    "max_var_breach_count": self.max_var_breach_count,
                },
            },
        }

        return report

    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print human-readable report summary."""
        status = report["status"]
        pnl_est = report["pnl_estimates"]
        portfolio = report["portfolio_metrics"]

        print(f"\n📊 PnL Report Summary - {report['timestamp']}")
        print(f"{'='*50}")
        print(f"Status: {status}")
        print(f"Total Position Value: ${portfolio.get('total_position_value', 0):.2f}")
        print(f"Estimated 12h PnL: ${pnl_est.get('estimated_12h_pnl_usd', 0):.2f}")
        print(f"Estimated Daily PnL: ${pnl_est.get('estimated_daily_pnl_usd', 0):.2f}")
        print(f"Current VaR: {portfolio.get('current_var_pct', 0):.2f}%")
        print(f"Average Edge: {portfolio.get('avg_edge_bps', 0):.2f} bps")
        print(f"Max Drawdown: {pnl_est.get('max_drawdown_pct', 0):.2f}%")
        print(f"Sharpe Estimate: {pnl_est.get('sharpe_estimate', 0):.2f}")

        if report["validation"]["issues"]:
            print(f"\n⚠️  Issues Detected:")
            for issue in report["validation"]["issues"]:
                print(f"  • {issue}")
        else:
            print(f"\n✅ All validation checks passed")


def main():
    """Main entry point for PnL reporting."""
    try:
        # Initialize reporter
        reporter = PnLReporter()

        # Generate report
        report = reporter.generate_report()

        # Print summary to stdout
        reporter.print_summary(report)

        # Output status for PagerDuty integration
        status = report["status"]
        print(f"\nFINAL_STATUS: {status}")

        # Write detailed report to log
        logger.info(f"PnL Report: {json.dumps(report, indent=2)}")

        # Exit with appropriate code
        sys.exit(0 if status == "OK" else 1)

    except Exception as e:
        logger.error(f"❌ PnL reporting failed: {e}")
        print(f"FINAL_STATUS: FAIL")
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
