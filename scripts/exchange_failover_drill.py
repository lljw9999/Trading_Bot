#!/usr/bin/env python3
"""
Venue Outage Failover Drill
Simulate WS drop + rate-limit (429) on primary venue for 5 min
"""

import os
import sys
import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

import redis
import requests
import websocket

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("exchange_failover_drill")


class ExchangeFailoverDrill:
    """Simulates exchange outages and tests failover mechanisms."""

    def __init__(self):
        """Initialize failover drill."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Drill configuration
        self.config = {
            "drill_duration_minutes": 5,
            "primary_venue": "binance",
            "secondary_venue": "coinbase",
            "tertiary_venue": "dydx",
            "simulate_ws_drop": True,
            "simulate_rate_limit": True,
            "simulate_api_errors": True,
            "recovery_timeout_seconds": 30,
            "max_stale_orders": 0,  # Zero tolerance for stale orders
            "recon_breach_threshold": 0,  # Zero tolerance for recon breaches
        }

        # Test symbols and order scenarios
        self.test_scenarios = [
            {"symbol": "BTCUSDT", "side": "buy", "size": 0.01},
            {"symbol": "ETHUSDT", "side": "sell", "size": 0.1},
            {"symbol": "SOLUSDT", "side": "buy", "size": 1.0},
        ]

        # Venue configurations
        self.venues = {
            "binance": {
                "ws_url": "wss://stream.binance.com:9443/ws/btcusdt@ticker",
                "api_url": "https://api.binance.com/api/v3/exchangeInfo",
                "router_key": "router:weight:binance",
            },
            "coinbase": {
                "ws_url": "wss://ws-feed.exchange.coinbase.com",
                "api_url": "https://api.exchange.coinbase.com/products",
                "router_key": "router:weight:coinbase",
            },
            "dydx": {
                "ws_url": "wss://api.dydx.exchange/v3/ws",
                "api_url": "https://api.dydx.exchange/v3/markets",
                "router_key": "router:weight:dydx",
            },
        }

        logger.info("🎯 Exchange Failover Drill initialized")

    def get_baseline_metrics(self) -> Dict[str, Any]:
        """Get baseline system metrics before drill."""
        try:
            metrics = {
                "timestamp": time.time(),
                "recon_breaches": int(self.redis.get("recon:breaches_24h") or 0),
                "position_mismatches": int(
                    self.redis.get("recon:position_mismatches") or 0
                ),
                "stale_orders": int(self.redis.get("orders:stale_count") or 0),
                "venue_weights": {},
                "active_connections": {},
                "last_price_updates": {},
            }

            # Get venue router weights
            for venue, config in self.venues.items():
                weight = float(self.redis.get(config["router_key"]) or 0.0)
                metrics["venue_weights"][venue] = weight

                # Check connection status
                conn_key = f"ws:{venue}:connected"
                connected = bool(int(self.redis.get(conn_key) or 0))
                metrics["active_connections"][venue] = connected

                # Get last price update time
                price_key = f"price:{venue}:BTCUSDT:timestamp"
                last_update = float(self.redis.get(price_key) or 0)
                metrics["last_price_updates"][venue] = last_update

            logger.info("📊 Captured baseline metrics")
            return metrics

        except Exception as e:
            logger.error(f"Error getting baseline metrics: {e}")
            return {"error": str(e)}

    def simulate_ws_disconnect(
        self, venue: str, duration_seconds: int
    ) -> Dict[str, Any]:
        """Simulate WebSocket disconnect for a venue."""
        try:
            logger.info(
                f"🔌 Simulating WS disconnect for {venue} ({duration_seconds}s)"
            )

            # Set disconnect flag in Redis (mock simulation)
            disconnect_key = f"ws:{venue}:simulate_disconnect"
            self.redis.setex(disconnect_key, duration_seconds, 1)

            # Set connection status to false
            conn_key = f"ws:{venue}:connected"
            self.redis.set(conn_key, 0)

            # Stop price updates (simulate stale data)
            price_key = f"price:{venue}:*:stale"
            self.redis.set(f"price:{venue}:BTCUSDT:stale", 1)
            self.redis.set(f"price:{venue}:ETHUSDT:stale", 1)

            return {
                "action": "ws_disconnect",
                "venue": venue,
                "duration": duration_seconds,
                "status": "simulated",
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error simulating WS disconnect for {venue}: {e}")
            return {
                "action": "ws_disconnect",
                "venue": venue,
                "status": "error",
                "error": str(e),
            }

    def simulate_rate_limiting(
        self, venue: str, duration_seconds: int
    ) -> Dict[str, Any]:
        """Simulate API rate limiting (429 errors) for a venue."""
        try:
            logger.info(
                f"⚡ Simulating rate limiting for {venue} ({duration_seconds}s)"
            )

            # Set rate limit flag
            rate_limit_key = f"api:{venue}:rate_limited"
            self.redis.setex(rate_limit_key, duration_seconds, 1)

            # Simulate 429 error responses
            error_key = f"api:{venue}:last_error"
            self.redis.set(error_key, "429 Too Many Requests")

            # Increment rate limit counter
            counter_key = f"api:{venue}:rate_limit_count"
            self.redis.incr(counter_key)

            return {
                "action": "rate_limiting",
                "venue": venue,
                "duration": duration_seconds,
                "status": "simulated",
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error simulating rate limiting for {venue}: {e}")
            return {
                "action": "rate_limiting",
                "venue": venue,
                "status": "error",
                "error": str(e),
            }

    def simulate_api_errors(self, venue: str, duration_seconds: int) -> Dict[str, Any]:
        """Simulate general API errors for a venue."""
        try:
            logger.info(f"❌ Simulating API errors for {venue} ({duration_seconds}s)")

            # Set API error flags
            error_key = f"api:{venue}:errors_enabled"
            self.redis.setex(error_key, duration_seconds, 1)

            # Simulate various error types
            errors = [
                "Connection timeout",
                "Internal server error",
                "Service temporarily unavailable",
                "Invalid request",
            ]

            for i, error in enumerate(errors):
                error_entry_key = f"api:{venue}:error:{i}"
                self.redis.setex(error_entry_key, duration_seconds, error)

            return {
                "action": "api_errors",
                "venue": venue,
                "duration": duration_seconds,
                "error_types": len(errors),
                "status": "simulated",
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error simulating API errors for {venue}: {e}")
            return {
                "action": "api_errors",
                "venue": venue,
                "status": "error",
                "error": str(e),
            }

    def check_router_failover(self) -> Dict[str, Any]:
        """Check if smart order router failed over to secondary venue."""
        try:
            # Check current venue weights
            primary_weight = float(
                self.redis.get(self.venues[self.config["primary_venue"]]["router_key"])
                or 0.0
            )
            secondary_weight = float(
                self.redis.get(
                    self.venues[self.config["secondary_venue"]]["router_key"]
                )
                or 0.0
            )

            # Router should have reduced primary and increased secondary
            failover_detected = primary_weight < 0.3 and secondary_weight > 0.4

            # Check routing decisions
            last_route = self.redis.get("router:last_venue") or "unknown"
            routing_to_secondary = last_route != self.config["primary_venue"]

            # Check venue health scores
            primary_health = float(
                self.redis.get(f"health:{self.config['primary_venue']}") or 0.0
            )
            secondary_health = float(
                self.redis.get(f"health:{self.config['secondary_venue']}") or 0.0
            )

            health_aware = primary_health < secondary_health

            result = {
                "failover_detected": failover_detected,
                "routing_to_secondary": routing_to_secondary,
                "health_aware": health_aware,
                "weights": {"primary": primary_weight, "secondary": secondary_weight},
                "last_route": last_route,
                "health_scores": {
                    "primary": primary_health,
                    "secondary": secondary_health,
                },
            }

            logger.info(
                f"🔄 Router failover check: {'✅' if failover_detected else '❌'}"
            )
            return result

        except Exception as e:
            logger.error(f"Error checking router failover: {e}")
            return {"failover_detected": False, "error": str(e)}

    def check_stale_order_cleanup(self) -> Dict[str, Any]:
        """Check if TTL watchdog cleared stale orders."""
        try:
            stale_count = int(self.redis.get("orders:stale_count") or 0)
            cleaned_count = int(self.redis.get("orders:cleaned_count") or 0)

            # Check if cleanup process ran recently
            last_cleanup = float(self.redis.get("orders:last_cleanup_time") or 0)
            cleanup_recent = (time.time() - last_cleanup) < 300  # Within 5 minutes

            # Check for any remaining stale orders
            stale_orders_cleared = stale_count <= self.config["max_stale_orders"]

            result = {
                "stale_orders_cleared": stale_orders_cleared,
                "cleanup_recent": cleanup_recent,
                "stale_count": stale_count,
                "cleaned_count": cleaned_count,
                "last_cleanup_time": last_cleanup,
                "max_allowed": self.config["max_stale_orders"],
            }

            logger.info(
                f"🧹 Stale order cleanup: {'✅' if stale_orders_cleared else '❌'}"
            )
            return result

        except Exception as e:
            logger.error(f"Error checking stale order cleanup: {e}")
            return {"stale_orders_cleared": False, "error": str(e)}

    def check_reconciliation_integrity(self) -> Dict[str, Any]:
        """Check if reconciliation stayed clean during outage."""
        try:
            current_breaches = int(self.redis.get("recon:breaches_24h") or 0)
            current_mismatches = int(self.redis.get("recon:position_mismatches") or 0)

            total_issues = current_breaches + current_mismatches
            recon_clean = total_issues <= self.config["recon_breach_threshold"]

            # Check recon process health
            last_recon = float(self.redis.get("recon:last_run_time") or 0)
            recon_recent = (time.time() - last_recon) < 600  # Within 10 minutes

            result = {
                "recon_clean": recon_clean,
                "recon_recent": recon_recent,
                "current_breaches": current_breaches,
                "current_mismatches": current_mismatches,
                "total_issues": total_issues,
                "threshold": self.config["recon_breach_threshold"],
                "last_recon_time": last_recon,
            }

            logger.info(f"🔍 Reconciliation check: {'✅' if recon_clean else '❌'}")
            return result

        except Exception as e:
            logger.error(f"Error checking reconciliation: {e}")
            return {"recon_clean": False, "error": str(e)}

    def wait_for_recovery(self, timeout_seconds: int) -> Dict[str, Any]:
        """Wait for system to recover and measure recovery time."""
        try:
            logger.info(f"⏱️ Waiting for system recovery (timeout: {timeout_seconds}s)")

            start_time = time.time()
            end_time = start_time + timeout_seconds

            while time.time() < end_time:
                # Check if primary venue is healthy again
                primary_connected = bool(
                    int(
                        self.redis.get(f"ws:{self.config['primary_venue']}:connected")
                        or 0
                    )
                )
                primary_not_rate_limited = not self.redis.exists(
                    f"api:{self.config['primary_venue']}:rate_limited"
                )

                # Check if router weights are rebalanced
                primary_weight = float(
                    self.redis.get(
                        self.venues[self.config["primary_venue"]]["router_key"]
                    )
                    or 0.0
                )
                weights_restored = (
                    primary_weight > 0.4
                )  # Primary should be getting traffic again

                if primary_connected and primary_not_rate_limited and weights_restored:
                    recovery_time = time.time() - start_time
                    logger.info(f"✅ System recovered in {recovery_time:.1f}s")

                    return {
                        "recovered": True,
                        "recovery_time": recovery_time,
                        "primary_connected": primary_connected,
                        "not_rate_limited": primary_not_rate_limited,
                        "weights_restored": weights_restored,
                    }

                time.sleep(1)  # Check every second

            # Timeout reached
            recovery_time = time.time() - start_time
            logger.warning(f"⏰ Recovery timeout after {recovery_time:.1f}s")

            return {"recovered": False, "recovery_time": recovery_time, "timeout": True}

        except Exception as e:
            logger.error(f"Error waiting for recovery: {e}")
            return {"recovered": False, "error": str(e)}

    def run_failover_drill(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run complete failover drill."""
        try:
            drill_start = time.time()
            logger.info("🎯 Starting exchange failover drill...")

            if dry_run:
                logger.info("🧪 DRY RUN MODE - Simulating drill without real outages")

            drill_result = {
                "status": "in_progress",
                "dry_run": dry_run,
                "start_time": drill_start,
                "config": self.config,
                "phases": [],
            }

            # Phase 1: Capture baseline
            logger.info("📊 Phase 1: Capturing baseline metrics")
            baseline_metrics = self.get_baseline_metrics()
            drill_result["baseline"] = baseline_metrics
            drill_result["phases"].append("baseline")

            # Phase 2: Simulate outages
            logger.info("🔥 Phase 2: Simulating venue outages")
            primary_venue = self.config["primary_venue"]
            drill_duration = self.config["drill_duration_minutes"] * 60

            outage_results = []

            if self.config["simulate_ws_drop"]:
                ws_result = self.simulate_ws_disconnect(primary_venue, drill_duration)
                outage_results.append(ws_result)

            if self.config["simulate_rate_limit"]:
                rate_limit_result = self.simulate_rate_limiting(
                    primary_venue, drill_duration
                )
                outage_results.append(rate_limit_result)

            if self.config["simulate_api_errors"]:
                api_error_result = self.simulate_api_errors(
                    primary_venue, drill_duration
                )
                outage_results.append(api_error_result)

            drill_result["outage_simulations"] = outage_results
            drill_result["phases"].append("simulate_outages")

            # Phase 3: Monitor failover (wait a bit for systems to react)
            logger.info("🔄 Phase 3: Monitoring failover response")
            time.sleep(10)  # Give systems time to react

            failover_check = self.check_router_failover()
            drill_result["failover_check"] = failover_check
            drill_result["phases"].append("check_failover")

            # Phase 4: Check system integrity during outage
            logger.info("🧪 Phase 4: Checking system integrity")

            # Wait through most of the drill duration
            remaining_time = drill_duration - 20  # Reserve 20s for final checks
            if remaining_time > 0:
                logger.info(f"⏱️ Waiting {remaining_time}s for drill duration...")
                time.sleep(remaining_time)

            stale_orders_check = self.check_stale_order_cleanup()
            recon_check = self.check_reconciliation_integrity()

            drill_result["stale_orders_check"] = stale_orders_check
            drill_result["recon_check"] = recon_check
            drill_result["phases"].append("integrity_check")

            # Phase 5: Wait for recovery
            logger.info("🔄 Phase 5: Waiting for recovery")
            recovery_result = self.wait_for_recovery(
                self.config["recovery_timeout_seconds"]
            )
            drill_result["recovery"] = recovery_result
            drill_result["phases"].append("recovery")

            # Phase 6: Final assessment
            logger.info("📋 Phase 6: Final assessment")

            # Determine overall drill success
            success_criteria = {
                "router_failover": failover_check.get("failover_detected", False),
                "stale_orders_clean": stale_orders_check.get(
                    "stale_orders_cleared", False
                ),
                "reconciliation_clean": recon_check.get("recon_clean", False),
                "system_recovered": recovery_result.get("recovered", False),
            }

            all_passed = all(success_criteria.values())
            drill_result["success_criteria"] = success_criteria
            drill_result["overall_success"] = all_passed

            if all_passed:
                drill_result["status"] = "PASS"
            else:
                drill_result["status"] = "FAIL"

            drill_duration_total = time.time() - drill_start
            drill_result["total_duration"] = drill_duration_total
            drill_result["phases"].append("assessment")

            logger.info(
                f"🎯 Failover drill completed: {drill_result['status']} "
                f"in {drill_duration_total:.1f}s"
            )

            return drill_result

        except Exception as e:
            logger.error(f"Error in failover drill: {e}")
            return {"status": "ERROR", "error": str(e), "timestamp": time.time()}

    def generate_drill_report(self, drill_result: Dict[str, Any]) -> str:
        """Generate comprehensive drill report."""
        try:
            report_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

            report = f"""# Exchange Failover Drill Report

**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
**Drill ID:** failover_drill_{report_timestamp}
**Status:** {drill_result.get('status', 'UNKNOWN')}
**Duration:** {drill_result.get('total_duration', 0):.1f} seconds

## Executive Summary

| Criterion | Result |
|-----------|---------|"""

            if "success_criteria" in drill_result:
                for criterion, passed in drill_result["success_criteria"].items():
                    status = "✅ PASS" if passed else "❌ FAIL"
                    report += (
                        f"\n| **{criterion.replace('_', ' ').title()}** | {status} |"
                    )

            report += f"""

**Overall Result:** {'🟢 PASS' if drill_result.get('overall_success', False) else '🔴 FAIL'}

## Drill Configuration

- **Primary Venue:** {self.config['primary_venue']}
- **Secondary Venue:** {self.config['secondary_venue']}
- **Drill Duration:** {self.config['drill_duration_minutes']} minutes
- **Recovery Timeout:** {self.config['recovery_timeout_seconds']} seconds

## Simulated Outages

"""

            if "outage_simulations" in drill_result:
                for outage in drill_result["outage_simulations"]:
                    action = outage.get("action", "unknown")
                    venue = outage.get("venue", "unknown")
                    status = outage.get("status", "unknown")
                    report += f"- **{action.replace('_', ' ').title()}** on {venue}: {status}\n"

            report += "\n## Test Results\n\n"

            # Failover Check
            if "failover_check" in drill_result:
                failover = drill_result["failover_check"]
                report += f"""### Router Failover
- **Failover Detected:** {'✅ Yes' if failover.get('failover_detected', False) else '❌ No'}
- **Routing to Secondary:** {'✅ Yes' if failover.get('routing_to_secondary', False) else '❌ No'}
- **Health Awareness:** {'✅ Yes' if failover.get('health_aware', False) else '❌ No'}
- **Primary Weight:** {failover.get('weights', {}).get('primary', 0):.2f}
- **Secondary Weight:** {failover.get('weights', {}).get('secondary', 0):.2f}

"""

            # Stale Orders Check
            if "stale_orders_check" in drill_result:
                stale = drill_result["stale_orders_check"]
                report += f"""### Stale Order Management
- **Orders Cleaned:** {'✅ Yes' if stale.get('stale_orders_cleared', False) else '❌ No'}
- **Cleanup Recent:** {'✅ Yes' if stale.get('cleanup_recent', False) else '❌ No'}
- **Stale Count:** {stale.get('stale_count', 0)}
- **Cleaned Count:** {stale.get('cleaned_count', 0)}

"""

            # Reconciliation Check
            if "recon_check" in drill_result:
                recon = drill_result["recon_check"]
                report += f"""### Reconciliation Integrity
- **Reconciliation Clean:** {'✅ Yes' if recon.get('recon_clean', False) else '❌ No'}
- **Process Recent:** {'✅ Yes' if recon.get('recon_recent', False) else '❌ No'}
- **Current Breaches:** {recon.get('current_breaches', 0)}
- **Position Mismatches:** {recon.get('current_mismatches', 0)}

"""

            # Recovery Check
            if "recovery" in drill_result:
                recovery = drill_result["recovery"]
                report += f"""### System Recovery
- **System Recovered:** {'✅ Yes' if recovery.get('recovered', False) else '❌ No'}
- **Recovery Time:** {recovery.get('recovery_time', 0):.1f} seconds
- **Primary Connected:** {'✅ Yes' if recovery.get('primary_connected', False) else '❌ No'}
- **Rate Limits Cleared:** {'✅ Yes' if recovery.get('not_rate_limited', False) else '❌ No'}

"""

            # Recommendations
            report += "## Recommendations\n\n"

            if drill_result.get("overall_success", False):
                report += """### 🟢 Drill Passed
The failover drill completed successfully. The system demonstrated:
- Automatic failover to secondary venue
- Proper cleanup of stale orders
- Maintained reconciliation integrity
- Successful recovery within timeout

**Next Steps:**
- Schedule regular failover drills (monthly)
- Monitor recovery time trends
- Consider reducing recovery timeout if consistently fast

"""
            else:
                report += """### 🔴 Drill Failed
The failover drill identified issues that require attention:

"""

                # Specific failure recommendations
                if "success_criteria" in drill_result:
                    criteria = drill_result["success_criteria"]
                    if not criteria.get("router_failover", False):
                        report += "- **Router Failover Failed:** Review smart order router logic and venue health scoring\n"
                    if not criteria.get("stale_orders_clean", False):
                        report += "- **Stale Orders Not Cleaned:** Review TTL watchdog configuration and order cleanup logic\n"
                    if not criteria.get("reconciliation_clean", False):
                        report += "- **Reconciliation Breached:** Review position tracking and reconciliation processes\n"
                    if not criteria.get("system_recovered", False):
                        report += "- **Recovery Failed:** Review venue health monitoring and weight restoration logic\n"

            report += f"""
---

*Report generated by Exchange Failover Drill*
*Next drill: {(datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")}*
*For questions: trading-infrastructure@company.com*
"""

            return report

        except Exception as e:
            logger.error(f"Error generating drill report: {e}")
            return f"# Drill Report Error\n\nError generating report: {e}\n"

    def save_drill_report(self, drill_result: Dict[str, Any]) -> str:
        """Save drill report to file."""
        try:
            # Create reports directory
            reports_dir = Path(__file__).parent.parent / "reports" / "drills"
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Generate report filename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"failover_drill_{timestamp}.md"

            # Generate and save report
            report_content = self.generate_drill_report(drill_result)
            with open(report_file, "w") as f:
                f.write(report_content)

            logger.info(f"💾 Saved drill report: {report_file}")
            return str(report_file)

        except Exception as e:
            logger.error(f"Error saving drill report: {e}")
            return ""


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Exchange Failover Drill")
    parser.add_argument("--run", action="store_true", help="Run failover drill")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (simulate without real outages)",
    )
    parser.add_argument(
        "--duration", type=int, default=5, help="Drill duration in minutes (default: 5)"
    )
    parser.add_argument(
        "--primary",
        type=str,
        default="binance",
        help="Primary venue to test (default: binance)",
    )
    parser.add_argument(
        "--secondary",
        type=str,
        default="coinbase",
        help="Secondary venue for failover (default: coinbase)",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    drill = ExchangeFailoverDrill()

    # Update config with command line arguments
    if args.duration:
        drill.config["drill_duration_minutes"] = args.duration
    if args.primary:
        drill.config["primary_venue"] = args.primary
    if args.secondary:
        drill.config["secondary_venue"] = args.secondary

    if args.run or args.dry_run or not sys.argv[1:]:  # Default to dry run
        result = drill.run_failover_drill(dry_run=args.dry_run or not args.run)

        # Save drill report
        report_file = drill.save_drill_report(result)
        result["report_file"] = report_file

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            status = result.get("status", "UNKNOWN")
            emoji = "✅" if status == "PASS" else ("❌" if status == "FAIL" else "❓")
            duration = result.get("total_duration", 0)

            print(f"{emoji} Failover Drill: {status} ({duration:.1f}s)")

            if "success_criteria" in result:
                print("Results:")
                for criterion, passed in result["success_criteria"].items():
                    check_emoji = "✅" if passed else "❌"
                    print(f"  {check_emoji} {criterion.replace('_', ' ').title()}")

            if report_file:
                print(f"📄 Report: {report_file}")

        # Exit code based on drill success
        if result.get("status") == "PASS":
            sys.exit(0)
        else:
            sys.exit(1)

    parser.print_help()


if __name__ == "__main__":
    main()
