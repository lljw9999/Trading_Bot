#!/usr/bin/env python3
"""
Preflight Supercheck
10-second sanity sweep to fail fast if anything is off before enabling live flow
"""

import sys
import time
import os
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any

import redis
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("preflight_supercheck")


class PreflightSupercheck:
    """Comprehensive pre-launch system validation."""

    def __init__(self):
        """Initialize preflight checker."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.checks_passed = True
        self.check_results = []

        # Critical thresholds
        self.thresholds = {
            "feed_staleness_max_s": 5.0,
            "recon_mismatches_max": 0,
            "prom_push_max_age_s": 90.0,
            "service_heartbeat_max_age_s": 30.0,
            "capital_effective_min": 0.05,  # Must have at least 5% capital
            "capital_effective_max": 1.0,  # Must not exceed 100%
        }

    def check(self, name: str, condition: bool, details: str = "") -> bool:
        """Record a check result."""
        status = "OK" if condition else "FAIL"
        self.check_results.append((name, status, details))
        if not condition:
            self.checks_passed = False
        return condition

    def check_redis_connectivity(self) -> bool:
        """Check Redis is up and responding."""
        try:
            result = self.redis.ping()
            return self.check("redis_ping", result, "Redis connectivity")
        except Exception as e:
            return self.check("redis_ping", False, f"Redis error: {e}")

    def check_feed_freshness(self) -> bool:
        """Check market data feeds are fresh."""
        try:
            staleness = float(self.redis.get("feed:staleness:max_s") or 999)
            is_fresh = staleness < self.thresholds["feed_staleness_max_s"]
            return self.check(
                "feeds_fresh", is_fresh, f"Max staleness: {staleness:.1f}s"
            )
        except Exception as e:
            return self.check("feeds_fresh", False, f"Feed check error: {e}")

    def check_reconciliation_status(self) -> bool:
        """Check no reconciliation breaches."""
        try:
            # Check position mismatches
            position_mismatches = int(self.redis.get("recon:position_mismatches") or 0)
            pos_ok = position_mismatches <= self.thresholds["recon_mismatches_max"]

            # Check reconciliation breaches in last 24h
            recon_breaches = int(self.redis.get("recon:breaches_24h") or 0)
            breach_ok = recon_breaches == 0

            # Overall recon status
            recon_ok = pos_ok and breach_ok
            details = f"Position mismatches: {position_mismatches}, 24h breaches: {recon_breaches}"

            return self.check("recon_clean", recon_ok, details)
        except Exception as e:
            return self.check("recon_clean", False, f"Recon check error: {e}")

    def check_feature_flags(self) -> bool:
        """Check feature flags are in expected state."""
        try:
            # Check system mode is auto (not halt/manual)
            mode = self.redis.get("mode") or "unknown"
            mode_ok = mode == "auto"

            # Check critical feature flags exist
            flags = self.redis.hgetall("features:flags") or {}

            # Ensure shadow flags are set correctly for production readiness
            expected_flags = {
                "EXEC_RL_SHADOW": "1",  # RL should start in shadow
                "BANDIT_WEIGHTS": "0",  # Will be enabled by cutover
                "LLM_SENTIMENT": "0",  # Will be enabled by cutover
                "HEDGE_ENABLED": "1",  # Hedging should be ready
            }

            flags_ok = True
            flag_details = []
            for flag, expected in expected_flags.items():
                actual = flags.get(flag, "0")
                if actual != expected:
                    flags_ok = False
                flag_details.append(f"{flag}={actual}")

            overall_ok = mode_ok and flags_ok
            details = f"Mode: {mode}, Flags: {', '.join(flag_details)}"

            return self.check("feature_flags", overall_ok, details)
        except Exception as e:
            return self.check("feature_flags", False, f"Flag check error: {e}")

    def check_prometheus_exporter(self) -> bool:
        """Check Prometheus metrics are being exported."""
        try:
            last_push = float(self.redis.get("prom:last_push_s") or 999)
            current_time = time.time()
            age = current_time - last_push

            is_recent = age < self.thresholds["prom_push_max_age_s"]
            return self.check("prom_exporter", is_recent, f"Last push {age:.1f}s ago")
        except Exception as e:
            return self.check("prom_exporter", False, f"Prometheus check error: {e}")

    def check_service_heartbeats(self) -> bool:
        """Check critical services are alive."""
        try:
            services = ["trading_bot", "ops_bot", "risk_monitor"]
            all_alive = True
            service_details = []

            for service in services:
                heartbeat_key = f"service:{service}:heartbeat"
                last_heartbeat = self.redis.get(heartbeat_key)

                if last_heartbeat:
                    age = time.time() - float(last_heartbeat)
                    is_alive = age < self.thresholds["service_heartbeat_max_age_s"]
                    service_details.append(f"{service}:{age:.0f}s")
                else:
                    is_alive = False
                    service_details.append(f"{service}:MISSING")

                if not is_alive:
                    all_alive = False

            return self.check(
                "service_heartbeats", all_alive, ", ".join(service_details)
            )
        except Exception as e:
            return self.check(
                "service_heartbeats", False, f"Heartbeat check error: {e}"
            )

    def check_capital_allocation(self) -> bool:
        """Check capital allocation is in safe range."""
        try:
            capital_effective = float(self.redis.get("risk:capital_effective") or 0.4)

            in_range = (
                self.thresholds["capital_effective_min"]
                <= capital_effective
                <= self.thresholds["capital_effective_max"]
            )

            return self.check(
                "capital_range", in_range, f"Capital effective: {capital_effective:.1%}"
            )
        except Exception as e:
            return self.check("capital_range", False, f"Capital check error: {e}")

    def check_external_tokens(self) -> bool:
        """Check Slack and Grafana tokens are present."""
        try:
            slack_token = os.getenv("SLACK_BOT_TOKEN", "")
            grafana_token = os.getenv("GRAFANA_API_KEY", "")

            slack_ok = (
                slack_token
                and len(slack_token) > 10
                and not slack_token.startswith("xoxb-test")
            )
            grafana_ok = grafana_token and len(grafana_token) > 10

            tokens_ok = slack_ok and grafana_ok
            details = f"Slack: {'OK' if slack_ok else 'MISSING'}, Grafana: {'OK' if grafana_ok else 'MISSING'}"

            return self.check("external_tokens", tokens_ok, details)
        except Exception as e:
            return self.check("external_tokens", False, f"Token check error: {e}")

    def check_system_resources(self) -> bool:
        """Check system has adequate resources."""
        try:
            # Check Redis memory usage
            try:
                redis_info = self.redis.info("memory")
                used_memory_mb = redis_info.get("used_memory", 0) / (1024 * 1024)
                memory_ok = used_memory_mb < 1000  # Less than 1GB
            except:
                memory_ok = True  # Assume OK if we can't check

            # Check system load if available
            try:
                with open("/proc/loadavg", "r") as f:
                    load_1min = float(f.read().split()[0])
                load_ok = load_1min < 4.0  # Reasonable load threshold
            except:
                load_ok = True  # Assume OK if we can't check

            resources_ok = memory_ok and load_ok
            details = f"Redis memory: {used_memory_mb:.0f}MB"
            if not memory_ok or not load_ok:
                details += f", Load: {load_1min:.1f}" if "load_1min" in locals() else ""

            return self.check("system_resources", resources_ok, details)
        except Exception as e:
            return self.check("system_resources", False, f"Resource check error: {e}")

    def check_risk_controls(self) -> bool:
        """Check risk control systems are active."""
        try:
            # Check if risk controls are enabled
            risk_enabled = bool(int(self.redis.get("risk:enabled") or 1))

            # Check position limits are set
            position_limit = float(self.redis.get("risk:position_limit_usd") or 0)
            limits_ok = position_limit > 1000  # At least $1k position limit

            # Check drawdown monitoring
            max_drawdown = float(self.redis.get("risk:max_drawdown_pct") or 0.05)
            drawdown_ok = 0.01 <= max_drawdown <= 0.20  # 1-20% range

            risk_ok = risk_enabled and limits_ok and drawdown_ok
            details = f"Enabled: {risk_enabled}, Pos limit: ${position_limit:,.0f}, Max DD: {max_drawdown:.1%}"

            return self.check("risk_controls", risk_ok, details)
        except Exception as e:
            return self.check("risk_controls", False, f"Risk check error: {e}")

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all preflight checks."""
        try:
            start_time = time.time()
            logger.info("🚀 Starting preflight supercheck...")

            # Run all checks
            self.check_redis_connectivity()
            self.check_feed_freshness()
            self.check_reconciliation_status()
            self.check_feature_flags()
            self.check_prometheus_exporter()
            self.check_service_heartbeats()
            self.check_capital_allocation()
            self.check_external_tokens()
            self.check_system_resources()
            self.check_risk_controls()

            # Calculate results
            total_checks = len(self.check_results)
            passed_checks = sum(
                1 for _, status, _ in self.check_results if status == "OK"
            )
            failed_checks = total_checks - passed_checks

            duration = time.time() - start_time

            # Generate summary
            result = {
                "status": "PASS" if self.checks_passed else "FAIL",
                "total_checks": total_checks,
                "passed": passed_checks,
                "failed": failed_checks,
                "duration_seconds": duration,
                "timestamp": time.time(),
                "checks": self.check_results,
            }

            # Print results table
            self._print_results_table()

            logger.info(
                f"🎯 Preflight complete: {passed_checks}/{total_checks} checks passed "
                f"in {duration:.1f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error in preflight checks: {e}")
            return {"status": "ERROR", "error": str(e), "timestamp": time.time()}

    def _print_results_table(self):
        """Print formatted results table."""
        print("\n" + "=" * 60)
        print(f"{'PREFLIGHT SUPERCHECK RESULTS':^60}")
        print("=" * 60)
        print(f"{'CHECK':<25} {'STATUS':<10} {'DETAILS':<25}")
        print("-" * 60)

        for name, status, details in self.check_results:
            status_color = "🟢" if status == "OK" else "🔴"
            print(f"{name:<25} {status_color} {status:<8} {details:<25}")

        print("-" * 60)

        total = len(self.check_results)
        passed = sum(1 for _, status, _ in self.check_results if status == "OK")
        failed = total - passed

        overall_status = "🟢 PASS" if self.checks_passed else "🔴 FAIL"
        print(f"{'OVERALL':<25} {overall_status:<10} {passed}/{total} checks passed")
        print("=" * 60)

        if not self.checks_passed:
            print("\n❌ PREFLIGHT FAILED - DO NOT PROCEED TO LIVE TRADING")
            print("Fix the failed checks above before continuing.")
        else:
            print("\n✅ PREFLIGHT PASSED - SYSTEM READY FOR CUTOVER")

        print()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Preflight Supercheck")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--silent", action="store_true", help="Suppress table output (for scripts)"
    )

    args = parser.parse_args()

    # Create checker and run
    checker = PreflightSupercheck()
    result = checker.run_all_checks()

    if args.json:
        import json

        print(json.dumps(result, indent=2, default=str))
    elif args.silent:
        # Just print pass/fail for script usage
        print("PASS" if result["status"] == "PASS" else "FAIL")

    # Exit with appropriate code
    if result["status"] == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
