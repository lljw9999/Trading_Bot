#!/usr/bin/env python3
"""
Chaos Monkey - Service Resilience Testing
Randomly kills services to test system recovery and auto-healing
"""

import random
import subprocess
import time
import logging
import redis
import json
import os
import signal
from typing import List, Dict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("chaos_monkey")


class ChaosMonkey:
    """Chaos engineering tool for testing service resilience."""

    def __init__(self):
        """Initialize Chaos Monkey."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Services to target for chaos testing
        self.victim_services = [
            "policy-blue",
            "policy-green",
            "price-ws",
            "risk",
            "whale_stream",
            "ops_bot",
            "redis-server",
        ]

        # Chaos settings
        self.min_wait_seconds = 3600  # 1 hour minimum
        self.max_wait_seconds = 7200  # 2 hours maximum
        self.enabled = True

        logger.info("🐒 Chaos Monkey initialized")
        logger.info(f"Target services: {', '.join(self.victim_services)}")

    def is_chaos_enabled(self) -> bool:
        """Check if chaos testing is enabled via Redis flag."""
        try:
            enabled = self.redis.get("chaos:enabled")
            return enabled == "true" if enabled else True
        except Exception:
            return self.enabled

    def set_chaos_enabled(self, enabled: bool):
        """Enable or disable chaos testing."""
        try:
            self.redis.set("chaos:enabled", "true" if enabled else "false")
            self.enabled = enabled
            logger.info(f"🎛️ Chaos testing {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            logger.error(f"Failed to set chaos state: {e}")

    def get_service_status(self, service: str) -> Dict[str, any]:
        """Get service status information."""
        try:
            # Check if service is running
            result = subprocess.run(
                ["pgrep", "-f", service], capture_output=True, text=True
            )

            is_running = result.returncode == 0

            return {
                "service": service,
                "running": is_running,
                "pid": result.stdout.strip() if is_running else None,
                "check_time": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error checking service {service}: {e}")
            return {
                "service": service,
                "running": False,
                "error": str(e),
                "check_time": datetime.now().isoformat(),
            }

    def kill_service(self, service: str, method: str = "TERM") -> bool:
        """
        Kill a service using various methods.

        Args:
            service: Service name to kill
            method: Kill method (TERM, KILL, systemctl)
        """
        try:
            logger.info(f"🎯 Targeting service: {service} (method: {method})")

            if method == "systemctl":
                # Use systemctl to kill service (production method)
                cmd = ["sudo", "systemctl", "kill", f"{service}.service"]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    logger.info(f"💀 Killed {service} via systemctl")
                    return True
                else:
                    logger.warning(
                        f"⚠️ systemctl kill failed for {service}: {result.stderr}"
                    )

            elif method in ["TERM", "KILL"]:
                # Use pkill to terminate/kill processes
                signal_flag = "-TERM" if method == "TERM" else "-KILL"
                cmd = ["pkill", signal_flag, "-f", service]

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    logger.info(f"💀 Killed {service} via pkill {signal_flag}")
                    return True
                else:
                    logger.warning(f"⚠️ pkill failed for {service} (may not be running)")

            return False

        except Exception as e:
            logger.error(f"❌ Failed to kill service {service}: {e}")
            return False

    def log_chaos_event(self, service: str, method: str, success: bool):
        """Log chaos event to Redis for analysis."""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "service": service,
                "method": method,
                "success": success,
                "monkey_version": "1.0",
            }

            # Store in Redis stream for monitoring
            self.redis.xadd("chaos:events", event, maxlen=1000)

            # Update chaos statistics
            stats_key = "chaos:stats"
            self.redis.hincrby(stats_key, "total_events", 1)

            if success:
                self.redis.hincrby(stats_key, "successful_kills", 1)
            else:
                self.redis.hincrby(stats_key, "failed_kills", 1)

            self.redis.hincrby(stats_key, f"service_{service}", 1)

        except Exception as e:
            logger.error(f"Failed to log chaos event: {e}")

    def wait_random_interval(self):
        """Wait for a random interval between min and max wait times."""
        wait_time = random.randint(self.min_wait_seconds, self.max_wait_seconds)
        wait_hours = wait_time / 3600

        logger.info(f"😴 Sleeping for {wait_hours:.1f} hours ({wait_time}s)")

        # Sleep in smaller intervals to allow for interruption
        sleep_interval = 60  # 1 minute chunks
        elapsed = 0

        while elapsed < wait_time:
            if not self.is_chaos_enabled():
                logger.info("🛑 Chaos testing disabled, stopping wait")
                break

            actual_sleep = min(sleep_interval, wait_time - elapsed)
            time.sleep(actual_sleep)
            elapsed += actual_sleep

            # Log progress every hour
            if elapsed % 3600 == 0:
                remaining_hours = (wait_time - elapsed) / 3600
                logger.debug(f"⏳ {remaining_hours:.1f} hours remaining")

    def select_victim(self) -> str:
        """Select a random service to target."""
        # Get service status to prefer running services
        running_services = []

        for service in self.victim_services:
            status = self.get_service_status(service)
            if status.get("running", False):
                running_services.append(service)

        # Prefer running services, fallback to all services
        candidates = running_services if running_services else self.victim_services

        victim = random.choice(candidates)
        logger.info(f"🎲 Selected victim: {victim}")

        return victim

    def run_chaos_cycle(self):
        """Run one chaos testing cycle."""
        if not self.is_chaos_enabled():
            logger.info("🚫 Chaos testing disabled, skipping cycle")
            return

        logger.info("🚀 Starting chaos cycle")

        # Select victim service
        victim = self.select_victim()

        # Choose kill method randomly
        methods = ["TERM", "KILL", "systemctl"]
        method = random.choice(methods)

        # Record pre-kill status
        pre_status = self.get_service_status(victim)

        # Execute chaos
        success = self.kill_service(victim, method)

        # Log the event
        self.log_chaos_event(victim, method, success)

        if success:
            logger.info(f"💥 Chaos executed: killed {victim} with {method}")

            # Wait a bit and check if service recovered
            time.sleep(30)  # 30 seconds recovery check

            post_status = self.get_service_status(victim)
            if post_status.get("running", False):
                logger.info(f"✅ Service {victim} recovered automatically")
            else:
                logger.warning(f"⚠️ Service {victim} did not recover automatically")
        else:
            logger.warning(f"⚠️ Chaos failed: could not kill {victim}")

        logger.info("🏁 Chaos cycle completed")

    def run_forever(self):
        """Run chaos monkey continuously."""
        logger.info("🐒 Starting Chaos Monkey main loop")

        try:
            while True:
                # Wait random interval
                self.wait_random_interval()

                # Check if still enabled
                if not self.is_chaos_enabled():
                    logger.info("🛑 Chaos testing disabled, exiting")
                    break

                # Run chaos cycle
                self.run_chaos_cycle()

        except KeyboardInterrupt:
            logger.info("🛑 Chaos Monkey stopped by user")
        except Exception as e:
            logger.error(f"❌ Unexpected error in chaos loop: {e}")
            raise

    def get_stats(self) -> Dict[str, any]:
        """Get chaos testing statistics."""
        try:
            stats = self.redis.hgetall("chaos:stats")

            # Convert string values to integers
            numeric_stats = {}
            for key, value in stats.items():
                try:
                    numeric_stats[key] = int(value)
                except (ValueError, TypeError):
                    numeric_stats[key] = value

            return numeric_stats

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Chaos Monkey for service resilience testing"
    )
    parser.add_argument("--enable", action="store_true", help="Enable chaos testing")
    parser.add_argument("--disable", action="store_true", help="Disable chaos testing")
    parser.add_argument(
        "--once", action="store_true", help="Run one chaos cycle and exit"
    )
    parser.add_argument("--stats", action="store_true", help="Show chaos statistics")
    parser.add_argument("--target", help="Specific service to target for testing")
    parser.add_argument(
        "--method",
        choices=["TERM", "KILL", "systemctl"],
        default="TERM",
        help="Kill method",
    )

    args = parser.parse_args()

    monkey = ChaosMonkey()

    if args.enable:
        monkey.set_chaos_enabled(True)
        return

    if args.disable:
        monkey.set_chaos_enabled(False)
        return

    if args.stats:
        stats = monkey.get_stats()
        if stats:
            print("📊 Chaos Monkey Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        else:
            print("No chaos statistics available")
        return

    if args.once:
        logger.info("🐒 Running single chaos cycle")
        monkey.run_chaos_cycle()
        return

    if args.target:
        logger.info(f"🎯 Targeting specific service: {args.target}")
        success = monkey.kill_service(args.target, args.method)
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        return

    # Default: run continuously
    monkey.run_forever()


if __name__ == "__main__":
    main()
