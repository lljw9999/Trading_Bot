#!/usr/bin/env python3
"""
Economic Event Guard

Gates position sizing around major economic events:
- CPI, FOMC, earnings releases
- Sets risk:event_lock 15-30 minutes before/after events
- Caps positions to 25% of normal during events
- Exempts basis carry and market making if desired
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import pytz

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import redis
    import requests

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger("economic_event_guard")


class EconomicEventGuard:
    """
    Monitors economic events and gates position sizing during high-impact periods.
    Protects against adverse selection during volatile news periods.
    """

    def __init__(self):
        """Initialize economic event guard."""
        self.redis_client = None
        if DEPS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        # Event guard configuration
        self.config = {
            "event_sources": [
                "fed",
                "bls",
                "earnings",
            ],  # Fed, Bureau of Labor Statistics, earnings
            "high_impact_events": [
                "FOMC Rate Decision",
                "FOMC Minutes",
                "Consumer Price Index",
                "Core CPI",
                "Non-Farm Payrolls",
                "Federal Reserve Speech",
                "GDP Release",
                "Inflation Data",
            ],
            "guard_window_minutes": {
                "before": 30,  # 30 min before
                "after": 15,  # 15 min after
            },
            "position_cap_during_event": 0.25,  # 25% of normal sizing
            "exempt_strategies": [
                "basis_carry",
                "market_maker",
            ],  # Strategies exempt from caps
            "timezone": "America/New_York",  # Eastern time for US events
        }

        logger.info("Initialized economic event guard")

    def fetch_economic_calendar(self, days_ahead: int = 7) -> List[Dict[str, any]]:
        """
        Fetch economic calendar from data source.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of economic events
        """
        try:
            logger.info(f"📅 Fetching economic calendar for next {days_ahead} days")

            # Mock implementation - in production would fetch from:
            # - Federal Reserve API
            # - Economic calendar APIs
            # - Earnings calendar APIs
            events = self._get_mock_economic_calendar(days_ahead)

            logger.info(
                f"Found {len(events)} economic events in next {days_ahead} days"
            )
            return events

        except Exception as e:
            logger.error(f"Error fetching economic calendar: {e}")
            return []

    def _get_mock_economic_calendar(self, days_ahead: int) -> List[Dict[str, any]]:
        """Get mock economic calendar for testing."""
        from datetime import datetime
        import random

        events = []
        et = pytz.timezone(self.config["timezone"])

        # Generate some mock events over next few days
        for day_offset in range(days_ahead):
            event_date = datetime.now(et) + timedelta(days=day_offset)

            # Add some random high-impact events
            if random.random() < 0.3:  # 30% chance of event per day
                event_time = event_date.replace(
                    hour=random.choice([8, 10, 14]),  # Common announcement times
                    minute=random.choice([0, 30]),
                    second=0,
                    microsecond=0,
                )

                event_type = random.choice(self.config["high_impact_events"])

                events.append(
                    {
                        "event_id": f"event_{day_offset}_{hash(event_type) % 1000}",
                        "event_type": event_type,
                        "impact": "high",
                        "scheduled_time": event_time.isoformat(),
                        "currency": "USD",
                        "source": "mock_calendar",
                    }
                )

        return events

    def check_current_event_status(self) -> Dict[str, any]:
        """
        Check if we're currently in an event guard period.

        Returns:
            Current event status and any active guards
        """
        try:
            current_time = datetime.now(pytz.timezone(self.config["timezone"]))

            status = {
                "timestamp": current_time.isoformat(),
                "event_guard_active": False,
                "active_events": [],
                "next_events": [],
                "position_cap_factor": 1.0,
            }

            # Get upcoming events
            events = self.fetch_economic_calendar()

            # Check for active event periods
            for event in events:
                event_time = datetime.fromisoformat(event["scheduled_time"])

                # Calculate guard window
                guard_start = event_time - timedelta(
                    minutes=self.config["guard_window_minutes"]["before"]
                )
                guard_end = event_time + timedelta(
                    minutes=self.config["guard_window_minutes"]["after"]
                )

                if guard_start <= current_time <= guard_end:
                    # We're in an active event guard period
                    status["event_guard_active"] = True
                    status["position_cap_factor"] = self.config[
                        "position_cap_during_event"
                    ]
                    status["active_events"].append(
                        {
                            "event": event,
                            "guard_start": guard_start.isoformat(),
                            "guard_end": guard_end.isoformat(),
                            "minutes_remaining": (
                                guard_end - current_time
                            ).total_seconds()
                            / 60,
                        }
                    )

                elif current_time < guard_start:
                    # Upcoming event
                    minutes_until_guard = (
                        guard_start - current_time
                    ).total_seconds() / 60
                    if minutes_until_guard <= 120:  # Show events within 2 hours
                        status["next_events"].append(
                            {
                                "event": event,
                                "guard_start": guard_start.isoformat(),
                                "minutes_until_guard": minutes_until_guard,
                            }
                        )

            return status

        except Exception as e:
            logger.error(f"Error checking event status: {e}")
            return {"error": str(e), "event_guard_active": False}

    def apply_event_guard(self, event_status: Dict[str, any]) -> Dict[str, any]:
        """
        Apply event guard by setting Redis flags.

        Args:
            event_status: Current event status

        Returns:
            Guard application results
        """
        try:
            guard_results = {
                "timestamp": datetime.now().isoformat(),
                "actions_taken": [],
            }

            if not self.redis_client:
                logger.warning("Redis unavailable - cannot apply event guard")
                return {"error": "Redis unavailable", "actions_taken": []}

            if event_status.get("event_guard_active", False):
                # Set event lock flag
                self.redis_client.set("risk:event_lock", "1")

                # Set position cap factor
                cap_factor = event_status.get("position_cap_factor", 0.25)
                self.redis_client.set("risk:event_position_cap", str(cap_factor))

                # Store active event info
                active_events = event_status.get("active_events", [])
                if active_events:
                    event_info = {
                        "events": active_events,
                        "set_timestamp": datetime.now().isoformat(),
                    }
                    self.redis_client.set("risk:active_events", json.dumps(event_info))

                guard_results["actions_taken"].extend(
                    [
                        f"Set risk:event_lock = 1",
                        f"Set risk:event_position_cap = {cap_factor}",
                        f"Stored {len(active_events)} active events",
                    ]
                )

                logger.warning(
                    f"🔒 Event guard ACTIVATED: {len(active_events)} active events, "
                    f"position cap = {cap_factor:.0%}"
                )

            else:
                # Clear event lock if no active events
                current_lock = self.redis_client.get("risk:event_lock")
                if current_lock == "1":
                    self.redis_client.set("risk:event_lock", "0")
                    self.redis_client.delete("risk:event_position_cap")
                    self.redis_client.delete("risk:active_events")

                    guard_results["actions_taken"].extend(
                        [
                            "Cleared risk:event_lock",
                            "Removed risk:event_position_cap",
                            "Cleared risk:active_events",
                        ]
                    )

                    logger.info("🔓 Event guard DEACTIVATED: no active events")

            return guard_results

        except Exception as e:
            logger.error(f"Error applying event guard: {e}")
            return {"error": str(e), "actions_taken": []}

    def check_strategy_exemption(self, strategy_name: str) -> bool:
        """
        Check if strategy is exempt from event guards.

        Args:
            strategy_name: Strategy to check

        Returns:
            True if strategy is exempt
        """
        return strategy_name.lower() in [
            s.lower() for s in self.config["exempt_strategies"]
        ]

    def get_effective_position_cap(self, strategy_name: str) -> float:
        """
        Get effective position cap for strategy during events.

        Args:
            strategy_name: Strategy name

        Returns:
            Position cap factor (1.0 = normal, 0.25 = 25% cap)
        """
        try:
            if not self.redis_client:
                return 1.0

            # Check if event lock is active
            event_lock = self.redis_client.get("risk:event_lock")
            if event_lock != "1":
                return 1.0

            # Check strategy exemption
            if self.check_strategy_exemption(strategy_name):
                return 1.0  # Exempt strategies get full sizing

            # Get event position cap
            cap_str = self.redis_client.get("risk:event_position_cap")
            if cap_str:
                return float(cap_str)

            return self.config["position_cap_during_event"]

        except Exception as e:
            logger.error(f"Error getting position cap: {e}")
            return 1.0  # Default to full sizing on error

    def run_event_monitor_cycle(self) -> Dict[str, any]:
        """Run single event monitoring cycle."""
        try:
            logger.debug("📅 Running economic event monitor cycle")

            cycle_results = {
                "timestamp": datetime.now().isoformat(),
                "cycle_type": "event_monitoring",
            }

            # Check current event status
            event_status = self.check_current_event_status()
            cycle_results["event_status"] = event_status

            # Apply event guards
            guard_results = self.apply_event_guard(event_status)
            cycle_results["guard_results"] = guard_results

            # Log significant events
            if event_status.get("event_guard_active", False):
                active_count = len(event_status.get("active_events", []))
                logger.info(
                    f"Event guard active: {active_count} events, "
                    f"cap = {event_status.get('position_cap_factor', 1.0):.0%}"
                )

            return cycle_results

        except Exception as e:
            logger.error(f"Error in event monitor cycle: {e}")
            return {"error": str(e)}

    def run_event_monitor_daemon(self):
        """Run event monitor as continuous daemon."""
        logger.info("📅 Starting economic event monitor daemon")

        try:
            while True:
                cycle_results = self.run_event_monitor_cycle()

                # Log guard changes
                if "guard_results" in cycle_results and cycle_results[
                    "guard_results"
                ].get("actions_taken"):
                    actions = cycle_results["guard_results"]["actions_taken"]
                    logger.info(f"Event guard actions: {actions}")

                # Wait before next cycle
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("Event monitor daemon stopped by user")
        except Exception as e:
            logger.error(f"Event monitor daemon error: {e}")

    def get_event_calendar_summary(self, days: int = 3) -> Dict[str, any]:
        """Get summary of upcoming economic events."""
        try:
            events = self.fetch_economic_calendar(days)
            et = pytz.timezone(self.config["timezone"])
            current_time = datetime.now(et)

            summary = {
                "timestamp": current_time.isoformat(),
                "days_ahead": days,
                "total_events": len(events),
                "high_impact_events": [],
                "today_events": [],
                "tomorrow_events": [],
            }

            for event in events:
                event_time = datetime.fromisoformat(event["scheduled_time"])

                if event["impact"] == "high":
                    summary["high_impact_events"].append(
                        {
                            "event_type": event["event_type"],
                            "scheduled_time": event_time.strftime("%Y-%m-%d %H:%M %Z"),
                            "hours_from_now": (
                                event_time - current_time
                            ).total_seconds()
                            / 3600,
                        }
                    )

                # Categorize by day
                days_from_now = (event_time.date() - current_time.date()).days
                if days_from_now == 0:
                    summary["today_events"].append(event)
                elif days_from_now == 1:
                    summary["tomorrow_events"].append(event)

            return summary

        except Exception as e:
            logger.error(f"Error getting calendar summary: {e}")
            return {"error": str(e)}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Economic Event Guard")

    parser.add_argument(
        "--mode",
        choices=["check", "monitor", "daemon", "calendar"],
        default="check",
        help="Guard mode",
    )
    parser.add_argument(
        "--strategy", type=str, help="Check position cap for specific strategy"
    )
    parser.add_argument("--days", type=int, default=3, help="Days ahead for calendar")
    parser.add_argument("--output", type=str, help="Output file for results")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info("📅 Starting Economic Event Guard")

    try:
        guard = EconomicEventGuard()

        if args.mode == "check":
            results = guard.check_current_event_status()
            print(f"\n📅 EVENT STATUS:")
            print(json.dumps(results, indent=2))

        elif args.mode == "monitor":
            results = guard.run_event_monitor_cycle()
            print(f"\n📊 MONITOR CYCLE:")
            print(json.dumps(results, indent=2))

        elif args.mode == "calendar":
            results = guard.get_event_calendar_summary(args.days)
            print(f"\n📋 ECONOMIC CALENDAR ({args.days} days):")
            print(json.dumps(results, indent=2))

        elif args.mode == "daemon":
            guard.run_event_monitor_daemon()
            return 0

        # Check strategy-specific cap if requested
        if args.strategy:
            cap = guard.get_effective_position_cap(args.strategy)
            print(f"\n📊 Position cap for {args.strategy}: {cap:.0%}")
            results["strategy_position_cap"] = {"strategy": args.strategy, "cap": cap}

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Error in economic event guard: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
