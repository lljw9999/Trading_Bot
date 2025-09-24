#!/usr/bin/env python3
"""
Market Hours Guard with Holiday and LULD Edge Cases

Manages trading hours and market conditions:
- NYSE holiday and early close calendar integration
- LULD (Limit Up Limit Down) monitoring and order suppression
- SSR (Short Sale Rule) detection and handling
- Automatic trading resumption when conditions clear
- Market microstructure anomaly detection
"""

import argparse
import json
import logging
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import pytz

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import redis
    import pandas as pd

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger("market_hours_guard")


class MarketHoursGuard:
    """
    Guards trading operations based on market hours, holidays, and market conditions.
    Ensures compliance with market rules and prevents trading during disrupted conditions.
    """

    def __init__(self):
        """Initialize market hours guard."""
        self.redis_client = None
        if DEPS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        # Market configuration
        self.config = {
            "exchanges": {
                "NYSE": {
                    "regular_hours": {"open": "09:30", "close": "16:00"},
                    "early_close": {"open": "09:30", "close": "13:00"},
                    "timezone": "America/New_York",
                },
                "NASDAQ": {
                    "regular_hours": {"open": "09:30", "close": "16:00"},
                    "early_close": {"open": "09:30", "close": "13:00"},
                    "timezone": "America/New_York",
                },
            },
            "holidays_2025": [
                {"date": "2025-01-01", "name": "New Year's Day", "type": "full_close"},
                {
                    "date": "2025-01-20",
                    "name": "Martin Luther King Jr. Day",
                    "type": "full_close",
                },
                {"date": "2025-02-17", "name": "Presidents' Day", "type": "full_close"},
                {"date": "2025-04-18", "name": "Good Friday", "type": "full_close"},
                {"date": "2025-05-26", "name": "Memorial Day", "type": "full_close"},
                {"date": "2025-06-19", "name": "Juneteenth", "type": "full_close"},
                {
                    "date": "2025-07-04",
                    "name": "Independence Day",
                    "type": "full_close",
                },
                {"date": "2025-09-01", "name": "Labor Day", "type": "full_close"},
                {"date": "2025-11-27", "name": "Thanksgiving", "type": "full_close"},
                {
                    "date": "2025-11-28",
                    "name": "Day after Thanksgiving",
                    "type": "early_close",
                },
                {"date": "2025-12-24", "name": "Christmas Eve", "type": "early_close"},
                {"date": "2025-12-25", "name": "Christmas Day", "type": "full_close"},
            ],
            "luld_thresholds": {
                "tier1": {"up": 0.05, "down": -0.05},  # 5% for Tier 1 NMS stocks
                "tier2": {"up": 0.10, "down": -0.10},  # 10% for other NMS stocks
                "pause_duration_minutes": 5,  # LULD pause duration
                "monitoring_window_minutes": 15,  # Monitor for LULD conditions
            },
            "ssr_config": {
                "trigger_threshold": -0.10,  # SSR triggered at 10% decline
                "duration_hours": 24,  # SSR active until end of next trading day
                "affected_order_types": ["short_sell", "short_exempt"],
            },
            "trading_control": {
                "halt_on_luld": True,
                "halt_on_ssr": False,  # Allow long orders during SSR
                "halt_on_holiday": True,
                "halt_on_early_close": False,  # Allow trading during early close hours
                "resume_delay_seconds": 30,  # Delay before resuming after conditions clear
            },
        }

        logger.info("Initialized market hours guard")

    def get_market_status(
        self, target_date: Optional[datetime] = None
    ) -> Dict[str, any]:
        """
        Get current market status including holidays, trading hours, and special conditions.

        Args:
            target_date: Date to check (defaults to today)

        Returns:
            Market status information
        """
        try:
            if target_date is None:
                target_date = datetime.now(pytz.timezone("America/New_York"))

            status = {
                "timestamp": target_date.isoformat(),
                "date": target_date.date().isoformat(),
                "market_open": False,
                "holiday_status": {},
                "trading_hours": {},
                "luld_status": {},
                "ssr_status": {},
                "trading_allowed": True,
                "halt_reasons": [],
            }

            # Check holiday status
            holiday_status = self._check_holiday_status(target_date.date())
            status["holiday_status"] = holiday_status

            # Check trading hours
            trading_hours = self._check_trading_hours(target_date, holiday_status)
            status["trading_hours"] = trading_hours
            status["market_open"] = trading_hours.get("currently_open", False)

            # Check LULD conditions
            luld_status = self._check_luld_conditions(target_date)
            status["luld_status"] = luld_status

            # Check SSR conditions
            ssr_status = self._check_ssr_conditions(target_date)
            status["ssr_status"] = ssr_status

            # Determine if trading should be halted
            trading_decision = self._determine_trading_status(
                holiday_status, trading_hours, luld_status, ssr_status
            )
            status["trading_allowed"] = trading_decision["allowed"]
            status["halt_reasons"] = trading_decision["halt_reasons"]

            return status

        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {
                "error": str(e),
                "trading_allowed": False,
                "halt_reasons": ["System error"],
            }

    def _check_holiday_status(self, check_date) -> Dict[str, any]:
        """Check if date is a market holiday."""
        try:
            date_str = check_date.isoformat()

            # Find matching holiday
            for holiday in self.config["holidays_2025"]:
                if holiday["date"] == date_str:
                    return {
                        "is_holiday": True,
                        "holiday_name": holiday["name"],
                        "holiday_type": holiday["type"],
                        "full_close": holiday["type"] == "full_close",
                        "early_close": holiday["type"] == "early_close",
                    }

            return {
                "is_holiday": False,
                "holiday_name": None,
                "holiday_type": None,
                "full_close": False,
                "early_close": False,
            }

        except Exception as e:
            return {"error": str(e), "is_holiday": False}

    def _check_trading_hours(
        self, check_datetime: datetime, holiday_status: Dict
    ) -> Dict[str, any]:
        """Check current trading hours status."""
        try:
            et = pytz.timezone("America/New_York")
            current_time = check_datetime.astimezone(et)

            # Determine market hours based on holiday status
            if holiday_status.get("full_close", False):
                return {
                    "market_type": "closed_holiday",
                    "currently_open": False,
                    "next_open": None,
                    "reason": f"Market closed for {holiday_status.get('holiday_name', 'holiday')}",
                }

            # Use early close hours if applicable
            if holiday_status.get("early_close", False):
                hours = self.config["exchanges"]["NYSE"]["early_close"]
                market_type = "early_close"
            else:
                hours = self.config["exchanges"]["NYSE"]["regular_hours"]
                market_type = "regular_hours"

            # Parse market hours
            open_time = datetime.strptime(hours["open"], "%H:%M").time()
            close_time = datetime.strptime(hours["close"], "%H:%M").time()

            current_time_only = current_time.time()

            # Check if currently within trading hours
            currently_open = (
                current_time.weekday() < 5  # Monday = 0, Friday = 4
                and open_time <= current_time_only <= close_time
            )

            # Calculate next market open
            next_open = self._calculate_next_market_open(current_time)

            return {
                "market_type": market_type,
                "currently_open": currently_open,
                "market_open_time": hours["open"],
                "market_close_time": hours["close"],
                "current_time": current_time.strftime("%H:%M:%S"),
                "is_weekend": current_time.weekday() >= 5,
                "next_open": next_open.isoformat() if next_open else None,
            }

        except Exception as e:
            return {"error": str(e), "currently_open": False}

    def _calculate_next_market_open(self, current_time: datetime) -> Optional[datetime]:
        """Calculate next market open time."""
        try:
            et = pytz.timezone("America/New_York")

            # Start with next business day
            next_day = current_time.date() + timedelta(days=1)

            # Find next non-holiday, non-weekend day
            for days_ahead in range(10):  # Look up to 10 days ahead
                check_date = current_time.date() + timedelta(days=days_ahead + 1)

                # Skip weekends
                if datetime.combine(check_date, time()).weekday() >= 5:
                    continue

                # Check if it's a holiday
                holiday_status = self._check_holiday_status(check_date)
                if holiday_status.get("full_close", False):
                    continue

                # This is a valid trading day
                open_time = datetime.strptime("09:30", "%H:%M").time()
                next_open = et.localize(datetime.combine(check_date, open_time))
                return next_open

            return None  # Couldn't find next open within 10 days

        except Exception as e:
            logger.error(f"Error calculating next market open: {e}")
            return None

    def _check_luld_conditions(self, check_datetime: datetime) -> Dict[str, any]:
        """Check for LULD (Limit Up Limit Down) conditions."""
        try:
            # Mock LULD implementation - real version would:
            # 1. Monitor stock price movements vs. reference prices
            # 2. Track LULD pause notifications from SIP feeds
            # 3. Identify which stocks are in LULD pause
            # 4. Calculate when pauses will expire

            luld_status = {
                "monitoring_active": True,
                "stocks_in_pause": [],
                "recent_luld_events": [],
                "pause_threshold_breaches": 0,
            }

            # Mock some LULD conditions for testing
            if self.redis_client:
                # Check for LULD simulation
                luld_simulation = self.redis_client.get("market:luld_simulation")
                if luld_simulation == "active":
                    luld_status["stocks_in_pause"] = ["AAPL", "MSFT", "NVDA"]
                    luld_status["recent_luld_events"] = [
                        {
                            "symbol": "AAPL",
                            "trigger_time": (
                                check_datetime - timedelta(minutes=2)
                            ).isoformat(),
                            "trigger_reason": "price_moved_above_upper_band",
                            "pause_expires": (
                                check_datetime + timedelta(minutes=3)
                            ).isoformat(),
                        }
                    ]
                    luld_status["pause_threshold_breaches"] = len(
                        luld_status["stocks_in_pause"]
                    )

            return luld_status

        except Exception as e:
            return {"error": str(e), "monitoring_active": False}

    def _check_ssr_conditions(self, check_datetime: datetime) -> Dict[str, any]:
        """Check for SSR (Short Sale Rule) conditions."""
        try:
            # Mock SSR implementation - real version would:
            # 1. Monitor stocks for 10% price declines from previous day close
            # 2. Track SSR activation notifications
            # 3. Maintain list of stocks currently under SSR
            # 4. Calculate when SSR periods will expire

            ssr_status = {
                "monitoring_active": True,
                "stocks_under_ssr": [],
                "recent_ssr_activations": [],
                "ssr_threshold_breaches": 0,
            }

            # Mock SSR conditions
            if self.redis_client:
                # Check for SSR simulation
                ssr_simulation = self.redis_client.get("market:ssr_simulation")
                if ssr_simulation == "active":
                    ssr_status["stocks_under_ssr"] = ["XYZ", "ABC"]
                    ssr_status["recent_ssr_activations"] = [
                        {
                            "symbol": "XYZ",
                            "activation_time": (
                                check_datetime - timedelta(hours=2)
                            ).isoformat(),
                            "trigger_reason": "declined_10_percent_from_previous_close",
                            "expires": (
                                check_datetime + timedelta(hours=22)
                            ).isoformat(),
                        }
                    ]
                    ssr_status["ssr_threshold_breaches"] = len(
                        ssr_status["stocks_under_ssr"]
                    )

            return ssr_status

        except Exception as e:
            return {"error": str(e), "monitoring_active": False}

    def _determine_trading_status(
        self,
        holiday_status: Dict,
        trading_hours: Dict,
        luld_status: Dict,
        ssr_status: Dict,
    ) -> Dict[str, any]:
        """Determine if trading should be allowed based on all conditions."""
        try:
            allowed = True
            halt_reasons = []

            # Check holiday halts
            if self.config["trading_control"]["halt_on_holiday"] and holiday_status.get(
                "full_close", False
            ):
                allowed = False
                halt_reasons.append(
                    f"Market closed for {holiday_status.get('holiday_name', 'holiday')}"
                )

            # Check market hours
            if not trading_hours.get("currently_open", False):
                if (
                    trading_hours.get("market_type") != "early_close"
                    or not self.config["trading_control"]["halt_on_early_close"]
                ):
                    allowed = False
                    halt_reasons.append("Market closed - outside trading hours")

            # Check LULD halts
            if (
                self.config["trading_control"]["halt_on_luld"]
                and len(luld_status.get("stocks_in_pause", [])) > 0
            ):
                allowed = False
                halt_reasons.append(
                    f"LULD pause active for {len(luld_status['stocks_in_pause'])} stocks"
                )

            # Check SSR halts (typically only affects short sales, not all trading)
            if (
                self.config["trading_control"]["halt_on_ssr"]
                and len(ssr_status.get("stocks_under_ssr", [])) > 0
            ):
                allowed = False
                halt_reasons.append(
                    f"SSR active for {len(ssr_status['stocks_under_ssr'])} stocks"
                )

            return {"allowed": allowed, "halt_reasons": halt_reasons}

        except Exception as e:
            return {
                "allowed": False,
                "halt_reasons": [f"Error determining trading status: {e}"],
            }

    def simulate_luld_event(
        self, symbol: str, duration_minutes: int = 5
    ) -> Dict[str, any]:
        """
        Simulate LULD event for testing.

        Args:
            symbol: Stock symbol to simulate LULD for
            duration_minutes: Duration of LULD pause

        Returns:
            Simulation results
        """
        try:
            logger.warning(f"🚨 Simulating LULD event for {symbol}")

            if not self.redis_client:
                return {"error": "Redis unavailable for simulation"}

            # Set LULD simulation flag
            self.redis_client.set("market:luld_simulation", "active")
            self.redis_client.expire("market:luld_simulation", duration_minutes * 60)

            # Record simulation event
            simulation_event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "luld_simulation",
                "symbol": symbol,
                "duration_minutes": duration_minutes,
                "trigger_reason": "manual_simulation",
            }

            self.redis_client.lpush(
                "market:simulation_events", json.dumps(simulation_event)
            )
            self.redis_client.ltrim(
                "market:simulation_events", 0, 99
            )  # Keep last 100 events

            logger.warning(
                f"LULD simulation active for {symbol} for {duration_minutes} minutes"
            )

            return {
                "success": True,
                "symbol": symbol,
                "duration_minutes": duration_minutes,
                "simulation_expires": (
                    datetime.now() + timedelta(minutes=duration_minutes)
                ).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error simulating LULD event: {e}")
            return {"error": str(e), "success": False}

    def simulate_ssr_event(
        self, symbol: str, duration_hours: int = 24
    ) -> Dict[str, any]:
        """
        Simulate SSR event for testing.

        Args:
            symbol: Stock symbol to simulate SSR for
            duration_hours: Duration of SSR period

        Returns:
            Simulation results
        """
        try:
            logger.warning(f"📉 Simulating SSR event for {symbol}")

            if not self.redis_client:
                return {"error": "Redis unavailable for simulation"}

            # Set SSR simulation flag
            self.redis_client.set("market:ssr_simulation", "active")
            self.redis_client.expire("market:ssr_simulation", duration_hours * 3600)

            # Record simulation event
            simulation_event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "ssr_simulation",
                "symbol": symbol,
                "duration_hours": duration_hours,
                "trigger_reason": "declined_10_percent_simulation",
            }

            self.redis_client.lpush(
                "market:simulation_events", json.dumps(simulation_event)
            )
            self.redis_client.ltrim("market:simulation_events", 0, 99)

            logger.warning(
                f"SSR simulation active for {symbol} for {duration_hours} hours"
            )

            return {
                "success": True,
                "symbol": symbol,
                "duration_hours": duration_hours,
                "simulation_expires": (
                    datetime.now() + timedelta(hours=duration_hours)
                ).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error simulating SSR event: {e}")
            return {"error": str(e), "success": False}

    def clear_simulations(self) -> Dict[str, any]:
        """Clear all active simulations."""
        try:
            if not self.redis_client:
                return {"error": "Redis unavailable"}

            # Clear simulation flags
            self.redis_client.delete("market:luld_simulation")
            self.redis_client.delete("market:ssr_simulation")

            logger.info("✅ Cleared all market simulations")

            return {"success": True, "timestamp": datetime.now().isoformat()}

        except Exception as e:
            logger.error(f"Error clearing simulations: {e}")
            return {"error": str(e), "success": False}

    def run_market_monitoring_cycle(self) -> Dict[str, any]:
        """Run single market monitoring cycle."""
        try:
            logger.debug("📊 Running market monitoring cycle")

            cycle_results = {
                "timestamp": datetime.now().isoformat(),
                "cycle_type": "market_monitoring",
            }

            # Get current market status
            market_status = self.get_market_status()
            cycle_results["market_status"] = market_status

            # Store current status in Redis
            if self.redis_client:
                self.redis_client.set(
                    "market:current_status", json.dumps(market_status)
                )
                self.redis_client.expire("market:current_status", 300)  # 5 minute TTL

            # Log significant market events
            if not market_status.get("trading_allowed", True):
                halt_reasons = ", ".join(market_status.get("halt_reasons", []))
                logger.warning(f"Trading halted: {halt_reasons}")

            return cycle_results

        except Exception as e:
            logger.error(f"Error in market monitoring cycle: {e}")
            return {"error": str(e)}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Market Hours Guard")

    parser.add_argument(
        "--action",
        choices=["status", "monitor", "simulate-luld", "simulate-ssr", "clear-sim"],
        default="status",
        help="Action to perform",
    )
    parser.add_argument(
        "--symbol", type=str, default="AAPL", help="Symbol for simulation"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Simulation duration (minutes for LULD, hours for SSR)",
    )
    parser.add_argument("--date", type=str, help="Date to check (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, help="Output file for results")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info("📊 Starting Market Hours Guard")

    try:
        guard = MarketHoursGuard()

        if args.action == "status":
            # Parse date if provided
            target_date = None
            if args.date:
                target_date = datetime.strptime(args.date, "%Y-%m-%d")
                target_date = pytz.timezone("America/New_York").localize(target_date)

            results = guard.get_market_status(target_date)
            print(f"\n📊 MARKET STATUS:")
            print(json.dumps(results, indent=2))

        elif args.action == "monitor":
            results = guard.run_market_monitoring_cycle()
            print(f"\n🔍 MONITORING CYCLE:")
            print(json.dumps(results, indent=2))

        elif args.action == "simulate-luld":
            results = guard.simulate_luld_event(args.symbol, args.duration)
            print(f"\n🚨 LULD SIMULATION:")
            print(json.dumps(results, indent=2))

        elif args.action == "simulate-ssr":
            results = guard.simulate_ssr_event(args.symbol, args.duration)
            print(f"\n📉 SSR SIMULATION:")
            print(json.dumps(results, indent=2))

        elif args.action == "clear-sim":
            results = guard.clear_simulations()
            print(f"\n✅ CLEAR SIMULATIONS:")
            print(json.dumps(results, indent=2))

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        # Return appropriate exit code
        if args.action in ["simulate-luld", "simulate-ssr", "clear-sim"]:
            return 0 if results.get("success", False) else 1
        else:
            return 0 if not results.get("error") else 1

    except Exception as e:
        logger.error(f"Error in market hours guard: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
