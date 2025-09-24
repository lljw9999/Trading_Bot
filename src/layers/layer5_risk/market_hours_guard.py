"""
Market Hours Guard for US Equities

Enforces trading only during NYSE/NASDAQ market hours with configurable
pre-market and post-market windows. Integrates with existing risk system.
"""

import pytz
from datetime import datetime, time, timezone
from typing import Optional, Dict, Any
import pandas_market_calendars as mcal

from ...utils.logger import get_logger


class MarketHoursGuard:
    """Guards against trading outside market hours"""

    def __init__(
        self,
        tz: str = "America/New_York",
        pre: str = "09:25",
        post: str = "16:05",
        calendar_name: str = "NYSE",
    ):
        """
        Initialize market hours guard.

        Args:
            tz: Timezone string (default: America/New_York)
            pre: Pre-market start time (default: 09:25)
            post: Post-market end time (default: 16:05)
            calendar_name: Market calendar to use (default: NYSE)
        """
        self.logger = get_logger(self.__class__.__name__)
        self.timezone = pytz.timezone(tz)

        # Parse time strings
        try:
            pre_hour, pre_min = map(int, pre.split(":"))
            post_hour, post_min = map(int, post.split(":"))
            self.pre_market_start = time(pre_hour, pre_min)
            self.post_market_end = time(post_hour, post_min)
        except ValueError as e:
            self.logger.error(f"Invalid time format: {e}")
            raise

        # Market calendar
        try:
            self.calendar = mcal.get_calendar(calendar_name)
        except Exception as e:
            self.logger.warning(f"Could not load {calendar_name} calendar: {e}")
            self.calendar = None

        self.logger.info(f"Market Hours Guard initialized: {pre}-{post} {tz}")

    def _normalize_time(self, dt: Optional[datetime]) -> datetime:
        """Normalize input datetime to the guard's timezone."""
        if dt is None:
            dt = datetime.now(timezone.utc)
        elif dt.tzinfo is None:
            dt = self.timezone.localize(dt)
        else:
            dt = dt.astimezone(self.timezone)
        return dt

    def is_open_now(self, now: Optional[datetime] = None) -> bool:
        """
        Check if market is currently open.

        Args:
            now: Current time (defaults to utcnow)

        Returns:
            True if market is open, False otherwise
        """
        market_now = self._normalize_time(now)

        # Check if it's a trading day
        if self.calendar:
            today = market_now.date()
            trading_days = self.calendar.valid_days(start_date=today, end_date=today)
            if len(trading_days) == 0:
                return False  # Holiday or weekend

        # Check time window
        current_time = market_now.time()
        return self.pre_market_start <= current_time <= self.post_market_end

    def should_block_trading(self, now: Optional[datetime] = None) -> bool:
        """
        Check if trading should be blocked due to market hours.

        Args:
            now: Current time (defaults to utcnow)

        Returns:
            True if trading should be blocked, False otherwise
        """
        return not self.is_open_now(now)

    def is_opening_auction_window(self, now: Optional[datetime] = None) -> bool:
        """
        Check if currently in opening auction window (9:25-9:30 ET).

        Args:
            now: Current time (defaults to utcnow)

        Returns:
            True if in opening auction window
        """
        market_now = self._normalize_time(now)
        current_time = market_now.time()

        opening_start = time(9, 25)
        opening_end = time(9, 30)

        return opening_start <= current_time <= opening_end

    def is_closing_auction_window(self, now: Optional[datetime] = None) -> bool:
        """
        Check if currently in closing auction window (15:58-16:00 ET).

        Args:
            now: Current time (defaults to utcnow)

        Returns:
            True if in closing auction window
        """
        market_now = self._normalize_time(now)
        current_time = market_now.time()

        closing_start = time(15, 58)
        closing_end = time(16, 0)

        return closing_start <= current_time <= closing_end

    def get_market_status(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get comprehensive market status information.

        Args:
            now: Current time (defaults to utcnow)

        Returns:
            Dictionary with market status details
        """
        market_now = self._normalize_time(now)

        return {
            "is_open": self.is_open_now(now),
            "should_block": self.should_block_trading(now),
            "is_opening_auction": self.is_opening_auction_window(now),
            "is_closing_auction": self.is_closing_auction_window(now),
            "market_time": market_now.strftime("%H:%M:%S %Z"),
            "market_date": market_now.strftime("%Y-%m-%d"),
            "timezone": str(self.timezone),
        }


def create_market_hours_guard(
    tz: str = "America/New_York", pre: str = "09:25", post: str = "16:05"
) -> MarketHoursGuard:
    """Factory function to create MarketHoursGuard instance."""
    return MarketHoursGuard(tz=tz, pre=pre, post=post)


if __name__ == "__main__":
    # Test the guard
    guard = create_market_hours_guard()
    status = guard.get_market_status()
    print(f"Market Status: {status}")
    print(f"Should block trading: {guard.should_block_trading()}")
