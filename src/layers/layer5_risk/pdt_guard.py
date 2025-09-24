"""
Pattern Day Trader (PDT) Guard

Enforces PDT rules for accounts under $25k equity to prevent
regulatory violations and account restrictions.
"""

from typing import Dict, Any
from datetime import datetime, timedelta, timezone

from ...utils.logger import get_logger


class PDTGuard:
    """Guards against Pattern Day Trading violations"""

    def __init__(
        self,
        min_equity_usd: float = 26000.0,
        max_daytrades_5d: int = 3,
        margin_buffer: float = 0.1,
    ):
        """
        Initialize PDT guard.

        Args:
            min_equity_usd: Minimum equity to avoid PDT restrictions (default: $26k)
            max_daytrades_5d: Max day trades in 5 days for accounts < min_equity (default: 3)
            margin_buffer: Safety margin (10% buffer above $25k requirement)
        """
        self.logger = get_logger(self.__class__.__name__)
        self.min_equity_usd = min_equity_usd
        self.max_daytrades_5d = max_daytrades_5d
        self.margin_buffer = margin_buffer

        self.logger.info(
            f"PDT Guard initialized: min_equity=${min_equity_usd:,.0f}, max_daytrades={max_daytrades_5d}"
        )

    def check(
        self, account_equity_usd: float, last_5d_daytrades: int
    ) -> Dict[str, Any]:
        """
        Check PDT status and determine if trading should be restricted.

        Args:
            account_equity_usd: Current account equity in USD
            last_5d_daytrades: Number of day trades in last 5 business days

        Returns:
            Dict with PDT status and recommendations
        """
        # Determine if account is subject to PDT rules
        is_pdt_account = account_equity_usd < self.min_equity_usd

        # Check current day trade count
        daytrades_remaining = max(0, self.max_daytrades_5d - last_5d_daytrades)

        # Determine if we should block intraday trading
        should_block = is_pdt_account and last_5d_daytrades >= self.max_daytrades_5d

        # Calculate how much equity needed to escape PDT
        equity_needed = max(0, self.min_equity_usd - account_equity_usd)

        # Risk level assessment
        if should_block:
            risk_level = "HIGH"
            recommendation = "BLOCK_INTRADAY"
        elif is_pdt_account and daytrades_remaining <= 1:
            risk_level = "MEDIUM"
            recommendation = "LIMIT_INTRADAY"
        elif is_pdt_account:
            risk_level = "LOW"
            recommendation = "MONITOR"
        else:
            risk_level = "NONE"
            recommendation = "NO_RESTRICTIONS"

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "account_equity_usd": account_equity_usd,
            "is_pdt_account": is_pdt_account,
            "last_5d_daytrades": last_5d_daytrades,
            "daytrades_remaining": daytrades_remaining,
            "should_block_intraday": should_block,
            "equity_needed_to_escape": equity_needed,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "min_equity_threshold": self.min_equity_usd,
            "max_daytrade_threshold": self.max_daytrades_5d,
        }

        if should_block:
            self.logger.warning(
                f"🚨 PDT BLOCK: Equity ${account_equity_usd:,.0f} < ${self.min_equity_usd:,.0f}, "
                f"day trades {last_5d_daytrades}/{self.max_daytrades_5d}"
            )
        elif is_pdt_account:
            self.logger.info(
                f"⚠️ PDT WATCH: Equity ${account_equity_usd:,.0f}, "
                f"day trades {last_5d_daytrades}/{self.max_daytrades_5d} ({daytrades_remaining} remaining)"
            )

        return result

    def is_daytrade(self, open_time: datetime, close_time: datetime) -> bool:
        """
        Determine if opening and closing a position constitutes a day trade.

        Args:
            open_time: Time position was opened
            close_time: Time position was closed

        Returns:
            True if this is a day trade (same trading day)
        """
        # Convert to market timezone for day comparison
        # For simplicity, assume same calendar day = day trade
        return open_time.date() == close_time.date()

    def get_daytrade_projection(self, pending_closes: int) -> Dict[str, Any]:
        """
        Project day trade count if pending position closes execute.

        Args:
            pending_closes: Number of positions that might close today

        Returns:
            Projection of day trade impact
        """
        # This would integrate with position tracking in real implementation
        return {
            "current_open_positions": 0,  # Would come from position tracker
            "pending_closes_today": pending_closes,
            "projected_additional_daytrades": pending_closes,
            "safe_to_open_new": pending_closes == 0,
        }


def create_pdt_guard(**kwargs) -> PDTGuard:
    """Factory function to create PDTGuard instance."""
    return PDTGuard(**kwargs)
