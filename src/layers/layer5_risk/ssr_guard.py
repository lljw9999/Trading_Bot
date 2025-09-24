"""
Short Sale Restriction (SSR) Guard

Monitors for SSR conditions and blocks short selling when restrictions are active.
SSR is triggered when a stock drops >=10% from previous close or broker flags SSR.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone

from ...utils.logger import get_logger


class SSRGuard:
    """Guards against short selling during SSR periods"""

    def __init__(
        self, ssr_threshold_pct: float = -10.0, grace_period_pct: float = -8.0
    ):
        """
        Initialize SSR guard.

        Args:
            ssr_threshold_pct: Price drop % that triggers SSR (default: -10%)
            grace_period_pct: Warning threshold before SSR trigger (default: -8%)
        """
        self.logger = get_logger(self.__class__.__name__)
        self.ssr_threshold_pct = ssr_threshold_pct / 100.0  # Convert to decimal
        self.grace_period_pct = grace_period_pct / 100.0

        self.logger.info(
            f"SSR Guard initialized: trigger={ssr_threshold_pct}%, grace={grace_period_pct}%"
        )

    def evaluate(
        self,
        symbol: str,
        last_price: float,
        prev_close: float,
        broker_ssr_flag: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate SSR status for a symbol.

        Args:
            symbol: Stock symbol
            last_price: Current/last traded price
            prev_close: Previous session close price
            broker_ssr_flag: SSR flag from broker (if available)

        Returns:
            Dict with SSR evaluation results
        """
        # Calculate price change
        price_change_pct = (
            (last_price - prev_close) / prev_close if prev_close > 0 else 0.0
        )

        # Determine SSR status
        is_ssr_triggered = False
        ssr_reason = None

        # Check price-based trigger
        if price_change_pct <= self.ssr_threshold_pct:
            is_ssr_triggered = True
            ssr_reason = "PRICE_DROP"

        # Check broker flag (overrides price-based logic)
        if broker_ssr_flag is True:
            is_ssr_triggered = True
            ssr_reason = "BROKER_FLAG"
        elif broker_ssr_flag is False:
            is_ssr_triggered = False
            ssr_reason = None

        # Warning status
        is_approaching_ssr = (
            not is_ssr_triggered and price_change_pct <= self.grace_period_pct
        )

        # Calculate distance to trigger
        distance_to_ssr_pct = (self.ssr_threshold_pct - price_change_pct) * 100.0

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "last_price": last_price,
            "prev_close": prev_close,
            "price_change_pct": price_change_pct * 100.0,
            "is_ssr_active": is_ssr_triggered,
            "ssr_reason": ssr_reason,
            "is_approaching_ssr": is_approaching_ssr,
            "distance_to_ssr_pct": distance_to_ssr_pct,
            "should_block_shorts": is_ssr_triggered,
            "broker_ssr_flag": broker_ssr_flag,
        }

        # Log significant events
        if is_ssr_triggered:
            self.logger.warning(
                f"🚨 SSR ACTIVE for {symbol}: {price_change_pct:.1%} drop, reason: {ssr_reason}"
            )
        elif is_approaching_ssr:
            self.logger.info(
                f"⚠️ SSR WARNING for {symbol}: {price_change_pct:.1%} drop, {abs(distance_to_ssr_pct):.1f}% from trigger"
            )

        return result

    def should_block_short_sale(self, symbol: str, order_side: str) -> bool:
        """
        Check if a short sale should be blocked for a symbol.

        Args:
            symbol: Stock symbol
            order_side: Order side ('buy' or 'sell')

        Returns:
            True if short sale should be blocked
        """
        if order_side.upper() not in ["SELL", "SELL_SHORT"]:
            return False  # Not a short sale

        # This would integrate with Redis in practice
        # For now, return False to allow development
        return False

    def get_ssr_symbols(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all symbols currently under SSR.

        Returns:
            Dict mapping symbol to SSR status
        """
        # This would query Redis for all risk:ssr:<symbol> keys in practice
        return {}


def create_ssr_guard(**kwargs) -> SSRGuard:
    """Factory function to create SSRGuard instance."""
    return SSRGuard(**kwargs)
