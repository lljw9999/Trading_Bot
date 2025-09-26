#!/usr/bin/env python3
"""
Tail-Risk Hedge Overlay
Simple rule-based hedging to protect against tail risk events
"""

import time
import logging
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


@dataclass
class HedgePosition:
    """Represents a hedge position."""

    symbol: str
    size: float
    entry_price: float
    entry_time: datetime
    position_type: str  # "put_spread", "perp_short"
    pnl: float = 0.0


class TailRiskHedgeOverlay:
    """Tail risk hedge overlay with simple ES-EVT and IV-based rules."""

    def __init__(
        self,
        max_hedge_ratio: float = 0.25,
        es_enter_threshold: float = 0.03,
        es_exit_threshold: float = 0.02,
        iv_sigma_threshold: float = 2.0,
    ):
        """
        Initialize tail risk hedge overlay.

        Args:
            max_hedge_ratio: Maximum hedge size as fraction of gross exposure
            es_enter_threshold: ES-EVT threshold to enter hedge (3%)
            es_exit_threshold: ES-EVT threshold to exit hedge (2%)
            iv_sigma_threshold: IV change threshold in sigmas
        """
        self.max_hedge_ratio = max_hedge_ratio
        self.es_enter_threshold = es_enter_threshold
        self.es_exit_threshold = es_exit_threshold
        self.iv_sigma_threshold = iv_sigma_threshold

        # Hedge state
        self.active_hedges: Dict[str, HedgePosition] = {}
        self.total_hedge_notional = 0.0
        self.total_hedge_pnl = 0.0

        # Risk metrics cache
        self.last_es_evt = 0.0
        self.last_iv_change_sigma = 0.0
        self.last_gross_exposure = 0.0

        logger.info("🛡️ TailRiskHedgeOverlay initialized")
        logger.info(
            f"   ES enter/exit: {es_enter_threshold:.1%}/{es_exit_threshold:.1%}"
        )
        logger.info(f"   IV threshold: {iv_sigma_threshold}σ")
        logger.info(f"   Max hedge ratio: {max_hedge_ratio:.1%}")

    def update_risk_metrics(
        self, es_evt_95: float, iv_change_sigma: float, gross_exposure: float
    ) -> None:
        """Update current risk metrics."""
        self.last_es_evt = es_evt_95
        self.last_iv_change_sigma = iv_change_sigma
        self.last_gross_exposure = gross_exposure

        logger.debug(
            f"Risk metrics: ES-EVT={es_evt_95:.3f}, "
            f"IV_chg={iv_change_sigma:.2f}σ, "
            f"gross=${gross_exposure:,.0f}"
        )

    def hedge_signal(self, es_evt: float, iv_chg_sigma: float) -> Tuple[bool, bool]:
        """
        Generate hedge enter/exit signals.

        Args:
            es_evt: Current ES-EVT 95th percentile
            iv_chg_sigma: IV change in sigmas

        Returns:
            Tuple of (enter_signal, exit_signal)
        """
        enter_signal = (
            es_evt > self.es_enter_threshold and iv_chg_sigma > self.iv_sigma_threshold
        )

        exit_signal = es_evt < self.es_exit_threshold

        return enter_signal, exit_signal

    def calculate_optimal_hedge_size(self, gross_exposure: float) -> float:
        """Calculate optimal hedge size based on current exposure."""
        if gross_exposure <= 0:
            return 0.0

        # Simple rule: hedge 25% of gross exposure
        target_hedge_size = gross_exposure * self.max_hedge_ratio

        # Account for existing hedges
        current_hedge_size = abs(self.total_hedge_notional)
        additional_hedge_needed = max(0, target_hedge_size - current_hedge_size)

        return additional_hedge_needed

    def enter_hedge_position(
        self,
        symbol: str,
        size: float,
        position_type: str = "perp_short",
        current_price: float = None,
    ) -> bool:
        """
        Enter a new hedge position.

        Args:
            symbol: Hedge instrument symbol
            size: Position size (positive for long, negative for short)
            position_type: Type of hedge ("perp_short", "put_spread")
            current_price: Current market price

        Returns:
            True if hedge entered successfully
        """
        try:
            # Use mock price if not provided
            if current_price is None:
                current_price = 50000.0  # Mock BTC price

            hedge_position = HedgePosition(
                symbol=symbol,
                size=size,
                entry_price=current_price,
                entry_time=datetime.now(timezone.utc),
                position_type=position_type,
            )

            # Store hedge position
            position_id = f"{symbol}_{int(time.time())}"
            self.active_hedges[position_id] = hedge_position

            # Update totals
            self.total_hedge_notional += abs(size * current_price)

            logger.info(
                f"🛡️ Entered hedge: {position_type} {symbol} "
                f"size={size:,.0f} @ {current_price:,.2f}"
            )

            return True

        except Exception as e:
            logger.error(f"Error entering hedge position: {e}")
            return False

    def exit_hedge_position(
        self, position_id: str, current_price: float = None
    ) -> bool:
        """
        Exit a hedge position.

        Args:
            position_id: ID of position to exit
            current_price: Current market price for P&L calculation

        Returns:
            True if position exited successfully
        """
        try:
            if position_id not in self.active_hedges:
                logger.warning(f"Hedge position {position_id} not found")
                return False

            position = self.active_hedges[position_id]

            # Use mock price if not provided
            if current_price is None:
                current_price = 51000.0  # Mock price change

            # Calculate P&L (simplified)
            if position.position_type == "perp_short":
                # Short position profits when price falls
                pnl = -position.size * (current_price - position.entry_price)
            else:
                # Long position (put spread) profits when price falls (simplified)
                pnl = position.size * (position.entry_price - current_price) * 0.5

            position.pnl = pnl
            self.total_hedge_pnl += pnl

            # Update totals
            self.total_hedge_notional -= abs(position.size * position.entry_price)

            logger.info(
                f"🛡️ Exited hedge: {position.symbol} "
                f"P&L=${pnl:,.2f}, total_pnl=${self.total_hedge_pnl:,.2f}"
            )

            # Remove from active hedges
            del self.active_hedges[position_id]

            return True

        except Exception as e:
            logger.error(f"Error exiting hedge position: {e}")
            return False

    def exit_all_hedges(self, current_price: float = None) -> int:
        """Exit all active hedge positions."""
        exited_count = 0
        position_ids = list(self.active_hedges.keys())

        for position_id in position_ids:
            if self.exit_hedge_position(position_id, current_price):
                exited_count += 1

        return exited_count

    def tick(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main tick function called by risk monitor.

        Args:
            risk_metrics: Dictionary containing current risk metrics

        Returns:
            Dictionary with hedge overlay actions and status
        """
        try:
            # Extract risk metrics
            es_evt = risk_metrics.get("es_evt_95", 0.0)
            iv_change_sigma = risk_metrics.get("iv_change_sigma", 0.0)
            gross_exposure = risk_metrics.get("gross_exposure", 0.0)
            current_price = risk_metrics.get("btc_price", 50000.0)

            # Update internal state
            self.update_risk_metrics(es_evt, iv_change_sigma, gross_exposure)

            # Generate signals
            enter_signal, exit_signal = self.hedge_signal(es_evt, iv_change_sigma)

            actions = []

            # Handle exit signal (priority over enter)
            if exit_signal and self.active_hedges:
                exited_count = self.exit_all_hedges(current_price)
                if exited_count > 0:
                    actions.append(f"Exited {exited_count} hedge positions")

            # Handle enter signal
            elif enter_signal and not self.active_hedges:
                # Calculate hedge size
                hedge_size = self.calculate_optimal_hedge_size(gross_exposure)

                if hedge_size > 10000:  # Minimum hedge size $10k
                    # Enter perp short hedge
                    hedge_contracts = -hedge_size / current_price  # Short position

                    if self.enter_hedge_position(
                        symbol="BTCUSDT",
                        size=hedge_contracts,
                        position_type="perp_short",
                        current_price=current_price,
                    ):
                        actions.append(
                            f"Entered perp hedge: ${hedge_size:,.0f} notional"
                        )

            # Update P&L for existing positions
            for position in self.active_hedges.values():
                if position.position_type == "perp_short":
                    position.pnl = -position.size * (
                        current_price - position.entry_price
                    )

            # Calculate total current P&L
            total_unrealized_pnl = sum(pos.pnl for pos in self.active_hedges.values())
            total_pnl = self.total_hedge_pnl + total_unrealized_pnl

            return {
                "actions": actions,
                "hedge_active": len(self.active_hedges) > 0,
                "hedge_count": len(self.active_hedges),
                "hedge_notional_usd": self.total_hedge_notional,
                "hedge_pnl_realized": self.total_hedge_pnl,
                "hedge_pnl_unrealized": total_unrealized_pnl,
                "hedge_pnl_total": total_pnl,
                "enter_signal": enter_signal,
                "exit_signal": exit_signal,
                "es_evt_95": es_evt,
                "iv_change_sigma": iv_change_sigma,
            }

        except Exception as e:
            logger.error(f"Error in hedge overlay tick: {e}")
            return {"actions": [], "hedge_active": False, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get current hedge overlay status."""
        try:
            total_unrealized_pnl = sum(pos.pnl for pos in self.active_hedges.values())

            return {
                "overlay": "tail_risk_hedge",
                "active_hedges": len(self.active_hedges),
                "total_notional": self.total_hedge_notional,
                "realized_pnl": self.total_hedge_pnl,
                "unrealized_pnl": total_unrealized_pnl,
                "total_pnl": self.total_hedge_pnl + total_unrealized_pnl,
                "positions": [
                    {
                        "symbol": pos.symbol,
                        "size": pos.size,
                        "entry_price": pos.entry_price,
                        "type": pos.position_type,
                        "pnl": pos.pnl,
                        "age_minutes": int(
                            (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 60
                        ),
                    }
                    for pos in self.active_hedges.values()
                ],
                "last_metrics": {
                    "es_evt_95": self.last_es_evt,
                    "iv_change_sigma": self.last_iv_change_sigma,
                    "gross_exposure": self.last_gross_exposure,
                },
                "thresholds": {
                    "es_enter": self.es_enter_threshold,
                    "es_exit": self.es_exit_threshold,
                    "iv_sigma": self.iv_sigma_threshold,
                    "max_hedge_ratio": self.max_hedge_ratio,
                },
            }

        except Exception as e:
            logger.error(f"Error getting hedge status: {e}")
            return {"error": str(e)}
