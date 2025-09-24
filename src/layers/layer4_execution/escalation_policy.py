#!/usr/bin/env python3
"""
Escalation Policy: Smart Order Type Progression
State machine for escalating from post-only to aggressive fills based on timing and adverse selection.
"""
import os
import sys
import json
import datetime
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class OrderState(Enum):
    """Order execution states."""

    POST_ONLY = "post_only"
    MID_POINT = "mid_point"
    CROSS_SPREAD = "cross_spread"
    FAILED = "failed"
    FILLED = "filled"


class EscalationReason(Enum):
    """Reasons for escalating order type."""

    SLA_TIMEOUT = "sla_timeout"
    ADVERSE_MOVEMENT = "adverse_movement"
    SPREAD_WIDENING = "spread_widening"
    QUEUE_DETERIORATION = "queue_deterioration"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class EscalationConfig:
    """Configuration for escalation policy."""

    # SLA timeouts by asset type
    crypto_sla_seconds: float = 120
    equity_sla_seconds: float = 180

    # Adverse selection thresholds
    adverse_score_threshold: float = 0.7
    max_adverse_bps: float = 20.0

    # Spread-based controls
    spread_widening_threshold: float = 1.5  # 1.5x spread widening triggers backoff
    min_tick_improvement: float = 0.0001  # Minimum price improvement for mid/cross

    # Position sizing
    max_aggressive_pct: float = 0.3  # Max 30% of order can be aggressive
    escalation_slice_pct: float = 0.1  # Escalate 10% at a time

    # Backoff parameters
    backoff_cool_down_seconds: float = 30
    max_escalation_attempts: int = 3

    # Cancel backoff jitter (M16.1 optimization)
    cancel_backoff_jitter_min_ms: float = 150  # Min jitter after cancel
    cancel_backoff_jitter_max_ms: float = 350  # Max jitter after cancel


@dataclass
class OrderContext:
    """Context for order being managed."""

    order_id: str
    asset: str
    side: str  # "buy" or "sell"
    total_size: float
    filled_size: float
    remaining_size: float
    avg_fill_price: float
    start_time: datetime.datetime
    sla_deadline: datetime.datetime
    current_state: OrderState
    escalation_attempts: int
    last_escalation_time: Optional[datetime.datetime]


class EscalationPolicy:
    """Smart order execution policy with timing and adverse selection awareness."""

    def __init__(self, config: EscalationConfig = None):
        self.config = config or EscalationConfig()
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

        # Load queue timing for ETA predictions
        try:
            sys.path.insert(0, str(self.base_dir))
            from src.layers.layer4_execution.queue_timing_v2 import QueueTimingV2

            self.queue_timing = QueueTimingV2()
        except ImportError:
            self.queue_timing = None
            print("⚠️ Queue timing not available")

    def calculate_adverse_score(
        self, order_ctx: OrderContext, market_data: Dict[str, Any]
    ) -> float:
        """Calculate adverse selection score (0-1, higher = more adverse)."""

        # Get current mid price
        bid = market_data.get("bid", 0)
        ask = market_data.get("ask", 0)
        mid_price = (
            (bid + ask) / 2 if bid > 0 and ask > 0 else market_data.get("last_price", 0)
        )

        if mid_price <= 0 or order_ctx.avg_fill_price <= 0:
            return 0.0

        # Calculate unrealized P&L
        if order_ctx.side.lower() == "buy":
            # For buys, adverse if price has moved up
            price_move_bps = (
                (mid_price - order_ctx.avg_fill_price) / order_ctx.avg_fill_price
            ) * 10000
        else:
            # For sells, adverse if price has moved down
            price_move_bps = (
                (order_ctx.avg_fill_price - mid_price) / order_ctx.avg_fill_price
            ) * 10000

        # Convert to adverse score (0-1)
        adverse_score = max(0, price_move_bps) / self.config.max_adverse_bps

        # Add time pressure component
        elapsed_pct = self.get_elapsed_time_pct(order_ctx)
        time_pressure = min(1.0, elapsed_pct * 2)  # Ramps up as we approach deadline

        # Combine price and time components
        combined_score = min(1.0, adverse_score * 0.7 + time_pressure * 0.3)

        return combined_score

    def get_elapsed_time_pct(self, order_ctx: OrderContext) -> float:
        """Get percentage of SLA time elapsed."""
        now = datetime.datetime.now(datetime.timezone.utc)
        total_duration = (order_ctx.sla_deadline - order_ctx.start_time).total_seconds()
        elapsed_duration = (now - order_ctx.start_time).total_seconds()

        return (
            min(1.0, elapsed_duration / total_duration) if total_duration > 0 else 1.0
        )

    def check_spread_conditions(
        self, market_data: Dict[str, Any], baseline_spread: Optional[float] = None
    ) -> Dict[str, Any]:
        """Check spread conditions for escalation decisions."""

        current_spread_bps = market_data.get("spread_bps", 10)

        spread_analysis = {
            "current_spread_bps": current_spread_bps,
            "is_wide": current_spread_bps > 15,
            "spread_widened": False,
            "spread_change_ratio": 1.0,
        }

        if baseline_spread and baseline_spread > 0:
            spread_change_ratio = current_spread_bps / baseline_spread
            spread_analysis["spread_change_ratio"] = spread_change_ratio
            spread_analysis["spread_widened"] = (
                spread_change_ratio > self.config.spread_widening_threshold
            )

        return spread_analysis

    def should_escalate(
        self,
        order_ctx: OrderContext,
        market_data: Dict[str, Any],
        baseline_spread: Optional[float] = None,
    ) -> Tuple[bool, List[EscalationReason]]:
        """Determine if order should escalate from current state."""

        reasons = []
        now = datetime.datetime.now(datetime.timezone.utc)

        # Check SLA timeout
        if now >= order_ctx.sla_deadline:
            reasons.append(EscalationReason.SLA_TIMEOUT)

        # Check adverse selection if we have fills
        if order_ctx.filled_size > 0:
            adverse_score = self.calculate_adverse_score(order_ctx, market_data)
            if adverse_score >= self.config.adverse_score_threshold:
                reasons.append(EscalationReason.ADVERSE_MOVEMENT)

        # Check spread conditions for backoff
        spread_analysis = self.check_spread_conditions(market_data, baseline_spread)
        if spread_analysis["spread_widened"]:
            # Don't escalate if spread has widened significantly
            return False, [EscalationReason.SPREAD_WIDENING]

        # Check queue timing if available
        if self.queue_timing and order_ctx.current_state == OrderState.POST_ONLY:
            try:
                eta_prediction = self.queue_timing.predict_time_to_fill(
                    order_ctx.asset,
                    order_ctx.remaining_size,
                    order_ctx.side,
                    market_data,
                )

                sla_seconds = (order_ctx.sla_deadline - now).total_seconds()

                if self.queue_timing.should_escalate_urgency(
                    eta_prediction, sla_seconds
                ):
                    reasons.append(EscalationReason.QUEUE_DETERIORATION)

            except Exception as e:
                print(f"⚠️ Queue timing error: {e}")

        # Check escalation limits
        if order_ctx.escalation_attempts >= self.config.max_escalation_attempts:
            return False, reasons

        # Check cool-down period
        if (
            order_ctx.last_escalation_time
            and (now - order_ctx.last_escalation_time).total_seconds()
            < self.config.backoff_cool_down_seconds
        ):
            return False, reasons

        return len(reasons) > 0, reasons

    def get_next_order_state(
        self, current_state: OrderState, escalation_reasons: List[EscalationReason]
    ) -> OrderState:
        """Determine next order state based on current state and escalation reasons."""

        if EscalationReason.SPREAD_WIDENING in escalation_reasons:
            # Back off to previous state or stay put
            return current_state

        # Progressive escalation
        if current_state == OrderState.POST_ONLY:
            return OrderState.MID_POINT
        elif current_state == OrderState.MID_POINT:
            return OrderState.CROSS_SPREAD
        else:
            # Already at most aggressive state
            return current_state

    def calculate_escalation_price(
        self,
        order_ctx: OrderContext,
        new_state: OrderState,
        market_data: Dict[str, Any],
    ) -> float:
        """Calculate appropriate price for escalated order."""

        bid = market_data.get("bid", 0)
        ask = market_data.get("ask", 0)

        if bid <= 0 or ask <= 0:
            return 0.0

        mid_price = (bid + ask) / 2
        tick_size = market_data.get("tick_size", 0.01)
        min_improvement = max(tick_size, self.config.min_tick_improvement)

        if order_ctx.side.lower() == "buy":
            if new_state == OrderState.MID_POINT:
                # Mid-point order
                return mid_price
            elif new_state == OrderState.CROSS_SPREAD:
                # Cross spread but with minimum improvement
                return ask + min_improvement
            else:
                # Post-only
                return bid
        else:  # sell
            if new_state == OrderState.MID_POINT:
                # Mid-point order
                return mid_price
            elif new_state == OrderState.CROSS_SPREAD:
                # Cross spread but with minimum improvement
                return bid - min_improvement
            else:
                # Post-only
                return ask

    def calculate_escalation_size(
        self, order_ctx: OrderContext, new_state: OrderState
    ) -> float:
        """Calculate size for escalated order slice."""

        remaining_size = order_ctx.remaining_size

        if new_state == OrderState.POST_ONLY:
            # Use full remaining size for post-only
            return remaining_size

        # For aggressive orders, limit size
        max_aggressive_size = order_ctx.total_size * self.config.max_aggressive_pct

        # Calculate how much we've already sent aggressively
        aggressive_filled = order_ctx.filled_size  # Simplified assumption
        remaining_aggressive_budget = max(0, max_aggressive_size - aggressive_filled)

        # Use escalation slice percentage
        slice_size = remaining_size * self.config.escalation_slice_pct

        # Don't exceed aggressive budget
        final_size = min(slice_size, remaining_aggressive_budget, remaining_size)

        return max(0, final_size)

    def apply_cancel_backoff_jitter(self) -> float:
        """Apply random jitter after order cancellation to avoid venue microbursts."""
        jitter_ms = random.uniform(
            self.config.cancel_backoff_jitter_min_ms,
            self.config.cancel_backoff_jitter_max_ms,
        )
        return jitter_ms / 1000.0  # Convert to seconds

    def create_escalation_decision(
        self,
        order_ctx: OrderContext,
        market_data: Dict[str, Any],
        baseline_spread: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create comprehensive escalation decision."""

        # Determine if we should escalate
        should_escalate, reasons = self.should_escalate(
            order_ctx, market_data, baseline_spread
        )

        if not should_escalate:
            return {
                "action": "hold",
                "current_state": order_ctx.current_state.value,
                "next_state": order_ctx.current_state.value,
                "reasons": [r.value for r in reasons],
                "order_size": 0,
                "order_price": 0,
                "adverse_score": (
                    self.calculate_adverse_score(order_ctx, market_data)
                    if order_ctx.filled_size > 0
                    else 0
                ),
                "elapsed_pct": self.get_elapsed_time_pct(order_ctx),
            }

        # Determine next state
        next_state = self.get_next_order_state(order_ctx.current_state, reasons)

        # Calculate order parameters
        escalation_price = self.calculate_escalation_price(
            order_ctx, next_state, market_data
        )
        escalation_size = self.calculate_escalation_size(order_ctx, next_state)

        # Apply cancel backoff jitter for resubmission timing
        cancel_jitter_seconds = self.apply_cancel_backoff_jitter()

        return {
            "action": "escalate",
            "current_state": order_ctx.current_state.value,
            "next_state": next_state.value,
            "reasons": [r.value for r in reasons],
            "order_size": escalation_size,
            "order_price": escalation_price,
            "adverse_score": (
                self.calculate_adverse_score(order_ctx, market_data)
                if order_ctx.filled_size > 0
                else 0
            ),
            "elapsed_pct": self.get_elapsed_time_pct(order_ctx),
            "spread_analysis": self.check_spread_conditions(
                market_data, baseline_spread
            ),
            "cancel_jitter_seconds": cancel_jitter_seconds,  # M16.1 optimization
        }

    def create_policy_report(
        self, test_scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create comprehensive escalation policy test report."""

        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "config": {
                "crypto_sla_seconds": self.config.crypto_sla_seconds,
                "equity_sla_seconds": self.config.equity_sla_seconds,
                "adverse_score_threshold": self.config.adverse_score_threshold,
                "spread_widening_threshold": self.config.spread_widening_threshold,
                "max_aggressive_pct": self.config.max_aggressive_pct,
            },
            "scenarios": [],
            "summary": {},
        }

        escalation_count = 0
        hold_count = 0

        for scenario in test_scenarios:
            order_ctx = scenario["order_context"]
            market_data = scenario["market_data"]
            baseline_spread = scenario.get("baseline_spread")

            decision = self.create_escalation_decision(
                order_ctx, market_data, baseline_spread
            )

            scenario_result = {
                "scenario_name": scenario.get("name", "unnamed"),
                "order_context": {
                    "asset": order_ctx.asset,
                    "side": order_ctx.side,
                    "total_size": order_ctx.total_size,
                    "filled_size": order_ctx.filled_size,
                    "current_state": order_ctx.current_state.value,
                    "elapsed_pct": self.get_elapsed_time_pct(order_ctx),
                },
                "market_data": market_data,
                "decision": decision,
            }

            report["scenarios"].append(scenario_result)

            if decision["action"] == "escalate":
                escalation_count += 1
            else:
                hold_count += 1

        # Summary statistics
        total_scenarios = len(test_scenarios)
        if total_scenarios > 0:
            report["summary"] = {
                "total_scenarios": total_scenarios,
                "escalation_rate": escalation_count / total_scenarios,
                "hold_rate": hold_count / total_scenarios,
                "avg_adverse_score": np.mean(
                    [s["decision"]["adverse_score"] for s in report["scenarios"]]
                ),
                "avg_elapsed_pct": np.mean(
                    [s["decision"]["elapsed_pct"] for s in report["scenarios"]]
                ),
            }

        return report


def test_escalation_policy():
    """Test escalation policy with various scenarios."""

    print("🚀 Testing Escalation Policy State Machine")
    print("=" * 45)

    policy = EscalationPolicy()

    # Create test scenarios
    current_time = datetime.datetime.now(datetime.timezone.utc)

    test_scenarios = [
        {
            "name": "crypto_sla_timeout",
            "order_context": OrderContext(
                order_id="test_1",
                asset="BTC-USD",
                side="buy",
                total_size=1000,
                filled_size=200,
                remaining_size=800,
                avg_fill_price=50000,
                start_time=current_time - datetime.timedelta(seconds=130),  # Past SLA
                sla_deadline=current_time - datetime.timedelta(seconds=10),
                current_state=OrderState.POST_ONLY,
                escalation_attempts=0,
                last_escalation_time=None,
            ),
            "market_data": {
                "bid": 50100,
                "ask": 50120,
                "spread_bps": 4.0,
                "last_price": 50110,
                "tick_size": 1.0,
            },
            "baseline_spread": 4.0,
        },
        {
            "name": "equity_adverse_movement",
            "order_context": OrderContext(
                order_id="test_2",
                asset="NVDA",
                side="sell",
                total_size=500,
                filled_size=100,
                remaining_size=400,
                avg_fill_price=500.00,
                start_time=current_time - datetime.timedelta(seconds=60),
                sla_deadline=current_time + datetime.timedelta(seconds=120),
                current_state=OrderState.POST_ONLY,
                escalation_attempts=0,
                last_escalation_time=None,
            ),
            "market_data": {
                "bid": 495.00,
                "ask": 495.50,
                "spread_bps": 10.0,
                "last_price": 495.25,
                "tick_size": 0.01,
            },
            "baseline_spread": 10.0,
        },
        {
            "name": "spread_widening_backoff",
            "order_context": OrderContext(
                order_id="test_3",
                asset="ETH-USD",
                side="buy",
                total_size=10,
                filled_size=0,
                remaining_size=10,
                avg_fill_price=0,
                start_time=current_time - datetime.timedelta(seconds=90),
                sla_deadline=current_time + datetime.timedelta(seconds=30),
                current_state=OrderState.MID_POINT,
                escalation_attempts=1,
                last_escalation_time=current_time - datetime.timedelta(seconds=60),
            ),
            "market_data": {
                "bid": 3000,
                "ask": 3030,
                "spread_bps": 100.0,  # Very wide spread
                "last_price": 3015,
                "tick_size": 0.01,
            },
            "baseline_spread": 50.0,  # Spread has doubled
        },
    ]

    # Generate policy report
    report = policy.create_policy_report(test_scenarios)

    print("\n📋 Escalation Decisions:")
    for scenario in report["scenarios"]:
        decision = scenario["decision"]
        order_ctx = scenario["order_context"]

        print(f"\n  {scenario['scenario_name']}:")
        print(f"    Asset: {order_ctx['asset']} ({order_ctx['side']})")
        print(f"    Current state: {order_ctx['current_state']}")
        print(f"    Action: {decision['action'].upper()}")

        if decision["action"] == "escalate":
            print(f"    Next state: {decision['next_state']}")
            print(f"    Reasons: {', '.join(decision['reasons'])}")
            print(
                f"    Order: {decision['order_size']:.0f} @ ${decision['order_price']:.2f}"
            )
        else:
            print(
                f"    Reasons: {', '.join(decision['reasons']) if decision['reasons'] else 'No escalation needed'}"
            )

        print(f"    Adverse score: {decision['adverse_score']:.2f}")
        print(f"    Time elapsed: {decision['elapsed_pct']:.1%}")

    print(f"\n📊 Summary:")
    print(f"  Escalation rate: {report['summary']['escalation_rate']:.1%}")
    print(f"  Average adverse score: {report['summary']['avg_adverse_score']:.2f}")
    print(f"  Average time elapsed: {report['summary']['avg_elapsed_pct']:.1%}")

    print(f"\n✅ Escalation policy test complete!")

    return report


if __name__ == "__main__":
    test_escalation_policy()
