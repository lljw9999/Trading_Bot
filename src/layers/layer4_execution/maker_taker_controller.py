#!/usr/bin/env python3
"""
Maker/Taker Mode Controller
Decides post-only vs cross fills using fill probability & adverse selection scoring.
Optimizes for maker rebates while respecting SLA constraints.
"""
import os
import sys
import json
import time
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class FillMode(Enum):
    POST_ONLY = "post_only"  # Maker-only, will not cross spread
    CROSS = "cross"  # Taker, can cross spread immediately
    ADAPTIVE = "adaptive"  # Try post first, fall back to cross


@dataclass
class MarketData:
    """Market data for maker/taker decision."""

    asset: str
    venue: str
    timestamp: datetime.datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread_bps: float
    depth_bps: float
    recent_fill_rate: float
    volatility_5m: float


@dataclass
class OrderContext:
    """Order context for execution decision."""

    side: str  # "buy" or "sell"
    size: float
    urgency_score: float  # 0-1, higher = more urgent
    max_delay_seconds: int
    expected_adverse_move_bps: float


@dataclass
class MakerTakerDecision:
    """Decision output from maker/taker controller."""

    fill_mode: FillMode
    post_price: Optional[float]
    cross_price: Optional[float]
    fill_probability: float
    expected_delay_seconds: float
    adverse_selection_score: float
    spread_capture_target_bps: float
    reason: str


class MakerTakerController:
    def __init__(self):
        self.rebate_targets = {
            "coinbase": {"crypto": -1.0, "stocks": 0.0},  # -1bp rebate for crypto
            "binance": {"crypto": -0.5, "stocks": 0.0},  # -0.5bp rebate for crypto
            "alpaca": {"crypto": 0.0, "stocks": -0.2},  # -0.2bp rebate for stocks
        }

        # Thresholds for decision making
        self.min_spread_for_post_bps = 3.0  # Minimum spread to attempt posting
        self.max_adverse_selection_bps = 2.0  # Max acceptable adverse selection
        self.target_maker_ratio = 0.60  # Target 60% maker fills

        # SLA constraints
        self.max_fill_delay_seconds = 180  # 3 minutes max delay

    def calculate_fill_probability(
        self, market_data: MarketData, order_ctx: OrderContext
    ) -> float:
        """Calculate probability of fill if posting at best bid/ask."""

        # Base fill probability from recent market activity
        base_prob = market_data.recent_fill_rate

        # Adjust based on order size vs available depth
        if order_ctx.side == "buy":
            size_ratio = order_ctx.size / max(market_data.bid_size, 1.0)
        else:
            size_ratio = order_ctx.size / max(market_data.ask_size, 1.0)

        # Larger orders relative to depth have lower fill probability
        size_adjustment = np.exp(-size_ratio * 2)

        # Adjust based on spread width (wider spreads = higher fill probability)
        spread_adjustment = min(1.0, market_data.spread_bps / 5.0)

        # Volatility reduces fill probability (market moving away)
        vol_adjustment = np.exp(-market_data.volatility_5m * 100)

        fill_prob = base_prob * size_adjustment * spread_adjustment * vol_adjustment
        return max(0.01, min(0.99, fill_prob))  # Clamp to reasonable range

    def calculate_adverse_selection_score(
        self, market_data: MarketData, order_ctx: OrderContext
    ) -> float:
        """Calculate adverse selection score (higher = more adverse selection risk)."""

        # Base adverse selection from expected move
        base_adverse = order_ctx.expected_adverse_move_bps / 10.0  # Normalize

        # High volatility increases adverse selection
        vol_factor = market_data.volatility_5m * 1000  # Scale to reasonable range

        # Wide spreads reduce adverse selection (more protection)
        spread_protection = max(0.1, market_data.spread_bps / 10.0)

        # Order size vs depth affects adverse selection
        if order_ctx.side == "buy":
            depth_ratio = order_ctx.size / max(market_data.ask_size, 1.0)
        else:
            depth_ratio = order_ctx.size / max(market_data.bid_size, 1.0)

        # Larger orders create more adverse selection
        size_factor = depth_ratio * 0.5

        adverse_score = (base_adverse + vol_factor + size_factor) / spread_protection
        return max(0.0, min(1.0, adverse_score))

    def calculate_expected_delay(
        self, fill_probability: float, market_data: MarketData
    ) -> float:
        """Calculate expected delay to fill in seconds."""

        if fill_probability <= 0.01:
            return self.max_fill_delay_seconds

        # Base delay from fill probability (exponential relationship)
        base_delay = -np.log(fill_probability) * 30  # 30 seconds per probability unit

        # Adjust based on market activity
        activity_factor = 1.0 / max(market_data.recent_fill_rate, 0.1)

        expected_delay = base_delay * activity_factor
        return min(expected_delay, self.max_fill_delay_seconds)

    def calculate_spread_capture_target(
        self, market_data: MarketData, order_ctx: OrderContext
    ) -> float:
        """Calculate target spread capture in basis points."""

        # Start with rebate target
        rebate_bps = self.get_rebate_target(market_data.asset, market_data.venue)

        # Add buffer for adverse selection
        adverse_buffer = order_ctx.expected_adverse_move_bps * 0.5

        # Add urgency penalty (urgent orders get less spread capture)
        urgency_penalty = order_ctx.urgency_score * market_data.spread_bps * 0.3

        # Target spread capture
        target_capture = abs(rebate_bps) + adverse_buffer - urgency_penalty

        # Cap at reasonable percentage of spread
        max_capture = market_data.spread_bps * 0.7
        return max(0.1, min(target_capture, max_capture))

    def get_rebate_target(self, asset: str, venue: str) -> float:
        """Get rebate target for asset/venue combination."""
        venue_rebates = self.rebate_targets.get(venue, {"crypto": 0.0, "stocks": 0.0})

        if asset in ["SOL-USD", "BTC-USD", "ETH-USD"]:
            return venue_rebates["crypto"]
        else:
            return venue_rebates["stocks"]

    def make_decision(
        self, market_data: MarketData, order_ctx: OrderContext
    ) -> MakerTakerDecision:
        """Make maker/taker decision based on market conditions and order context."""

        # Calculate key metrics
        fill_prob = self.calculate_fill_probability(market_data, order_ctx)
        adverse_score = self.calculate_adverse_selection_score(market_data, order_ctx)
        expected_delay = self.calculate_expected_delay(fill_prob, market_data)
        spread_capture_target = self.calculate_spread_capture_target(
            market_data, order_ctx
        )

        # Decision logic
        reasons = []

        # Check if spread is wide enough for posting
        if market_data.spread_bps < self.min_spread_for_post_bps:
            fill_mode = FillMode.CROSS
            reasons.append(f"spread_too_narrow_{market_data.spread_bps:.1f}bp")

        # Check SLA constraints
        elif expected_delay > order_ctx.max_delay_seconds:
            fill_mode = FillMode.CROSS
            reasons.append(
                f"sla_breach_{expected_delay:.0f}s>{order_ctx.max_delay_seconds}s"
            )

        # Check adverse selection limits
        elif adverse_score > (self.max_adverse_selection_bps / 10.0):
            fill_mode = FillMode.CROSS
            reasons.append(f"adverse_selection_high_{adverse_score:.2f}")

        # Check urgency
        elif order_ctx.urgency_score > 0.8:
            fill_mode = FillMode.CROSS
            reasons.append(f"high_urgency_{order_ctx.urgency_score:.2f}")

        # Conditions favor posting
        else:
            # Check if we should try adaptive (post first, cross on timeout)
            if fill_prob > 0.6 and adverse_score < 0.3:
                fill_mode = FillMode.POST_ONLY
                reasons.append(
                    f"favorable_conditions_prob_{fill_prob:.2f}_adverse_{adverse_score:.2f}"
                )
            else:
                fill_mode = FillMode.ADAPTIVE
                reasons.append(
                    f"moderate_conditions_prob_{fill_prob:.2f}_adverse_{adverse_score:.2f}"
                )

        # Calculate prices
        post_price = None
        cross_price = None

        if fill_mode in [FillMode.POST_ONLY, FillMode.ADAPTIVE]:
            if order_ctx.side == "buy":
                # Post at bid to capture spread
                post_price = market_data.bid_price
                cross_price = market_data.ask_price
            else:
                # Post at ask to capture spread
                post_price = market_data.ask_price
                cross_price = market_data.bid_price
        else:
            # Cross immediately
            if order_ctx.side == "buy":
                cross_price = market_data.ask_price
            else:
                cross_price = market_data.bid_price

        return MakerTakerDecision(
            fill_mode=fill_mode,
            post_price=post_price,
            cross_price=cross_price,
            fill_probability=fill_prob,
            expected_delay_seconds=expected_delay,
            adverse_selection_score=adverse_score,
            spread_capture_target_bps=spread_capture_target,
            reason="; ".join(reasons),
        )

    def update_fill_statistics(
        self,
        asset: str,
        venue: str,
        was_maker: bool,
        rebate_bps: float,
        timestamp: datetime.datetime,
    ):
        """Update running statistics for maker/taker performance."""

        # In production, this would update a Redis-backed statistics store
        # For now, we'll log the statistics

        stats = {
            "timestamp": timestamp.isoformat() + "Z",
            "asset": asset,
            "venue": venue,
            "was_maker": was_maker,
            "rebate_bps": rebate_bps,
            "fill_type": "maker" if was_maker else "taker",
        }

        # Log to audit trail
        audit_dir = "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D/artifacts/audit"
        os.makedirs(audit_dir, exist_ok=True)

        audit_file = f"{audit_dir}/maker_taker_fills_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(audit_file, "a") as f:
            f.write(json.dumps(stats) + "\n")


def test_maker_taker_controller():
    """Test the maker/taker controller with sample data."""
    controller = MakerTakerController()

    # Test case 1: Wide spread, good conditions for posting
    market_data = MarketData(
        asset="SOL-USD",
        venue="coinbase",
        timestamp=datetime.datetime.now(),
        bid_price=100.0,
        ask_price=100.5,  # 50bp spread
        bid_size=1000,
        ask_size=1200,
        spread_bps=5.0,
        depth_bps=20.0,
        recent_fill_rate=0.7,
        volatility_5m=0.001,
    )

    order_ctx = OrderContext(
        side="buy",
        size=500,
        urgency_score=0.3,
        max_delay_seconds=120,
        expected_adverse_move_bps=1.0,
    )

    decision = controller.make_decision(market_data, order_ctx)

    print("🧪 Maker/Taker Controller Test Results:")
    print(f"Decision: {decision.fill_mode.value}")
    print(f"Post Price: {decision.post_price}")
    print(f"Cross Price: {decision.cross_price}")
    print(f"Fill Probability: {decision.fill_probability:.2f}")
    print(f"Expected Delay: {decision.expected_delay_seconds:.1f}s")
    print(f"Adverse Selection Score: {decision.adverse_selection_score:.3f}")
    print(f"Spread Capture Target: {decision.spread_capture_target_bps:.1f}bp")
    print(f"Reason: {decision.reason}")


if __name__ == "__main__":
    test_maker_taker_controller()
