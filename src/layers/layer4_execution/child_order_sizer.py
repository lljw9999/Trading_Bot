#!/usr/bin/env python3
"""
Adaptive Child Order Sizer
Dynamically sizes order slices based on volatility, queue depth, and liquidity regime.
"""
import numpy as np
import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class LiquidityRegime(Enum):
    THIN = "thin"
    NORMAL = "normal"
    HEAVY = "heavy"


@dataclass
class MarketState:
    """Current market microstructure state."""

    bid_size: float
    ask_size: float
    bid_ask_spread_bps: float
    recent_volume: float
    price_volatility: float
    regime: LiquidityRegime


@dataclass
class ChildOrderParams:
    """Child order sizing parameters."""

    slice_size: float
    max_participation_rate: float
    post_only_prob: float
    min_fill_prob: float
    adverse_selection_score: float


class ChildOrderSizer:
    def __init__(self):
        # Configuration parameters (would be loaded from config)
        self.base_slice_pct = 0.1  # 10% of parent order
        self.max_slice_pct = 0.5  # 50% max
        self.min_slice_pct = 0.02  # 2% min

        # Regime-specific parameters
        self.regime_params = {
            LiquidityRegime.THIN: {
                "slice_multiplier": 0.3,  # Smaller slices in thin markets
                "max_participation": 0.05,  # 5% max participation
                "post_only_bias": 0.8,  # Prefer post-only
            },
            LiquidityRegime.NORMAL: {
                "slice_multiplier": 1.0,  # Base sizing
                "max_participation": 0.15,  # 15% participation
                "post_only_bias": 0.5,  # Balanced
            },
            LiquidityRegime.HEAVY: {
                "slice_multiplier": 2.0,  # Larger slices possible
                "max_participation": 0.25,  # 25% participation
                "post_only_bias": 0.2,  # More aggressive crossing
            },
        }

    def classify_regime(self, market_state: MarketState) -> LiquidityRegime:
        """Classify current liquidity regime."""
        # Simple regime classification based on book depth and spread
        total_book_size = market_state.bid_size + market_state.ask_size
        spread_bps = market_state.bid_ask_spread_bps

        # Thresholds (would be asset-specific in production)
        if total_book_size < 1000 or spread_bps > 20:
            return LiquidityRegime.THIN
        elif total_book_size > 5000 and spread_bps < 5:
            return LiquidityRegime.HEAVY
        else:
            return LiquidityRegime.NORMAL

    def estimate_adverse_selection_risk(
        self, market_state: MarketState, side: str
    ) -> float:
        """Estimate adverse selection risk for crossing the spread."""
        # Simplified adverse selection model
        spread_bps = market_state.bid_ask_spread_bps
        volatility = market_state.price_volatility

        # Higher spread and volatility = higher adverse selection risk
        base_risk = min(1.0, (spread_bps / 10.0) + (volatility / 0.02))

        # Adjust for regime
        regime_multipliers = {
            LiquidityRegime.THIN: 1.5,  # Higher risk in thin markets
            LiquidityRegime.NORMAL: 1.0,
            LiquidityRegime.HEAVY: 0.7,  # Lower risk in deep markets
        }

        regime = self.classify_regime(market_state)
        adjusted_risk = base_risk * regime_multipliers[regime]

        return np.clip(adjusted_risk, 0.0, 1.0)

    def calculate_optimal_slice_size(
        self, parent_order_size: float, market_state: MarketState, side: str
    ) -> float:
        """Calculate optimal slice size based on market conditions."""

        regime = self.classify_regime(market_state)
        regime_config = self.regime_params[regime]

        # Base slice size
        base_slice = parent_order_size * self.base_slice_pct

        # Adjust for regime
        regime_adjusted = base_slice * regime_config["slice_multiplier"]

        # Volatility adjustment - smaller slices in high vol
        vol_adjustment = 1.0 - min(0.7, market_state.price_volatility / 0.03)
        vol_adjusted = regime_adjusted * vol_adjustment

        # Book depth adjustment
        total_book_size = market_state.bid_size + market_state.ask_size
        if side == "buy":
            relevant_book = market_state.ask_size
        else:
            relevant_book = market_state.bid_size

        # Don't exceed reasonable fraction of available liquidity
        max_from_book = relevant_book * regime_config["max_participation"]

        # Final slice size
        optimal_slice = min(
            max(vol_adjusted, parent_order_size * self.min_slice_pct),
            parent_order_size * self.max_slice_pct,
            max_from_book,
        )

        return optimal_slice

    def calculate_post_only_probability(
        self, market_state: MarketState, adverse_selection_score: float
    ) -> float:
        """Calculate probability of using post-only orders."""

        regime = self.classify_regime(market_state)
        base_post_prob = self.regime_params[regime]["post_only_bias"]

        # Higher adverse selection = higher post-only probability
        adverse_selection_adjustment = adverse_selection_score * 0.4

        # Spread adjustment - wider spreads favor posting
        spread_adjustment = min(0.3, market_state.bid_ask_spread_bps / 50)

        final_prob = base_post_prob + adverse_selection_adjustment + spread_adjustment
        return np.clip(final_prob, 0.1, 0.95)

    def estimate_fill_probability(
        self, market_state: MarketState, slice_size: float, side: str, post_only: bool
    ) -> float:
        """Estimate probability of order fill."""

        if post_only:
            # For post-only orders, fill prob depends on queue position and flow
            regime = self.classify_regime(market_state)
            base_fill_prob = {
                LiquidityRegime.THIN: 0.3,
                LiquidityRegime.NORMAL: 0.6,
                LiquidityRegime.HEAVY: 0.8,
            }[regime]

            # Adjust for slice size relative to typical flow
            size_penalty = min(0.4, slice_size / market_state.recent_volume)
            return max(0.1, base_fill_prob - size_penalty)

        else:
            # For market/IOC orders, fill prob is high but depends on book depth
            if side == "buy":
                available_liquidity = market_state.ask_size
            else:
                available_liquidity = market_state.bid_size

            if slice_size <= available_liquidity:
                return 0.95  # High confidence for crossing
            else:
                # Partial fill expected
                return 0.7 * (available_liquidity / slice_size)

    def size_child_order(
        self, parent_order_size: float, market_state: MarketState, side: str
    ) -> ChildOrderParams:
        """Main function to size child orders."""

        # Classify current regime
        regime = self.classify_regime(market_state)
        market_state.regime = regime

        # Calculate optimal slice size
        slice_size = self.calculate_optimal_slice_size(
            parent_order_size, market_state, side
        )

        # Estimate adverse selection risk
        adverse_selection_score = self.estimate_adverse_selection_risk(
            market_state, side
        )

        # Calculate post-only probability
        post_only_prob = self.calculate_post_only_probability(
            market_state, adverse_selection_score
        )

        # Estimate fill probabilities
        fill_prob_post = self.estimate_fill_probability(
            market_state, slice_size, side, post_only=True
        )
        fill_prob_cross = self.estimate_fill_probability(
            market_state, slice_size, side, post_only=False
        )

        # Choose minimum acceptable fill probability
        min_fill_prob = max(fill_prob_post, fill_prob_cross * 0.7)

        # Max participation rate
        regime_config = self.regime_params[regime]
        max_participation = regime_config["max_participation"]

        return ChildOrderParams(
            slice_size=slice_size,
            max_participation_rate=max_participation,
            post_only_prob=post_only_prob,
            min_fill_prob=min_fill_prob,
            adverse_selection_score=adverse_selection_score,
        )

    def get_execution_recommendation(
        self, child_params: ChildOrderParams, market_state: MarketState
    ) -> Dict[str, Any]:
        """Get execution strategy recommendation."""

        # Decision logic
        should_post = np.random.random() < child_params.post_only_prob

        if should_post:
            strategy = "POST_ONLY"
            expected_fill_prob = self.estimate_fill_probability(
                market_state, child_params.slice_size, "buy", post_only=True
            )
        else:
            if child_params.adverse_selection_score > 0.7:
                strategy = "IOC"  # Immediate or cancel to limit adverse selection
            else:
                strategy = "MARKET"  # Full market order

            expected_fill_prob = self.estimate_fill_probability(
                market_state, child_params.slice_size, "buy", post_only=False
            )

        return {
            "strategy": strategy,
            "slice_size": child_params.slice_size,
            "expected_fill_prob": expected_fill_prob,
            "regime": market_state.regime.value,
            "adverse_selection_score": child_params.adverse_selection_score,
            "reasoning": self._get_reasoning(child_params, market_state, strategy),
        }

    def _get_reasoning(
        self, child_params: ChildOrderParams, market_state: MarketState, strategy: str
    ) -> str:
        """Generate human-readable reasoning for the sizing decision."""

        reasons = []

        # Regime-based reasoning
        if market_state.regime == LiquidityRegime.THIN:
            reasons.append(
                "Thin market detected - using smaller slices and post-only bias"
            )
        elif market_state.regime == LiquidityRegime.HEAVY:
            reasons.append("Heavy liquidity - allowing larger slices and crossing")
        else:
            reasons.append("Normal market conditions - balanced approach")

        # Adverse selection reasoning
        if child_params.adverse_selection_score > 0.6:
            reasons.append(
                f"High adverse selection risk ({child_params.adverse_selection_score:.2f}) - favoring passive execution"
            )

        # Strategy-specific reasoning
        if strategy == "POST_ONLY":
            reasons.append(
                "Using post-only to capture spread and avoid adverse selection"
            )
        elif strategy == "IOC":
            reasons.append("Using IOC to limit exposure while crossing spread")
        else:
            reasons.append("Market conditions favorable for aggressive crossing")

        return "; ".join(reasons)


# Example usage and testing functions
def create_example_market_state() -> MarketState:
    """Create example market state for testing."""
    return MarketState(
        bid_size=2500.0,
        ask_size=1800.0,
        bid_ask_spread_bps=8.5,
        recent_volume=15000.0,
        price_volatility=0.015,  # 1.5% volatility
        regime=LiquidityRegime.NORMAL,
    )


def run_sizing_example():
    """Run example of child order sizing."""
    sizer = ChildOrderSizer()
    market_state = create_example_market_state()
    parent_order_size = 5000.0
    side = "buy"

    # Size the child order
    child_params = sizer.size_child_order(parent_order_size, market_state, side)

    # Get execution recommendation
    recommendation = sizer.get_execution_recommendation(child_params, market_state)

    print("Child Order Sizing Results:")
    print(f"  Parent Order Size: {parent_order_size:,.0f}")
    print(f"  Optimal Slice Size: {child_params.slice_size:,.0f}")
    print(f"  Regime: {market_state.regime.value}")
    print(f"  Strategy: {recommendation['strategy']}")
    print(f"  Expected Fill Prob: {recommendation['expected_fill_prob']:.1%}")
    print(f"  Adverse Selection Score: {child_params.adverse_selection_score:.3f}")
    print(f"  Reasoning: {recommendation['reasoning']}")


if __name__ == "__main__":
    run_sizing_example()
