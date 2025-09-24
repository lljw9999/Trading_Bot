#!/usr/bin/env python3
"""
RL Policy Integration for Execution Layer
Safely integrates RL policy outputs with baseline execution strategies
"""
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from src.rl.action_mixer import ActionMixer, BaselineStrategies


@dataclass
class ExecutionDecision:
    """Execution decision with safety metadata."""

    size_fraction: float  # Fraction of target quantity per order
    timing_delay: float  # Delay between orders (seconds)
    aggression: float  # Market impact tolerance [-1, 1]
    venue_preference: str  # Preferred venue
    influence_used: float  # Actual influence weight applied
    source: str  # "baseline", "mixed", or "policy"


class RLExecutionIntegrator:
    """
    Integrates RL policy recommendations with safe baseline execution strategies.

    Safety features:
    - Always produces valid execution parameters
    - Falls back to baseline on any policy failures
    - Logs all decisions for audit
    - Respects current influence weight
    """

    def __init__(self):
        self.mixer = ActionMixer()
        self.logger = logging.getLogger("rl_execution")

    def get_execution_decision(
        self,
        symbol: str,
        target_quantity: float,
        time_horizon_sec: float,
        market_conditions: Optional[Dict[str, Any]] = None,
        policy_recommendation: Optional[Dict[str, float]] = None,
    ) -> ExecutionDecision:
        """
        Get execution decision by mixing RL policy with safe baseline.

        Args:
            symbol: Trading symbol
            target_quantity: Total quantity to execute
            time_horizon_sec: Maximum execution time
            market_conditions: Current market state (spread, volatility, etc.)
            policy_recommendation: RL policy output (optional)

        Returns:
            ExecutionDecision with safe parameters
        """
        try:
            # Get baseline strategy based on market conditions
            baseline_params = self._get_baseline_strategy(
                symbol, target_quantity, time_horizon_sec, market_conditions
            )

            # Get current influence weight for logging
            mixing_stats = self.mixer.get_mixing_stats()
            influence_weight = mixing_stats["influence_weight"]

            if influence_weight <= 0.0:
                # Pure baseline mode
                self.logger.info(
                    f"Using baseline execution for {symbol} (influence: 0%)"
                )
                return ExecutionDecision(
                    size_fraction=baseline_params["size_fraction"],
                    timing_delay=baseline_params["timing_delay"],
                    aggression=baseline_params["aggression"],
                    venue_preference=baseline_params.get("venue", "best"),
                    influence_used=0.0,
                    source="baseline",
                )

            elif policy_recommendation is None:
                # No policy recommendation available
                self.logger.warning(
                    f"No policy recommendation for {symbol} - using baseline"
                )
                return ExecutionDecision(
                    size_fraction=baseline_params["size_fraction"],
                    timing_delay=baseline_params["timing_delay"],
                    aggression=baseline_params["aggression"],
                    venue_preference=baseline_params.get("venue", "best"),
                    influence_used=0.0,
                    source="baseline",
                )

            else:
                # Mix policy with baseline
                mixed_params = self.mixer.choose_action(
                    policy_recommendation,
                    baseline_params,
                    continuous=True,
                    action_type="execution_params",
                )

                self.logger.info(
                    f"Mixed execution for {symbol} (influence: {influence_weight:.1%})"
                )
                return ExecutionDecision(
                    size_fraction=self._clamp_size_fraction(
                        mixed_params["size_fraction"]
                    ),
                    timing_delay=max(0, mixed_params["timing_delay"]),
                    aggression=max(-1, min(1, mixed_params["aggression"])),
                    venue_preference=mixed_params.get("venue", "best"),
                    influence_used=influence_weight,
                    source="mixed",
                )

        except Exception as e:
            self.logger.error(
                f"Execution integration failed: {e} - using safe baseline"
            )
            baseline_params = BaselineStrategies.conservative_twap()

            return ExecutionDecision(
                size_fraction=baseline_params["size_fraction"],
                timing_delay=baseline_params["timing_delay"],
                aggression=baseline_params["aggression"],
                venue_preference="best",
                influence_used=0.0,
                source="baseline_fallback",
            )

    def _get_baseline_strategy(
        self,
        symbol: str,
        target_quantity: float,
        time_horizon_sec: float,
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Select appropriate baseline strategy based on market conditions.

        Args:
            symbol: Trading symbol
            target_quantity: Target quantity
            time_horizon_sec: Time horizon
            market_conditions: Market state

        Returns:
            Baseline execution parameters
        """
        if market_conditions is None:
            market_conditions = {}

        spread = market_conditions.get("spread", 0.001)  # Default 10 bps
        volatility = market_conditions.get("volatility", 0.02)  # Default 2%

        # Adjust strategy based on market conditions
        if spread > 0.005 or volatility > 0.05:  # High spread or volatility
            # Use more conservative approach
            return {
                "size_fraction": 0.05,  # Very small chunks
                "timing_delay": 120.0,  # 2 minutes between orders
                "aggression": -0.5,  # Well inside spread
                "venue": "best",
            }
        elif time_horizon_sec < 300:  # Short time horizon (< 5 minutes)
            # Use faster execution
            return {
                "size_fraction": 0.2,  # Larger chunks
                "timing_delay": 30.0,  # 30 seconds between orders
                "aggression": 0.2,  # Slightly aggressive
                "venue": "primary",
            }
        else:
            # Standard TWAP approach
            return BaselineStrategies.conservative_twap()

    def _clamp_size_fraction(self, size_fraction: float) -> float:
        """Clamp size fraction to safe range."""
        return max(0.01, min(0.5, size_fraction))  # Between 1% and 50%

    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status for monitoring."""
        mixing_stats = self.mixer.get_mixing_stats()

        return {
            "influence_active": mixing_stats["influence_weight"] > 0,
            "influence_percentage": mixing_stats["influence_percentage"],
            "ttl_remaining": mixing_stats.get("ttl_seconds", -1),
            "integration_mode": (
                "active" if mixing_stats["influence_weight"] > 0 else "baseline_only"
            ),
            "safety_checks": "enabled",
            "fallback_strategy": "conservative_twap",
        }


# Example integration for paper trading
def demo_paper_execution():
    """Demonstrate safe RL integration for paper trading."""
    integrator = RLExecutionIntegrator()

    # Example market conditions
    market_conditions = {
        "spread": 0.002,  # 20 bps spread
        "volatility": 0.03,  # 3% volatility
        "liquidity": "high",
    }

    # Example policy recommendation (could be from RL model)
    policy_rec = {
        "size_fraction": 0.8,  # Policy wants large orders
        "timing_delay": 0.0,  # Policy wants immediate execution
        "aggression": 0.9,  # Policy wants aggressive execution
        "venue": "fastest",
    }

    # Get execution decision
    decision = integrator.get_execution_decision(
        symbol="BTC-USD",
        target_quantity=1.0,
        time_horizon_sec=600,  # 10 minutes
        market_conditions=market_conditions,
        policy_recommendation=policy_rec,
    )

    print(f"Execution Decision: {decision}")
    print(f"Integration Status: {integrator.get_integration_status()}")

    return decision


if __name__ == "__main__":
    demo_paper_execution()
