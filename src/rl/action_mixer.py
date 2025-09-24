#!/usr/bin/env python3
"""
Safe Action Mixer for RL Policy Integration
Blends RL policy actions with safe baseline actions based on influence weight
"""
import random
import numpy as np
from typing import Union, Dict, Any, Optional
from src.rl.influence_controller import InfluenceController
import logging


class ActionMixer:
    """
    Safe action mixer that blends RL policy actions with baseline actions.

    Safety features:
    - Always defaults to baseline when influence is 0%
    - Supports both continuous and discrete mixing modes
    - Logs all mixing decisions for audit
    - Graceful fallback to baseline on errors
    """

    def __init__(self, ttl_sec: int = 3600):
        """
        Initialize action mixer.

        Args:
            ttl_sec: TTL for influence controller
        """
        self.influence_controller = InfluenceController(ttl_sec)
        self.logger = logging.getLogger("action_mixer")

    def choose_action(
        self,
        policy_action: Union[float, np.ndarray, Dict[str, Any]],
        baseline_action: Union[float, np.ndarray, Dict[str, Any]],
        continuous: bool = True,
        action_type: str = "generic",
    ) -> Union[float, np.ndarray, Dict[str, Any]]:
        """
        Choose action by mixing policy and baseline based on current influence weight.

        Args:
            policy_action: Action from RL policy
            baseline_action: Safe baseline action (e.g., TWAP, do-nothing)
            continuous: If True, linear blend. If False, epsilon-greedy selection
            action_type: Type of action for logging (e.g., "order_size", "timing")

        Returns:
            Mixed action based on influence weight
        """
        try:
            # Get current influence weight
            weight = self.influence_controller.get_weight()

            # Safety: always use baseline if weight is 0
            if weight <= 0.0:
                self.logger.debug(f"Using baseline {action_type} (influence: 0%)")
                return baseline_action

            # Log non-zero influence usage
            self.logger.info(f"Mixing {action_type} with {weight:.1%} policy influence")

            if continuous:
                return self._continuous_mix(policy_action, baseline_action, weight)
            else:
                return self._discrete_mix(policy_action, baseline_action, weight)

        except Exception as e:
            self.logger.error(f"Action mixing failed: {e} - falling back to baseline")
            return baseline_action

    def _continuous_mix(
        self,
        policy_action: Union[float, np.ndarray, Dict[str, Any]],
        baseline_action: Union[float, np.ndarray, Dict[str, Any]],
        weight: float,
    ) -> Union[float, np.ndarray, Dict[str, Any]]:
        """
        Continuous linear blending of actions.

        Formula: (1 - weight) * baseline + weight * policy
        """
        if isinstance(policy_action, dict) and isinstance(baseline_action, dict):
            # Handle dictionary actions (e.g., multiple trading parameters)
            mixed_action = {}
            for key in baseline_action.keys():
                if key in policy_action:
                    mixed_action[key] = (1 - weight) * baseline_action[
                        key
                    ] + weight * policy_action[key]
                else:
                    # Use baseline if policy doesn't have this parameter
                    mixed_action[key] = baseline_action[key]
            return mixed_action

        elif isinstance(policy_action, (int, float)) and isinstance(
            baseline_action, (int, float)
        ):
            # Handle scalar actions
            return (1 - weight) * baseline_action + weight * policy_action

        elif isinstance(policy_action, np.ndarray) and isinstance(
            baseline_action, np.ndarray
        ):
            # Handle array actions (e.g., multi-dimensional order parameters)
            if policy_action.shape != baseline_action.shape:
                self.logger.warning("Action shape mismatch - using baseline")
                return baseline_action
            return (1 - weight) * baseline_action + weight * policy_action

        else:
            self.logger.warning(
                f"Unsupported action types: {type(policy_action)}, {type(baseline_action)}"
            )
            return baseline_action

    def _discrete_mix(
        self,
        policy_action: Union[float, np.ndarray, Dict[str, Any]],
        baseline_action: Union[float, np.ndarray, Dict[str, Any]],
        weight: float,
    ) -> Union[float, np.ndarray, Dict[str, Any]]:
        """
        Discrete epsilon-greedy selection between policy and baseline.

        Args:
            weight: Probability of choosing policy action
        """
        use_policy = random.random() < weight

        if use_policy:
            self.logger.debug("Selected policy action")
            return policy_action
        else:
            self.logger.debug("Selected baseline action")
            return baseline_action

    def get_mixing_stats(self) -> Dict[str, Any]:
        """Get current mixing configuration and stats."""
        status = self.influence_controller.get_status()
        return {
            "influence_weight": status["weight"],
            "influence_percentage": status["percentage"],
            "ttl_seconds": status.get("ttl_seconds", -1),
            "key_exists": status["key_exists"],
            "mixing_mode": "active" if status["weight"] > 0 else "baseline_only",
        }


# Convenience functions for integration
def mix_order_size(
    policy_size: float, baseline_size: float, continuous: bool = True
) -> float:
    """Mix order sizes using current influence weight."""
    mixer = ActionMixer()
    return mixer.choose_action(policy_size, baseline_size, continuous, "order_size")


def mix_timing_decision(
    policy_timing: float, baseline_timing: float, continuous: bool = True
) -> float:
    """Mix timing decisions using current influence weight."""
    mixer = ActionMixer()
    return mixer.choose_action(policy_timing, baseline_timing, continuous, "timing")


def mix_execution_params(
    policy_params: Dict[str, float],
    baseline_params: Dict[str, float],
    continuous: bool = True,
) -> Dict[str, float]:
    """Mix execution parameters using current influence weight."""
    mixer = ActionMixer()
    return mixer.choose_action(
        policy_params, baseline_params, continuous, "execution_params"
    )


# Example baseline strategies
class BaselineStrategies:
    """Collection of safe baseline strategies for different action types."""

    @staticmethod
    def conservative_twap() -> Dict[str, float]:
        """Conservative TWAP baseline parameters."""
        return {
            "size_fraction": 0.1,  # Small chunks
            "timing_delay": 60.0,  # 1 minute between orders
            "aggression": 0.0,  # Passive orders only
        }

    @staticmethod
    def do_nothing() -> Dict[str, float]:
        """Do-nothing baseline (no trading)."""
        return {"size_fraction": 0.0, "timing_delay": float("inf"), "aggression": 0.0}

    @staticmethod
    def safe_limit_order(spread_fraction: float = 0.5) -> Dict[str, float]:
        """Safe limit order baseline."""
        return {
            "size_fraction": 0.05,  # Very small size
            "timing_delay": 30.0,  # Short delay
            "aggression": -spread_fraction,  # Inside spread
        }


if __name__ == "__main__":
    # Demo usage
    mixer = ActionMixer()

    # Example: Mix order sizing
    policy_size = 1000.0  # Policy wants large order
    baseline_size = 100.0  # Baseline prefers small orders

    mixed_size = mixer.choose_action(
        policy_size, baseline_size, continuous=True, action_type="order_size"
    )
    print(f"Mixed order size: {mixed_size}")

    # Example: Mix execution parameters
    policy_params = {"size_fraction": 0.8, "timing_delay": 0.0, "aggression": 0.9}
    baseline_params = BaselineStrategies.conservative_twap()

    mixed_params = mixer.choose_action(
        policy_params, baseline_params, continuous=True, action_type="execution"
    )
    print(f"Mixed parameters: {mixed_params}")

    # Show current stats
    stats = mixer.get_mixing_stats()
    print(f"Mixing stats: {stats}")
