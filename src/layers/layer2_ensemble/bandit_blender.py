#!/usr/bin/env python3
"""
Contextual Bandit Ensemble Weights (LinUCB)
Dynamic ensemble weight selection using contextual multi-armed bandits
"""

import json
import time
import redis
import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

try:
    from contextualbandits import online

    HAS_CONTEXTUAL_BANDITS = True
except ImportError:
    HAS_CONTEXTUAL_BANDITS = False
    online = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bandit_blender")


class BanditBlender:
    """Contextual bandit for dynamic ensemble weight selection."""

    def __init__(
        self,
        arms: List[str] = None,
        alpha: float = 0.25,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        """
        Initialize contextual bandit blender.

        Args:
            arms: List of alpha model names
            alpha: LinUCB exploration parameter (higher = more exploration)
            redis_host: Redis server host
            redis_port: Redis server port
        """
        # Default arms (alpha models)
        self.arms = arms or ["obp", "mam", "lstm", "news", "onchain"]
        self.n_arms = len(self.arms)
        self.alpha = alpha

        # Redis connection
        self.redis = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )

        # Initialize LinUCB contextual bandit
        # Initialize bandit model
        if HAS_CONTEXTUAL_BANDITS:
            self.bandit = online.LinUCB(
                nchoices=self.n_arms,
                alpha=self.alpha,
                fit_intercept=True,
                random_state=42,
            )
        else:
            # Fallback to uniform weights when contextual bandits unavailable
            self.bandit = None
            logger.warning("contextualbandits not available, using uniform weights")

        # Performance tracking
        self.total_rounds = 0
        self.cumulative_reward = 0.0
        self.arm_counts = np.zeros(self.n_arms)
        self.arm_rewards = np.zeros(self.n_arms)

        # Load persisted bandit state if available
        self.load_bandit_state()

        logger.info(f"🎰 BanditBlender initialized")
        logger.info(f"   Arms: {self.arms}")
        logger.info(f"   Alpha (exploration): {self.alpha}")
        logger.info(f"   Total rounds: {self.total_rounds}")

    def extract_context_features(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Extract context features for bandit decision.

        Args:
            state: Current market state dictionary

        Returns:
            Context feature vector
        """
        try:
            features = [
                float(state.get("vol_20", 0.0)),  # 20-period volatility
                float(state.get("sent_bull", 0.0)),  # Bullish sentiment
                float(state.get("sent_bear", 0.0)),  # Bearish sentiment
                float(state.get("iv_slope", 0.0)),  # IV term structure slope
                float(state.get("rsi", 50.0)),  # RSI momentum
                float(state.get("volume_ratio", 1.0)),  # Volume ratio
                float(state.get("spread_pct", 0.0)),  # Bid-ask spread
                float(state.get("market_cap_flow", 0.0)),  # Market cap weighted flow
                float(state.get("funding_rate", 0.0)),  # Perpetual funding rate
                float(state.get("oi_change", 0.0)),  # Open interest change
            ]

            return np.array(features, dtype=np.float64)

        except Exception as e:
            logger.warning(f"Error extracting context features: {e}")
            # Return zero features as fallback
            return np.zeros(10, dtype=np.float64)

    def choose_weights(self, context: np.ndarray) -> Dict[str, float]:
        """
        Choose ensemble weights using contextual bandit.

        Args:
            context: Context feature vector

        Returns:
            Dictionary mapping arm names to weights
        """
        try:
            # Reshape context for bandit
            context_matrix = context.reshape(1, -1)

            # Get bandit prediction (chosen arm index)
            chosen_arm = self.bandit.predict(context_matrix)[0]

            # Create weight vector with high weight for chosen arm
            weights = np.full(self.n_arms, 0.05)  # Baseline weight
            weights[chosen_arm] = 0.8  # High weight for chosen arm

            # Normalize weights
            weights = weights / weights.sum()

            # Create arm -> weight mapping
            weight_dict = {arm: float(weights[i]) for i, arm in enumerate(self.arms)}

            # Store in Redis for immediate use
            self.redis.hset("ensemble:weights", mapping=weight_dict)

            # Store metadata (convert to strings for Redis)
            metadata = {
                "chosen_arm": str(chosen_arm),
                "chosen_model": str(self.arms[chosen_arm]),
                "alpha": str(self.alpha),
                "total_rounds": str(self.total_rounds),
                "timestamp": str(time.time()),
            }
            self.redis.hset("ensemble:bandit_meta", mapping=metadata)

            logger.info(f"Bandit choice: {self.arms[chosen_arm]} (arm {chosen_arm})")
            return weight_dict

        except Exception as e:
            logger.error(f"Error in choose_weights: {e}")
            # Return uniform weights as fallback
            uniform_weight = 1.0 / self.n_arms
            return {arm: uniform_weight for arm in self.arms}

    def update_bandit(self, context: np.ndarray, reward_vector: np.ndarray) -> None:
        """
        Update bandit with observed rewards.

        Args:
            context: Context features used for last decision
            reward_vector: Reward for each arm (PnL or hit rate)
        """
        try:
            # Find which arm achieved the best reward
            best_arm = np.argmax(reward_vector)
            best_reward = float(reward_vector[best_arm])

            # Reshape context for bandit update
            context_matrix = context.reshape(1, -1)

            # Update bandit with partial feedback
            # Note: LinUCB needs the reward for the chosen arm, not all arms
            self.bandit.partial_fit(
                context_matrix, np.array([best_arm]), np.array([best_reward])
            )

            # Update tracking statistics
            self.total_rounds += 1
            self.cumulative_reward += best_reward
            self.arm_counts[best_arm] += 1
            self.arm_rewards[best_arm] += best_reward

            # Persist bandit state
            self.save_bandit_state()

            logger.info(
                f"Bandit updated: arm {best_arm} ({self.arms[best_arm]}) "
                f"reward {best_reward:.4f}, total rounds: {self.total_rounds}"
            )

        except Exception as e:
            logger.error(f"Error updating bandit: {e}")

    def update_from_pnl_feedback(self, last_pnl: float, context: np.ndarray) -> None:
        """
        Update bandit based on PnL feedback.

        Args:
            last_pnl: PnL from last period
            context: Context features from last decision
        """
        try:
            # Get the last chosen arm from Redis
            meta = self.redis.hgetall("ensemble:bandit_meta")
            last_chosen_arm = int(meta.get("chosen_arm", 0))

            # Create reward vector with PnL for chosen arm
            reward_vector = np.zeros(self.n_arms)
            reward_vector[last_chosen_arm] = last_pnl

            # Update bandit
            self.update_bandit(context, reward_vector)

        except Exception as e:
            logger.error(f"Error in PnL feedback update: {e}")

    def set_exploration_mode(self, explore: bool = True) -> None:
        """
        Toggle exploration vs exploitation mode.

        Args:
            explore: If True, increase alpha for more exploration
        """
        if explore:
            self.bandit.alpha = 1.0  # High exploration
            logger.info("🔍 Bandit set to exploration mode (alpha=1.0)")
        else:
            self.bandit.alpha = self.alpha  # Normal exploration
            logger.info(f"🎯 Bandit set to exploitation mode (alpha={self.alpha})")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get bandit performance statistics."""
        try:
            avg_reward = self.cumulative_reward / max(1, self.total_rounds)

            arm_stats = {}
            for i, arm in enumerate(self.arms):
                count = int(self.arm_counts[i])
                total_reward = float(self.arm_rewards[i])
                avg_arm_reward = total_reward / max(1, count)

                arm_stats[arm] = {
                    "selections": count,
                    "total_reward": total_reward,
                    "avg_reward": avg_arm_reward,
                    "selection_rate": count / max(1, self.total_rounds),
                }

            return {
                "total_rounds": self.total_rounds,
                "cumulative_reward": self.cumulative_reward,
                "average_reward": avg_reward,
                "exploration_parameter": self.bandit.alpha,
                "arm_statistics": arm_stats,
            }

        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {"error": str(e)}

    def save_bandit_state(self) -> None:
        """Save bandit state to Redis for persistence."""
        try:
            # Serialize bandit object (convert all to strings for Redis)
            bandit_data = {
                "bandit_pickle": pickle.dumps(self.bandit).hex(),
                "total_rounds": str(self.total_rounds),
                "cumulative_reward": str(float(self.cumulative_reward)),
                "arm_counts": json.dumps(self.arm_counts.tolist()),
                "arm_rewards": json.dumps(self.arm_rewards.tolist()),
                "alpha": str(self.alpha),
                "arms": json.dumps(self.arms),
            }

            self.redis.hset("bandit:state", mapping=bandit_data)

        except Exception as e:
            logger.warning(f"Failed to save bandit state: {e}")

    def load_bandit_state(self) -> bool:
        """Load bandit state from Redis."""
        try:
            state = self.redis.hgetall("bandit:state")
            if not state:
                logger.info("No saved bandit state found, starting fresh")
                return False

            # Restore bandit object
            bandit_bytes = bytes.fromhex(state["bandit_pickle"])
            self.bandit = pickle.loads(bandit_bytes)

            # Restore statistics
            self.total_rounds = int(state["total_rounds"])
            self.cumulative_reward = float(state["cumulative_reward"])
            self.arm_counts = np.array(json.loads(state["arm_counts"]))
            self.arm_rewards = np.array(json.loads(state["arm_rewards"]))

            logger.info(f"✅ Loaded bandit state: {self.total_rounds} rounds")
            return True

        except Exception as e:
            logger.warning(f"Failed to load bandit state: {e}")
            return False


def run_bandit_refresh_cron():
    """CRON job for 30-minute bandit refresh based on PnL performance."""
    logger.info("🔄 Running bandit refresh CRON job")

    try:
        redis_client = redis.Redis(decode_responses=True)
        blender = BanditBlender()

        # Get PnL from last 30 minutes
        pnl_30m = redis_client.get("pnl:last_30m")
        if pnl_30m is None:
            logger.warning("No PnL data available for bandit refresh")
            return

        pnl_30m = float(pnl_30m)
        logger.info(f"Last 30m PnL: {pnl_30m:.4f}")

        # Set exploration mode based on performance
        if pnl_30m > 0:
            # Keep exploiting current strategy
            blender.set_exploration_mode(explore=False)
            logger.info("✅ Positive PnL: continuing exploitation")
        else:
            # Explore new strategies
            blender.set_exploration_mode(explore=True)
            logger.info("🔍 Negative PnL: switching to exploration")

        # Update Redis with refresh timestamp
        redis_client.set("bandit:last_refresh", int(time.time()))

    except Exception as e:
        logger.error(f"Error in bandit refresh CRON: {e}")


def main():
    """Main entry point for testing and utilities."""
    import argparse

    parser = argparse.ArgumentParser(description="Contextual Bandit Blender")
    parser.add_argument(
        "--test", action="store_true", help="Test bandit with synthetic data"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show performance statistics"
    )
    parser.add_argument("--refresh", action="store_true", help="Run CRON refresh job")
    parser.add_argument("--reset", action="store_true", help="Reset bandit state")

    args = parser.parse_args()

    blender = BanditBlender()

    if args.test:
        logger.info("🧪 Testing bandit blender with synthetic data")

        # Test with random contexts and rewards
        for round_num in range(10):
            # Generate synthetic context
            context = np.random.randn(10)

            # Get weights
            weights = blender.choose_weights(context)
            logger.info(f"Round {round_num}: weights = {weights}")

            # Simulate rewards
            reward_vector = np.random.randn(len(blender.arms)) * 0.01
            blender.update_bandit(context, reward_vector)

            time.sleep(0.5)

    elif args.stats:
        stats = blender.get_performance_stats()
        print(json.dumps(stats, indent=2))

    elif args.refresh:
        run_bandit_refresh_cron()

    elif args.reset:
        redis_client = redis.Redis(decode_responses=True)
        redis_client.delete("bandit:state")
        redis_client.delete("ensemble:weights")
        redis_client.delete("ensemble:bandit_meta")
        logger.info("🗑️ Bandit state reset")

    else:
        print("Use --test, --stats, --refresh, or --reset")


if __name__ == "__main__":
    main()
