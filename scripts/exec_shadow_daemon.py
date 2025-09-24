#!/usr/bin/env python3
"""
RL Execution Shadow Agent Daemon
Continuously runs RL agent in shadow mode for performance comparison
"""

import os
import sys
import json
import time
import redis
import logging
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from stable_baselines3 import PPO
from envs.orderbook_env import LiveExecEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("exec_shadow")


class ExecShadowDaemon:
    """Shadow execution daemon for RL agent testing."""

    def __init__(
        self, model_path: str = "/models/exec_agent_v1", update_interval: float = 0.05
    ):
        """
        Initialize shadow execution daemon.

        Args:
            model_path: Path to trained RL model
            update_interval: Seconds between predictions
        """
        self.model_path = model_path
        self.update_interval = update_interval

        # Redis connection
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Load trained RL model
        self.model = None
        self.env = LiveExecEnv()

        # Performance tracking
        self.predictions_count = 0
        self.start_time = time.time()

        logger.info(f"🤖 ExecShadowDaemon initialized")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Update interval: {update_interval}s")

    def load_model(self):
        """Load the trained RL model."""
        try:
            if not os.path.exists(self.model_path + ".zip"):
                logger.error(f"Model file not found: {self.model_path}.zip")
                return False

            self.model = PPO.load(self.model_path, env=None)
            logger.info(f"✅ Loaded RL model from {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False

    def get_current_state(self) -> np.ndarray:
        """Get current market state from live environment."""
        try:
            state = self.env.get_state()
            return state.astype(np.float32)

        except Exception as e:
            logger.warning(f"Error getting live state, using fallback: {e}")
            # Return fallback state if live data unavailable
            return np.zeros(16, dtype=np.float32)

    def predict_action(self, state: np.ndarray) -> np.ndarray:
        """Get RL agent prediction for current state."""
        try:
            if self.model is None:
                return np.array([0.0, 0.0, 0.0])  # Neutral action

            action, _ = self.model.predict(state, deterministic=True)
            return action

        except Exception as e:
            logger.warning(f"Prediction error: {e}")
            return np.array([0.0, 0.0, 0.0])  # Neutral action

    def log_shadow_execution(self, state: np.ndarray, action: np.ndarray):
        """Log shadow execution data to Redis."""
        try:
            # Create shadow execution payload
            payload = {
                "ts": time.time(),
                "action_timing": float(action[0]),
                "action_size": float(action[1]),
                "action_aggression": float(action[2]),
                "state_spread": float(state[2]) if len(state) > 2 else 0.0,
                "state_imbalance": float(state[14]) if len(state) > 14 else 0.0,
                "shadow_agent": "rl_exec_v1",
                "predictions_count": self.predictions_count,
            }

            # Store in Redis stream
            self.redis.xadd("shadow_exec", payload, maxlen=10000)

            # Update latest shadow state for monitoring
            self.redis.hset(
                "shadow:latest",
                mapping={
                    "last_action": json.dumps(action.tolist()),
                    "last_prediction_ts": int(time.time()),
                    "predictions_count": self.predictions_count,
                    "uptime_seconds": int(time.time() - self.start_time),
                },
            )

        except Exception as e:
            logger.warning(f"Failed to log shadow execution: {e}")

    def get_daemon_stats(self) -> dict:
        """Get daemon performance statistics."""
        uptime = time.time() - self.start_time
        predictions_per_sec = self.predictions_count / max(1, uptime)

        return {
            "service": "exec_shadow_daemon",
            "status": "active",
            "model_path": self.model_path,
            "model_loaded": self.model is not None,
            "predictions_count": self.predictions_count,
            "uptime_seconds": uptime,
            "predictions_per_second": predictions_per_sec,
            "update_interval": self.update_interval,
        }

    def run(self):
        """Main daemon loop."""
        logger.info("🚀 Starting RL execution shadow daemon")

        # Load model
        if not self.load_model():
            logger.error("Failed to load model, exiting")
            return False

        try:
            while True:
                loop_start = time.time()

                # Get current market state
                state = self.get_current_state()

                # Predict optimal execution action
                action = self.predict_action(state)

                # Log shadow execution for analysis
                self.log_shadow_execution(state, action)

                # Update counters
                self.predictions_count += 1

                # Log periodically
                if self.predictions_count % 200 == 0:
                    stats = self.get_daemon_stats()
                    logger.info(
                        f"Shadow predictions: {stats['predictions_count']:,} "
                        f"({stats['predictions_per_second']:.1f}/sec)"
                    )

                # Sleep until next update
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.update_interval - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("🛑 Shadow daemon stopped by user")
            return True
        except Exception as e:
            logger.error(f"❌ Fatal error in shadow daemon: {e}")
            return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="RL Execution Shadow Daemon")
    parser.add_argument(
        "--model-path", default="/models/exec_agent_v1", help="Path to trained RL model"
    )
    parser.add_argument(
        "--interval", type=float, default=0.05, help="Update interval in seconds"
    )
    parser.add_argument("--stats", action="store_true", help="Show daemon statistics")
    parser.add_argument("--test", action="store_true", help="Test model loading only")

    args = parser.parse_args()

    daemon = ExecShadowDaemon(model_path=args.model_path, update_interval=args.interval)

    if args.stats:
        # Show current statistics
        try:
            redis_client = redis.Redis(decode_responses=True)
            latest = redis_client.hgetall("shadow:latest")
            print("Shadow Daemon Statistics:")
            print(json.dumps(latest, indent=2))
        except Exception as e:
            print(f"Error getting stats: {e}")
        return

    if args.test:
        # Test model loading
        success = daemon.load_model()
        if success:
            print("✅ Model loaded successfully")

            # Test prediction
            test_state = np.random.randn(16).astype(np.float32)
            test_action = daemon.predict_action(test_state)
            print(f"Test prediction: {test_action}")
        else:
            print("❌ Model loading failed")
        return

    # Run daemon
    success = daemon.run()
    exit_code = 0 if success else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
