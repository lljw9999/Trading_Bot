#!/usr/bin/env python3
"""
RL Execution Agent Training Script
Train PPO agent for optimal execution using order book environment
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from envs.orderbook_env import OrderBookEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("exec_agent")


class ExecutionAgentTrainer:
    """Trainer for RL execution agent."""

    def __init__(
        self,
        parquet_dir: str = "/data/binance_ticks",
        model_save_path: str = "/models/exec_agent_v1",
        n_envs: int = 4,
        total_timesteps: int = 4_000_000,
    ):
        """
        Initialize execution agent trainer.

        Args:
            parquet_dir: Directory with historical tick data
            model_save_path: Path to save trained model
            n_envs: Number of parallel environments
            total_timesteps: Total training timesteps
        """
        self.parquet_dir = parquet_dir
        self.model_save_path = model_save_path
        self.n_envs = n_envs
        self.total_timesteps = total_timesteps

        # Create models directory
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        logger.info(f"🤖 ExecutionAgentTrainer initialized")
        logger.info(f"   Data: {parquet_dir}")
        logger.info(f"   Save path: {model_save_path}")
        logger.info(f"   Environments: {n_envs}")
        logger.info(f"   Total timesteps: {total_timesteps:,}")

    def create_env(self):
        """Create training environment."""

        def _init():
            env = OrderBookEnv(parquet_dir=self.parquet_dir, mode="train")
            env = Monitor(env)  # Monitor for logging
            return env

        return _init

    def create_training_env(self):
        """Create vectorized training environment."""
        if self.n_envs == 1:
            env = DummyVecEnv([self.create_env()])
        else:
            # Use SubprocVecEnv for true parallelism (slower startup but faster training)
            env = SubprocVecEnv([self.create_env() for _ in range(self.n_envs)])

        logger.info(
            f"✅ Created vectorized environment with {self.n_envs} parallel envs"
        )
        return env

    def create_model(self, env):
        """Create PPO model with optimized hyperparameters."""
        model = PPO(
            policy="MlpPolicy",
            env=env,
            batch_size=65536,  # Large batch size as specified
            n_steps=4096,  # Steps per environment per update
            learning_rate=3e-4,  # Standard learning rate
            n_epochs=10,  # Number of epochs per update
            gamma=0.99,  # Discount factor
            gae_lambda=0.95,  # GAE lambda
            clip_range=0.2,  # PPO clip range
            ent_coef=0.01,  # Entropy coefficient
            vf_coef=0.5,  # Value function coefficient
            max_grad_norm=0.5,  # Gradient clipping
            verbose=1,  # Logging verbosity
            tensorboard_log="./logs/exec_agent_tensorboard/",
            device="auto",  # Use GPU if available
        )

        logger.info("🧠 PPO model created with optimized hyperparameters")
        return model

    def setup_callbacks(self, eval_env=None):
        """Set up training callbacks."""
        callbacks = []

        # Checkpoint callback - save model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,  # Save every 50k steps
            save_path="./models/checkpoints/",
            name_prefix="exec_agent",
        )
        callbacks.append(checkpoint_callback)

        # Evaluation callback (if eval env provided)
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path="./models/best/",
                log_path="./logs/eval/",
                eval_freq=25000,  # Evaluate every 25k steps
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)

        return callbacks

    def train(self):
        """Train the execution agent."""
        logger.info("🚀 Starting RL execution agent training")

        # Create training environment
        train_env = self.create_training_env()

        # Create evaluation environment (single env)
        eval_env = DummyVecEnv(
            [lambda: Monitor(OrderBookEnv(parquet_dir=self.parquet_dir, mode="eval"))]
        )

        # Create model
        model = self.create_model(train_env)

        # Set up callbacks
        callbacks = self.setup_callbacks(eval_env)

        # Log training configuration
        logger.info("📋 Training Configuration:")
        logger.info(f"   Total timesteps: {self.total_timesteps:,}")
        logger.info(f"   Batch size: {model.batch_size}")
        logger.info(f"   N steps: {model.n_steps}")
        logger.info(f"   Learning rate: {model.learning_rate}")
        logger.info(f"   Environments: {self.n_envs}")

        # Start training
        start_time = time.time()

        try:
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                progress_bar=True,
            )

            training_time = time.time() - start_time
            logger.info(f"✅ Training completed in {training_time/3600:.1f} hours")

            # Save final model
            model.save(self.model_save_path)
            logger.info(f"💾 Model saved to {self.model_save_path}")

            # Clean up environments
            train_env.close()
            eval_env.close()

            return model

        except KeyboardInterrupt:
            logger.info("🛑 Training interrupted by user")
            # Save intermediate model
            model.save(self.model_save_path + "_interrupted")
            logger.info(
                f"💾 Intermediate model saved to {self.model_save_path}_interrupted"
            )

            train_env.close()
            eval_env.close()
            raise

    def test_trained_model(self, model_path: str = None, n_episodes: int = 10):
        """Test the trained model."""
        model_path = model_path or self.model_save_path

        if not os.path.exists(model_path + ".zip"):
            logger.error(f"Model not found at {model_path}")
            return

        logger.info(f"🧪 Testing trained model: {model_path}")

        # Load model
        model = PPO.load(model_path)

        # Create test environment
        test_env = OrderBookEnv(parquet_dir=self.parquet_dir, mode="eval")

        total_rewards = []
        total_slippages = []

        for episode in range(n_episodes):
            obs, _ = test_env.reset()
            episode_reward = 0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            completion_rate = info.get("completion_rate", 0)
            executed_qty = info.get("executed_quantity", 0)

            total_rewards.append(episode_reward)
            logger.info(
                f"Episode {episode + 1}: reward={episode_reward:.2f}, "
                f"completion={completion_rate:.1%}, executed={executed_qty:.0f}"
            )

        # Summary statistics
        avg_reward = sum(total_rewards) / len(total_rewards)
        logger.info(f"📊 Test Results ({n_episodes} episodes):")
        logger.info(f"   Average reward: {avg_reward:.2f}")
        logger.info(f"   Reward std: {np.std(total_rewards):.2f}")

        return {
            "average_reward": avg_reward,
            "rewards": total_rewards,
            "n_episodes": n_episodes,
        }


def main():
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train RL Execution Agent")
    parser.add_argument(
        "--data-dir",
        default="/data/binance_ticks",
        help="Directory with parquet tick data",
    )
    parser.add_argument(
        "--save-path",
        default="/models/exec_agent_v1",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--timesteps", type=int, default=4_000_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--n-envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Only test existing model"
    )
    parser.add_argument(
        "--quick-test", action="store_true", help="Quick test with reduced timesteps"
    )

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        args.timesteps = 50000
        args.n_envs = 2
        logger.info("🏃 Quick test mode enabled")

    # Create trainer
    trainer = ExecutionAgentTrainer(
        parquet_dir=args.data_dir,
        model_save_path=args.save_path,
        n_envs=args.n_envs,
        total_timesteps=args.timesteps,
    )

    if args.test_only:
        # Test existing model
        trainer.test_trained_model()
    else:
        # Train new model
        model = trainer.train()

        # Test trained model
        logger.info("🧪 Testing trained model...")
        trainer.test_trained_model()


if __name__ == "__main__":
    # Import numpy here to avoid issues with multiprocessing
    import numpy as np

    main()
