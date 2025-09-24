#!/usr/bin/env python3
"""
Order Book Environment for RL Execution Agent
Custom Gymnasium environment for training execution strategies
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import redis
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger("orderbook_env")


class OrderBookEnv(gym.Env):
    """Order Book Environment for RL execution agent training."""

    def __init__(self, parquet_dir: str = "/data/binance_ticks", mode: str = "train"):
        super().__init__()

        self.parquet_dir = parquet_dir
        self.mode = mode

        # State space: [bid_levels(5), ask_levels(5), spread, mid_price_change, volume_imbalance, time_features(3)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )

        # Action space: [timing(0-1), size_fraction(0-1), aggression(0-1)]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = 1000
        self.target_quantity = 1000.0
        self.executed_quantity = 0.0

        logger.info(f"📊 OrderBookEnv initialized (mode: {mode})")
        self._generate_mock_data()

    def _generate_mock_data(self):
        """Generate mock order book data."""
        np.random.seed(42)
        n_ticks = 10000

        base_price = 50000.0
        price_changes = np.random.normal(0, 0.001, n_ticks)
        mid_prices = base_price * np.exp(np.cumsum(price_changes))

        self.tick_data = []
        for i in range(n_ticks):
            mid_price = mid_prices[i]
            spread = mid_price * np.random.uniform(2, 10) / 10000

            tick = {
                "mid_price": mid_price,
                "spread": spread,
                "volume_imbalance": np.random.uniform(-0.5, 0.5),
            }
            self.tick_data.append(tick)

    def _get_observation(self) -> np.ndarray:
        """Get current environment observation."""
        if self.current_step >= len(self.tick_data):
            return np.zeros(16, dtype=np.float32)

        tick = self.tick_data[self.current_step]
        mid_price = tick["mid_price"]
        spread = tick["spread"]

        # Mock bid/ask levels
        bid_levels = [
            (mid_price - spread / 2 - i * spread * 0.1, 1000 * (1 - i * 0.1))
            for i in range(5)
        ]
        ask_levels = [
            (mid_price + spread / 2 + i * spread * 0.1, 1000 * (1 - i * 0.1))
            for i in range(5)
        ]

        # Price change
        price_change = 0.0
        if self.current_step > 0:
            prev_price = self.tick_data[self.current_step - 1]["mid_price"]
            price_change = (mid_price - prev_price) / prev_price

        # Time features
        progress = self.current_step / self.max_steps
        remaining_qty = max(
            0, (self.target_quantity - self.executed_quantity) / self.target_quantity
        )
        exec_rate = self.executed_quantity / max(1, self.current_step)

        # Normalize bid/ask levels
        bid_norm = [(bp - mid_price) / mid_price for bp, _ in bid_levels]
        ask_norm = [(ap - mid_price) / mid_price for ap, _ in ask_levels]

        observation = np.array(
            [
                *bid_norm,
                *ask_norm,
                spread / mid_price,
                price_change,
                tick["volume_imbalance"],
                progress,
                remaining_qty,
                exec_rate,
            ],
            dtype=np.float32,
        )

        return observation

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.executed_quantity = 0.0
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        timing, size_fraction, aggression = action

        reward = 0.0
        remaining_qty = self.target_quantity - self.executed_quantity

        if timing > 0.5 and remaining_qty > 0:
            exec_qty = min(remaining_qty, size_fraction * remaining_qty)
            if exec_qty > 0:
                tick = self.tick_data[self.current_step]
                mid_price = tick["mid_price"]
                spread = tick["spread"]

                # Calculate execution cost
                execution_price = mid_price + spread / 2 + aggression * spread
                slippage = spread / mid_price

                reward = -slippage * 10000 - self.current_step * 0.01
                if self.executed_quantity + exec_qty >= self.target_quantity:
                    reward += 10.0

                self.executed_quantity += exec_qty

        self.current_step += 1

        terminated = (
            self.current_step >= self.max_steps
            or self.executed_quantity >= self.target_quantity
        )

        observation = self._get_observation()
        info = {"executed_quantity": self.executed_quantity}

        return observation, reward, terminated, False, info


class LiveExecEnv:
    """Live execution environment for shadow mode."""

    def __init__(self):
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        logger.info("📈 LiveExecEnv initialized")

    def get_state(self) -> np.ndarray:
        """Get current market state."""
        # Mock state for demo
        return np.random.randn(16).astype(np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    env = OrderBookEnv(mode="test")
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, done={done}")
        if done:
            break
