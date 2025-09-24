"""
Equity Market RL Environment

Market hours-aware RL environment for equities trading.
Episodes run during market hours only, with equity-specific rewards and observations.
"""

import numpy as np
import redis
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

from ..src.layers.layer5_risk.market_hours_guard import create_market_hours_guard
from ..src.utils.logger import get_logger


class EquityMarketEnv(gym.Env):
    """RL Environment for equity trading with market hours awareness"""

    def __init__(
        self,
        symbols: list = None,
        episode_length_minutes: int = 60,
        observation_window: int = 20,
        redis_url: str = "redis://localhost:6379",
    ):
        super().__init__()

        self.logger = get_logger(self.__class__.__name__)
        self.symbols = symbols or ["AAPL", "MSFT", "NVDA", "SPY"]
        self.episode_length_minutes = episode_length_minutes
        self.observation_window = observation_window

        # Market hours guard
        self.market_guard = create_market_hours_guard()

        # Redis for market data and risk flags
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)

        # Define action and observation spaces
        # Actions: [-1, 1] for each symbol (short to long position)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.symbols),), dtype=np.float32
        )

        # Observations: price features + risk flags + time features
        obs_size = (
            len(self.symbols) * 5  # price, volume, bid/ask spread, momentum, volatility
            + len(self.symbols) * 3  # SSR, halt, PDT flags per symbol
            + 4  # market time features: time_to_close, vol_bucket, event_window, auction_phase
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.episode_start_time = None
        self.positions = np.zeros(len(self.symbols))
        self.cash = 100000.0  # Starting cash
        self.last_portfolio_value = self.cash

        self.logger.info(
            f"Equity RL Environment initialized: {len(self.symbols)} symbols, {episode_length_minutes}min episodes"
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode"""
        # Only start episodes during market hours
        if self.market_guard.should_block_trading():
            self.logger.info("Market closed - delaying episode start")
            # Return dummy observation and wait
            return self._get_dummy_observation(), {"market_closed": True}

        self.current_step = 0
        self.episode_start_time = datetime.utcnow()
        self.positions = np.zeros(len(self.symbols))
        self.cash = 100000.0
        self.last_portfolio_value = self.cash

        observation = self._get_observation()
        info = {
            "episode_start": self.episode_start_time.isoformat(),
            "symbols": self.symbols,
            "market_status": self.market_guard.get_market_status(),
        }

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.current_step += 1

        # Check if market is still open
        if self.market_guard.should_block_trading():
            # Market closed during episode - terminate
            return self._get_observation(), 0.0, True, False, {"market_closed": True}

        # Execute action (update positions)
        old_positions = self.positions.copy()
        self.positions = np.clip(action, -1.0, 1.0)  # Normalize positions

        # Calculate reward
        reward = self._calculate_reward(old_positions, self.positions)

        # Check if episode should end
        episode_duration = datetime.utcnow() - self.episode_start_time
        terminated = episode_duration.total_seconds() > (
            self.episode_length_minutes * 60
        )
        truncated = False

        # End episode if market close approaching (last 5 minutes)
        market_status = self.market_guard.get_market_status()
        if market_status["is_closing_auction"]:
            terminated = True

        observation = self._get_observation()
        info = {
            "step": self.current_step,
            "positions": self.positions.tolist(),
            "portfolio_value": self._get_portfolio_value(),
            "market_status": market_status,
            "episode_duration_min": episode_duration.total_seconds() / 60,
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""
        obs_components = []

        # Price features for each symbol
        for symbol in self.symbols:
            price_features = self._get_price_features(symbol)
            obs_components.extend(price_features)

        # Risk flags for each symbol
        for symbol in self.symbols:
            risk_flags = self._get_risk_flags(symbol)
            obs_components.extend(risk_flags)

        # Market time features
        time_features = self._get_time_features()
        obs_components.extend(time_features)

        return np.array(obs_components, dtype=np.float32)

    def _get_price_features(self, symbol: str) -> list:
        """Get price-related features for a symbol"""
        # Mock implementation - would pull from market data feed
        return [
            175.50,  # current_price
            1000000,  # volume
            0.01,  # bid_ask_spread_pct
            0.02,  # momentum_5min
            0.25,  # volatility_estimate
        ]

    def _get_risk_flags(self, symbol: str) -> list:
        """Get risk flags for a symbol"""
        # Check Redis for risk flags
        ssr_flag = float(self.redis_client.get(f"risk:ssr:{symbol}") or 0)
        halt_flag = float(self.redis_client.get(f"risk:halted:{symbol}") or 0)
        pdt_flag = float(self.redis_client.get("risk:pdt_block") or 0)

        return [ssr_flag, halt_flag, pdt_flag]

    def _get_time_features(self) -> list:
        """Get market time-related features"""
        market_status = self.market_guard.get_market_status()

        # Time to close (normalized 0-1)
        if market_status["is_open"]:
            # Rough calculation - would be more precise with actual market hours
            time_to_close = 0.5  # Mock: halfway through trading day
        else:
            time_to_close = 0.0

        return [
            time_to_close,
            0.5,  # vol_bucket (0=low, 1=high)
            0.0,  # event_window (0=normal, 1=earnings/events)
            1.0 if market_status["is_opening_auction"] else 0.0,  # auction_phase
        ]

    def _get_dummy_observation(self) -> np.ndarray:
        """Return dummy observation when market is closed"""
        obs_size = self.observation_space.shape[0]
        return np.zeros(obs_size, dtype=np.float32)

    def _calculate_reward(
        self, old_positions: np.ndarray, new_positions: np.ndarray
    ) -> float:
        """Calculate reward for the action taken"""
        # Mock reward calculation
        # Real implementation would calculate:
        # - PnL from position changes
        # - Transaction costs
        # - Slippage costs
        # - Auction/overnight penalties
        # - Risk-adjusted returns

        position_change = np.abs(new_positions - old_positions).sum()
        base_reward = np.random.normal(0, 0.1)  # Mock market returns

        # Penalize excessive position changes (transaction costs)
        transaction_penalty = position_change * 0.001

        # Penalize trading near market close (auction risk)
        market_status = self.market_guard.get_market_status()
        auction_penalty = 0.01 if market_status["is_closing_auction"] else 0.0

        return base_reward - transaction_penalty - auction_penalty

    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        # Mock calculation - would use real prices and positions
        position_value = np.abs(self.positions).sum() * 1000  # Mock value per position
        return self.cash + position_value

    def render(self, mode: str = "human") -> None:
        """Render environment state"""
        if mode == "human":
            market_status = self.market_guard.get_market_status()
            portfolio_value = self._get_portfolio_value()

            print(f"Step: {self.current_step}")
            print(f"Market Status: {'OPEN' if market_status['is_open'] else 'CLOSED'}")
            print(f"Positions: {self.positions}")
            print(f"Portfolio Value: ${portfolio_value:,.2f}")


def create_equity_market_env(**kwargs) -> EquityMarketEnv:
    """Factory function to create EquityMarketEnv instance."""
    return EquityMarketEnv(**kwargs)


if __name__ == "__main__":
    # Test the environment
    env = create_equity_market_env()
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Info: {info}")

    # Take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}, Terminated: {terminated}")
    print(f"Info: {info}")
