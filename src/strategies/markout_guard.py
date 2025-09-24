#!/usr/bin/env python3
"""
Short-Horizon Markout Guard

Monitors post-trade price impact (markout) at short horizons (5s, 30s) and
dynamically adjusts entry thresholds when performance deteriorates.
"""

import os
import sys
import time
import asyncio
import logging
import json
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.utils.aredis import (
        get_redis,
        get_batch_writer,
        set_metric,
        get_config_value,
    )

    ASYNC_REDIS_AVAILABLE = True
except ImportError:
    ASYNC_REDIS_AVAILABLE = False
    import redis

logger = logging.getLogger("markout_guard")


class MarkoutHorizon(Enum):
    """Markout measurement horizons."""

    MO_5S = "5s"
    MO_30S = "30s"
    MO_1M = "1m"
    MO_5M = "5m"


@dataclass
class FillEvent:
    """Represents a trade fill for markout analysis."""

    fill_id: str
    symbol: str
    side: str  # "buy" or "sell"
    price: float
    quantity: float
    timestamp: float
    strategy: str
    venue: str


@dataclass
class MarkoutMeasurement:
    """Markout measurement result."""

    fill_id: str
    symbol: str
    horizon: MarkoutHorizon
    markout_bps: float
    ref_price: float
    measurement_time: float
    valid: bool


class MarkoutCalculator:
    """
    Calculates markout (price impact) at various horizons.

    Markout = (CurrentPrice - FillPrice) / FillPrice * 10000 * sign
    where sign = +1 for buys, -1 for sells (positive markout = adverse impact)
    """

    def __init__(self, horizons: List[MarkoutHorizon] = None):
        """
        Initialize markout calculator.

        Args:
            horizons: List of horizons to measure (default: [5s, 30s])
        """
        if horizons is None:
            horizons = [MarkoutHorizon.MO_5S, MarkoutHorizon.MO_30S]

        self.horizons = horizons
        self.horizon_seconds = {
            MarkoutHorizon.MO_5S: 5,
            MarkoutHorizon.MO_30S: 30,
            MarkoutHorizon.MO_1M: 60,
            MarkoutHorizon.MO_5M: 300,
        }

        # Track pending fills for markout measurement
        self.pending_fills: Dict[str, FillEvent] = {}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.markout_results: Dict[str, List[MarkoutMeasurement]] = defaultdict(list)

        logger.info(
            f"Initialized markout calculator for horizons: {[h.value for h in horizons]}"
        )

    def add_fill(self, fill_event: FillEvent):
        """Add a fill for markout tracking."""
        try:
            # Store fill for later markout measurement
            self.pending_fills[fill_event.fill_id] = fill_event

            logger.debug(
                f"Added fill for markout tracking: {fill_event.fill_id} "
                f"{fill_event.symbol} {fill_event.side} {fill_event.quantity} @ {fill_event.price}"
            )

        except Exception as e:
            logger.error(f"Error adding fill: {e}")

    def add_price_tick(self, symbol: str, price: float, timestamp: float):
        """Add price tick for markout reference."""
        try:
            self.price_history[symbol].append((timestamp, price))

            # Check if any pending fills can be measured
            self._check_pending_markouts(symbol, price, timestamp)

        except Exception as e:
            logger.error(f"Error adding price tick: {e}")

    def _check_pending_markouts(
        self, symbol: str, current_price: float, current_time: float
    ):
        """Check if any pending fills can have markout measured."""
        try:
            fills_to_remove = []

            for fill_id, fill_event in self.pending_fills.items():
                if fill_event.symbol != symbol:
                    continue

                # Check each horizon
                for horizon in self.horizons:
                    horizon_seconds = self.horizon_seconds[horizon]
                    time_elapsed = current_time - fill_event.timestamp

                    # If we've reached the measurement horizon
                    if time_elapsed >= horizon_seconds:
                        markout = self._calculate_markout(
                            fill_event, current_price, current_time, horizon
                        )

                        if markout:
                            self.markout_results[symbol].append(markout)

                            # Keep only recent results
                            if len(self.markout_results[symbol]) > 10000:
                                self.markout_results[symbol] = self.markout_results[
                                    symbol
                                ][-5000:]

                # Remove fills that are too old (beyond longest horizon)
                max_horizon = max(self.horizon_seconds.values())
                if current_time - fill_event.timestamp > max_horizon * 2:
                    fills_to_remove.append(fill_id)

            # Clean up old fills
            for fill_id in fills_to_remove:
                del self.pending_fills[fill_id]

        except Exception as e:
            logger.error(f"Error checking pending markouts: {e}")

    def _calculate_markout(
        self,
        fill_event: FillEvent,
        ref_price: float,
        ref_time: float,
        horizon: MarkoutHorizon,
    ) -> Optional[MarkoutMeasurement]:
        """Calculate markout for a specific fill and horizon."""
        try:
            # Sign convention: positive markout = adverse impact
            side_multiplier = 1.0 if fill_event.side == "buy" else -1.0

            # Markout in basis points
            price_diff = ref_price - fill_event.price
            markout_bps = (price_diff / fill_event.price) * 10000 * side_multiplier

            measurement = MarkoutMeasurement(
                fill_id=fill_event.fill_id,
                symbol=fill_event.symbol,
                horizon=horizon,
                markout_bps=markout_bps,
                ref_price=ref_price,
                measurement_time=ref_time,
                valid=True,
            )

            logger.debug(
                f"Measured markout: {fill_event.fill_id} {horizon.value} = {markout_bps:.2f}bps"
            )

            return measurement

        except Exception as e:
            logger.error(f"Error calculating markout: {e}")
            return None

    def get_recent_markouts(
        self, symbol: str, horizon: MarkoutHorizon, lookback_minutes: int = 60
    ) -> List[MarkoutMeasurement]:
        """Get recent markout measurements for analysis."""
        try:
            cutoff_time = time.time() - (lookback_minutes * 60)

            recent_markouts = [
                m
                for m in self.markout_results[symbol]
                if m.horizon == horizon and m.measurement_time >= cutoff_time
            ]

            return recent_markouts

        except Exception as e:
            logger.error(f"Error getting recent markouts: {e}")
            return []

    def get_markout_statistics(
        self, symbol: str, horizon: MarkoutHorizon, lookback_minutes: int = 60
    ) -> Dict[str, float]:
        """Get markout statistics for a symbol and horizon."""
        try:
            markouts = self.get_recent_markouts(symbol, horizon, lookback_minutes)

            if not markouts:
                return {"count": 0, "mean_bps": 0.0, "median_bps": 0.0, "std_bps": 0.0}

            markout_values = [m.markout_bps for m in markouts]

            stats = {
                "count": len(markout_values),
                "mean_bps": np.mean(markout_values),
                "median_bps": np.median(markout_values),
                "std_bps": np.std(markout_values),
                "p25_bps": np.percentile(markout_values, 25),
                "p75_bps": np.percentile(markout_values, 75),
                "p95_bps": np.percentile(markout_values, 95),
                "worst_bps": np.max(markout_values) if markout_values else 0.0,
                "best_bps": np.min(markout_values) if markout_values else 0.0,
            }

            return stats

        except Exception as e:
            logger.error(f"Error calculating markout statistics: {e}")
            return {"count": 0, "mean_bps": 0.0, "median_bps": 0.0, "std_bps": 0.0}


class MarkoutGuard:
    """
    Markout-based guard that adjusts strategy parameters when performance deteriorates.

    Monitors short-horizon markouts and boosts entry thresholds when fills
    consistently show adverse price impact.
    """

    def __init__(self, symbols: List[str] = None):
        """
        Initialize markout guard.

        Args:
            symbols: List of symbols to monitor (default: ["BTC", "ETH", "SOL"])
        """
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL"]

        self.symbols = symbols
        self.calculator = MarkoutCalculator()

        # Guard configuration
        self.config = {
            "mo5s_threshold_bps": -1.0,  # Threshold for 5s markout (negative = adverse)
            "mo30s_threshold_bps": -1.5,  # Threshold for 30s markout
            "min_samples": 5,  # Minimum fills to trigger guard
            "lookback_minutes": 30,  # Lookback window for analysis
            "boost_bps": 3.0,  # Threshold boost when guard triggers
            "cool_down_minutes": 10,  # Cooldown before guard can trigger again
            "decay_minutes": 60,  # Time for boost to decay
            "max_boost_bps": 10.0,  # Maximum cumulative boost
        }

        # Guard state
        self.active_boosts: Dict[str, Dict[str, Any]] = {}  # symbol -> boost_info
        self.last_trigger: Dict[str, float] = {}  # symbol -> last_trigger_time
        self.guard_events: List[Dict[str, Any]] = []

        logger.info(f"Initialized markout guard for symbols: {symbols}")
        logger.info(
            f"  Thresholds: 5s<{self.config['mo5s_threshold_bps']}bps, "
            f"30s<{self.config['mo30s_threshold_bps']}bps"
        )
        logger.info(
            f"  Boost: {self.config['boost_bps']}bps for {self.config['decay_minutes']}min"
        )

    async def process_fill_event(self, fill_data: Dict[str, Any]):
        """Process a fill event for markout tracking."""
        try:
            fill_event = FillEvent(
                fill_id=fill_data.get("fill_id", ""),
                symbol=fill_data.get("symbol", ""),
                side=fill_data.get("side", ""),
                price=float(fill_data.get("price", 0)),
                quantity=float(fill_data.get("quantity", 0)),
                timestamp=float(fill_data.get("timestamp", time.time())),
                strategy=fill_data.get("strategy", ""),
                venue=fill_data.get("venue", ""),
            )

            # Only track fills from strategies we care about
            if fill_event.symbol in self.symbols:
                self.calculator.add_fill(fill_event)
                logger.debug(f"Tracking fill for markout: {fill_event.fill_id}")

        except Exception as e:
            logger.error(f"Error processing fill event: {e}")

    async def process_price_tick(
        self, symbol: str, price: float, timestamp: float = None
    ):
        """Process price tick and check for guard triggers."""
        try:
            if timestamp is None:
                timestamp = time.time()

            if symbol in self.symbols:
                # Update calculator
                self.calculator.add_price_tick(symbol, price, timestamp)

                # Check if guard should trigger
                await self._check_guard_trigger(symbol, timestamp)

                # Update boost decay
                await self._update_boost_decay(symbol, timestamp)

        except Exception as e:
            logger.error(f"Error processing price tick: {e}")

    async def _check_guard_trigger(self, symbol: str, current_time: float):
        """Check if markout guard should trigger for a symbol."""
        try:
            # Check cooldown
            last_trigger = self.last_trigger.get(symbol, 0)
            if current_time - last_trigger < self.config["cool_down_minutes"] * 60:
                return

            # Get recent markout statistics
            mo5s_stats = self.calculator.get_markout_statistics(
                symbol, MarkoutHorizon.MO_5S, self.config["lookback_minutes"]
            )
            mo30s_stats = self.calculator.get_markout_statistics(
                symbol, MarkoutHorizon.MO_30S, self.config["lookback_minutes"]
            )

            # Check if we have enough samples
            if (
                mo5s_stats["count"] < self.config["min_samples"]
                or mo30s_stats["count"] < self.config["min_samples"]
            ):
                return

            # Check if both horizons show adverse markout
            mo5s_bad = mo5s_stats["median_bps"] > self.config["mo5s_threshold_bps"]
            mo30s_bad = mo30s_stats["median_bps"] > self.config["mo30s_threshold_bps"]

            if mo5s_bad and mo30s_bad:
                await self._trigger_guard(symbol, current_time, mo5s_stats, mo30s_stats)

        except Exception as e:
            logger.error(f"Error checking guard trigger: {e}")

    async def _trigger_guard(
        self,
        symbol: str,
        trigger_time: float,
        mo5s_stats: Dict[str, float],
        mo30s_stats: Dict[str, float],
    ):
        """Trigger markout guard for a symbol."""
        try:
            # Calculate boost amount
            current_boost = self.get_current_threshold_boost(symbol)
            new_boost = min(
                current_boost + self.config["boost_bps"], self.config["max_boost_bps"]
            )

            # Update boost state
            boost_info = {
                "boost_bps": new_boost,
                "trigger_time": trigger_time,
                "expire_time": trigger_time + (self.config["decay_minutes"] * 60),
                "mo5s_median": mo5s_stats["median_bps"],
                "mo30s_median": mo30s_stats["median_bps"],
                "trigger_count": self.active_boosts.get(symbol, {}).get(
                    "trigger_count", 0
                )
                + 1,
            }

            self.active_boosts[symbol] = boost_info
            self.last_trigger[symbol] = trigger_time

            # Log guard event
            event = {
                "symbol": symbol,
                "action": "trigger",
                "boost_bps": new_boost,
                "trigger_time": trigger_time,
                "mo5s_stats": mo5s_stats,
                "mo30s_stats": mo30s_stats,
                "trigger_count": boost_info["trigger_count"],
            }
            self.guard_events.append(event)

            # Publish boost to Redis with TTL
            if ASYNC_REDIS_AVAILABLE:
                redis = await get_redis()
                boost_key = f"basis:guard:thresh_boost:{symbol}"
                ttl_seconds = int(self.config["decay_minutes"] * 60)

                if hasattr(redis, "setex"):
                    await redis.setex(boost_key, ttl_seconds, new_boost)
                else:
                    redis.setex(boost_key, ttl_seconds, new_boost)

                # Publish metrics
                await set_metric(f"markout_guard_boost_{symbol.lower()}", new_boost)
                await set_metric(
                    f"markout_guard_triggers_{symbol.lower()}",
                    boost_info["trigger_count"],
                )

            logger.warning(
                f"🛡️ Markout guard triggered for {symbol}: "
                f"boost +{new_boost:.1f}bps (5s: {mo5s_stats['median_bps']:.1f}bps, "
                f"30s: {mo30s_stats['median_bps']:.1f}bps, samples: {mo5s_stats['count']})"
            )

        except Exception as e:
            logger.error(f"Error triggering guard: {e}")

    async def _update_boost_decay(self, symbol: str, current_time: float):
        """Update boost decay for a symbol."""
        try:
            if symbol not in self.active_boosts:
                return

            boost_info = self.active_boosts[symbol]

            # Check if boost has expired
            if current_time >= boost_info["expire_time"]:
                del self.active_boosts[symbol]

                # Clear Redis key
                if ASYNC_REDIS_AVAILABLE:
                    redis = await get_redis()
                    boost_key = f"basis:guard:thresh_boost:{symbol}"
                    if hasattr(redis, "delete"):
                        await redis.delete(boost_key)
                    else:
                        redis.delete(boost_key)

                    await set_metric(f"markout_guard_boost_{symbol.lower()}", 0.0)

                logger.info(f"Markout guard boost expired for {symbol}")

                # Log expiration event
                event = {
                    "symbol": symbol,
                    "action": "expire",
                    "boost_bps": 0.0,
                    "expire_time": current_time,
                }
                self.guard_events.append(event)

        except Exception as e:
            logger.error(f"Error updating boost decay: {e}")

    def get_current_threshold_boost(self, symbol: str) -> float:
        """Get current threshold boost for a symbol."""
        try:
            if symbol not in self.active_boosts:
                return 0.0

            boost_info = self.active_boosts[symbol]
            current_time = time.time()

            # Check if expired
            if current_time >= boost_info["expire_time"]:
                return 0.0

            return boost_info["boost_bps"]

        except Exception as e:
            logger.error(f"Error getting threshold boost: {e}")
            return 0.0

    def get_guard_status(self) -> Dict[str, Any]:
        """Get comprehensive guard status."""
        try:
            current_time = time.time()

            status = {
                "symbols": self.symbols,
                "config": self.config.copy(),
                "active_boosts": {},
                "recent_events": self.guard_events[-10:],
                "statistics": {},
            }

            # Add current boost status
            for symbol in self.symbols:
                boost = self.get_current_threshold_boost(symbol)
                status["active_boosts"][symbol] = {
                    "boost_bps": boost,
                    "active": boost > 0,
                    "info": self.active_boosts.get(symbol, {}),
                }

                # Add markout statistics
                mo5s_stats = self.calculator.get_markout_statistics(
                    symbol, MarkoutHorizon.MO_5S, self.config["lookback_minutes"]
                )
                mo30s_stats = self.calculator.get_markout_statistics(
                    symbol, MarkoutHorizon.MO_30S, self.config["lookback_minutes"]
                )

                status["statistics"][symbol] = {
                    "mo5s": mo5s_stats,
                    "mo30s": mo30s_stats,
                }

            return status

        except Exception as e:
            logger.error(f"Error getting guard status: {e}")
            return {"error": str(e)}


async def main():
    """Test the markout guard."""
    import argparse

    parser = argparse.ArgumentParser(description="Markout Guard")
    parser.add_argument("--symbol", default="BTC", help="Symbol to monitor")
    parser.add_argument(
        "--test", action="store_true", help="Run test with synthetic data"
    )
    parser.add_argument("--monitor", action="store_true", help="Start monitoring mode")

    args = parser.parse_args()

    if args.test:
        # Test with synthetic fills and prices
        guard = MarkoutGuard([args.symbol])

        base_price = 50000
        fill_count = 0

        for i in range(100):
            current_time = time.time() + i

            # Add some price movement
            price_noise = np.random.normal(0, 50)
            current_price = base_price + price_noise

            # Occasionally add fills with adverse markout
            if i % 20 == 0:
                fill_count += 1
                fill_data = {
                    "fill_id": f"test_fill_{fill_count}",
                    "symbol": args.symbol,
                    "side": "buy",
                    "price": current_price + 10,  # Adverse fill (buy high)
                    "quantity": 1.0,
                    "timestamp": current_time,
                    "strategy": "basis_carry",
                    "venue": "test",
                }
                await guard.process_fill_event(fill_data)

            # Process price tick
            await guard.process_price_tick(args.symbol, current_price, current_time)

            # Print status every 30 ticks
            if i % 30 == 0:
                status = guard.get_guard_status()
                boost = status["active_boosts"].get(args.symbol, {}).get("boost_bps", 0)
                stats = status["statistics"].get(args.symbol, {})
                mo5s = stats.get("mo5s", {})

                print(
                    f"Tick {i}: price=${current_price:.0f}, boost={boost:.1f}bps, "
                    f"5s_markout={mo5s.get('median_bps', 0):.1f}bps "
                    f"(samples={mo5s.get('count', 0)})"
                )

        # Final status
        final_status = guard.get_guard_status()
        print(f"\nFinal guard status:")
        print(json.dumps(final_status, indent=2, default=str))

    elif args.monitor:
        # Start monitoring (would integrate with live data feed)
        guard = MarkoutGuard([args.symbol])
        logger.info(f"Starting markout guard monitoring for {args.symbol}...")

        try:
            while True:
                # In real implementation, this would process live market data
                # and fill events from Redis streams
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Markout guard monitoring stopped")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
