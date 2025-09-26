"""
Feature Bus - Real-time Feature Computation Engine

Consumes raw market data and computes technical indicators and features
for downstream alpha models.
"""

import asyncio
import json
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Any, Optional, Deque
import numpy as np
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import orjson

from .schemas import MarketTick, FeatureSnapshot
from ...utils.logger import get_logger
from ...utils.metrics import get_metrics


class FeatureBus:
    """
    Real-time feature computation engine.

    Processes market ticks and computes technical indicators and features
    for alpha models.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize the feature bus.

        Args:
            max_history: Maximum number of ticks to keep in memory
        """
        self.max_history = max_history
        self.logger = get_logger("feature_bus")

        # Historical data storage per symbol
        self.price_history: Dict[str, Deque[tuple[datetime, Decimal]]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self.volume_history: Dict[str, Deque[tuple[datetime, Decimal]]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self.tick_history: Dict[str, Deque[MarketTick]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

        # Soft information storage (sentiment + fundamentals)
        self.sentiment_history: Dict[str, Deque[tuple[datetime, float]]] = defaultdict(
            lambda: deque(maxlen=100)  # Keep 100 sentiment records per symbol
        )
        self.fundamental_history: Dict[str, Deque[tuple[datetime, float]]] = (
            defaultdict(
                lambda: deque(maxlen=50)  # Keep 50 fundamental records per symbol
            )
        )

        # Feature computation statistics
        self.features_computed = 0
        self.computation_times = deque(maxlen=100)  # Last 100 computation times

        # Soft information lookup tolerance (±90 seconds)
        self.sentiment_tolerance = timedelta(seconds=90)

        self.logger.info(f"Feature Bus initialized with max_history={max_history}")

    async def update_sentiment(
        self, symbol: str, sentiment_score: float, timestamp: datetime = None
    ):
        """Update sentiment score for a symbol."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        self.sentiment_history[symbol].append((timestamp, sentiment_score))
        self.logger.debug(f"Updated sentiment for {symbol}: {sentiment_score:.3f}")

    async def update_fundamental(
        self, symbol: str, pe_ratio: float, timestamp: datetime = None
    ):
        """Update fundamental P/E ratio for a symbol."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        self.fundamental_history[symbol].append((timestamp, pe_ratio))
        self.logger.debug(f"Updated P/E for {symbol}: {pe_ratio:.2f}")

    def _get_latest_sentiment(
        self, symbol: str, reference_time: datetime
    ) -> Optional[float]:
        """Get latest sentiment score within tolerance window."""
        sentiment_hist = self.sentiment_history[symbol]
        if not sentiment_hist:
            return None

        # Look for sentiment within ±90 seconds
        for timestamp, sentiment in reversed(sentiment_hist):
            time_diff = abs((reference_time - timestamp).total_seconds())
            if time_diff <= self.sentiment_tolerance.total_seconds():
                return sentiment

        return None  # No sentiment within tolerance

    def _get_latest_fundamental(
        self, symbol: str, reference_time: datetime
    ) -> Optional[float]:
        """Get latest fundamental P/E ratio within tolerance window."""
        fund_hist = self.fundamental_history[symbol]
        if not fund_hist:
            return None

        # Look for fundamental data within ±90 seconds (fundamentals change slowly)
        for timestamp, pe_ratio in reversed(fund_hist):
            time_diff = abs((reference_time - timestamp).total_seconds())
            if time_diff <= self.sentiment_tolerance.total_seconds():
                return pe_ratio

        return None

    async def process_tick(self, tick: MarketTick) -> Optional[FeatureSnapshot]:
        """
        Process a market tick and compute features.

        Args:
            tick: Market tick to process

        Returns:
            FeatureSnapshot if successfully computed, None otherwise
        """
        start_time = time.perf_counter()

        try:
            # Store tick in history
            self._store_tick(tick)

            # Compute features
            features = await self._compute_features(tick)

            # Track computation time
            computation_time = (
                time.perf_counter() - start_time
            ) * 1_000_000  # microseconds
            self.computation_times.append(computation_time)
            self.features_computed += 1

            # Log performance warning if too slow
            if computation_time > 300:  # 300 microseconds threshold
                self.logger.warning(
                    f"Feature computation took {computation_time:.1f}µs for {tick.symbol} "
                    f"(threshold: 300µs)"
                )

            return features

        except Exception as e:
            self.logger.error(f"Error processing tick for {tick.symbol}: {e}")
            return None

    def _store_tick(self, tick: MarketTick) -> None:
        """Store tick data in historical buffers."""
        symbol = tick.symbol

        # Store price data
        price_value = None
        if getattr(tick, "mid", None):
            price_value = Decimal(str(tick.mid))
        elif getattr(tick, "last", None):
            price_value = Decimal(str(tick.last))

        if price_value is not None:
            self.price_history[symbol].append((tick.timestamp, price_value))

        # Store volume data
        if getattr(tick, "volume", None):
            volume_value = Decimal(str(tick.volume))
            self.volume_history[symbol].append((tick.timestamp, volume_value))

        # Store complete tick
        self.tick_history[symbol].append(tick)

    async def _compute_features(self, tick: MarketTick) -> FeatureSnapshot:
        """Compute technical features for the given tick."""
        symbol = tick.symbol
        timestamp = tick.timestamp

        # Basic price features
        mid_price = tick.mid
        spread = tick.spread
        spread_bps = tick.spread_bps

        # Time-based returns
        return_1m = self._compute_return(symbol, timedelta(minutes=1))
        return_5m = self._compute_return(symbol, timedelta(minutes=5))
        return_15m = self._compute_return(symbol, timedelta(minutes=15))

        # Volatility features
        volatility_5m = self._compute_volatility(symbol, timedelta(minutes=5))
        volatility_15m = self._compute_volatility(symbol, timedelta(minutes=15))
        volatility_1h = self._compute_volatility(symbol, timedelta(hours=1))

        # Order book features
        order_book_imbalance = self._compute_order_book_imbalance(tick)
        order_book_pressure = self._compute_order_book_pressure(tick)

        # Volume features
        volume_1m = self._compute_volume(symbol, timedelta(minutes=1))
        volume_5m = self._compute_volume(symbol, timedelta(minutes=5))
        volume_ratio = self._compute_volume_ratio(volume_1m, volume_5m)

        # Technical indicators
        sma_5 = self._compute_sma(symbol, 5)
        sma_20 = self._compute_sma(symbol, 20)
        rsi_14 = self._compute_rsi(symbol, 14)

        # Soft information features (sentiment + fundamentals)
        sent_score = self._get_latest_sentiment(symbol, timestamp)
        fund_pe = self._get_latest_fundamental(symbol, timestamp)

        # Default to neutral sentiment if no data available
        if sent_score is None:
            sent_score = 0.0

        return FeatureSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            mid_price=mid_price,
            spread=spread,
            spread_bps=spread_bps,
            return_1m=return_1m,
            return_5m=return_5m,
            return_15m=return_15m,
            volatility_5m=volatility_5m,
            volatility_15m=volatility_15m,
            volatility_1h=volatility_1h,
            order_book_imbalance=order_book_imbalance,
            order_book_pressure=order_book_pressure,
            volume_1m=volume_1m,
            volume_5m=volume_5m,
            volume_ratio=volume_ratio,
            sma_5=sma_5,
            sma_20=sma_20,
            rsi_14=rsi_14,
            sent_score=sent_score,
            fund_pe=fund_pe,
        )

    def _compute_return(self, symbol: str, lookback: timedelta) -> Optional[float]:
        """Compute return over specified lookback period."""
        price_hist = self.price_history[symbol]
        if len(price_hist) < 2:
            return None

        current_time = price_hist[-1][0]
        current_price = price_hist[-1][1]

        # Find price at lookback time
        target_time = current_time - lookback

        for timestamp, price in reversed(price_hist):
            if timestamp <= target_time:
                if price and price > 0:
                    return float((current_price - price) / price)
                break

        return None

    def _compute_volatility(self, symbol: str, window: timedelta) -> Optional[float]:
        """Compute realized volatility over specified window."""
        price_hist = self.price_history[symbol]
        if len(price_hist) < 10:  # Need minimum data points
            return None

        current_time = price_hist[-1][0]
        target_time = current_time - window

        # Collect prices in window
        prices = []
        for timestamp, price in reversed(price_hist):
            if timestamp >= target_time:
                prices.append(float(price))
            else:
                break

        if len(prices) < 2:
            return None

        # Compute log returns
        prices = np.array(prices[::-1])  # Reverse to chronological order
        log_returns = np.diff(np.log(prices))

        # Annualized volatility
        if len(log_returns) > 1:
            return float(np.std(log_returns) * np.sqrt(365 * 24 * 60))  # Annualized

        return None

    def _compute_order_book_imbalance(self, tick: MarketTick) -> Optional[float]:
        """Compute order book imbalance from bid/ask sizes."""
        bid_size = getattr(tick, "bid_size", None)
        ask_size = getattr(tick, "ask_size", None)

        if bid_size is None or ask_size is None:
            return None

        try:
            total_size = float(bid_size) + float(ask_size)
        except (TypeError, ValueError):
            return None

        if total_size > 0:
            return float((float(bid_size) - float(ask_size)) / total_size)

        return None

    def _compute_order_book_pressure(self, tick: MarketTick) -> Optional[float]:
        """Compute order book pressure from full order book."""
        bids = getattr(tick, "bids", None)
        asks = getattr(tick, "asks", None)

        if not bids or not asks:
            return None

        if not isinstance(bids, (list, tuple)) or not isinstance(asks, (list, tuple)):
            return None

        # Calculate weighted pressure based on order book depth
        bid_pressure = sum(float(price * size) for price, size in bids[:5])
        ask_pressure = sum(float(price * size) for price, size in asks[:5])

        total_pressure = bid_pressure + ask_pressure
        if total_pressure > 0:
            return (bid_pressure - ask_pressure) / total_pressure

        return None

    def _compute_volume(self, symbol: str, window: timedelta) -> Optional[Decimal]:
        """Compute total volume over specified window."""
        volume_hist = self.volume_history[symbol]
        if not volume_hist:
            return None

        current_time = volume_hist[-1][0]
        target_time = current_time - window

        total_volume = Decimal("0")
        for timestamp, volume in reversed(volume_hist):
            if timestamp >= target_time:
                total_volume += volume
            else:
                break

        return total_volume if total_volume > 0 else None

    def _compute_volume_ratio(
        self, volume_1m: Optional[Decimal], volume_5m: Optional[Decimal]
    ) -> Optional[float]:
        """Compute 1m/5m volume ratio."""
        if volume_1m and volume_5m and volume_5m > 0:
            return float(volume_1m / volume_5m)
        return None

    def _compute_sma(self, symbol: str, periods: int) -> Optional[Decimal]:
        """Compute Simple Moving Average."""
        price_hist = self.price_history[symbol]
        if len(price_hist) < periods:
            return None

        # Get last N prices
        recent_prices = [price for _, price in list(price_hist)[-periods:]]

        if len(recent_prices) == periods:
            return sum(recent_prices) / len(recent_prices)

        return None

    def _compute_rsi(self, symbol: str, periods: int) -> Optional[float]:
        """Compute Relative Strength Index."""
        price_hist = self.price_history[symbol]
        if len(price_hist) < periods + 1:
            return None

        # Get price changes
        prices = [float(price) for _, price in list(price_hist)[-(periods + 1) :]]
        changes = np.diff(prices)

        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def get_stats(self) -> Dict[str, Any]:
        """Get feature bus performance statistics."""
        avg_computation_time = (
            np.mean(self.computation_times) if self.computation_times else 0
        )
        max_computation_time = (
            max(self.computation_times) if self.computation_times else 0
        )

        return {
            "features_computed": self.features_computed,
            "avg_computation_time_us": round(avg_computation_time, 2),
            "max_computation_time_us": round(max_computation_time, 2),
            "symbols_tracked": len(self.price_history),
            "total_price_points": sum(
                len(hist) for hist in self.price_history.values()
            ),
            "performance_target_met": avg_computation_time < 300,  # 300µs target
        }


class FeatureBusManager:
    """Manager for the feature bus with Kafka integration."""

    def __init__(self, kafka_brokers: list[str] = None):
        """
        Initialize feature bus manager.

        Args:
            kafka_brokers: List of Kafka broker addresses
        """
        self.kafka_brokers = kafka_brokers or ["localhost:9092"]
        self.feature_bus = FeatureBus()
        self.logger = get_logger("feature_bus_manager")

        self.consumer: AIOKafkaConsumer | None = None
        self.producer: AIOKafkaProducer | None = None

        # Topics
        self.input_topics = [
            "market.raw.crypto",  # Coinbase
            "market.raw.crypto.binance",  # Binance (L0-2)
        ]
        self.output_topic = "features.raw.crypto"

        # Metrics
        self.metrics = get_metrics()
        self.latency_hist = self.metrics.register_histogram(
            "feature_latency_ms",
            "tick→feature latency (ms)",
            buckets=(1, 2, 5, 10, 20, 50),
        )

        self.logger.info("Feature Bus Manager initialized")

    async def start(self):
        """Start the feature bus manager."""
        self.logger.info("Starting Feature Bus Manager…")

        # Initialize Kafka
        self.consumer = AIOKafkaConsumer(
            *self.input_topics,
            bootstrap_servers=self.kafka_brokers,
            value_deserializer=lambda v: orjson.loads(v),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        await self.consumer.start()

        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_brokers,
            value_serializer=lambda v: orjson.dumps(v),
            compression_type="gzip",
            batch_size=16384,
            linger_ms=5,
        )
        await self.producer.start()

        self.logger.info("Kafka consumer/producer started")

        # Consume loop
        asyncio.create_task(self._consume_loop())

    async def _consume_loop(self):
        assert self.consumer is not None
        async for msg in self.consumer:
            tick_data = msg.value
            start = time.perf_counter()
            tick = self._tick_from_json(tick_data)
            if not tick:
                continue
            features = await self.feature_bus.process_tick(tick)
            if features:
                await self._publish_features(features)
                latency_ms = (time.perf_counter() - start) * 1_000
                self.latency_hist.observe(
                    latency_ms / 1000
                )  # seconds histogram uses seconds

    def _tick_from_json(self, data: Dict[str, Any]) -> Optional[MarketTick]:
        try:
            mid = (
                (data["bid"] + data["ask"]) / 2
                if data.get("bid") and data.get("ask")
                else None
            )
            return MarketTick(
                symbol=data["symbol"],
                exchange=data["exchange"],
                asset_type="crypto",
                timestamp=datetime.now(timezone.utc),
                bid=Decimal(str(data["bid"])) if data.get("bid") else None,
                ask=Decimal(str(data["ask"])) if data.get("ask") else None,
                bid_size=(
                    Decimal(str(data["bid_size"])) if data.get("bid_size") else None
                ),
                ask_size=(
                    Decimal(str(data["ask_size"])) if data.get("ask_size") else None
                ),
                last=None,
                volume=None,
                metadata=data,
                mid=Decimal(str(mid)) if mid else None,
            )
        except Exception as exc:
            self.logger.error(f"Malformed tick: {exc}")
            return None

    async def _publish_features(self, features: FeatureSnapshot) -> None:
        """Publish limited feature snapshot per FB-1 spec."""
        try:
            mid = float(features.mid_price) if features.mid_price else None
            payload = {
                "ts": features.timestamp.timestamp(),
                "sym": features.symbol,
                "mid": mid,
                "spread_bps": features.spread_bps,
                "ret_60s": features.return_1m,
                "sigma_60s": features.volatility_1h,  # Using 1h as placeholder; can compute 60s
            }
            await self.producer.send_and_wait(self.output_topic, value=payload)
        except Exception as exc:
            self.logger.error(f"Publish features error: {exc}")

    async def stop(self):
        self.logger.info("Stopping Feature Bus Manager…")
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()
        self.logger.info("Feature Bus Manager stopped")


async def test_feature_bus():
    """Test the feature bus with synthetic data."""
    feature_bus = FeatureBus()

    # Create synthetic ticks
    base_price = Decimal("50000")

    for i in range(100):
        price = base_price + Decimal(str(i * 10))
        tick = MarketTick(
            symbol="BTC-USD",
            exchange="test",
            asset_type="crypto",
            timestamp=datetime.now(timezone.utc),
            bid=price - Decimal("5"),
            ask=price + Decimal("5"),
            volume=Decimal("100"),
            bid_size=Decimal("1.0"),
            ask_size=Decimal("1.5"),
        )

        features = await feature_bus.process_tick(tick)
        if features:
            print(
                f"Tick {i}: mid={features.mid_price}, spread_bps={features.spread_bps}"
            )

    print(f"Feature bus stats: {feature_bus.get_stats()}")


if __name__ == "__main__":
    asyncio.run(test_feature_bus())
