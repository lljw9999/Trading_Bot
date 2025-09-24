#!/usr/bin/env python3
"""
Feature Cache
Batch action calls where safe (vectorize per-symbol); cache last state slice for 50–100ms to reduce FeatureBus work
"""

import time
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

import redis
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("feature_cache")


@dataclass
class CachedFeature:
    """Cached feature data with timestamp."""

    data: Dict[str, Any]
    timestamp: float
    ttl_ms: int

    @property
    def is_expired(self) -> bool:
        """Check if cached feature is expired."""
        return (time.time() * 1000 - self.timestamp) > self.ttl_ms

    @property
    def age_ms(self) -> float:
        """Get age of cached feature in milliseconds."""
        return time.time() * 1000 - self.timestamp


@dataclass
class BatchRequest:
    """Batched feature request."""

    symbol: str
    feature_names: List[str]
    timestamp: float
    future: asyncio.Future
    priority: int = 1


class FeatureCache:
    """High-performance feature cache with batching and TTL."""

    def __init__(self):
        """Initialize feature cache."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Cache configuration
        self.config = {
            "default_ttl_ms": 100,  # 100ms default TTL
            "max_cache_size": 10000,  # Max cached features per symbol
            "batch_size": 32,  # Max batch size for feature requests
            "batch_timeout_ms": 50,  # Max wait time to collect batch
            "cache_hit_ratio_target": 0.85,  # Target cache hit ratio
            "enable_vectorization": True,
            "enable_compression": False,  # Compress cached data
            "stats_interval_seconds": 60,  # Stats reporting interval
        }

        # In-memory cache
        self.cache: Dict[str, Dict[str, CachedFeature]] = defaultdict(dict)
        self.cache_access_times: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Batching system
        self.pending_requests: Dict[str, List[BatchRequest]] = defaultdict(list)
        self.batch_timers: Dict[str, asyncio.Handle] = {}

        # Performance metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "features_served": 0,
            "batches_processed": 0,
            "batch_sizes": [],
            "feature_computation_times": [],
            "cache_evictions": 0,
            "cpu_savings_pct": 0.0,
        }

        # Start background tasks
        self.stats_task = None

        logger.info("💾 Feature Cache initialized")

    def start_background_tasks(self):
        """Start background tasks."""
        self.stats_task = asyncio.create_task(self._stats_reporter())

    def stop_background_tasks(self):
        """Stop background tasks."""
        if self.stats_task:
            self.stats_task.cancel()

    async def _stats_reporter(self):
        """Report cache statistics periodically."""
        while True:
            try:
                await asyncio.sleep(self.config["stats_interval_seconds"])
                await self._update_performance_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats reporter: {e}")

    def _calculate_cache_key(self, symbol: str, feature_name: str) -> str:
        """Calculate cache key for feature."""
        return f"feature_cache:{symbol}:{feature_name}"

    def _evict_expired_features(self, symbol: str):
        """Evict expired features for a symbol."""
        if symbol not in self.cache:
            return

        current_time = time.time() * 1000
        expired_keys = []

        for feature_name, cached_feature in self.cache[symbol].items():
            if cached_feature.is_expired:
                expired_keys.append(feature_name)

        for key in expired_keys:
            del self.cache[symbol][key]
            self.metrics["cache_evictions"] += 1

        # Also evict LRU if cache is too large
        if len(self.cache[symbol]) > self.config["max_cache_size"]:
            # Sort by access time (oldest first)
            access_times = self.cache_access_times[symbol]
            if access_times:
                # Remove oldest entries
                excess_count = len(self.cache[symbol]) - self.config["max_cache_size"]
                for _ in range(excess_count):
                    if access_times:
                        oldest_feature = access_times.popleft()
                        if oldest_feature in self.cache[symbol]:
                            del self.cache[symbol][oldest_feature]
                            self.metrics["cache_evictions"] += 1

    def get_cached_feature(
        self, symbol: str, feature_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get feature from cache if available and not expired."""
        try:
            if symbol in self.cache and feature_name in self.cache[symbol]:
                cached_feature = self.cache[symbol][feature_name]

                if not cached_feature.is_expired:
                    # Update access time
                    self.cache_access_times[symbol].append(feature_name)
                    self.metrics["cache_hits"] += 1
                    return cached_feature.data
                else:
                    # Remove expired feature
                    del self.cache[symbol][feature_name]
                    self.metrics["cache_evictions"] += 1

            self.metrics["cache_misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Error getting cached feature {symbol}:{feature_name}: {e}")
            return None

    def cache_feature(
        self,
        symbol: str,
        feature_name: str,
        data: Dict[str, Any],
        ttl_ms: Optional[int] = None,
    ):
        """Cache a feature with TTL."""
        try:
            if ttl_ms is None:
                ttl_ms = self.config["default_ttl_ms"]

            cached_feature = CachedFeature(
                data=data, timestamp=time.time() * 1000, ttl_ms=ttl_ms
            )

            # Store in cache
            self.cache[symbol][feature_name] = cached_feature
            self.cache_access_times[symbol].append(feature_name)

            # Evict expired/old features periodically
            if len(self.cache[symbol]) % 100 == 0:  # Every 100 entries
                self._evict_expired_features(symbol)

        except Exception as e:
            logger.error(f"Error caching feature {symbol}:{feature_name}: {e}")

    def compute_single_feature(self, symbol: str, feature_name: str) -> Dict[str, Any]:
        """Compute a single feature (mock implementation)."""
        try:
            start_time = time.time()

            # Mock feature computation based on feature name
            if feature_name == "price_features":
                data = {
                    "current_price": np.random.uniform(30000, 50000),
                    "volume_24h": np.random.uniform(1000000, 10000000),
                    "price_change_1h": np.random.uniform(-0.05, 0.05),
                    "price_change_24h": np.random.uniform(-0.2, 0.2),
                }
            elif feature_name == "technical_indicators":
                data = {
                    "rsi": np.random.uniform(20, 80),
                    "macd": np.random.uniform(-100, 100),
                    "bollinger_upper": np.random.uniform(31000, 52000),
                    "bollinger_lower": np.random.uniform(29000, 48000),
                    "sma_20": np.random.uniform(30000, 50000),
                    "ema_50": np.random.uniform(30000, 50000),
                }
            elif feature_name == "market_microstructure":
                data = {
                    "bid_ask_spread": np.random.uniform(0.01, 1.0),
                    "order_book_imbalance": np.random.uniform(-0.5, 0.5),
                    "trade_intensity": np.random.uniform(0.1, 10.0),
                    "market_impact": np.random.uniform(0.001, 0.01),
                }
            elif feature_name == "volatility_features":
                data = {
                    "realized_volatility": np.random.uniform(0.1, 2.0),
                    "garch_volatility": np.random.uniform(0.1, 2.0),
                    "vix_equivalent": np.random.uniform(10, 80),
                    "volatility_skew": np.random.uniform(-0.5, 0.5),
                }
            else:
                # Generic feature
                data = {
                    "value": np.random.uniform(0, 100),
                    "timestamp": time.time(),
                    "symbol": symbol,
                }

            computation_time = (time.time() - start_time) * 1000
            self.metrics["feature_computation_times"].append(computation_time)

            return data

        except Exception as e:
            logger.error(f"Error computing feature {symbol}:{feature_name}: {e}")
            return {"error": str(e)}

    def compute_features_batch(
        self, requests: List[BatchRequest]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute features in batch for efficiency."""
        try:
            start_time = time.time()

            # Group requests by symbol for vectorization
            symbol_features = defaultdict(set)
            for req in requests:
                symbol_features[req.symbol].update(req.feature_names)

            results = {}

            # Process each symbol's features together (vectorization opportunity)
            for symbol, feature_names in symbol_features.items():
                symbol_results = {}

                if self.config["enable_vectorization"] and len(feature_names) > 1:
                    # Vectorized computation (mock - could call optimized C++/CUDA code)
                    logger.debug(
                        f"🚀 Vectorized computation for {symbol}: {list(feature_names)}"
                    )

                    # In reality, this would be a single optimized call
                    for feature_name in feature_names:
                        symbol_results[feature_name] = self.compute_single_feature(
                            symbol, feature_name
                        )
                else:
                    # Individual feature computation
                    for feature_name in feature_names:
                        symbol_results[feature_name] = self.compute_single_feature(
                            symbol, feature_name
                        )

                results[symbol] = symbol_results

            batch_time = (time.time() - start_time) * 1000
            self.metrics["batches_processed"] += 1
            self.metrics["batch_sizes"].append(len(requests))

            # Calculate CPU savings from batching/caching
            expected_individual_time = (
                len(requests) * 5
            )  # Assume 5ms per individual request
            cpu_savings = max(
                0, (expected_individual_time - batch_time) / expected_individual_time
            )
            self.metrics["cpu_savings_pct"] = cpu_savings * 100

            return results

        except Exception as e:
            logger.error(f"Error computing features batch: {e}")
            return {}

    async def _process_batch(self, symbol: str):
        """Process a batch of requests for a symbol."""
        try:
            if symbol not in self.pending_requests or not self.pending_requests[symbol]:
                return

            # Get pending requests
            requests = self.pending_requests[symbol]
            self.pending_requests[symbol] = []

            # Clear timer
            if symbol in self.batch_timers:
                self.batch_timers[symbol].cancel()
                del self.batch_timers[symbol]

            if not requests:
                return

            logger.debug(f"📦 Processing batch for {symbol}: {len(requests)} requests")

            # Compute features
            results = self.compute_features_batch(requests)

            # Cache results and fulfill requests
            for request in requests:
                try:
                    symbol_results = results.get(request.symbol, {})
                    request_results = {}

                    for feature_name in request.feature_names:
                        if feature_name in symbol_results:
                            feature_data = symbol_results[feature_name]

                            # Cache the feature
                            self.cache_feature(
                                request.symbol, feature_name, feature_data
                            )
                            request_results[feature_name] = feature_data
                        else:
                            logger.warning(
                                f"⚠️ Feature {feature_name} not computed for {request.symbol}"
                            )

                    # Fulfill the request
                    if not request.future.done():
                        request.future.set_result(request_results)

                except Exception as e:
                    if not request.future.done():
                        request.future.set_exception(e)

        except Exception as e:
            logger.error(f"Error processing batch for {symbol}: {e}")

            # Set exception for all pending requests
            if symbol in self.pending_requests:
                for request in self.pending_requests[symbol]:
                    if not request.future.done():
                        request.future.set_exception(e)
                self.pending_requests[symbol] = []

    async def get_features(
        self, symbol: str, feature_names: List[str], priority: int = 1
    ) -> Dict[str, Any]:
        """Get features with caching and batching."""
        try:
            # Check cache first
            cached_results = {}
            missing_features = []

            for feature_name in feature_names:
                cached_data = self.get_cached_feature(symbol, feature_name)
                if cached_data is not None:
                    cached_results[feature_name] = cached_data
                else:
                    missing_features.append(feature_name)

            # If all features are cached, return immediately
            if not missing_features:
                self.metrics["features_served"] += len(feature_names)
                return cached_results

            # Create batch request for missing features
            future = asyncio.Future()
            request = BatchRequest(
                symbol=symbol,
                feature_names=missing_features,
                timestamp=time.time(),
                future=future,
                priority=priority,
            )

            # Add to pending requests
            self.pending_requests[symbol].append(request)

            # Set timer for batch processing if not already set
            if symbol not in self.batch_timers:

                def trigger_batch():
                    asyncio.create_task(self._process_batch(symbol))

                loop = asyncio.get_event_loop()
                self.batch_timers[symbol] = loop.call_later(
                    self.config["batch_timeout_ms"] / 1000, trigger_batch
                )

            # Process immediately if batch is full
            if len(self.pending_requests[symbol]) >= self.config["batch_size"]:
                if symbol in self.batch_timers:
                    self.batch_timers[symbol].cancel()
                await self._process_batch(symbol)

            # Wait for results
            computed_results = await future

            # Combine cached and computed results
            final_results = {**cached_results, **computed_results}
            self.metrics["features_served"] += len(feature_names)

            return final_results

        except Exception as e:
            logger.error(f"Error getting features for {symbol}: {e}")
            raise

    async def _update_performance_metrics(self):
        """Update performance metrics in Redis."""
        try:
            total_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
            cache_hit_ratio = self.metrics["cache_hits"] / max(total_requests, 1)

            # Calculate averages
            avg_batch_size = (
                np.mean(self.metrics["batch_sizes"])
                if self.metrics["batch_sizes"]
                else 0
            )
            avg_computation_time = (
                np.mean(self.metrics["feature_computation_times"])
                if self.metrics["feature_computation_times"]
                else 0
            )

            metrics = {
                "cache_hit_ratio": cache_hit_ratio,
                "cache_hits": self.metrics["cache_hits"],
                "cache_misses": self.metrics["cache_misses"],
                "features_served": self.metrics["features_served"],
                "batches_processed": self.metrics["batches_processed"],
                "avg_batch_size": avg_batch_size,
                "avg_computation_time_ms": avg_computation_time,
                "cache_evictions": self.metrics["cache_evictions"],
                "cpu_savings_pct": self.metrics["cpu_savings_pct"],
                "cached_symbols": len(self.cache),
                "total_cached_features": sum(
                    len(features) for features in self.cache.values()
                ),
            }

            # Store in Redis for monitoring
            for metric_name, value in metrics.items():
                self.redis.set(f"featurebus:{metric_name}", value)

            # Store CPU savings percentage for monitoring
            self.redis.set("featurebus_cpu_pct_saved", self.metrics["cpu_savings_pct"])

            logger.info(
                f"📊 Feature cache metrics: "
                f"hit_ratio={cache_hit_ratio:.2%}, "
                f"features_served={self.metrics['features_served']}, "
                f"cpu_savings={self.metrics['cpu_savings_pct']:.1f}%"
            )

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics."""
        try:
            total_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
            cache_hit_ratio = self.metrics["cache_hits"] / max(total_requests, 1)

            return {
                "cache_hit_ratio": cache_hit_ratio,
                "cache_hits": self.metrics["cache_hits"],
                "cache_misses": self.metrics["cache_misses"],
                "features_served": self.metrics["features_served"],
                "batches_processed": self.metrics["batches_processed"],
                "avg_batch_size": (
                    np.mean(self.metrics["batch_sizes"])
                    if self.metrics["batch_sizes"]
                    else 0
                ),
                "cache_evictions": self.metrics["cache_evictions"],
                "cpu_savings_pct": self.metrics["cpu_savings_pct"],
                "cached_symbols": len(self.cache),
                "total_cached_features": sum(
                    len(features) for features in self.cache.values()
                ),
                "pending_requests": sum(
                    len(reqs) for reqs in self.pending_requests.values()
                ),
                "config": self.config,
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cache for specific symbol or all symbols."""
        try:
            if symbol:
                if symbol in self.cache:
                    del self.cache[symbol]
                if symbol in self.cache_access_times:
                    del self.cache_access_times[symbol]
                logger.info(f"🗑️ Cleared cache for symbol: {symbol}")
            else:
                self.cache.clear()
                self.cache_access_times.clear()
                logger.info("🗑️ Cleared entire feature cache")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


# Global feature cache instance
feature_cache = FeatureCache()


async def get_features_cached(
    symbol: str, feature_names: List[str], priority: int = 1
) -> Dict[str, Any]:
    """Convenience function to get features with caching."""
    return await feature_cache.get_features(symbol, feature_names, priority)


def get_cache_statistics() -> Dict[str, Any]:
    """Get cache statistics."""
    return feature_cache.get_cache_stats()


def clear_feature_cache(symbol: Optional[str] = None):
    """Clear feature cache."""
    feature_cache.clear_cache(symbol)


# Example usage
async def main():
    """Example usage of feature cache."""
    # Start background tasks
    feature_cache.start_background_tasks()

    try:
        # Example: Get features for multiple symbols
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        feature_names = [
            "price_features",
            "technical_indicators",
            "market_microstructure",
        ]

        # Concurrent requests to test batching
        tasks = []
        for symbol in symbols:
            task = get_features_cached(symbol, feature_names)
            tasks.append(task)

        # Wait for all requests
        results = await asyncio.gather(*tasks)

        for i, symbol in enumerate(symbols):
            print(f"📊 Features for {symbol}: {list(results[i].keys())}")

        # Show cache stats
        stats = get_cache_statistics()
        print(f"📈 Cache stats: {stats}")

    finally:
        # Stop background tasks
        feature_cache.stop_background_tasks()


if __name__ == "__main__":
    asyncio.run(main())
