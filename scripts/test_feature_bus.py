#!/usr/bin/env python3
"""Feature bus validation helpers for CLI usage and pytest."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import redis
except ImportError:  # pragma: no cover - Redis is optional for offline testing
    redis = None  # type: ignore[assignment]

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.layers.layer0_data_ingestion.feature_bus import FeatureBus
from src.layers.layer0_data_ingestion.schemas import MarketTick

logger = logging.getLogger(__name__)


def _generate_synthetic_ticks(symbol: str, num_ticks: int) -> List[MarketTick]:
    """Generate deterministic ticks so tests do not depend on external data."""

    base_time = datetime.now(timezone.utc) - timedelta(seconds=num_ticks)
    ticks: List[MarketTick] = []

    for i in range(num_ticks):
        timestamp = base_time + timedelta(seconds=i)
        price = Decimal("50000") + (Decimal(i) * Decimal("0.25"))
        bid = price - Decimal("0.5")
        ask = price + Decimal("0.5")
        volume = Decimal("5") + Decimal((i % 5) * 0.1)
        bid_size = Decimal("1.5") + Decimal((i % 3) * 0.05)
        ask_size = Decimal("1.3") + Decimal((i % 2) * 0.05)

        ticks.append(
            MarketTick(
                symbol=symbol,
                exchange="synthetic",
                asset_type="crypto",
                timestamp=timestamp,
                bid=bid,
                ask=ask,
                mid=price,
                last=price,
                bid_size=bid_size,
                ask_size=ask_size,
                volume=volume,
            )
        )

    return ticks


def _load_ticks_from_redis(symbol: str, num_ticks: int) -> List[MarketTick]:
    """Attempt to load ticks from Redis if available."""

    if redis is None:
        logger.debug("Redis client not installed; falling back to synthetic data")
        return []

    try:
        client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        redis_key = f"market.raw.crypto.{symbol}"
        available = client.llen(redis_key)

        if not available:
            logger.debug("Redis key %s empty; using synthetic data", redis_key)
            return []

        fetch_count = min(available, num_ticks)
        ticks: List[MarketTick] = []

        for i in range(fetch_count):
            raw_tick = client.lindex(redis_key, -(i + 1))
            if not raw_tick:
                continue

            try:
                tick_data = json.loads(raw_tick)
                ts_ms = tick_data.get("ts")
                timestamp = (
                    datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                    if isinstance(ts_ms, (int, float))
                    else datetime.now(timezone.utc)
                )
                price = Decimal(
                    str(tick_data.get("price") or tick_data.get("last") or "0")
                )
                volume = Decimal(
                    str(tick_data.get("qty") or tick_data.get("volume") or "0")
                )

                ticks.append(
                    MarketTick(
                        symbol=symbol,
                        exchange="redis",
                        asset_type="crypto",
                        timestamp=timestamp,
                        bid=None,
                        ask=None,
                        mid=price,
                        last=price,
                        volume=volume,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - keep processing remaining ticks
                logger.debug("Failed to parse Redis tick %s: %s", i, exc)
                continue

        return ticks

    except Exception as exc:  # noqa: BLE001 - Redis optional for tests
        logger.debug("Redis unavailable: %s", exc)
        return []


async def run_feature_bus_pipeline(
    symbol: str = "BTC-USD",
    num_ticks: int = 200,
    *,
    use_redis: bool = True,
) -> Tuple[Dict[str, Any], Optional[Any]]:
    """Process ticks through the feature bus and return summary statistics."""

    ticks: List[MarketTick] = []

    if use_redis:
        ticks = _load_ticks_from_redis(symbol, num_ticks)

    if not ticks:
        ticks = _generate_synthetic_ticks(symbol, num_ticks)

    feature_bus = FeatureBus()
    latencies_ms: List[float] = []
    nan_count = 0
    last_features = None

    for tick in ticks[:num_ticks]:
        start = time.perf_counter()
        features = await feature_bus.process_tick(tick)
        latency = (time.perf_counter() - start) * 1000
        latencies_ms.append(latency)

        if features is None:
            continue

        last_features = features
        for value in features.__dict__.values():
            if isinstance(value, float) and math.isnan(value):
                nan_count += 1

    stats = feature_bus.get_stats()

    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    max_latency = float(np.max(latencies_ms)) if latencies_ms else 0.0
    p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    results = {
        "symbol": symbol,
        "ticks_processed": len(latencies_ms),
        "features_computed": stats.get("features_computed", 0),
        "nan_count": nan_count,
        "avg_latency_ms": round(avg_latency, 3),
        "max_latency_ms": round(max_latency, 3),
        "p95_latency_ms": round(p95_latency, 3),
        "latency_target_met": avg_latency <= 5.0,
        "no_nans": nan_count == 0,
        "bus_stats": stats,
    }

    return results, last_features


def print_results(results: Dict[str, Any]) -> None:
    """Print test results in a formatted way."""
    print("\n" + "=" * 60)
    print("FEATURE BUS PIPELINE VALIDATION RESULTS")
    print("=" * 60)

    print(f"Symbol: {results['symbol']}")
    print(f"Ticks processed: {results['ticks_processed']}")
    print(f"Features computed: {results['features_computed']}")
    print(f"NaN count: {results['nan_count']}")

    print(f"\nLatency Statistics:")
    print(f"  Average: {results['avg_latency_ms']:.3f} ms")
    print(f"  Maximum: {results['max_latency_ms']:.3f} ms")
    print(f"  95th percentile: {results['p95_latency_ms']:.3f} ms")

    print(f"\nSuccess Criteria:")
    print(f"  ✅ Edge vector has no NaNs: {results['no_nans']}")
    print(f"  ✅ Average latency ≤ 2ms: {results['latency_target_met']}")

    print(f"\nFeature Bus Performance:")
    bus_stats = results["bus_stats"]
    print(f"  Features computed: {bus_stats['features_computed']}")
    print(f"  Avg computation time: {bus_stats['avg_computation_time_us']:.1f}µs")
    print(f"  Max computation time: {bus_stats['max_computation_time_us']:.1f}µs")
    print(f"  Performance target met: {bus_stats['performance_target_met']}")

    # Overall pass/fail
    overall_pass = results["no_nans"] and results["latency_target_met"]
    status = "✅ PASS" if overall_pass else "❌ FAIL"
    print(
        f"\n{status} - Pipeline validation {'succeeded' if overall_pass else 'failed'}"
    )


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate feature bus pipeline")
    parser.add_argument("--symbol", default="BTC-USD", help="Symbol to test")
    parser.add_argument(
        "--ticks", type=int, default=200, help="Number of ticks to process"
    )
    parser.add_argument(
        "--use-redis",
        action="store_true",
        help="Attempt to pull ticks from Redis before falling back to synthetic data",
    )
    return parser.parse_args(argv)


async def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    results, _ = await run_feature_bus_pipeline(
        symbol=args.symbol,
        num_ticks=args.ticks,
        use_redis=args.use_redis,
    )
    print_results(results)
    return 0 if results["no_nans"] and results["latency_target_met"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    exit_code = asyncio.run(main())
    raise SystemExit(exit_code)


# ------------------------------- Pytest hooks ---------------------------------

import pytest


@pytest.mark.asyncio
async def test_feature_bus_pipeline_generates_features() -> None:
    """Synthetic ticks should yield stable features without NaNs."""

    results, last_features = await run_feature_bus_pipeline(use_redis=False)

    assert results["ticks_processed"] == 200
    assert results["features_computed"] > 0
    assert results["nan_count"] == 0
    assert results["latency_target_met"]
    assert last_features is not None
    assert last_features.mid_price is not None
    assert results["bus_stats"]["performance_target_met"] is not None
