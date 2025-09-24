#!/usr/bin/env python3
"""
Async Redis Connection Pool and Utilities

High-performance async Redis interface with connection pooling,
batched writes, and feature flag support.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from contextlib import asynccontextmanager

try:
    import aioredis

    AIOREDIS_AVAILABLE = True
except ImportError:
    AIOREDIS_AVAILABLE = False
    import redis  # Fallback to sync redis

logger = logging.getLogger(__name__)

# Global connection pool
_pool: Optional[aioredis.Redis] = None
_fallback_redis: Optional[redis.Redis] = None


async def get_redis() -> Union[aioredis.Redis, redis.Redis]:
    """Get async Redis connection pool, with fallback to sync."""
    global _pool, _fallback_redis

    if not AIOREDIS_AVAILABLE:
        logger.warning("aioredis not available, using sync redis")
        if _fallback_redis is None:
            _fallback_redis = redis.Redis(
                host="localhost", port=6379, decode_responses=True, max_connections=32
            )
        return _fallback_redis

    if _pool is None:
        try:
            _pool = await aioredis.from_url(
                "redis://localhost:6379",
                decode_responses=True,
                max_connections=32,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )
            logger.info("Async Redis pool initialized")
        except Exception as e:
            logger.error(f"Failed to create async Redis pool: {e}")
            # Fallback to sync redis
            if _fallback_redis is None:
                _fallback_redis = redis.Redis(
                    host="localhost",
                    port=6379,
                    decode_responses=True,
                    max_connections=32,
                )
            return _fallback_redis

    return _pool


async def check_feature_flag(flag: str, default: bool = False) -> bool:
    """Check if feature flag is enabled."""
    try:
        r = await get_redis()
        if hasattr(r, "get"):
            if asyncio.iscoroutinefunction(r.get):
                value = await r.get(f"features:{flag}")
            else:
                value = r.get(f"features:{flag}")
        else:
            value = None

        if value is None:
            return default
        return str(value).lower() in ("1", "true", "yes", "on")
    except Exception as e:
        logger.error(f"Error checking feature flag {flag}: {e}")
        return default


class AsyncBatchWriter:
    """Batch writer for high-throughput Redis operations."""

    def __init__(self, batch_size: int = 100, flush_interval: float = 0.1):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batch: List[Tuple[str, str, Dict[str, Any]]] = []
        self.last_flush = time.time()
        self._lock = asyncio.Lock()

    async def add(self, stream: str, data: Dict[str, Any], maxlen: int = 20000) -> None:
        """Add item to batch."""
        async with self._lock:
            self.batch.append((stream, "XADD", {"data": data, "maxlen": maxlen}))

            # Auto-flush if batch full or time elapsed
            should_flush = (
                len(self.batch) >= self.batch_size
                or time.time() - self.last_flush > self.flush_interval
            )

            if should_flush:
                await self._flush()

    async def set_batch(self, key: str, value: Any) -> None:
        """Add SET operation to batch."""
        async with self._lock:
            self.batch.append((key, "SET", {"value": value}))

            if len(self.batch) >= self.batch_size:
                await self._flush()

    async def _flush(self) -> None:
        """Flush batch to Redis."""
        if not self.batch:
            return

        try:
            r = await get_redis()

            if hasattr(r, "pipeline") and asyncio.iscoroutinefunction(r.pipeline):
                # Async Redis pipeline
                async with r.pipeline(transaction=False) as pipe:
                    for key, op, params in self.batch:
                        if op == "XADD":
                            pipe.xadd(
                                key,
                                params["data"],
                                maxlen=params["maxlen"],
                                approximate=True,
                            )
                        elif op == "SET":
                            pipe.set(key, params["value"])

                    await pipe.execute()
            else:
                # Sync Redis fallback
                pipe = r.pipeline(transaction=False)
                for key, op, params in self.batch:
                    if op == "XADD":
                        pipe.xadd(
                            key,
                            params["data"],
                            maxlen=params["maxlen"],
                            approximate=True,
                        )
                    elif op == "SET":
                        pipe.set(key, params["value"])

                pipe.execute()

            batch_size = len(self.batch)
            self.batch.clear()
            self.last_flush = time.time()

            logger.debug(f"Flushed batch of {batch_size} Redis operations")

        except Exception as e:
            logger.error(f"Error flushing Redis batch: {e}")
            # Don't clear batch on error - will retry next time

    async def flush(self) -> None:
        """Manual flush."""
        async with self._lock:
            await self._flush()


# Global batch writer instance
_batch_writer: Optional[AsyncBatchWriter] = None


async def get_batch_writer() -> AsyncBatchWriter:
    """Get global batch writer instance."""
    global _batch_writer
    if _batch_writer is None:
        _batch_writer = AsyncBatchWriter()
    return _batch_writer


async def publish_metrics_batch(
    metrics: Dict[str, Any], stream: str = "metrics"
) -> None:
    """Publish metrics using batch writer."""
    try:
        writer = await get_batch_writer()
        await writer.add(stream, {"timestamp": time.time(), **metrics})
    except Exception as e:
        logger.error(f"Error publishing metrics batch: {e}")


async def publish_trade_event(
    event_data: Dict[str, Any], stream: str = "exec:events"
) -> None:
    """Publish trade event using batch writer."""
    try:
        writer = await get_batch_writer()
        await writer.add(stream, {"timestamp": time.time(), **event_data})
    except Exception as e:
        logger.error(f"Error publishing trade event: {e}")


async def set_metric(key: str, value: Any) -> None:
    """Set metric using batch writer."""
    try:
        writer = await get_batch_writer()
        await writer.set_batch(f"metric:{key}", value)
    except Exception as e:
        logger.error(f"Error setting metric {key}: {e}")


@asynccontextmanager
async def redis_transaction():
    """Context manager for Redis transactions."""
    r = await get_redis()

    if hasattr(r, "pipeline") and asyncio.iscoroutinefunction(r.pipeline):
        async with r.pipeline(transaction=True) as pipe:
            yield pipe
    else:
        # Sync fallback
        pipe = r.pipeline(transaction=True)
        try:
            yield pipe
            pipe.execute()
        except Exception:
            pipe.discard()
            raise


async def get_config_value(key: str, default: Any = None, cast_type: type = str) -> Any:
    """Get configuration value from Redis."""
    try:
        r = await get_redis()

        if hasattr(r, "get") and asyncio.iscoroutinefunction(r.get):
            value = await r.get(f"config:{key}")
        else:
            value = r.get(f"config:{key}")

        if value is None:
            return default

        if cast_type == bool:
            return str(value).lower() in ("1", "true", "yes", "on")
        elif cast_type == dict:
            return json.loads(value)
        else:
            return cast_type(value)

    except Exception as e:
        logger.error(f"Error getting config {key}: {e}")
        return default


async def set_config_value(key: str, value: Any) -> None:
    """Set configuration value in Redis."""
    try:
        writer = await get_batch_writer()

        if isinstance(value, (dict, list)):
            json_value = json.dumps(value)
        else:
            json_value = str(value)

        await writer.set_batch(f"config:{key}", json_value)

    except Exception as e:
        logger.error(f"Error setting config {key}: {e}")


async def get_market_data(symbol: str) -> Dict[str, Any]:
    """Get market data for symbol."""
    try:
        r = await get_redis()
        data = {}

        # Get spot price
        if hasattr(r, "get") and asyncio.iscoroutinefunction(r.get):
            spot_price = await r.get(f"price:{symbol.lower()}:spot")
            perp_price = await r.get(f"price:{symbol.lower()}:perp")
            funding = await r.get(f"funding:{symbol.lower()}:annual")
        else:
            spot_price = r.get(f"price:{symbol.lower()}:spot")
            perp_price = r.get(f"price:{symbol.lower()}:perp")
            funding = r.get(f"funding:{symbol.lower()}:annual")

        if spot_price:
            data["spot_price"] = float(spot_price)
        if perp_price:
            data["perp_price"] = float(perp_price)
        if funding:
            data["funding_annual"] = float(funding)

        return data

    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        return {}


async def cleanup():
    """Cleanup connections."""
    global _pool, _batch_writer

    try:
        if _batch_writer:
            await _batch_writer.flush()

        if _pool and hasattr(_pool, "close"):
            await _pool.close()

        logger.info("Async Redis cleanup completed")

    except Exception as e:
        logger.error(f"Error during Redis cleanup: {e}")


# Utility functions for backward compatibility
async def aget(key: str) -> Optional[str]:
    """Async get."""
    r = await get_redis()
    if hasattr(r, "get") and asyncio.iscoroutinefunction(r.get):
        return await r.get(key)
    else:
        return r.get(key)


async def aset(key: str, value: Any) -> None:
    """Async set."""
    writer = await get_batch_writer()
    await writer.set_batch(key, value)


async def ahget(key: str, field: str) -> Optional[str]:
    """Async hash get."""
    r = await get_redis()
    if hasattr(r, "hget") and asyncio.iscoroutinefunction(r.hget):
        return await r.hget(key, field)
    else:
        return r.hget(key, field)


async def ahgetall(key: str) -> Dict[str, str]:
    """Async hash get all."""
    r = await get_redis()
    if hasattr(r, "hgetall") and asyncio.iscoroutinefunction(r.hgetall):
        return await r.hgetall(key)
    else:
        return r.hgetall(key)


async def axadd(stream: str, data: Dict[str, Any], maxlen: int = 20000) -> None:
    """Async stream add."""
    writer = await get_batch_writer()
    await writer.add(stream, data, maxlen)
