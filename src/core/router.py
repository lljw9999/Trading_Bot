#!/usr/bin/env python3
"""
Model Router - Dynamic Alpha Selection

Factory + strategy pattern for automatic model selection based on instrument 
metadata and trading horizon. Optimized for sub-50µs latency per call.

Usage:
    router = ModelRouter()
    model_id = router.select_model("BTC-USD", horizon_ms=30000)  # -> "tlob_tiny"
"""

import time
import logging
import json
import redis
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset class enumeration for routing rules."""

    CRYPTO = "crypto"
    US_STOCKS = "us_stocks"
    A_SHARES = "a_shares"
    FOREX = "forex"


class ModelFamily(Enum):
    """Available model families for routing."""

    TLOB_TINY = "tlob_tiny"
    PATCHTST_SMALL = "patchtst_small"
    TIMESNET_BASE = "timesnet_base"
    MAMBA_TS_SMALL = "mamba_ts_small"
    CHRONOS_BOLT_BASE = "chronos_bolt_base"


@dataclass
class InstrumentInfo:
    """Instrument metadata for routing decisions."""

    symbol: str
    asset_class: AssetClass
    exchange: str
    tick_size: float
    lot_size: float
    market_hours: Dict[str, str]

    @classmethod
    def from_redis_data(cls, symbol: str, data: Dict[str, Any]) -> "InstrumentInfo":
        """Create InstrumentInfo from Redis hash data."""
        return cls(
            symbol=symbol,
            asset_class=AssetClass(data.get("asset_class", "crypto")),
            exchange=data.get("exchange", "unknown"),
            tick_size=float(data.get("tick_size", 0.01)),
            lot_size=float(data.get("lot_size", 1.0)),
            market_hours=json.loads(data.get("market_hours", "{}")),
        )


@dataclass
class RoutingRule:
    """Routing rule for model selection."""

    asset_class: AssetClass
    horizon_min_ms: int
    horizon_max_ms: int
    model_id: ModelFamily
    priority: int = 100  # Lower = higher priority

    def matches(self, asset_class: AssetClass, horizon_ms: int) -> bool:
        """Check if this rule matches the given parameters."""
        return (
            self.asset_class == asset_class
            and self.horizon_min_ms <= horizon_ms <= self.horizon_max_ms
        )


class ModelRouterConfig(BaseModel):
    """Configuration for model router."""

    rules: List[Dict[str, Any]]
    default_model: str = "tlob_tiny"
    cache_ttl_seconds: int = 300  # 5 minutes
    redis_url: str = "redis://localhost:6379/0"


class ModelRouter:
    """
    Fast model router with pre-compiled rules for sub-50µs latency.

    Routes incoming trading signals to appropriate model families based on:
    - Asset class (crypto, stocks, etc.)
    - Trading horizon (milliseconds)
    - Instrument-specific metadata
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        config: Optional[ModelRouterConfig] = None,
    ):
        """Initialize model router with pre-compiled rules."""
        self.redis_client = redis_client or redis.Redis.from_url(
            "redis://localhost:6379/0"
        )
        self.config = config or self._load_default_config()

        # Pre-compile routing rules for fast lookup
        self._compiled_rules: List[RoutingRule] = []
        self._instrument_cache: Dict[str, InstrumentInfo] = {}
        self._cache_timestamps: Dict[str, float] = {}

        # Performance metrics
        self._call_count = 0
        self._total_latency_us = 0

        self._compile_rules()
        logger.info(f"ModelRouter initialized with {len(self._compiled_rules)} rules")

    def _load_default_config(self) -> ModelRouterConfig:
        """Load default routing configuration."""
        default_rules = [
            # Crypto high-frequency (< 1 minute)
            {
                "asset_class": "crypto",
                "horizon_min_ms": 0,
                "horizon_max_ms": 60000,  # 1 minute
                "model_id": "tlob_tiny",
                "priority": 10,
            },
            # Crypto medium-frequency (1 minute - 2 hours)
            {
                "asset_class": "crypto",
                "horizon_min_ms": 60000,  # 1 minute
                "horizon_max_ms": 7200000,  # 2 hours
                "model_id": "patchtst_small",
                "priority": 20,
            },
            # Crypto long-term (> 2 hours)
            {
                "asset_class": "crypto",
                "horizon_min_ms": 7200000,  # 2 hours
                "horizon_max_ms": 86400000,  # 24 hours
                "model_id": "mamba_ts_small",
                "priority": 30,
            },
            # US Stocks intraday (< 4 hours)
            {
                "asset_class": "us_stocks",
                "horizon_min_ms": 0,
                "horizon_max_ms": 14400000,  # 4 hours
                "model_id": "timesnet_base",
                "priority": 40,
            },
            # US Stocks overnight/swing (> 4 hours)
            {
                "asset_class": "us_stocks",
                "horizon_min_ms": 14400000,  # 4 hours
                "horizon_max_ms": 86400000,  # 24 hours
                "model_id": "mamba_ts_small",
                "priority": 50,
            },
            # A-shares intraday
            {
                "asset_class": "a_shares",
                "horizon_min_ms": 0,
                "horizon_max_ms": 14400000,  # 4 hours
                "model_id": "timesnet_base",
                "priority": 60,
            },
            # A-shares overnight
            {
                "asset_class": "a_shares",
                "horizon_min_ms": 14400000,  # 4 hours
                "horizon_max_ms": 86400000,  # 24 hours
                "model_id": "chronos_bolt_base",
                "priority": 70,
            },
            # Fallback rule
            {
                "asset_class": "crypto",  # Default to crypto rules
                "horizon_min_ms": 0,
                "horizon_max_ms": 86400000,
                "model_id": "tlob_tiny",
                "priority": 1000,
            },
        ]

        return ModelRouterConfig(rules=default_rules)

    def _compile_rules(self):
        """Pre-compile routing rules for fast lookup."""
        self._compiled_rules = []

        for rule_data in self.config.rules:
            rule = RoutingRule(
                asset_class=AssetClass(rule_data["asset_class"]),
                horizon_min_ms=rule_data["horizon_min_ms"],
                horizon_max_ms=rule_data["horizon_max_ms"],
                model_id=ModelFamily(rule_data["model_id"]),
                priority=rule_data.get("priority", 100),
            )
            self._compiled_rules.append(rule)

        # Sort by priority for fast matching
        self._compiled_rules.sort(key=lambda r: r.priority)

        logger.info(f"Compiled {len(self._compiled_rules)} routing rules")

    def _get_instrument_info(self, symbol: str) -> Optional[InstrumentInfo]:
        """Get instrument info with caching for performance."""
        current_time = time.time()

        # Check cache first
        if (
            symbol in self._instrument_cache
            and symbol in self._cache_timestamps
            and current_time - self._cache_timestamps[symbol]
            < self.config.cache_ttl_seconds
        ):
            return self._instrument_cache[symbol]

        # Fetch from Redis
        try:
            redis_key = f"instrument:info:{symbol}"
            data = self.redis_client.hgetall(redis_key)

            if not data:
                # Create default info if not found
                asset_class = self._infer_asset_class(symbol)
                default_info = InstrumentInfo(
                    symbol=symbol,
                    asset_class=asset_class,
                    exchange=self._infer_exchange(symbol),
                    tick_size=0.01,
                    lot_size=1.0,
                    market_hours={},
                )
                self._instrument_cache[symbol] = default_info
                self._cache_timestamps[symbol] = current_time
                return default_info

            # Convert bytes to strings for Redis data
            str_data = {
                k.decode() if isinstance(k, bytes) else k: (
                    v.decode() if isinstance(v, bytes) else v
                )
                for k, v in data.items()
            }

            instrument_info = InstrumentInfo.from_redis_data(symbol, str_data)
            self._instrument_cache[symbol] = instrument_info
            self._cache_timestamps[symbol] = current_time

            return instrument_info

        except Exception as e:
            logger.warning(f"Failed to fetch instrument info for {symbol}: {e}")
            return None

    def _infer_asset_class(self, symbol: str) -> AssetClass:
        """Infer asset class from symbol if not in Redis."""
        symbol_upper = symbol.upper()

        if any(
            crypto in symbol_upper for crypto in ["BTC", "ETH", "SOL", "USD", "USDT"]
        ):
            return AssetClass.CRYPTO
        elif symbol_upper.endswith(".SS") or symbol_upper.endswith(".SZ"):
            return AssetClass.A_SHARES
        elif any(
            stock_pattern in symbol_upper
            for stock_pattern in ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]
        ):
            return AssetClass.US_STOCKS
        else:
            # For truly unknown symbols, default to crypto to use fallback rule
            return AssetClass.CRYPTO

    def _infer_exchange(self, symbol: str) -> str:
        """Infer exchange from symbol."""
        symbol_upper = symbol.upper()

        if "USD" in symbol_upper:
            return "coinbase"
        elif symbol_upper.endswith(".SS"):
            return "shanghai"
        elif symbol_upper.endswith(".SZ"):
            return "shenzhen"
        else:
            return "nasdaq"

    def select_model(self, symbol: str, horizon_ms: int) -> str:
        """
        Fast model selection based on symbol and horizon.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD", "AAPL")
            horizon_ms: Trading horizon in milliseconds

        Returns:
            Model ID string (e.g., "tlob_tiny")
        """
        start_time = time.perf_counter()

        try:
            # Get instrument info
            instrument_info = self._get_instrument_info(symbol)
            if not instrument_info:
                logger.warning(f"No instrument info for {symbol}, using default")
                return self.config.default_model

            # Find matching rule (pre-sorted by priority)
            for rule in self._compiled_rules:
                if rule.matches(instrument_info.asset_class, horizon_ms):
                    model_id = rule.model_id.value

                    # Update performance metrics
                    latency_us = (time.perf_counter() - start_time) * 1_000_000
                    self._call_count += 1
                    self._total_latency_us += latency_us

                    # Log selection for monitoring
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Selected {model_id} for {symbol} "
                            f"(horizon={horizon_ms}ms, latency={latency_us:.1f}µs)"
                        )

                    return model_id

            # Fallback to default
            logger.warning(f"No matching rule for {symbol} horizon={horizon_ms}ms")
            return self.config.default_model

        except Exception as e:
            logger.error(f"Model selection failed for {symbol}: {e}")
            return self.config.default_model

    def publish_selection(
        self,
        symbol: str,
        model_id: str,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """Publish model selection to Redis channel."""
        try:
            channel = f"alpha.selected.{symbol}"
            data = {
                "symbol": symbol,
                "model_id": model_id,
                "timestamp": time.time(),
                "router_version": "1.0",
            }

            if additional_data:
                data.update(additional_data)

            message = json.dumps(data)
            self.redis_client.publish(channel, message)

            logger.debug(f"Published selection {model_id} for {symbol} to {channel}")

        except Exception as e:
            logger.error(f"Failed to publish selection for {symbol}: {e}")

    def get_performance_stats(self) -> Dict[str, float]:
        """Get router performance statistics."""
        if self._call_count == 0:
            return {"avg_latency_us": 0.0, "call_count": 0}

        avg_latency = self._total_latency_us / self._call_count
        return {
            "avg_latency_us": avg_latency,
            "call_count": self._call_count,
            "total_latency_us": self._total_latency_us,
        }

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self._call_count = 0
        self._total_latency_us = 0

    def reload_config(self, new_config: ModelRouterConfig):
        """Hot-reload router configuration."""
        logger.info("Reloading router configuration...")
        self.config = new_config
        self._compile_rules()
        self._instrument_cache.clear()  # Clear cache to pick up new rules
        self._cache_timestamps.clear()
        logger.info(f"Configuration reloaded with {len(self._compiled_rules)} rules")


# Factory function for easy instantiation
def create_model_router(
    redis_url: str = "redis://localhost:6379/0", config_path: Optional[str] = None
) -> ModelRouter:
    """Factory function to create ModelRouter instance."""
    redis_client = redis.Redis.from_url(redis_url)

    config = None
    if config_path:
        # TODO: Load config from YAML file
        pass

    return ModelRouter(redis_client=redis_client, config=config)
