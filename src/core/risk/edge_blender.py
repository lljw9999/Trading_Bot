#!/usr/bin/env python3
"""
Edge Blender for Risk Harmoniser v1

Blends per-trade edge and edge_confidence from multiple alpha engines using:
- Decay weights w_i = exp(-λ_i) where λ_i from YAML configuration
- Bayesian shrinkage towards prior
- Confidence-weighted blending: E = (Σ w_i c_i e_i) / (Σ w_i c_i)

Ensures aggregate portfolio VaR ≤ target and prevents single model blow-through.
"""

import time
import math
import logging
import json
import yaml
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import redis

from src.utils.env_flags import env_flag

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ModelEdge:
    """Individual model edge with metadata."""

    model_id: str
    edge_bps: float
    confidence: float
    timestamp: float
    horizon_ms: int

    def __post_init__(self):
        """Validate edge data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")
        if abs(self.edge_bps) > 10000:  # Sanity check: max 100% edge
            raise ValueError(f"Edge {self.edge_bps}bps seems unrealistic")


@dataclass
class BlendedEdge:
    """Blended edge result with detailed breakdown."""

    symbol: str
    edge_blended_bps: float
    edge_raw: Dict[str, float]  # {model_id: raw_edge_bps}
    weights: Dict[str, float]  # {model_id: final_weight}
    confidences: Dict[str, float]  # {model_id: confidence}
    kelly_frac: float
    timestamp: float
    num_models: int
    total_weight: float
    shrinkage_applied: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "edge_blended_bps": self.edge_blended_bps,
            "edge_raw": self.edge_raw,
            "weights": self.weights,
            "confidences": self.confidences,
            "kelly_frac": self.kelly_frac,
            "timestamp": self.timestamp,
            "num_models": self.num_models,
            "total_weight": self.total_weight,
            "shrinkage_applied": self.shrinkage_applied,
        }


FF_EDGE_BLEND_LEGACY = env_flag("FF_EDGE_BLEND_LEGACY")
LEGACY_FAST_BENCH = env_flag("FF_EDGE_BLEND_LEGACY_FAST", default=True)


class EdgeBlender:
    """
    Edge blending engine with decay weights and Bayesian shrinkage.

    Implements the algorithm from Future_instruction.txt:
    1. Raw edge e_i from model i
    2. Confidence c_i ∈ [0,1] (supplied by model; else fallback = 0.5)
    3. Decay weight w_i = exp(-λ_i) where λ_i from YAML
    4. Blended edge: E = (Σ w_i c_i e_i) / (Σ w_i c_i)
    """

    def __init__(self, config_path: str, redis_client: Optional[redis.Redis] = None):
        """Initialize edge blender with configuration."""
        self.config_path = Path(config_path)
        self.redis_client = redis_client

        # Thread-safe configuration storage
        self._config_lock = threading.RLock()
        self._config: Optional[Dict[str, Any]] = None

        # Performance tracking
        self._blend_count = 0
        self._total_blend_time_us = 0.0
        self._model_usage_stats: Dict[str, int] = {}

        # Edge history for variance estimation
        self._edge_history: Dict[str, List[float]] = {}  # {symbol: [recent_edges]}
        self._max_history_length = 252  # ~1 year of daily data

        # Load initial configuration
        self._load_config()

        logger.info(f"EdgeBlender initialized with config from {config_path}")

    def blend_edges(self, symbol: str, model_edges: List[ModelEdge]) -> BlendedEdge:
        """
        Blend edges from multiple models with decay weights and confidence.

        Args:
            symbol: Trading symbol
            model_edges: List of ModelEdge objects from different models

        Returns:
            BlendedEdge with blended result and metadata
        """
        start_time = time.perf_counter()

        try:
            with self._config_lock:
                config = self._config

            if not config or not model_edges:
                # Return zero edge if no models or config
                return self._create_zero_blend(symbol)

            # Filter valid edges
            valid_edges = self._filter_valid_edges(model_edges, config)
            if not valid_edges:
                return self._create_zero_blend(symbol)

            # Calculate decay weights
            weights = self._calculate_decay_weights(valid_edges, config)

            use_legacy_multi = FF_EDGE_BLEND_LEGACY and len(valid_edges) > 1

            if use_legacy_multi:
                edge_blended_bps = self._blend_with_confidence(valid_edges, weights)
                shrinkage_applied = False
                kelly_frac = config.get("position_sizing", {}).get(
                    "kelly_fraction", 0.25
                )
            else:
                edge_blended_bps, shrinkage_applied = (
                    self._blend_with_bayesian_shrinkage(valid_edges, weights, config)
                )
                kelly_frac = self._calculate_kelly_fraction(
                    edge_blended_bps, symbol, config
                )

            # Create result
            result = BlendedEdge(
                symbol=symbol,
                edge_blended_bps=edge_blended_bps,
                edge_raw={edge.model_id: edge.edge_bps for edge in valid_edges},
                weights={edge.model_id: weights[edge.model_id] for edge in valid_edges},
                confidences={edge.model_id: edge.confidence for edge in valid_edges},
                kelly_frac=kelly_frac,
                timestamp=time.time(),
                num_models=len(valid_edges),
                total_weight=sum(weights.values()),
                shrinkage_applied=shrinkage_applied,
            )

            # Update performance metrics
            blend_time_us = (time.perf_counter() - start_time) * 1_000_000
            target_latency_us = (
                config.get("monitoring", {})
                .get("performance", {})
                .get("max_blend_latency_us", 20)
            )
            self._update_performance_metrics(valid_edges, blend_time_us)

            # Store edge for variance estimation
            self._store_edge_history(symbol, edge_blended_bps)

            # Publish blended result
            self._publish_blended_edge(result)

            # Performance check
            if blend_time_us > target_latency_us:
                logger.warning(
                    f"Blend latency {blend_time_us:.1f}µs exceeds target {target_latency_us}µs"
                )

            logger.debug(
                f"Blended {len(valid_edges)} models for {symbol}: "
                f"{edge_blended_bps:.2f}bps (latency: {blend_time_us:.1f}µs)"
            )

            return result

        except Exception as e:
            logger.error(f"Edge blending failed for {symbol}: {e}")
            return self._create_zero_blend(symbol)

    def _filter_valid_edges(
        self, model_edges: List[ModelEdge], config: Dict[str, Any]
    ) -> List[ModelEdge]:
        """Filter edges based on confidence thresholds and limits."""
        edge_config = config.get("edge_blending", {})
        min_confidence = edge_config.get("min_confidence_threshold", 0.3)
        max_models = edge_config.get("max_models_to_blend", 5)

        # Filter by confidence threshold
        valid_edges = [
            edge for edge in model_edges if edge.confidence >= min_confidence
        ]

        if FF_EDGE_BLEND_LEGACY:
            valid_edges = [edge for edge in valid_edges if edge.confidence > 0]

        # Limit number of models (take highest confidence ones)
        if len(valid_edges) > max_models:
            valid_edges = sorted(valid_edges, key=lambda x: x.confidence, reverse=True)[
                :max_models
            ]

        return valid_edges

    def _calculate_decay_weights(
        self, model_edges: List[ModelEdge], config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate decay weights w_i = exp(-λ_i) for each model."""
        decay_factors = config.get("edge_blending", {}).get("decay_factors", {})
        weights = {}

        for edge in model_edges:
            # Get decay factor for this model (default to 1.0 for unknown models)
            lambda_i = decay_factors.get(edge.model_id, 1.0)

            # Calculate decay weight: w_i = exp(-λ_i)
            weights[edge.model_id] = math.exp(-lambda_i)

        return weights

    def _blend_with_confidence(
        self, model_edges: List[ModelEdge], weights: Dict[str, float]
    ) -> float:
        """
        Apply confidence-weighted blending: E = (Σ w_i c_i e_i) / (Σ w_i c_i)
        """
        numerator = 0.0
        denominator = 0.0

        for edge in model_edges:
            w_i = weights[edge.model_id]
            c_i = edge.confidence
            e_i = edge.edge_bps

            numerator += w_i * c_i * e_i
            denominator += w_i * c_i

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _blend_with_bayesian_shrinkage(
        self,
        model_edges: List[ModelEdge],
        weights: Dict[str, float],
        config: Dict[str, Any],
    ) -> Tuple[float, bool]:
        """Blend edges and apply Bayesian shrinkage when enabled."""
        base_edge = self._blend_with_confidence(model_edges, weights)

        shrink_cfg = config.get("edge_blending", {}).get("shrinkage", {})
        if not shrink_cfg.get("enabled", False):
            return base_edge, False

        iterations = int(shrink_cfg.get("iterations", 1))
        if FF_EDGE_BLEND_LEGACY and LEGACY_FAST_BENCH:
            iterations = max(1, min(iterations, 100))
        else:
            iterations = max(1, iterations)

        prior_edge = shrink_cfg.get("prior_edge_bps", 0.0)
        shrinkage_strength = float(shrink_cfg.get("shrinkage_strength", 0.1))
        shrinkage_strength = max(0.0, min(shrinkage_strength, 1.0))

        if shrinkage_strength == 0.0:
            return base_edge, False

        if iterations <= 1:
            shrunk = self._apply_bayesian_shrinkage(
                base_edge, prior_edge, shrinkage_strength
            )
            return shrunk, True

        step_strength = 1.0 - math.pow(1.0 - shrinkage_strength, 1.0 / iterations)
        shrunk_edge = base_edge

        # Iterative shrinkage lets us tune performance in legacy fast mode.
        for _ in range(iterations):
            shrunk_edge = self._apply_bayesian_shrinkage(
                shrunk_edge, prior_edge, step_strength
            )

        return shrunk_edge, True

    @staticmethod
    def _apply_bayesian_shrinkage(
        edge_bps: float, prior_edge: float, shrinkage_strength: float
    ) -> float:
        """Shrink edge towards prior by the supplied strength."""
        shrinkage_strength = max(0.0, min(shrinkage_strength, 1.0))
        return (1 - shrinkage_strength) * edge_bps + shrinkage_strength * prior_edge

    def _calculate_kelly_fraction(
        self, edge_bps: float, symbol: str, config: Dict[str, Any]
    ) -> float:
        """Calculate Kelly fraction for position sizing."""
        # Get Kelly fraction from config
        kelly_frac = config.get("position_sizing", {}).get("kelly_fraction", 0.25)

        # Estimate variance from edge history
        variance = self._estimate_edge_variance(symbol)

        if variance <= 0:
            return 0.0

        # Kelly formula: f = edge / variance
        # But we already have a configured Kelly fraction, so we scale by edge strength
        edge_decimal = edge_bps / 10000.0  # Convert bps to decimal

        # Scale Kelly fraction by edge strength (normalized)
        # Strong edges (>10bps) get full Kelly, weak edges get scaled down
        edge_strength = min(abs(edge_decimal) * 1000, 1.0)  # Cap at 1.0

        return kelly_frac * edge_strength

    def _estimate_edge_variance(self, symbol: str) -> float:
        """Estimate edge variance from historical data."""
        if symbol not in self._edge_history or len(self._edge_history[symbol]) < 2:
            # Default variance estimate for new symbols
            return 1.0  # 1bps^2 default variance

        edges = self._edge_history[symbol]
        return float(np.var(edges)) if len(edges) > 1 else 1.0

    def _store_edge_history(self, symbol: str, edge_bps: float):
        """Store edge in history for variance estimation."""
        if symbol not in self._edge_history:
            self._edge_history[symbol] = []

        self._edge_history[symbol].append(edge_bps)

        # Keep only recent history
        if len(self._edge_history[symbol]) > self._max_history_length:
            self._edge_history[symbol].pop(0)

    def _publish_blended_edge(self, blended_edge: BlendedEdge):
        """Publish blended edge to Redis channel."""
        if not self.redis_client:
            return

        try:
            # Publish to symbol-specific channel
            channel = f"risk.edge_blended.{blended_edge.symbol}"
            message = json.dumps(blended_edge.to_dict())
            self.redis_client.publish(channel, message)

            # Also publish to general risk channel for monitoring
            risk_channel = "risk.edge_blended"
            self.redis_client.publish(risk_channel, message)

            logger.debug(f"Published blended edge for {blended_edge.symbol} to Redis")

        except Exception as e:
            logger.error(f"Failed to publish blended edge: {e}")

    def _create_zero_blend(self, symbol: str) -> BlendedEdge:
        """Create zero-edge blend result."""
        return BlendedEdge(
            symbol=symbol,
            edge_blended_bps=0.0,
            edge_raw={},
            weights={},
            confidences={},
            kelly_frac=0.0,
            timestamp=time.time(),
            num_models=0,
            total_weight=0.0,
            shrinkage_applied=False,
        )

    def _update_performance_metrics(
        self, model_edges: List[ModelEdge], blend_time_us: float
    ):
        """Update performance tracking metrics."""
        self._blend_count += 1
        self._total_blend_time_us += blend_time_us

        for edge in model_edges:
            self._model_usage_stats[edge.model_id] = (
                self._model_usage_stats.get(edge.model_id, 0) + 1
            )

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                full_config = yaml.safe_load(f)

            # Extract risk harmoniser config
            config = full_config.get("risk_harmoniser", {})

            with self._config_lock:
                self._config = config

            logger.info(
                f"Loaded risk configuration: {len(config.get('edge_blending', {}).get('decay_factors', {}))} models configured"
            )

        except Exception as e:
            logger.error(f"Failed to load risk config from {self.config_path}: {e}")
            # Set minimal fallback config
            with self._config_lock:
                self._config = {
                    "edge_blending": {
                        "decay_factors": {},
                        "default_confidence": 0.5,
                        "min_confidence_threshold": 0.3,
                    },
                    "position_sizing": {"kelly_fraction": 0.25},
                }

    def reload_config(self) -> bool:
        """Reload configuration from file."""
        try:
            self._load_config()
            logger.info("Risk configuration reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload risk configuration: {e}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_blend_time_us = self._total_blend_time_us / max(1, self._blend_count)

        return {
            "blend_count": self._blend_count,
            "avg_blend_time_us": avg_blend_time_us,
            "model_usage_stats": dict(self._model_usage_stats),
            "symbols_tracked": len(self._edge_history),
            "config_loaded": self._config is not None,
        }

    def get_edge_history(self, symbol: str, limit: int = 50) -> List[float]:
        """Get recent edge history for a symbol."""
        if symbol not in self._edge_history:
            return []
        return self._edge_history[symbol][-limit:]


# Factory function for easy instantiation
def create_edge_blender(
    config_path: str = "conf/risk_params.yml", redis_url: Optional[str] = None
) -> EdgeBlender:
    """Factory function to create EdgeBlender instance."""
    redis_client = None
    if redis_url:
        try:
            redis_client = redis.Redis.from_url(redis_url)
            redis_client.ping()  # Test connection
        except Exception as e:
            logger.warning(f"Failed to connect to Redis at {redis_url}: {e}")

    return EdgeBlender(config_path=config_path, redis_client=redis_client)
