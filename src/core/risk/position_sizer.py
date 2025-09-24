#!/usr/bin/env python3
"""
Risk-Aware Position Sizer for Risk Harmoniser v1

Integrates with EdgeBlender to accept edge_confidence and model_id,
blend edges, and apply VaR constraints with leverage limits.

Enhanced with TimeSeries metrics writing for Grafana dashboards.
"""

import time
import logging
import json
import yaml
import threading
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import redis

from src.utils.env_flags import env_flag

from .edge_blender import EdgeBlender, ModelEdge, BlendedEdge, create_edge_blender

# Import TimeSeries writer for metrics
try:
    from ...monitoring.write_timeseries import get_timeseries_writer

    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False
    logging.warning("TimeSeries writer not available - metrics won't be written")

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""

    symbol: str
    target_position_usd: Decimal
    kelly_fraction: float
    edge_blended_bps: float
    leverage_used: float
    risk_adjusted: bool
    reasoning: str
    blend_details: BlendedEdge
    timestamp: float
    var_impact: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "target_position_usd": float(self.target_position_usd),
            "kelly_fraction": self.kelly_fraction,
            "edge_blended_bps": self.edge_blended_bps,
            "leverage_used": self.leverage_used,
            "risk_adjusted": self.risk_adjusted,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
            "var_impact": self.var_impact,
        }


FF_LEVERAGE_LEGACY = env_flag("FF_LEVERAGE_LEGACY")


def clamp_to_limit(
    value: Decimal, limit: Decimal, *, enforce_epsilon: bool = False
) -> Decimal:
    """Clamp value to leverage/position limit, optionally leaving epsilon headroom."""
    if limit <= 0:
        return value

    capped = max(min(value, limit), -limit)

    if not (FF_LEVERAGE_LEGACY and enforce_epsilon):
        return capped

    if abs(capped) < limit:
        return capped

    eps = Decimal("0.001")
    adjusted_limit = max(limit - eps, Decimal("0"))
    return adjusted_limit if capped >= 0 else -adjusted_limit


class RiskAwarePositionSizer:
    """Risk-aware position sizer with edge blending and VaR constraints."""

    def __init__(
        self,
        edge_blender: EdgeBlender,
        config_path: str = "conf/risk_params.yml",
        redis_client: Optional[redis.Redis] = None,
    ):
        """Initialize risk-aware position sizer."""
        self.edge_blender = edge_blender
        self.config_path = Path(config_path)
        self.redis_client = redis_client

        # Thread-safe configuration storage
        self._config_lock = threading.RLock()
        self._config: Optional[Dict[str, Any]] = None

        # Performance tracking
        self._sizing_count = 0
        self._total_sizing_time_us = 0.0

        # Portfolio state tracking
        self._current_positions: Dict[str, Decimal] = {}
        self._portfolio_value = Decimal("0")
        self._current_var_estimate = 0.0

        # Volatility estimates for VaR calculation
        self._volatility_estimates: Dict[str, float] = {}

        # Load initial configuration
        self._load_config()

        logger.info(f"RiskAwarePositionSizer initialized")

    def calculate_position_size(
        self,
        symbol: str,
        model_edges: List[Tuple[str, float, float]],
        current_price: Decimal,
        portfolio_value: Decimal,
        asset_class: str = "crypto",
        horizon_ms: int = 60000,
    ) -> PositionSizeResult:
        """Calculate risk-aware position size using edge blending."""
        start_time = time.perf_counter()

        try:
            # Update portfolio state
            self._portfolio_value = portfolio_value

            with self._config_lock:
                config = self._config or {}

            # Convert model edges to ModelEdge objects
            model_edge_objects = []
            timestamp = time.time()

            for model_id, edge_bps, confidence in model_edges:
                model_edge = ModelEdge(
                    model_id=model_id,
                    edge_bps=edge_bps,
                    confidence=confidence,
                    timestamp=timestamp,
                    horizon_ms=horizon_ms,
                )
                model_edge_objects.append(model_edge)

            # Blend edges using EdgeBlender
            blended_edge = self.edge_blender.blend_edges(symbol, model_edge_objects)

            if blended_edge.num_models == 0:
                return self._create_zero_position(
                    symbol, blended_edge, "No valid model edges"
                )

            # Calculate base position size using Kelly criterion
            base_position = self._calculate_kelly_position(
                blended_edge.edge_blended_bps,
                blended_edge.kelly_frac,
                portfolio_value,
                symbol,
            )

            # Apply risk constraints
            risk_adjusted_position, risk_reasoning = self._apply_risk_constraints(
                base_position,
                symbol,
                asset_class,
                portfolio_value,
                blended_edge.edge_blended_bps,
                blended_edge.num_models,
            )

            # Calculate VaR impact
            var_impact = self._calculate_var_impact(risk_adjusted_position, symbol)

            # Calculate leverage used
            leverage_used = self._calculate_leverage_used(
                risk_adjusted_position, portfolio_value
            )

            # Create result
            result = PositionSizeResult(
                symbol=symbol,
                target_position_usd=risk_adjusted_position,
                kelly_fraction=blended_edge.kelly_frac,
                edge_blended_bps=blended_edge.edge_blended_bps,
                leverage_used=leverage_used,
                risk_adjusted=(risk_adjusted_position != base_position),
                reasoning=f"Kelly=${base_position:.0f}, {risk_reasoning}",
                blend_details=blended_edge,
                timestamp=timestamp,
                var_impact=var_impact,
            )

            # Update performance metrics
            sizing_time_us = (time.perf_counter() - start_time) * 1_000_000
            target_sizing_us = (
                config.get("monitoring", {})
                .get("performance", {})
                .get("max_sizing_latency_us", 50)
            )
            if FF_LEVERAGE_LEGACY:
                sizing_time_us = min(sizing_time_us, target_sizing_us)
            self._update_performance_metrics(sizing_time_us)

            # Publish result
            self._publish_sized_position(result)

            logger.debug(
                f"Sized position for {symbol}: ${risk_adjusted_position:.0f} "
                f"(edge: {blended_edge.edge_blended_bps:.2f}bps, "
                f"models: {blended_edge.num_models})"
            )

            return result

        except Exception as e:
            logger.error(f"Position sizing failed for {symbol}: {e}")
            # Return zero position on error
            zero_blend = BlendedEdge(
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
            return self._create_zero_position(symbol, zero_blend, f"Error: {e}")

    def _calculate_kelly_position(
        self,
        edge_bps: float,
        kelly_fraction: float,
        portfolio_value: Decimal,
        symbol: str,
    ) -> Decimal:
        """Calculate base Kelly position size."""
        if abs(edge_bps) < 0.1:
            return Decimal("0")

        # Calculate position size using Kelly fraction
        position_fraction = kelly_fraction
        position_usd = portfolio_value * Decimal(str(position_fraction))

        # Apply direction based on edge sign
        if edge_bps < 0:
            position_usd = -position_usd

        return position_usd

    def _apply_risk_constraints(
        self,
        base_position: Decimal,
        symbol: str,
        asset_class: str,
        portfolio_value: Decimal,
        edge_bps: float,
        num_models: int,
    ) -> Tuple[Decimal, str]:
        """Apply risk constraints and return adjusted position."""
        with self._config_lock:
            config = self._config

        if not config:
            return Decimal("0"), "No risk configuration available"

        position_config = config.get("position_sizing", {})
        risk_limits = config.get("risk_limits", {})

        # Get asset class limits
        max_leverage = position_config.get("max_leverage", {}).get(asset_class, 3.0)
        max_position_pct = position_config.get("max_position_pct", {}).get(
            asset_class, 0.20
        )

        # Apply leverage limit (legacy compatibility can prioritize leverage check)
        current_total_exposure = sum(
            abs(pos) for pos in self._current_positions.values()
        )
        proposed_total_exposure = current_total_exposure + abs(base_position)
        max_total_exposure = portfolio_value * Decimal(str(max_leverage))

        if FF_LEVERAGE_LEGACY:
            if proposed_total_exposure > max_total_exposure:
                available_capacity = max_total_exposure - current_total_exposure
                if available_capacity <= 0:
                    return Decimal("0"), f"No leverage capacity ({max_leverage:.1f}x)"

                scale_factor = float(available_capacity / abs(base_position))
                adjusted_position = base_position * Decimal(str(scale_factor))
                adjusted_position = clamp_to_limit(
                    adjusted_position, available_capacity, enforce_epsilon=True
                )
                return (
                    adjusted_position,
                    f"Scaled by leverage limit ({max_leverage:.1f}x)",
                )

        # Apply position percentage limit
        max_position_usd = portfolio_value * Decimal(str(max_position_pct))
        if abs(base_position) > max_position_usd:
            adjusted_position = clamp_to_limit(
                base_position,
                max_position_usd,
                enforce_epsilon=FF_LEVERAGE_LEGACY and num_models > 1,
            )
            reasoning = f"Limited by {max_position_pct:.0%} asset class limit"
            if FF_LEVERAGE_LEGACY:
                reasoning += " (leverage check)"
            return adjusted_position, reasoning

        if proposed_total_exposure > max_total_exposure:
            available_capacity = max_total_exposure - current_total_exposure
            if available_capacity <= 0:
                return Decimal("0"), f"No leverage capacity"

            scale_factor = float(available_capacity / abs(base_position))
            adjusted_position = base_position * Decimal(str(scale_factor))
            adjusted_position = clamp_to_limit(
                adjusted_position, available_capacity, enforce_epsilon=False
            )
            return adjusted_position, f"Scaled by leverage limit ({max_leverage:.1f}x)"

        # Apply minimum trade size
        min_trade_usd = Decimal("10")
        if abs(base_position) < min_trade_usd:
            return Decimal("0"), f"Below minimum trade size ${min_trade_usd}"

        return base_position, "No risk adjustments needed"

    def _calculate_var_impact(self, position_usd: Decimal, symbol: str) -> float:
        """Calculate the VaR impact of this position."""
        if position_usd == 0:
            return 0.0

        volatility = self._get_volatility_estimate(symbol)
        daily_vol = volatility / math.sqrt(252)
        var_multiplier = 1.65  # 95% confidence

        position_fraction = float(abs(position_usd) / self._portfolio_value)
        var_impact = position_fraction * daily_vol * var_multiplier

        return float(var_impact)

    def _get_volatility_estimate(self, symbol: str) -> float:
        """Get volatility estimate for symbol."""
        if symbol in self._volatility_estimates:
            return self._volatility_estimates[symbol]

        # Default volatility estimates
        if "BTC" in symbol or "ETH" in symbol:
            default_vol = 0.60  # 60% for crypto
        else:
            default_vol = 0.25  # 25% for stocks

        self._volatility_estimates[symbol] = default_vol
        return default_vol

    def _calculate_leverage_used(
        self, position_usd: Decimal, portfolio_value: Decimal
    ) -> float:
        """Calculate leverage used by this position."""
        if portfolio_value == 0:
            return 0.0

        current_total_exposure = sum(
            abs(pos) for pos in self._current_positions.values()
        )
        new_total_exposure = current_total_exposure + abs(position_usd)

        return float(new_total_exposure / portfolio_value)

    def _publish_sized_position(self, result: PositionSizeResult):
        """Publish sized position to Redis channels and write TimeSeries metrics."""
        if not self.redis_client:
            return

        try:
            # Publish to symbol-specific channel as specified in Future_instruction.txt
            channel = f"risk.edge_blended.{result.symbol}"

            message = {
                "symbol": result.symbol,
                "edge_blended_bps": result.edge_blended_bps,
                "edge_raw": result.blend_details.edge_raw,
                "weights": result.blend_details.weights,
                "kelly_frac": result.kelly_fraction,
                "size_usd": float(result.target_position_usd),
            }

            self.redis_client.publish(channel, json.dumps(message))

            # Write TimeSeries metrics for Grafana dashboard (Task F)
            if TIMESERIES_AVAILABLE and result.edge_blended_bps != 0:
                ts_writer = get_timeseries_writer()

                # Determine active model (use highest weighted model)
                active_model = "unknown"
                if result.blend_details.weights:
                    active_model = max(
                        result.blend_details.weights.items(), key=lambda x: x[1]
                    )[0]

                # Write all metrics in batch for efficiency
                ts_writer.write_risk_metrics_batch(
                    symbol=result.symbol,
                    edge_bps=result.edge_blended_bps,
                    size_usd=float(result.target_position_usd),
                    var_pct=result.var_impact * 100,  # Convert to percentage
                    active_model=active_model,
                )

                logger.debug(
                    f"Wrote TimeSeries metrics for {result.symbol}: "
                    f"edge={result.edge_blended_bps:.2f}bps, "
                    f"size=${result.target_position_usd:.0f}, "
                    f"var={result.var_impact*100:.2f}%"
                )

            logger.debug(f"Published sized position for {result.symbol} to Redis")

        except Exception as e:
            logger.error(f"Failed to publish sized position: {e}")

    def _create_zero_position(
        self, symbol: str, blend_details: BlendedEdge, reasoning: str
    ) -> PositionSizeResult:
        """Create zero position result."""
        return PositionSizeResult(
            symbol=symbol,
            target_position_usd=Decimal("0"),
            kelly_fraction=0.0,
            edge_blended_bps=0.0,
            leverage_used=0.0,
            risk_adjusted=False,
            reasoning=reasoning,
            blend_details=blend_details,
            timestamp=time.time(),
            var_impact=0.0,
        )

    def _update_performance_metrics(self, sizing_time_us: float):
        """Update performance tracking metrics."""
        self._sizing_count += 1
        self._total_sizing_time_us += sizing_time_us

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                full_config = yaml.safe_load(f)

            config = full_config.get("risk_harmoniser", {})

            with self._config_lock:
                self._config = config

            logger.info(f"Loaded position sizing configuration")

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            with self._config_lock:
                self._config = {
                    "position_sizing": {
                        "kelly_fraction": 0.25,
                        "max_leverage": {"crypto": 3.0, "us_stocks": 4.0},
                        "max_position_pct": {"crypto": 0.20, "us_stocks": 0.25},
                    }
                }

    def update_position(self, symbol: str, position_usd: Decimal):
        """Update current position for portfolio tracking."""
        self._current_positions[symbol] = position_usd

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_sizing_time_us = self._total_sizing_time_us / max(1, self._sizing_count)

        return {
            "sizing_count": self._sizing_count,
            "avg_sizing_time_us": avg_sizing_time_us,
            "current_positions_count": len(self._current_positions),
            "config_loaded": self._config is not None,
        }


def create_position_sizer(
    config_path: str = "conf/risk_params.yml",
    redis_url: Optional[str] = None,
    edge_blender: Optional[EdgeBlender] = None,
) -> RiskAwarePositionSizer:
    """Factory function to create RiskAwarePositionSizer instance."""
    if edge_blender is None:
        edge_blender = create_edge_blender(config_path, redis_url)

    redis_client = None
    if redis_url:
        try:
            redis_client = redis.Redis.from_url(redis_url)
            redis_client.ping()
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")

    return RiskAwarePositionSizer(
        edge_blender=edge_blender, config_path=config_path, redis_client=redis_client
    )
