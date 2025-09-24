#!/usr/bin/env python3
"""
Async Spot-Perp Basis Carry Strategy

High-performance async version of the basis carry strategy with batched Redis operations,
ONNX inference support, and enhanced risk management.
"""

import os
import sys
import json
import time
import math
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.aredis import (
    get_redis,
    check_feature_flag,
    get_batch_writer,
    publish_metrics_batch,
    publish_trade_event,
    set_metric,
    get_market_data,
    get_config_value,
    set_config_value,
)

try:
    from src.models.onnx_runtime_runner import run_inference, get_model_stats

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("basis_carry_async")


class AsyncSpotPerpBasisCarryStrategy:
    """High-performance async basis carry strategy."""

    def __init__(self):
        """Initialize async basis carry strategy."""

        # Strategy configuration with feature flags
        self.config = {
            "symbols": ["BTC", "ETH", "SOL"],
            "max_gross_per_strategy": 0.20,
            "entry_basis_threshold": -5.0,
            "entry_funding_threshold": 0.10,
            "exit_basis_threshold": 1.0,
            "stop_loss_pct": -0.006,
            "hedge_divergence_limit": 15.0,
            "timeout_hours": 8,
            "tick_interval": 10,
            "min_notional_usd": 100,
            "max_notional_usd": 50000,
            # New async-specific config
            "batch_flush_interval": 0.1,
            "max_concurrent_operations": 10,
            "use_onnx_inference": False,
            "dynamic_hedge_ratio": False,
            "expected_edge_sizing": False,
            "markout_guard": False,
        }

        # Track active positions
        self.active_positions = {}
        self.position_history = []
        self.total_pnl = 0.0
        self.total_trades = 0
        self.win_trades = 0

        # Async-specific state
        self._redis = None
        self._batch_writer = None
        self._running = False
        self._tasks = set()

        # Performance metrics
        self.performance_stats = {
            "async_operations": 0,
            "batch_operations": 0,
            "redis_latency_ms": 0.0,
            "inference_latency_ms": 0.0,
            "tick_duration_ms": 0.0,
        }

        logger.info("📊 Async Spot-Perp Basis Carry Strategy initialized")

    async def initialize(self) -> bool:
        """Initialize async resources."""
        try:
            # Get Redis connection
            self._redis = await get_redis()
            self._batch_writer = await get_batch_writer()

            # Load configuration from Redis
            await self._load_config()

            # Initialize feature flags
            await self._check_feature_flags()

            logger.info("Async strategy initialization complete")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize async strategy: {e}")
            return False

    async def _load_config(self):
        """Load configuration from Redis."""
        try:
            # Load dynamic config values
            entry_min_bps = await get_config_value("basis:entry_min_bps", 6.0, float)
            slip_est_bps = await get_config_value("basis:slip_est_bps", 2.0, float)
            markout_config = await get_config_value("basis:markout_guard", {}, dict)

            # Update config
            self.config.update(
                {
                    "entry_min_bps": entry_min_bps,
                    "slip_est_bps": slip_est_bps,
                    "markout_guard_config": markout_config,
                }
            )

            logger.debug(
                f"Loaded config: entry_min_bps={entry_min_bps}, slip_est_bps={slip_est_bps}"
            )

        except Exception as e:
            logger.error(f"Error loading config: {e}")

    async def _check_feature_flags(self):
        """Check and update feature flags."""
        try:
            self.config.update(
                {
                    "use_onnx_inference": await check_feature_flag("onnx_infer", False),
                    "dynamic_hedge_ratio": await check_feature_flag(
                        "dynamic_hedge", False
                    ),
                    "expected_edge_sizing": await check_feature_flag(
                        "expected_edge", False
                    ),
                    "markout_guard": await check_feature_flag("markout_guard", False),
                    "async_redis": await check_feature_flag("async_redis", True),
                }
            )

            logger.info(
                f"Feature flags: ONNX={self.config['use_onnx_inference']}, "
                f"Dynamic hedge={self.config['dynamic_hedge_ratio']}, "
                f"Expected edge={self.config['expected_edge_sizing']}"
            )

        except Exception as e:
            logger.error(f"Error checking feature flags: {e}")

    def compute_basis_bps(self, spot_px: float, perp_px: float) -> float:
        """Compute basis in basis points."""
        try:
            if spot_px <= 0 or perp_px <= 0:
                return 0.0
            return (perp_px - spot_px) / spot_px * 10000
        except Exception as e:
            logger.error(f"Error computing basis: {e}")
            return 0.0

    async def get_market_data_async(self, symbol: str) -> Dict[str, float]:
        """Get market data with async Redis operations."""
        try:
            start_time = time.time()

            # Use async market data helper
            market_data = await get_market_data(symbol)

            # Fill in missing data with mocks for demo
            if not market_data.get("spot_price"):
                mock_spots = {"BTC": 97650.0, "ETH": 3520.0, "SOL": 185.5}
                market_data["spot_price"] = mock_spots.get(symbol, 100.0)

            if not market_data.get("perp_price"):
                spot_px = market_data["spot_price"]
                basis_variance = np.random.uniform(-8, 3)
                market_data["perp_price"] = spot_px * (1 + basis_variance / 10000)

            if not market_data.get("funding_annual"):
                mock_funding = {"BTC": 0.08, "ETH": 0.12, "SOL": 0.15}
                market_data["funding_annual"] = mock_funding.get(symbol, 0.10)

            # Calculate basis
            market_data["basis_bps"] = self.compute_basis_bps(
                market_data["spot_price"], market_data["perp_price"]
            )

            # Update performance stats
            duration = time.time() - start_time
            self.performance_stats["redis_latency_ms"] = duration * 1000

            return market_data

        except Exception as e:
            logger.error(f"Error getting async market data for {symbol}: {e}")
            return {}

    async def get_dynamic_hedge_ratio(self, symbol: str) -> float:
        """Get dynamic hedge ratio using GLS or fallback to Kalman."""
        try:
            if not self.config.get("dynamic_hedge_ratio"):
                return 1.0

            # Try to get calibrated beta from GLS calculator
            if hasattr(self._redis, "hget"):
                beta_data = await self._redis.hget(f"basis:beta:{symbol}", "beta")
                if beta_data:
                    return float(beta_data)

            # Fallback to existing Kalman beta
            if hasattr(self._redis, "hget"):
                calib_data = await self._redis.hgetall(f"basis:calib:{symbol}")
                if calib_data and "beta" in calib_data:
                    return float(calib_data["beta"])

            return 1.0

        except Exception as e:
            logger.error(f"Error getting dynamic hedge ratio for {symbol}: {e}")
            return 1.0

    async def compute_expected_edge(
        self, symbol: str, basis_bps: float, funding_annual: float
    ) -> float:
        """Compute expected edge after costs."""
        try:
            if not self.config.get("expected_edge_sizing"):
                return basis_bps  # Fallback to simple basis

            # Calculate net edge = basis + funding - fees - estimated slippage
            funding_bps_daily = funding_annual * 365 / 10000
            fees_bps = 0.08  # Typical maker/taker fees
            est_slippage_bps = self.config.get("slip_est_bps", 2.0)

            net_edge_bps = (
                abs(basis_bps) + funding_bps_daily - fees_bps - est_slippage_bps
            )

            logger.debug(
                f"{symbol} expected edge: basis={basis_bps:.1f}, "
                f"funding={funding_bps_daily:.1f}, fees={fees_bps:.1f}, "
                f"slip={est_slippage_bps:.1f} -> net={net_edge_bps:.1f}bps"
            )

            return net_edge_bps

        except Exception as e:
            logger.error(f"Error computing expected edge for {symbol}: {e}")
            return basis_bps

    async def check_markout_guard(self, symbol: str) -> Tuple[bool, float]:
        """Check markout guard and return (should_block, threshold_boost)."""
        try:
            if not self.config.get("markout_guard"):
                return False, 0.0

            guard_config = self.config.get("markout_guard_config", {})
            if not guard_config:
                return False, 0.0

            # Check for TTL boost key
            boost_key = f"basis:guard:thresh_boost:{symbol}"
            if hasattr(self._redis, "get"):
                boost_value = await self._redis.get(boost_key)
                if boost_value:
                    boost_bps = float(boost_value)
                    logger.warning(
                        f"Markout guard active for {symbol}: +{boost_bps}bps threshold boost"
                    )
                    return True, boost_bps

            return False, 0.0

        except Exception as e:
            logger.error(f"Error checking markout guard for {symbol}: {e}")
            return False, 0.0

    async def enhanced_entry_signal(
        self, basis_bps: float, funding_ann: float, symbol: str = "BTC"
    ) -> Tuple[bool, str]:
        """Enhanced entry signal with expected edge and markout guard."""
        try:
            # Check markout guard first
            guard_blocked, threshold_boost = await self.check_markout_guard(symbol)

            # Compute expected edge
            net_edge_bps = await self.compute_expected_edge(
                symbol, basis_bps, funding_ann
            )

            # Apply threshold boost from markout guard
            entry_min_bps = self.config.get("entry_min_bps", 6.0) + threshold_boost

            # Block entry if net edge too low
            if net_edge_bps < entry_min_bps:
                return False, f"low_edge_{net_edge_bps:.1f}bps"

            # Try calibrated z-score entry
            if hasattr(self._redis, "hgetall"):
                calib_data = await self._redis.hgetall(f"basis:calib:{symbol}")
                if calib_data and "z" in calib_data:
                    z_score = float(calib_data["z"])
                    if z_score < -1.2:
                        return True, f"z_score_entry_{z_score:.2f}"

            # Fallback to basis-based logic with enhanced thresholds
            if basis_bps < self.config["entry_basis_threshold"]:
                return True, "cheap_carry_enhanced"

            if funding_ann > self.config["entry_funding_threshold"]:
                return True, "high_funding_enhanced"

            return False, "no_entry_enhanced"

        except Exception as e:
            logger.error(f"Error in enhanced entry signal: {e}")
            return False, "error"

    async def calculate_kelly_position_size(
        self, symbol: str, market_data: Dict, net_edge_bps: float
    ) -> float:
        """Calculate position size using Kelly criterion."""
        try:
            if not self.config.get("expected_edge_sizing"):
                return await self.calculate_position_size_basic(symbol, market_data)

            # Get volatility estimate
            if hasattr(self._redis, "get"):
                vol_key = f"risk:vol:{symbol}:daily"
                vol_str = await self._redis.get(vol_key)
                if vol_str:
                    daily_vol = float(vol_str)
                else:
                    daily_vol = 0.02  # Default 2% daily vol
            else:
                daily_vol = 0.02

            # Kelly fraction = edge / (volatility^2)
            edge_frac = net_edge_bps / 10000
            kelly_frac = edge_frac / (daily_vol**2)

            # Cap Kelly fraction
            kelly_frac = np.clip(
                kelly_frac, 0, self.config["max_gross_per_strategy"] / 2
            )

            # Get equity
            if hasattr(self._redis, "get"):
                equity_str = await self._redis.get("risk:equity_usd")
                equity_usd = float(equity_str) if equity_str else 100000
            else:
                equity_usd = 100000

            # Calculate notional
            notional_usd = equity_usd * kelly_frac

            # Apply min/max limits
            notional_usd = np.clip(
                notional_usd,
                self.config["min_notional_usd"],
                self.config["max_notional_usd"],
            )

            logger.debug(
                f"Kelly sizing for {symbol}: edge={edge_frac:.4f}, vol={daily_vol:.4f}, "
                f"kelly_frac={kelly_frac:.4f}, notional=${notional_usd:,.0f}"
            )

            return notional_usd

        except Exception as e:
            logger.error(f"Error in Kelly position sizing: {e}")
            return await self.calculate_position_size_basic(symbol, market_data)

    async def calculate_position_size_basic(
        self, symbol: str, market_data: Dict
    ) -> float:
        """Basic position size calculation (fallback)."""
        try:
            # Get current equity
            if hasattr(self._redis, "get"):
                equity_str = await self._redis.get("risk:equity_usd")
                equity_usd = float(equity_str) if equity_str else 100000
            else:
                equity_usd = 100000

            # Calculate max notional per strategy
            max_notional = equity_usd * self.config["max_gross_per_strategy"]

            # Check current exposure
            current_notional = sum(
                pos["notional_usd"] for pos in self.active_positions.values()
            )
            available_notional = max_notional - current_notional

            # Use portion of available notional
            target_notional = min(
                available_notional * 0.5, self.config["max_notional_usd"]
            )

            if target_notional < self.config["min_notional_usd"]:
                return 0.0

            return target_notional

        except Exception as e:
            logger.error(f"Error calculating basic position size: {e}")
            return 0.0

    async def open_trade_async(
        self, symbol: str, notional_usd: float, market_data: Dict
    ) -> Dict[str, Any]:
        """Open trade with async operations and enhanced sizing."""
        try:
            position_id = f"{symbol}_basis_{int(time.time())}"
            entry_time = time.time()

            spot_price = market_data["spot_price"]
            perp_price = market_data["perp_price"]
            basis_bps = market_data["basis_bps"]
            funding_annual = market_data["funding_annual"]

            # Get dynamic hedge ratio
            hedge_ratio = await self.get_dynamic_hedge_ratio(symbol)

            # Calculate quantities
            spot_quantity = notional_usd / spot_price
            perp_quantity = spot_quantity * hedge_ratio

            # Determine trade direction
            if basis_bps < 0:
                spot_side, perp_side = "buy", "sell"
                strategy_type = "basis_convergence"
            else:
                spot_side, perp_side = "sell", "buy"
                strategy_type = "funding_capture"

            # Create position record
            position = {
                "position_id": position_id,
                "symbol": symbol,
                "strategy_type": strategy_type,
                "entry_time": entry_time,
                "notional_usd": notional_usd,
                "spot_leg": {
                    "side": spot_side,
                    "price": spot_price,
                    "quantity": spot_quantity,
                    "notional": spot_quantity * spot_price,
                },
                "perp_leg": {
                    "side": perp_side,
                    "price": perp_price,
                    "quantity": perp_quantity,
                    "notional": perp_quantity * perp_price,
                },
                "entry_basis_bps": basis_bps,
                "entry_funding_annual": funding_annual,
                "realized_pnl": 0.0,
                "status": "open",
                "hedge_ratio": hedge_ratio,
                "enhanced_features": {
                    "dynamic_hedge": self.config.get("dynamic_hedge_ratio", False),
                    "expected_edge": self.config.get("expected_edge_sizing", False),
                },
            }

            # Store in active positions
            self.active_positions[position_id] = position

            # Async Redis operations
            await asyncio.gather(
                self._batch_writer.set_batch(
                    f"strategy:basis:{symbol}:pos", json.dumps(position, default=str)
                ),
                publish_trade_event(
                    {
                        "position_id": position_id,
                        "symbol": symbol,
                        "action": "open",
                        "strategy_type": strategy_type,
                        "notional_usd": notional_usd,
                        "entry_basis_bps": basis_bps,
                        "entry_funding_annual": funding_annual,
                        "hedge_ratio": hedge_ratio,
                    },
                    "strategy:basis:events",
                ),
            )

            # Update metrics
            self.total_trades += 1
            await self._update_metrics_batch()

            logger.info(
                f"💰 Opened {strategy_type} trade {position_id}: "
                f"{spot_side.upper()} {spot_quantity:.6f} {symbol} spot @ ${spot_price:.2f}, "
                f"{perp_side.upper()} {perp_quantity:.6f} {symbol} perp @ ${perp_price:.2f} "
                f"(basis: {basis_bps:.1f}bps, hedge_ratio: {hedge_ratio:.4f})"
            )

            return {"success": True, "position_id": position_id, "position": position}

        except Exception as e:
            logger.error(f"Error opening async trade for {symbol}: {e}")
            return {"success": False, "error": str(e)}

    async def close_trade_async(
        self, position: Dict, market_data: Dict, reason: str
    ) -> Dict[str, Any]:
        """Close trade with async operations."""
        try:
            position_id = position["position_id"]
            symbol = position["symbol"]

            # Calculate realized PnL
            realized_pnl = self.calculate_position_pnl(position, market_data)

            # Update position record
            position.update(
                {
                    "exit_time": time.time(),
                    "exit_basis_bps": market_data["basis_bps"],
                    "realized_pnl": realized_pnl,
                    "status": "closed",
                    "exit_reason": reason,
                }
            )

            # Calculate metrics
            notional_usd = position["notional_usd"]
            pnl_pct = realized_pnl / notional_usd
            holding_time = position["exit_time"] - position["entry_time"]

            # Update totals
            self.total_pnl += realized_pnl
            if realized_pnl > 0:
                self.win_trades += 1

            # Remove from active positions
            if position_id in self.active_positions:
                del self.active_positions[position_id]

            # Store closed position
            self.position_history.append(position.copy())
            if len(self.position_history) > 1000:
                self.position_history = self.position_history[-500:]

            # Async Redis operations
            await asyncio.gather(
                self._batch_writer.set_batch(
                    f"strategy:basis:{symbol}:pnl", realized_pnl
                ),
                publish_trade_event(
                    {
                        "position_id": position_id,
                        "symbol": symbol,
                        "action": "close",
                        "reason": reason,
                        "realized_pnl": realized_pnl,
                        "pnl_pct": pnl_pct,
                        "holding_time_hours": holding_time / 3600,
                    },
                    "strategy:basis:events",
                ),
            )

            # Update metrics
            await self._update_metrics_batch()

            logger.info(
                f"🎯 Closed basis trade {position_id}: "
                f"${realized_pnl:+,.2f} ({pnl_pct:+.2%}) "
                f"in {holding_time/3600:.1f}h, reason: {reason}"
            )

            return {
                "success": True,
                "position_id": position_id,
                "realized_pnl": realized_pnl,
                "pnl_pct": pnl_pct,
            }

        except Exception as e:
            logger.error(
                f"Error closing async trade {position.get('position_id')}: {e}"
            )
            return {"success": False, "error": str(e)}

    def calculate_position_pnl(self, position: Dict, current_data: Dict) -> float:
        """Calculate position PnL (same as sync version)."""
        try:
            spot_leg = position["spot_leg"]
            perp_leg = position["perp_leg"]

            current_spot_price = current_data["spot_price"]
            current_perp_price = current_data["perp_price"]

            # Calculate P&L for each leg
            if spot_leg["side"] == "buy":
                spot_pnl = spot_leg["quantity"] * (
                    current_spot_price - spot_leg["price"]
                )
            else:
                spot_pnl = spot_leg["quantity"] * (
                    spot_leg["price"] - current_spot_price
                )

            if perp_leg["side"] == "buy":
                perp_pnl = perp_leg["quantity"] * (
                    current_perp_price - perp_leg["price"]
                )
            else:
                perp_pnl = perp_leg["quantity"] * (
                    perp_leg["price"] - current_perp_price
                )

            return spot_pnl + perp_pnl

        except Exception as e:
            logger.error(f"Error calculating position PnL: {e}")
            return 0.0

    async def _update_metrics_batch(self):
        """Update metrics using batch writer."""
        try:
            metrics = {
                "basis_open_trades": len(self.active_positions),
                "basis_notional_usd": sum(
                    pos["notional_usd"] for pos in self.active_positions.values()
                ),
                "basis_pnl_usd": self.total_pnl,
                "basis_win_rate": self.win_trades / max(1, self.total_trades),
                "basis_total_trades": self.total_trades,
                **self.performance_stats,
            }

            # Batch update all metrics
            tasks = [set_metric(k, v) for k, v in metrics.items()]
            await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"Error updating metrics batch: {e}")

    async def check_risk_limits_async(self) -> Tuple[bool, str]:
        """Async risk limits check."""
        try:
            # Get risk data in parallel
            equity_task = (
                self._redis.get("risk:equity_usd")
                if hasattr(self._redis, "get")
                else None
            )
            mode_task = self._redis.get("mode") if hasattr(self._redis, "get") else None
            capital_task = (
                self._redis.get("risk:capital_effective")
                if hasattr(self._redis, "get")
                else None
            )

            if equity_task and mode_task and capital_task:
                equity_str, mode, capital_str = await asyncio.gather(
                    equity_task, mode_task, capital_task
                )
            else:
                equity_str, mode, capital_str = "100000", "auto", "0.4"

            equity_usd = float(equity_str) if equity_str else 100000
            capital_effective = float(capital_str) if capital_str else 0.4

            # Check exposure
            total_notional = sum(
                pos["notional_usd"] for pos in self.active_positions.values()
            )
            exposure_pct = total_notional / equity_usd

            if exposure_pct > self.config["max_gross_per_strategy"]:
                return False, f"gross_exposure_exceeded_{exposure_pct:.1%}"

            if mode == "halt":
                return False, "system_halt"

            if capital_effective < 0.1:
                return False, f"low_capital_allocation_{capital_effective:.1%}"

            return True, "within_limits"

        except Exception as e:
            logger.error(f"Error checking async risk limits: {e}")
            return False, "risk_check_error"

    async def tick_async(self) -> Dict[str, Any]:
        """Main async strategy tick."""
        try:
            tick_start = time.time()
            actions_taken = {"opens": 0, "closes": 0, "holds": 0, "errors": 0}

            # Reload feature flags periodically
            await self._check_feature_flags()

            # Check risk limits
            within_limits, limit_reason = await self.check_risk_limits_async()
            if not within_limits:
                logger.warning(f"⚠️ Risk limits exceeded: {limit_reason}")

                # Close positions if halted
                if "halt" in limit_reason:
                    close_tasks = []
                    for position in list(self.active_positions.values()):
                        market_data = await self.get_market_data_async(
                            position["symbol"]
                        )
                        if market_data:
                            close_tasks.append(
                                self.close_trade_async(
                                    position, market_data, "system_halt"
                                )
                            )

                    if close_tasks:
                        await asyncio.gather(*close_tasks)
                        actions_taken["closes"] += len(close_tasks)

            # Process symbols concurrently
            symbol_tasks = []
            for symbol in self.config["symbols"]:
                task = self._process_symbol_async(symbol, within_limits, actions_taken)
                symbol_tasks.append(task)

            if symbol_tasks:
                await asyncio.gather(*symbol_tasks, return_exceptions=True)

            # Update performance stats
            tick_duration = time.time() - tick_start
            self.performance_stats["tick_duration_ms"] = tick_duration * 1000
            self.performance_stats["async_operations"] += len(symbol_tasks)

            # Batch update metrics
            await self._update_metrics_batch()

            # Flush batch writer
            await self._batch_writer.flush()

            return {
                "timestamp": tick_start,
                "status": "completed",
                "actions_taken": actions_taken,
                "active_positions": len(self.active_positions),
                "total_pnl": self.total_pnl,
                "tick_duration_ms": self.performance_stats["tick_duration_ms"],
                "within_limits": within_limits,
                "limit_reason": limit_reason,
                "performance_stats": self.performance_stats.copy(),
            }

        except Exception as e:
            logger.error(f"Error in async strategy tick: {e}")
            return {"timestamp": time.time(), "status": "error", "error": str(e)}

    async def _process_symbol_async(
        self, symbol: str, within_limits: bool, actions_taken: Dict
    ) -> None:
        """Process single symbol asynchronously."""
        try:
            # Get market data
            market_data = await self.get_market_data_async(symbol)
            if not market_data:
                return

            basis_bps = market_data["basis_bps"]
            funding_annual = market_data["funding_annual"]

            # Check existing positions
            symbol_positions = [
                pos for pos in self.active_positions.values() if pos["symbol"] == symbol
            ]

            if symbol_positions:
                # Manage existing positions
                for position in symbol_positions:
                    should_exit, exit_reason = await self.exit_signal_async(
                        position, market_data
                    )

                    if should_exit:
                        result = await self.close_trade_async(
                            position, market_data, exit_reason
                        )
                        if result["success"]:
                            actions_taken["closes"] += 1
                        else:
                            actions_taken["errors"] += 1
                    else:
                        actions_taken["holds"] += 1

                        # Update unrealized PnL
                        unrealized_pnl = self.calculate_position_pnl(
                            position, market_data
                        )
                        await self._batch_writer.set_batch(
                            f"strategy:basis:{symbol}:unrealized_pnl", unrealized_pnl
                        )
            else:
                # Check for new entries
                if within_limits:
                    should_enter, entry_reason = await self.enhanced_entry_signal(
                        basis_bps, funding_annual, symbol
                    )

                    if should_enter:
                        # Calculate expected edge for sizing
                        net_edge_bps = await self.compute_expected_edge(
                            symbol, basis_bps, funding_annual
                        )

                        if net_edge_bps > 0:
                            notional_usd = await self.calculate_kelly_position_size(
                                symbol, market_data, net_edge_bps
                            )

                            if notional_usd > 0:
                                result = await self.open_trade_async(
                                    symbol, notional_usd, market_data
                                )
                                if result["success"]:
                                    actions_taken["opens"] += 1
                                else:
                                    actions_taken["errors"] += 1

        except Exception as e:
            logger.error(f"Error processing {symbol} in async tick: {e}")
            actions_taken["errors"] += 1

    async def exit_signal_async(
        self, position: Dict, current_data: Dict
    ) -> Tuple[bool, str]:
        """Async version of exit signal check."""
        try:
            # Use same logic as sync version but with async Redis calls
            position_id = position["position_id"]
            symbol = position["symbol"]
            entry_time = position["entry_time"]
            entry_basis_bps = position["entry_basis_bps"]
            current_basis_bps = current_data["basis_bps"]

            # Calculate unrealized PnL
            unrealized_pnl = self.calculate_position_pnl(position, current_data)
            unrealized_pnl_pct = unrealized_pnl / position["notional_usd"]

            # Try calibrated z-score exit
            if hasattr(self._redis, "hgetall"):
                calib_data = await self._redis.hgetall(f"basis:calib:{symbol}")
                if calib_data and "z" in calib_data:
                    z_score = float(calib_data["z"])
                    if abs(z_score) < 0.3:
                        return True, f"z_score_revert_{z_score:.2f}"

            # Standard exit conditions
            if abs(current_basis_bps) < self.config["exit_basis_threshold"]:
                return True, "basis_mean_revert"

            if unrealized_pnl_pct < self.config["stop_loss_pct"]:
                return True, "stop_loss"

            hedge_divergence = abs(current_basis_bps - entry_basis_bps)
            if hedge_divergence > self.config["hedge_divergence_limit"]:
                return True, "hedge_divergence"

            position_age = time.time() - entry_time
            timeout_seconds = self.config["timeout_hours"] * 3600
            if position_age > timeout_seconds:
                return True, "timeout"

            return False, "hold"

        except Exception as e:
            logger.error(f"Error checking async exit signal: {e}")
            return True, "error_exit"

    async def run_async_loop(self):
        """Main async strategy loop."""
        logger.info("🚀 Starting async basis carry strategy loop")
        self._running = True

        try:
            # Initialize
            if not await self.initialize():
                logger.error("Failed to initialize async strategy")
                return

            while self._running:
                try:
                    # Run strategy tick
                    result = await self.tick_async()

                    if result["status"] == "completed":
                        total_actions = sum(result["actions_taken"].values())
                        if total_actions > 0 or len(self.active_positions) > 0:
                            logger.debug(
                                f"📊 Async tick: {total_actions} actions, "
                                f"{len(self.active_positions)} positions, "
                                f"P&L: ${self.total_pnl:+,.2f}, "
                                f"duration: {result['tick_duration_ms']:.1f}ms"
                            )

                    # Sleep until next tick
                    await asyncio.sleep(self.config["tick_interval"])

                except Exception as e:
                    logger.error(f"Error in async strategy loop: {e}")
                    await asyncio.sleep(30)

        except KeyboardInterrupt:
            logger.info("Async basis carry strategy stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in async strategy loop: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down async basis carry strategy...")
        self._running = False

        try:
            # Close all positions
            if self.active_positions:
                logger.info("Closing all active positions...")
                close_tasks = []
                for position in list(self.active_positions.values()):
                    market_data = await self.get_market_data_async(position["symbol"])
                    if market_data:
                        close_tasks.append(
                            self.close_trade_async(
                                position, market_data, "strategy_shutdown"
                            )
                        )

                if close_tasks:
                    await asyncio.gather(*close_tasks)

            # Flush final batch
            if self._batch_writer:
                await self._batch_writer.flush()

            # Cancel running tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()

            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)

            logger.info("Async strategy shutdown complete")

        except Exception as e:
            logger.error(f"Error during async shutdown: {e}")


async def main():
    """Main entry point for async basis carry strategy."""
    import argparse

    parser = argparse.ArgumentParser(description="Async Spot-Perp Basis Carry Strategy")
    parser.add_argument("--run", action="store_true", help="Run async strategy loop")
    parser.add_argument("--tick", action="store_true", help="Run single async tick")

    args = parser.parse_args()

    # Create async strategy
    strategy = AsyncSpotPerpBasisCarryStrategy()

    if args.tick:
        # Run single tick
        await strategy.initialize()
        result = await strategy.tick_async()
        print(json.dumps(result, indent=2, default=str))
        await strategy.shutdown()
        return

    if args.run:
        # Run async loop
        await strategy.run_async_loop()
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
