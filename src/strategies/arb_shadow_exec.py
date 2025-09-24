#!/usr/bin/env python3
"""
Arbitrage Shadow Executor

Shadow execution system for cross-exchange arbitrage opportunities.
Executes arbitrage trades at reduced size to measure real-world performance
before promoting to live execution.
"""

import os
import sys
import time
import asyncio
import logging
import json
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.utils.aredis import (
        get_redis,
        get_batch_writer,
        set_metric,
        get_config_value,
        publish_trade_event,
    )

    ASYNC_REDIS_AVAILABLE = True
except ImportError:
    ASYNC_REDIS_AVAILABLE = False
    import redis

from src.strategies.arb_scanner import ArbitrageOpportunity, ArbitrageScanner

logger = logging.getLogger("arb_shadow_exec")


class ExecutionStatus(Enum):
    """Execution status for arbitrage trades."""

    PENDING = "pending"
    PARTIAL = "partial"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ShadowTrade:
    """Represents a shadow arbitrage trade."""

    trade_id: str
    opportunity: ArbitrageOpportunity
    shadow_size: float  # Actual shadow trade size
    shadow_size_pct: float  # Percentage of full opportunity size
    status: ExecutionStatus
    created_time: float

    # Execution details
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    buy_fill_price: Optional[float] = None
    sell_fill_price: Optional[float] = None
    buy_fill_size: Optional[float] = None
    sell_fill_size: Optional[float] = None

    # Performance metrics
    realized_spread_bps: Optional[float] = None
    realized_profit_usd: Optional[float] = None
    execution_time_ms: Optional[float] = None
    slippage_bps: Optional[float] = None

    # Market impact measurement
    pre_trade_bid: Optional[float] = None
    pre_trade_ask: Optional[float] = None
    post_trade_bid: Optional[float] = None
    post_trade_ask: Optional[float] = None
    market_impact_bps: Optional[float] = None

    completion_time: Optional[float] = None
    error_message: Optional[str] = None


class ShadowExecutionEngine:
    """
    Shadow execution engine for arbitrage opportunities.

    Executes arbitrage trades at reduced size to measure performance
    without significant capital risk or market impact.
    """

    def __init__(self, scanner: ArbitrageScanner):
        """
        Initialize shadow execution engine.

        Args:
            scanner: Arbitrage scanner instance
        """
        self.scanner = scanner
        self.active_trades: Dict[str, ShadowTrade] = {}
        self.completed_trades: deque = deque(maxlen=10000)

        # Configuration
        self.config = {
            "shadow_size_pct": 0.01,  # 1% of opportunity size
            "min_shadow_size_usd": 50,  # Minimum shadow trade size
            "max_shadow_size_usd": 1000,  # Maximum shadow trade size
            "max_concurrent_trades": 5,  # Max concurrent shadow trades
            "opportunity_delay_seconds": 2,  # Delay before executing
            "execution_timeout_seconds": 30,  # Max execution time
            "min_confidence": 0.8,  # Minimum confidence to execute
            "min_spread_bps": 5.0,  # Minimum spread for shadow execution
            "enabled": True,  # Shadow execution enabled
        }

        # Performance tracking
        self.performance_stats = {
            "total_opportunities": 0,
            "executed_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "avg_realized_spread_bps": 0.0,
            "avg_slippage_bps": 0.0,
            "avg_execution_time_ms": 0.0,
            "success_rate": 0.0,
            "total_profit_usd": 0.0,
            "sharpe_ratio": 0.0,
        }

        # A/B testing metrics for promotion
        self.ab_metrics = {
            "recent_trades": deque(maxlen=100),
            "fill_rate": 0.0,
            "avg_improvement_bps": 0.0,
            "consistency_score": 0.0,
        }

        logger.info("Initialized arbitrage shadow executor")
        logger.info(f"  Shadow size: {self.config['shadow_size_pct']:.1%}")
        logger.info(f"  Max concurrent: {self.config['max_concurrent_trades']}")

    async def start_monitoring(self):
        """Start monitoring for arbitrage opportunities."""
        logger.info("🕵️ Starting arbitrage shadow execution monitoring")

        try:
            while self.config["enabled"]:
                await self._process_opportunities()
                await self._monitor_active_trades()
                await self._update_performance_metrics()
                await asyncio.sleep(1)  # Check every second

        except Exception as e:
            logger.error(f"Error in shadow execution monitoring: {e}")

    async def _process_opportunities(self):
        """Process new arbitrage opportunities."""
        try:
            for symbol in self.scanner.symbols:
                opportunities = self.scanner.get_top_opportunities(symbol, 3)

                for opp in opportunities:
                    if await self._should_execute_shadow(opp):
                        await self._execute_shadow_trade(opp)

        except Exception as e:
            logger.error(f"Error processing opportunities: {e}")

    async def _should_execute_shadow(self, opportunity: ArbitrageOpportunity) -> bool:
        """Determine if we should execute a shadow trade for this opportunity."""
        try:
            # Check if shadow execution is enabled
            if not self.config["enabled"]:
                return False

            # Check if we're at max concurrent trades
            if len(self.active_trades) >= self.config["max_concurrent_trades"]:
                return False

            # Check confidence threshold
            if opportunity.confidence < self.config["min_confidence"]:
                return False

            # Check spread threshold
            if opportunity.risk_adjusted_bps < self.config["min_spread_bps"]:
                return False

            # Check if we've already traded this opportunity recently
            trade_key = f"{opportunity.symbol}_{opportunity.buy_exchange}_{opportunity.sell_exchange}"
            recent_trades = [
                t
                for t in self.active_trades.values()
                if f"{t.opportunity.symbol}_{t.opportunity.buy_exchange}_{t.opportunity.sell_exchange}"
                == trade_key
            ]
            if recent_trades:
                return False

            # Check opportunity age (don't trade stale opportunities)
            age = time.time() - opportunity.timestamp
            if age > 10:  # 10 seconds
                return False

            # Check feature flag
            feature_enabled = await self._check_feature_flag("arb_shadow_exec", True)
            if not feature_enabled:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking shadow execution criteria: {e}")
            return False

    async def _check_feature_flag(self, flag: str, default: bool) -> bool:
        """Check feature flag from Redis."""
        try:
            if ASYNC_REDIS_AVAILABLE:
                redis = await get_redis()
                if hasattr(redis, "get"):
                    value = await redis.get(f"features:{flag}")
                    if value:
                        return str(value).lower() in ("1", "true", "yes", "on")

            return default

        except Exception as e:
            logger.error(f"Error checking feature flag: {e}")
            return default

    async def _execute_shadow_trade(self, opportunity: ArbitrageOpportunity):
        """Execute a shadow arbitrage trade."""
        try:
            # Calculate shadow trade size
            full_size = opportunity.max_size
            shadow_size = min(
                full_size * self.config["shadow_size_pct"],
                self.config["max_shadow_size_usd"] / opportunity.buy_price,
            )

            # Check minimum size
            shadow_value = shadow_size * opportunity.buy_price
            if shadow_value < self.config["min_shadow_size_usd"]:
                logger.debug(f"Shadow trade too small: ${shadow_value:.2f}")
                return

            # Create shadow trade
            trade_id = f"shadow_{opportunity.symbol}_{int(time.time() * 1000)}"
            shadow_trade = ShadowTrade(
                trade_id=trade_id,
                opportunity=opportunity,
                shadow_size=shadow_size,
                shadow_size_pct=self.config["shadow_size_pct"],
                status=ExecutionStatus.PENDING,
                created_time=time.time(),
            )

            # Store pre-trade market data for impact measurement
            await self._capture_pre_trade_data(shadow_trade)

            # Add to active trades
            self.active_trades[trade_id] = shadow_trade

            # Simulate execution (in real implementation, would place actual orders)
            await self._simulate_execution(shadow_trade)

            self.performance_stats["total_opportunities"] += 1

            logger.info(
                f"🧪 Shadow arbitrage: {opportunity.symbol} {opportunity.buy_exchange}→{opportunity.sell_exchange} "
                f"{shadow_size:.6f} @ {opportunity.risk_adjusted_bps:.1f}bps "
                f"(${shadow_value:.0f} value)"
            )

        except Exception as e:
            logger.error(f"Error executing shadow trade: {e}")

    async def _capture_pre_trade_data(self, shadow_trade: ShadowTrade):
        """Capture market data before trade execution."""
        try:
            symbol = shadow_trade.opportunity.symbol
            buy_exchange = shadow_trade.opportunity.buy_exchange
            sell_exchange = shadow_trade.opportunity.sell_exchange

            # Get current market data
            buy_data = self.scanner.market_data.get((symbol, buy_exchange))
            sell_data = self.scanner.market_data.get((symbol, sell_exchange))

            if buy_data and sell_data:
                shadow_trade.pre_trade_bid = buy_data.bid_price
                shadow_trade.pre_trade_ask = buy_data.ask_price
                # Store sell side data as well for comprehensive impact measurement

        except Exception as e:
            logger.error(f"Error capturing pre-trade data: {e}")

    async def _simulate_execution(self, shadow_trade: ShadowTrade):
        """
        Simulate trade execution with realistic fills and slippage.

        In a real implementation, this would:
        1. Place actual orders on exchanges at reduced size
        2. Monitor fills and partial fills
        3. Handle rejections and timeouts
        4. Measure real slippage and market impact
        """
        try:
            start_time = time.time()
            opp = shadow_trade.opportunity

            # Simulate execution delay
            await asyncio.sleep(0.1 + np.random.uniform(0, 0.2))  # 100-300ms

            # Simulate slippage (price movement during execution)
            base_slippage = np.random.uniform(0.5, 2.0)  # 0.5-2 bps base slippage
            market_impact = shadow_trade.shadow_size * 0.1  # Size-dependent impact
            total_slippage = base_slippage + market_impact

            # Simulate fills with slippage
            shadow_trade.buy_fill_price = opp.buy_price * (1 + total_slippage / 10000)
            shadow_trade.sell_fill_price = opp.sell_price * (1 - total_slippage / 10000)
            shadow_trade.buy_fill_size = shadow_trade.shadow_size
            shadow_trade.sell_fill_size = shadow_trade.shadow_size

            # Calculate performance metrics
            realized_spread = shadow_trade.sell_fill_price - shadow_trade.buy_fill_price
            shadow_trade.realized_spread_bps = (
                realized_spread / shadow_trade.buy_fill_price
            ) * 10000
            shadow_trade.realized_profit_usd = (
                realized_spread * shadow_trade.shadow_size
            )
            shadow_trade.slippage_bps = total_slippage

            # Execution time
            shadow_trade.execution_time_ms = (time.time() - start_time) * 1000
            shadow_trade.completion_time = time.time()

            # Determine success/failure
            if shadow_trade.realized_spread_bps > 0:
                shadow_trade.status = ExecutionStatus.COMPLETED
                self.performance_stats["successful_trades"] += 1
                self.performance_stats[
                    "total_profit_usd"
                ] += shadow_trade.realized_profit_usd
            else:
                shadow_trade.status = ExecutionStatus.FAILED
                shadow_trade.error_message = "Negative realized spread"
                self.performance_stats["failed_trades"] += 1

            self.performance_stats["executed_trades"] += 1

            # Move to completed trades
            del self.active_trades[shadow_trade.trade_id]
            self.completed_trades.append(shadow_trade)

            # Update A/B metrics
            self.ab_metrics["recent_trades"].append(shadow_trade)

            # Publish trade event
            await self._publish_trade_event(shadow_trade)

            logger.debug(
                f"Shadow trade {shadow_trade.trade_id} completed: "
                f"realized={shadow_trade.realized_spread_bps:.1f}bps, "
                f"slippage={shadow_trade.slippage_bps:.1f}bps, "
                f"profit=${shadow_trade.realized_profit_usd:.2f}"
            )

        except Exception as e:
            logger.error(f"Error simulating execution: {e}")
            shadow_trade.status = ExecutionStatus.FAILED
            shadow_trade.error_message = str(e)

    async def _monitor_active_trades(self):
        """Monitor active trades for timeouts and completion."""
        try:
            current_time = time.time()
            timeout_trades = []

            for trade_id, shadow_trade in self.active_trades.items():
                age = current_time - shadow_trade.created_time

                if age > self.config["execution_timeout_seconds"]:
                    timeout_trades.append(trade_id)

            # Handle timeouts
            for trade_id in timeout_trades:
                shadow_trade = self.active_trades[trade_id]
                shadow_trade.status = ExecutionStatus.FAILED
                shadow_trade.error_message = "Execution timeout"
                shadow_trade.completion_time = current_time

                self.completed_trades.append(shadow_trade)
                del self.active_trades[trade_id]

                self.performance_stats["failed_trades"] += 1
                self.performance_stats["executed_trades"] += 1

                logger.warning(f"Shadow trade {trade_id} timed out after {age:.1f}s")

        except Exception as e:
            logger.error(f"Error monitoring active trades: {e}")

    async def _update_performance_metrics(self):
        """Update performance metrics and A/B testing data."""
        try:
            if not self.completed_trades:
                return

            # Calculate aggregate metrics
            recent_trades = list(self.completed_trades)[-100:]  # Last 100 trades

            if recent_trades:
                successful_trades = [
                    t for t in recent_trades if t.status == ExecutionStatus.COMPLETED
                ]

                # Success rate
                success_rate = len(successful_trades) / len(recent_trades)
                self.performance_stats["success_rate"] = success_rate

                if successful_trades:
                    # Average metrics for successful trades
                    realized_spreads = [
                        t.realized_spread_bps for t in successful_trades
                    ]
                    slippages = [
                        t.slippage_bps for t in successful_trades if t.slippage_bps
                    ]
                    exec_times = [
                        t.execution_time_ms
                        for t in successful_trades
                        if t.execution_time_ms
                    ]
                    profits = [
                        t.realized_profit_usd
                        for t in successful_trades
                        if t.realized_profit_usd
                    ]

                    self.performance_stats["avg_realized_spread_bps"] = np.mean(
                        realized_spreads
                    )
                    if slippages:
                        self.performance_stats["avg_slippage_bps"] = np.mean(slippages)
                    if exec_times:
                        self.performance_stats["avg_execution_time_ms"] = np.mean(
                            exec_times
                        )

                    # Sharpe ratio (simplified)
                    if profits and len(profits) > 1:
                        profit_returns = np.array(profits)
                        self.performance_stats["sharpe_ratio"] = (
                            np.mean(profit_returns)
                            / np.std(profit_returns)
                            * np.sqrt(len(profit_returns))
                        )

                # Update A/B metrics for promotion decision
                await self._update_ab_metrics(recent_trades)

            # Publish metrics
            await self._publish_performance_metrics()

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    async def _update_ab_metrics(self, recent_trades: List[ShadowTrade]):
        """Update A/B testing metrics for promotion evaluation."""
        try:
            successful_trades = [
                t for t in recent_trades if t.status == ExecutionStatus.COMPLETED
            ]

            # Fill rate (success rate)
            fill_rate = (
                len(successful_trades) / len(recent_trades) if recent_trades else 0
            )
            self.ab_metrics["fill_rate"] = fill_rate

            # Average improvement over predicted spread
            if successful_trades:
                improvements = []
                for trade in successful_trades:
                    predicted_spread = trade.opportunity.risk_adjusted_bps
                    realized_spread = trade.realized_spread_bps or 0
                    improvement = realized_spread - predicted_spread
                    improvements.append(improvement)

                self.ab_metrics["avg_improvement_bps"] = np.mean(improvements)

                # Consistency score (how often we beat predictions)
                beat_count = len([i for i in improvements if i > 0])
                self.ab_metrics["consistency_score"] = beat_count / len(improvements)

        except Exception as e:
            logger.error(f"Error updating A/B metrics: {e}")

    async def _publish_trade_event(self, shadow_trade: ShadowTrade):
        """Publish shadow trade event to Redis."""
        try:
            event_data = {
                "trade_id": shadow_trade.trade_id,
                "symbol": shadow_trade.opportunity.symbol,
                "buy_exchange": shadow_trade.opportunity.buy_exchange,
                "sell_exchange": shadow_trade.opportunity.sell_exchange,
                "shadow_size": shadow_trade.shadow_size,
                "predicted_spread_bps": shadow_trade.opportunity.risk_adjusted_bps,
                "realized_spread_bps": shadow_trade.realized_spread_bps,
                "realized_profit_usd": shadow_trade.realized_profit_usd,
                "slippage_bps": shadow_trade.slippage_bps,
                "execution_time_ms": shadow_trade.execution_time_ms,
                "status": shadow_trade.status.value,
                "error_message": shadow_trade.error_message,
            }

            if ASYNC_REDIS_AVAILABLE:
                await publish_trade_event(event_data, "strategy:arb_shadow:events")

        except Exception as e:
            logger.error(f"Error publishing trade event: {e}")

    async def _publish_performance_metrics(self):
        """Publish performance metrics to Redis."""
        try:
            if ASYNC_REDIS_AVAILABLE:
                metrics = {
                    f"arb_shadow_{k}": v for k, v in self.performance_stats.items()
                }

                # Add A/B metrics
                metrics.update(
                    {
                        f"arb_shadow_ab_{k}": v
                        for k, v in self.ab_metrics.items()
                        if not isinstance(v, deque)
                    }
                )

                # Publish individual metrics
                for metric_name, value in metrics.items():
                    await set_metric(metric_name, value)

        except Exception as e:
            logger.error(f"Error publishing performance metrics: {e}")

    def get_promotion_readiness(self) -> Dict[str, Any]:
        """
        Evaluate readiness for promotion to live execution.

        Returns promotion criteria and current performance against thresholds.
        """
        try:
            # Promotion criteria
            criteria = {
                "min_trades": 50,  # Minimum trades for statistical significance
                "min_fill_rate": 0.95,  # 95% fill rate
                "min_improvement_bps": 3.0,  # 3 bps improvement over prediction
                "min_consistency": 0.70,  # 70% of trades beat prediction
                "max_avg_slippage": 5.0,  # Max 5 bps average slippage
                "min_sharpe": 1.0,  # Minimum Sharpe ratio
            }

            # Current performance
            recent_trades = (
                list(self.completed_trades)[-criteria["min_trades"] :]
                if self.completed_trades
                else []
            )
            current_performance = {
                "trades_count": len(recent_trades),
                "fill_rate": self.ab_metrics["fill_rate"],
                "avg_improvement_bps": self.ab_metrics["avg_improvement_bps"],
                "consistency_score": self.ab_metrics["consistency_score"],
                "avg_slippage_bps": self.performance_stats["avg_slippage_bps"],
                "sharpe_ratio": self.performance_stats["sharpe_ratio"],
            }

            # Check each criterion
            readiness = {
                "ready": True,
                "criteria": criteria,
                "performance": current_performance,
                "checks": {},
            }

            readiness["checks"]["sufficient_trades"] = (
                len(recent_trades) >= criteria["min_trades"]
            )
            readiness["checks"]["fill_rate_ok"] = (
                self.ab_metrics["fill_rate"] >= criteria["min_fill_rate"]
            )
            readiness["checks"]["improvement_ok"] = (
                self.ab_metrics["avg_improvement_bps"]
                >= criteria["min_improvement_bps"]
            )
            readiness["checks"]["consistency_ok"] = (
                self.ab_metrics["consistency_score"] >= criteria["min_consistency"]
            )
            readiness["checks"]["slippage_ok"] = (
                self.performance_stats["avg_slippage_bps"]
                <= criteria["max_avg_slippage"]
            )
            readiness["checks"]["sharpe_ok"] = (
                self.performance_stats["sharpe_ratio"] >= criteria["min_sharpe"]
            )

            # Overall readiness
            readiness["ready"] = all(readiness["checks"].values())

            return readiness

        except Exception as e:
            logger.error(f"Error evaluating promotion readiness: {e}")
            return {"ready": False, "error": str(e)}

    def get_execution_status(self) -> Dict[str, Any]:
        """Get comprehensive execution status."""
        try:
            return {
                "config": self.config.copy(),
                "performance_stats": self.performance_stats.copy(),
                "ab_metrics": {
                    k: v for k, v in self.ab_metrics.items() if not isinstance(v, deque)
                },
                "active_trades_count": len(self.active_trades),
                "completed_trades_count": len(self.completed_trades),
                "promotion_readiness": self.get_promotion_readiness(),
                "recent_trades": [
                    {
                        "trade_id": t.trade_id,
                        "symbol": t.opportunity.symbol,
                        "status": t.status.value,
                        "realized_spread_bps": t.realized_spread_bps,
                        "profit_usd": t.realized_profit_usd,
                        "completion_time": t.completion_time,
                    }
                    for t in list(self.completed_trades)[-10:]
                ],
            }

        except Exception as e:
            logger.error(f"Error getting execution status: {e}")
            return {"error": str(e)}


async def main():
    """Test the arbitrage shadow executor."""
    import argparse

    parser = argparse.ArgumentParser(description="Arbitrage Shadow Executor")
    parser.add_argument("--symbol", default="BTC", help="Symbol to execute")
    parser.add_argument(
        "--test", action="store_true", help="Run test with synthetic opportunities"
    )
    parser.add_argument("--monitor", action="store_true", help="Start monitoring mode")

    args = parser.parse_args()

    if args.test:
        # Test with synthetic opportunities
        from src.strategies.arb_scanner import ArbitrageScanner

        scanner = ArbitrageScanner([args.symbol])
        executor = ShadowExecutionEngine(scanner)

        # Generate some market data and opportunities
        base_price = 50000

        # Create price differences for arbitrage
        await scanner.update_market_data(
            args.symbol, "binance_spot", base_price - 5, base_price + 5, 10, 10
        )
        await scanner.update_market_data(
            args.symbol,
            "coinbase_spot",
            base_price + 15,
            base_price + 25,
            5,
            5,  # More expensive
        )

        # Run shadow execution for a few cycles
        for i in range(20):
            await executor._process_opportunities()
            await executor._monitor_active_trades()
            await executor._update_performance_metrics()

            if i % 5 == 0:
                status = executor.get_execution_status()
                print(
                    f"Cycle {i}: {status['completed_trades_count']} completed, "
                    f"success_rate={status['performance_stats']['success_rate']:.2f}"
                )

            await asyncio.sleep(0.1)

        # Final status and promotion readiness
        final_status = executor.get_execution_status()
        readiness = final_status["promotion_readiness"]

        print(f"\nFinal Status:")
        print(f"  Completed trades: {final_status['completed_trades_count']}")
        print(
            f"  Success rate: {final_status['performance_stats']['success_rate']:.2%}"
        )
        print(
            f"  Avg spread: {final_status['performance_stats']['avg_realized_spread_bps']:.1f}bps"
        )
        print(
            f"  Total profit: ${final_status['performance_stats']['total_profit_usd']:.2f}"
        )
        print(f"  Ready for promotion: {readiness['ready']}")

    elif args.monitor:
        # Start monitoring (would integrate with live scanner)
        from src.strategies.arb_scanner import ArbitrageScanner

        scanner = ArbitrageScanner([args.symbol])
        executor = ShadowExecutionEngine(scanner)

        logger.info(
            f"Starting arbitrage shadow executor monitoring for {args.symbol}..."
        )

        try:
            await executor.start_monitoring()
        except KeyboardInterrupt:
            logger.info("Arbitrage shadow executor stopped")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
