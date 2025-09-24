#!/usr/bin/env python3
"""
Smart Order Router (SOR)

Multi-exchange execution optimization system that routes orders intelligently
across multiple exchanges to minimize costs and maximize execution quality.

Features:
- Real-time liquidity analysis across exchanges
- Order fragmentation and TWAP strategies
- Latency-aware routing
- Impact cost modeling
- Dark pool integration
- Post-only and aggressive execution modes
"""

import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import json
from collections import defaultdict
import heapq

from ...utils.logger import get_logger


class ExchangeType(Enum):
    """Exchange type enumeration."""

    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    DARK_POOL = "dark_pool"


class OrderUrgency(Enum):
    """Order urgency levels."""

    IMMEDIATE = "immediate"  # Market orders, aggressive execution
    MODERATE = "moderate"  # Fill within 5-15 minutes
    PATIENT = "patient"  # TWAP over 30+ minutes
    PASSIVE = "passive"  # Post-only, no market impact


class VenueMetrics(NamedTuple):
    """Venue performance metrics."""

    fill_rate: float
    avg_latency_ms: float
    avg_spread_bps: float
    daily_volume: float
    uptime_pct: float
    maker_fee_bps: float
    taker_fee_bps: float


@dataclass
class LiquidityLevel:
    """Order book liquidity level."""

    price: Decimal
    size: Decimal
    cumulative_size: Decimal
    exchange: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RoutingFragment:
    """Order fragment for routing."""

    exchange: str
    symbol: str
    side: str
    size: Decimal
    order_type: str
    limit_price: Optional[Decimal] = None
    time_in_force: str = "IOC"
    expected_fill_rate: float = 1.0
    expected_cost_bps: float = 0.0
    priority: int = 0


@dataclass
class ExecutionPlan:
    """Complete execution plan for an order."""

    original_order_id: str
    total_size: Decimal
    fragments: List[RoutingFragment]
    estimated_cost_bps: float
    estimated_fill_time_seconds: float
    confidence_score: float
    created_at: datetime = field(default_factory=datetime.utcnow)


class SmartOrderRouter:
    """
    Smart Order Router for multi-exchange execution optimization.

    Routes orders across multiple exchanges to minimize execution costs
    and maximize fill probability while managing market impact.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Smart Order Router.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.logger = get_logger("execution.smart_router")

        # Initialize Redis for TCA integration
        import redis

        try:
            self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        except:
            self.redis = None
            self.logger.warning("Redis connection failed - TCA integration disabled")

        # Venue configuration
        self.venues = {
            "binance": {
                "type": ExchangeType.CENTRALIZED,
                "maker_fee_bps": 1.0,
                "taker_fee_bps": 1.0,
                "min_order_size": 10.0,
                "max_order_size": 1000000.0,
                "supported_symbols": {"BTCUSDT", "ETHUSDT", "ADAUSDT"},
                "latency_ms": 50,
            },
            "coinbase": {
                "type": ExchangeType.CENTRALIZED,
                "maker_fee_bps": 0.5,
                "taker_fee_bps": 0.5,
                "min_order_size": 1.0,
                "max_order_size": 500000.0,
                "supported_symbols": {"BTC-USD", "ETH-USD", "ADA-USD"},
                "latency_ms": 80,
            },
            "kraken": {
                "type": ExchangeType.CENTRALIZED,
                "maker_fee_bps": 1.6,
                "taker_fee_bps": 2.6,
                "min_order_size": 5.0,
                "max_order_size": 200000.0,
                "supported_symbols": {"XBTUSD", "ETHUSD", "ADAUSD"},
                "latency_ms": 120,
            },
            "uniswap_v3": {
                "type": ExchangeType.DECENTRALIZED,
                "maker_fee_bps": 0.0,
                "taker_fee_bps": 30.0,  # 0.3% swap fee
                "min_order_size": 1.0,
                "max_order_size": 100000.0,
                "supported_symbols": {"WBTC-USDC", "ETH-USDC", "ADA-USDC"},
                "latency_ms": 15000,  # Block time
            },
            "dark_pool_alpha": {
                "type": ExchangeType.DARK_POOL,
                "maker_fee_bps": 0.0,
                "taker_fee_bps": 0.5,
                "min_order_size": 50.0,
                "max_order_size": 50000.0,
                "supported_symbols": {"BTCUSDT", "ETHUSDT"},
                "latency_ms": 200,
            },
        }

        # Venue performance tracking
        self.venue_metrics = {}
        self._initialize_venue_metrics()

        # TCA integration
        self.tca_weights = {}
        self._update_tca_weights()

        # Market data cache
        self.order_books = {}  # {exchange: {symbol: {'bids': [], 'asks': []}}}
        self.last_prices = {}  # {symbol: price}
        self.volume_profiles = {}  # {symbol: {exchange: 24h_volume}}

        # Execution tracking
        self.active_plans = {}  # {plan_id: ExecutionPlan}
        self.execution_history = []

        # Performance counters
        self.total_orders_routed = 0
        self.total_cost_savings_bps = 0.0
        self.avg_fill_rate = 0.0

        self.logger.info(
            f"Smart Order Router initialized with {len(self.venues)} venues"
        )

    def _initialize_venue_metrics(self):
        """Initialize venue performance metrics with defaults."""
        for venue_id, venue_config in self.venues.items():
            self.venue_metrics[venue_id] = VenueMetrics(
                fill_rate=0.95,
                avg_latency_ms=venue_config["latency_ms"],
                avg_spread_bps=5.0,
                daily_volume=1000000.0,
                uptime_pct=99.5,
                maker_fee_bps=venue_config["maker_fee_bps"],
                taker_fee_bps=venue_config["taker_fee_bps"],
            )

    def _update_tca_weights(self):
        """Update venue weights from TCA analysis."""
        if not self.redis:
            # Default equal weights if no TCA data
            default_weight = 1.0 / len(self.venues)
            for venue in self.venues:
                self.tca_weights[venue] = default_weight
            return

        try:
            total_weight = 0.0
            raw_weights = {}

            # Get TCA scores for each venue
            for venue in self.venues:
                tca_data = self.redis.hgetall(f"tca:venue:{venue}")
                if tca_data and "score" in tca_data:
                    score = float(tca_data["score"])
                    # Convert score to weight (ensure positive)
                    weight = max(0.1, score + 1.0)  # Add 1.0 to ensure positive
                    raw_weights[venue] = weight
                    total_weight += weight
                else:
                    # Fallback weight
                    raw_weights[venue] = 0.25
                    total_weight += 0.25

            # Normalize weights
            if total_weight > 0:
                for venue, weight in raw_weights.items():
                    self.tca_weights[venue] = weight / total_weight
            else:
                # Fallback to equal weights
                equal_weight = 1.0 / len(self.venues)
                for venue in self.venues:
                    self.tca_weights[venue] = equal_weight

            self.logger.debug(f"Updated TCA weights: {self.tca_weights}")

        except Exception as e:
            self.logger.error(f"Error updating TCA weights: {e}")
            # Fallback to equal weights
            equal_weight = 1.0 / len(self.venues)
            for venue in self.venues:
                self.tca_weights[venue] = equal_weight

    async def route_order(
        self,
        symbol: str,
        side: str,
        size: Decimal,
        urgency: OrderUrgency = OrderUrgency.MODERATE,
        max_impact_bps: float = 10.0,
        allow_partial: bool = True,
    ) -> ExecutionPlan:
        """
        Create optimal execution plan for an order.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Order size in base currency
            urgency: Execution urgency level
            max_impact_bps: Maximum acceptable market impact in basis points
            allow_partial: Whether to allow partial fills

        Returns:
            ExecutionPlan with optimized routing strategy
        """
        try:
            self.logger.info(f"Routing order: {symbol} {side} {size} ({urgency.value})")

            # Get available venues for symbol
            available_venues = self._get_available_venues(symbol)
            if not available_venues:
                raise ValueError(f"No venues available for symbol {symbol}")

            # Update market data and TCA weights
            await self._update_market_data(symbol, available_venues)
            self._update_tca_weights()

            # Analyze liquidity across venues
            liquidity_analysis = self._analyze_cross_venue_liquidity(
                symbol, side, available_venues
            )

            # Generate execution plan based on urgency
            if urgency == OrderUrgency.IMMEDIATE:
                plan = self._create_aggressive_plan(
                    symbol, side, size, liquidity_analysis
                )
            elif urgency == OrderUrgency.MODERATE:
                plan = self._create_balanced_plan(
                    symbol, side, size, liquidity_analysis, max_impact_bps
                )
            elif urgency == OrderUrgency.PATIENT:
                plan = self._create_twap_plan(
                    symbol, side, size, liquidity_analysis, max_impact_bps
                )
            else:  # PASSIVE
                plan = self._create_passive_plan(symbol, side, size, liquidity_analysis)

            # Validate and optimize plan
            plan = self._optimize_execution_plan(plan, max_impact_bps)

            # Store plan for tracking
            self.active_plans[plan.original_order_id] = plan
            self.total_orders_routed += 1

            self.logger.info(
                f"Created execution plan: {len(plan.fragments)} fragments, "
                f"estimated cost: {plan.estimated_cost_bps:.1f}bps, "
                f"confidence: {plan.confidence_score:.1%}"
            )

            return plan

        except Exception as e:
            self.logger.error(f"Error routing order: {e}")
            raise

    def _get_available_venues(self, symbol: str) -> List[str]:
        """Get list of venues that support the given symbol."""
        available = []
        for venue_id, venue_config in self.venues.items():
            if symbol in venue_config["supported_symbols"]:
                metrics = self.venue_metrics[venue_id]
                # Only include venues with good uptime
                if metrics.uptime_pct > 95.0:
                    available.append(venue_id)
        return available

    async def _update_market_data(self, symbol: str, venues: List[str]):
        """Update market data for symbol across venues."""
        # Simulate market data updates
        # In production, this would fetch real order book data

        current_price = self.last_prices.get(symbol, Decimal("50000"))

        for venue in venues:
            if venue not in self.order_books:
                self.order_books[venue] = {}

            # Generate synthetic order book
            self.order_books[venue][symbol] = self._generate_synthetic_orderbook(
                venue, symbol, current_price
            )

        self.last_prices[symbol] = current_price

    def _generate_synthetic_orderbook(
        self, venue: str, symbol: str, mid_price: Decimal
    ) -> Dict:
        """Generate synthetic order book for testing."""
        venue_config = self.venues[venue]
        venue_metrics = self.venue_metrics[venue]

        # Calculate spread based on venue characteristics
        base_spread_bps = venue_metrics.avg_spread_bps
        spread = mid_price * Decimal(base_spread_bps / 10000)

        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2

        # Generate order book levels
        bids = []
        asks = []

        # Venue-specific liquidity characteristics
        liquidity_factor = {
            "binance": 2.0,
            "coinbase": 1.5,
            "kraken": 1.0,
            "uniswap_v3": 0.8,
            "dark_pool_alpha": 0.3,
        }.get(venue, 1.0)

        for i in range(10):
            # Bids (decreasing prices)
            price_offset = Decimal(i * 0.001) * mid_price
            bid_level_price = bid_price - price_offset
            bid_size = Decimal(np.random.exponential(10.0 * liquidity_factor))

            bids.append(
                LiquidityLevel(
                    price=bid_level_price,
                    size=bid_size,
                    cumulative_size=sum(b.size for b in bids) + bid_size,
                    exchange=venue,
                )
            )

            # Asks (increasing prices)
            ask_level_price = ask_price + price_offset
            ask_size = Decimal(np.random.exponential(10.0 * liquidity_factor))

            asks.append(
                LiquidityLevel(
                    price=ask_level_price,
                    size=ask_size,
                    cumulative_size=sum(a.size for a in asks) + ask_size,
                    exchange=venue,
                )
            )

        return {"bids": bids, "asks": asks}

    def _analyze_cross_venue_liquidity(
        self, symbol: str, side: str, venues: List[str]
    ) -> Dict:
        """Analyze liquidity across all venues for optimal routing."""
        analysis = {
            "total_liquidity": Decimal("0"),
            "weighted_avg_price": Decimal("0"),
            "best_venues": [],
            "liquidity_distribution": {},
            "impact_analysis": {},
        }

        all_levels = []

        # Collect all liquidity levels across venues
        for venue in venues:
            if venue in self.order_books and symbol in self.order_books[venue]:
                book = self.order_books[venue][symbol]
                levels = book["asks"] if side == "buy" else book["bids"]

                for level in levels:
                    all_levels.append(level)

        if not all_levels:
            return analysis

        # Sort levels by price (best first)
        if side == "buy":
            all_levels.sort(key=lambda x: x.price)  # Ascending for asks
        else:
            all_levels.sort(key=lambda x: x.price, reverse=True)  # Descending for bids

        # Calculate cumulative liquidity and impact
        cumulative_size = Decimal("0")
        cumulative_cost = Decimal("0")

        for level in all_levels:
            cumulative_size += level.size
            cumulative_cost += level.size * level.price

            analysis["liquidity_distribution"][level.exchange] = (
                analysis["liquidity_distribution"].get(level.exchange, Decimal("0"))
                + level.size
            )

        if cumulative_size > 0:
            analysis["total_liquidity"] = cumulative_size
            analysis["weighted_avg_price"] = cumulative_cost / cumulative_size

        # Rank venues by liquidity, cost, and TCA performance
        venue_scores = {}
        for venue in venues:
            liquidity = analysis["liquidity_distribution"].get(venue, Decimal("0"))
            metrics = self.venue_metrics[venue]
            tca_weight = self.tca_weights.get(venue, 0.25)

            # Score based on liquidity, fees, performance, and TCA analysis
            score = (
                float(liquidity) * 0.3  # Liquidity availability
                + (100 - metrics.taker_fee_bps) * 0.2  # Fee efficiency
                + metrics.fill_rate * 100 * 0.2  # Historical fill rate
                + (100 - metrics.avg_latency_ms / 10) * 0.1  # Latency performance
                + tca_weight * 100 * 0.2  # TCA-based venue score
            )
            venue_scores[venue] = score

        analysis["best_venues"] = sorted(
            venue_scores.items(), key=lambda x: x[1], reverse=True
        )

        return analysis

    def _create_aggressive_plan(
        self, symbol: str, side: str, size: Decimal, liquidity_analysis: Dict
    ) -> ExecutionPlan:
        """Create aggressive execution plan for immediate fills."""
        plan_id = f"aggressive_{int(time.time())}"
        fragments = []

        remaining_size = size
        estimated_cost = 0.0

        # Use best venues with taker orders
        for venue, score in liquidity_analysis["best_venues"][:3]:
            if remaining_size <= 0:
                break

            venue_liquidity = liquidity_analysis["liquidity_distribution"].get(
                venue, Decimal("0")
            )
            fragment_size = min(remaining_size, venue_liquidity * Decimal("0.8"))

            if fragment_size > 0:
                venue_config = self.venues[venue]
                fragment = RoutingFragment(
                    exchange=venue,
                    symbol=symbol,
                    side=side,
                    size=fragment_size,
                    order_type="market",
                    time_in_force="IOC",
                    expected_fill_rate=0.95,
                    expected_cost_bps=self.venue_metrics[venue].taker_fee_bps,
                    priority=1,  # High priority
                )

                fragments.append(fragment)
                remaining_size -= fragment_size
                estimated_cost += (
                    float(fragment_size) * self.venue_metrics[venue].taker_fee_bps
                )

        total_allocated = sum(f.size for f in fragments)
        avg_cost_bps = (
            estimated_cost / float(total_allocated) if total_allocated > 0 else 0.0
        )

        return ExecutionPlan(
            original_order_id=plan_id,
            total_size=size,
            fragments=fragments,
            estimated_cost_bps=avg_cost_bps,
            estimated_fill_time_seconds=5.0,
            confidence_score=0.9,
        )

    def _create_balanced_plan(
        self,
        symbol: str,
        side: str,
        size: Decimal,
        liquidity_analysis: Dict,
        max_impact_bps: float,
    ) -> ExecutionPlan:
        """Create balanced execution plan optimizing cost and speed."""
        plan_id = f"balanced_{int(time.time())}"
        fragments = []

        remaining_size = size
        estimated_cost = 0.0

        # Mix of maker and taker orders across venues
        for i, (venue, score) in enumerate(liquidity_analysis["best_venues"][:4]):
            if remaining_size <= 0:
                break

            venue_liquidity = liquidity_analysis["liquidity_distribution"].get(
                venue, Decimal("0")
            )
            fragment_size = min(remaining_size, venue_liquidity * Decimal("0.6"))

            if fragment_size > 0:
                # Alternate between maker and taker strategies
                if i % 2 == 0:
                    # Use maker orders for better fees
                    order_type = "limit"
                    tif = "GTC"
                    cost_bps = self.venue_metrics[venue].maker_fee_bps
                    fill_rate = 0.8
                else:
                    # Use taker orders for guaranteed fills
                    order_type = "market"
                    tif = "IOC"
                    cost_bps = self.venue_metrics[venue].taker_fee_bps
                    fill_rate = 0.95

                fragment = RoutingFragment(
                    exchange=venue,
                    symbol=symbol,
                    side=side,
                    size=fragment_size,
                    order_type=order_type,
                    time_in_force=tif,
                    expected_fill_rate=fill_rate,
                    expected_cost_bps=cost_bps,
                    priority=2 if order_type == "market" else 3,
                )

                fragments.append(fragment)
                remaining_size -= fragment_size
                estimated_cost += float(fragment_size) * cost_bps

        total_allocated = sum(f.size for f in fragments)
        avg_cost_bps = (
            estimated_cost / float(total_allocated) if total_allocated > 0 else 0.0
        )

        return ExecutionPlan(
            original_order_id=plan_id,
            total_size=size,
            fragments=fragments,
            estimated_cost_bps=avg_cost_bps,
            estimated_fill_time_seconds=60.0,
            confidence_score=0.85,
        )

    def _create_twap_plan(
        self,
        symbol: str,
        side: str,
        size: Decimal,
        liquidity_analysis: Dict,
        max_impact_bps: float,
    ) -> ExecutionPlan:
        """Create TWAP execution plan for patient execution."""
        plan_id = f"twap_{int(time.time())}"
        fragments = []

        # Split order into smaller time-weighted chunks
        num_slices = 10
        slice_size = size / num_slices
        estimated_cost = 0.0

        for i in range(num_slices):
            # Distribute across best venues
            for j, (venue, score) in enumerate(liquidity_analysis["best_venues"][:2]):
                if j > i % 2:  # Alternate venues
                    continue

                fragment = RoutingFragment(
                    exchange=venue,
                    symbol=symbol,
                    side=side,
                    size=(
                        slice_size / 2
                        if len(liquidity_analysis["best_venues"]) > 1
                        else slice_size
                    ),
                    order_type="limit",
                    time_in_force="GTC",
                    expected_fill_rate=0.7,
                    expected_cost_bps=self.venue_metrics[venue].maker_fee_bps,
                    priority=4 + i,  # Lower priority, staggered execution
                )

                fragments.append(fragment)
                estimated_cost += (
                    float(fragment.size) * self.venue_metrics[venue].maker_fee_bps
                )

        total_allocated = sum(f.size for f in fragments)
        avg_cost_bps = (
            estimated_cost / float(total_allocated) if total_allocated > 0 else 0.0
        )

        return ExecutionPlan(
            original_order_id=plan_id,
            total_size=size,
            fragments=fragments,
            estimated_cost_bps=avg_cost_bps,
            estimated_fill_time_seconds=1800.0,  # 30 minutes
            confidence_score=0.75,
        )

    def _create_passive_plan(
        self, symbol: str, side: str, size: Decimal, liquidity_analysis: Dict
    ) -> ExecutionPlan:
        """Create passive execution plan with post-only orders."""
        plan_id = f"passive_{int(time.time())}"
        fragments = []

        remaining_size = size
        estimated_cost = 0.0

        # Use only maker orders across top venues
        for venue, score in liquidity_analysis["best_venues"][:3]:
            if remaining_size <= 0:
                break

            venue_liquidity = liquidity_analysis["liquidity_distribution"].get(
                venue, Decimal("0")
            )
            fragment_size = min(remaining_size, venue_liquidity * Decimal("0.3"))

            if fragment_size > 0:
                fragment = RoutingFragment(
                    exchange=venue,
                    symbol=symbol,
                    side=side,
                    size=fragment_size,
                    order_type="limit",
                    time_in_force="GTC",
                    expected_fill_rate=0.6,
                    expected_cost_bps=self.venue_metrics[venue].maker_fee_bps,
                    priority=10,  # Lowest priority
                )

                fragments.append(fragment)
                remaining_size -= fragment_size
                estimated_cost += (
                    float(fragment_size) * self.venue_metrics[venue].maker_fee_bps
                )

        total_allocated = sum(f.size for f in fragments)
        avg_cost_bps = (
            estimated_cost / float(total_allocated) if total_allocated > 0 else 0.0
        )

        return ExecutionPlan(
            original_order_id=plan_id,
            total_size=size,
            fragments=fragments,
            estimated_cost_bps=avg_cost_bps,
            estimated_fill_time_seconds=3600.0,  # 1 hour
            confidence_score=0.6,
        )

    def _optimize_execution_plan(
        self, plan: ExecutionPlan, max_impact_bps: float
    ) -> ExecutionPlan:
        """Optimize execution plan for better performance."""
        # Sort fragments by priority and expected performance
        plan.fragments.sort(key=lambda f: (f.priority, -f.expected_fill_rate))

        # Remove fragments that don't meet minimum criteria
        optimized_fragments = []
        for fragment in plan.fragments:
            venue_metrics = self.venue_metrics[fragment.exchange]

            # Skip venues with poor performance
            if venue_metrics.fill_rate < 0.5 or venue_metrics.uptime_pct < 90:
                continue

            # Skip if cost is too high
            if fragment.expected_cost_bps > max_impact_bps:
                continue

            optimized_fragments.append(fragment)

        plan.fragments = optimized_fragments

        # Recalculate metrics
        if plan.fragments:
            total_allocated = sum(f.size for f in plan.fragments)
            total_cost = sum(
                float(f.size) * f.expected_cost_bps for f in plan.fragments
            )
            plan.estimated_cost_bps = (
                total_cost / float(total_allocated) if total_allocated > 0 else 0.0
            )

            avg_fill_rate = sum(f.expected_fill_rate for f in plan.fragments) / len(
                plan.fragments
            )
            plan.confidence_score = avg_fill_rate * 0.8  # Discount for execution risk

        return plan

    async def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute the routing plan across multiple venues."""
        self.logger.info(
            f"Executing plan {plan.original_order_id} with {len(plan.fragments)} fragments"
        )

        execution_results = {
            "plan_id": plan.original_order_id,
            "fragments_executed": 0,
            "total_filled": Decimal("0"),
            "avg_fill_price": Decimal("0"),
            "total_cost_bps": 0.0,
            "execution_time_seconds": 0.0,
            "success_rate": 0.0,
        }

        start_time = time.time()
        filled_amount = Decimal("0")
        total_cost = Decimal("0")
        successful_fragments = 0

        # Execute fragments in priority order
        for fragment in plan.fragments:
            try:
                self.logger.debug(
                    f"Executing fragment: {fragment.exchange} {fragment.size}"
                )

                # Simulate execution
                success = await self._execute_fragment(fragment)

                if success:
                    filled_amount += fragment.size
                    total_cost += fragment.size * Decimal(
                        fragment.expected_cost_bps / 10000
                    )
                    successful_fragments += 1

                execution_results["fragments_executed"] += 1

                # Add small delay between fragments
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error executing fragment: {e}")

        execution_time = time.time() - start_time

        # Calculate final metrics
        if filled_amount > 0:
            execution_results["total_filled"] = filled_amount
            execution_results["avg_fill_price"] = self.last_prices.get(
                plan.fragments[0].symbol, Decimal("0")
            )
            execution_results["total_cost_bps"] = float(
                total_cost / filled_amount * 10000
            )

        execution_results["execution_time_seconds"] = execution_time
        execution_results["success_rate"] = (
            successful_fragments / len(plan.fragments) if plan.fragments else 0.0
        )

        # Update performance tracking
        self.avg_fill_rate = (self.avg_fill_rate * 0.9) + (
            execution_results["success_rate"] * 0.1
        )

        self.logger.info(
            f"Plan execution completed: {execution_results['success_rate']:.1%} success rate, "
            f"{float(execution_results['total_filled']):.2f} filled, "
            f"{execution_results['total_cost_bps']:.1f}bps cost"
        )

        return execution_results

    async def _execute_fragment(self, fragment: RoutingFragment) -> bool:
        """Execute a single routing fragment."""
        # Simulate fragment execution
        venue_metrics = self.venue_metrics[fragment.exchange]

        # Simulate latency
        latency_seconds = venue_metrics.avg_latency_ms / 1000.0
        await asyncio.sleep(latency_seconds)

        # Simulate fill probability
        fill_probability = fragment.expected_fill_rate * venue_metrics.fill_rate
        success = np.random.random() < fill_probability

        self.logger.debug(
            f"Fragment execution: {fragment.exchange} {'SUCCESS' if success else 'FAILED'} "
            f"(prob: {fill_probability:.2f})"
        )

        return success

    def get_venue_status(self) -> Dict[str, Any]:
        """Get current status of all venues."""
        status = {}

        for venue_id, venue_config in self.venues.items():
            metrics = self.venue_metrics[venue_id]

            status[venue_id] = {
                "type": venue_config["type"].value,
                "supported_symbols": list(venue_config["supported_symbols"]),
                "performance": {
                    "fill_rate": f"{metrics.fill_rate:.1%}",
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "avg_spread_bps": metrics.avg_spread_bps,
                    "uptime_pct": f"{metrics.uptime_pct:.1f}%",
                },
                "fees": {
                    "maker_bps": metrics.maker_fee_bps,
                    "taker_bps": metrics.taker_fee_bps,
                },
            }

        return status

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        return {
            "total_orders_routed": self.total_orders_routed,
            "avg_fill_rate": f"{self.avg_fill_rate:.1%}",
            "active_plans": len(self.active_plans),
            "venue_count": len(self.venues),
            "supported_symbols": len(
                set().union(*[v["supported_symbols"] for v in self.venues.values()])
            ),
            "execution_modes": ["aggressive", "balanced", "twap", "passive"],
        }


async def main():
    """Test the Smart Order Router."""
    print("🚀 Testing Smart Order Router...")

    # Initialize router
    router = SmartOrderRouter()

    # Test venue status
    print("\n📊 Venue Status:")
    venue_status = router.get_venue_status()
    for venue, status in venue_status.items():
        print(
            f"  {venue}: {status['performance']['fill_rate']} fill rate, "
            f"{status['fees']['taker_bps']}bps taker fee"
        )

    # Test order routing
    test_cases = [
        ("BTCUSDT", "buy", Decimal("1000"), OrderUrgency.IMMEDIATE),
        ("ETHUSDT", "sell", Decimal("5000"), OrderUrgency.BALANCED),
        ("BTCUSDT", "buy", Decimal("50000"), OrderUrgency.PATIENT),
    ]

    for symbol, side, size, urgency in test_cases:
        print(f"\n📈 Testing {urgency.value} order: {symbol} {side} ${size}")

        try:
            # Route order
            plan = await router.route_order(symbol, side, size, urgency)

            print(f"  ✅ Created plan with {len(plan.fragments)} fragments")
            print(f"  💰 Estimated cost: {plan.estimated_cost_bps:.1f}bps")
            print(f"  ⏱️  Estimated time: {plan.estimated_fill_time_seconds:.0f}s")
            print(f"  🎯 Confidence: {plan.confidence_score:.1%}")

            # Execute plan
            results = await router.execute_plan(plan)
            print(
                f"  📊 Execution: {results['success_rate']:.1%} success, "
                f"${float(results['total_filled']):.0f} filled"
            )

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Show final stats
    print(f"\n📈 Routing Stats:")
    stats = router.get_routing_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("🎉 Smart Order Router test completed!")


if __name__ == "__main__":
    asyncio.run(main())
