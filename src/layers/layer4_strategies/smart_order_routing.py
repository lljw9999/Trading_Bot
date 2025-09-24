#!/usr/bin/env python3
"""
Enhanced Smart Order Routing with Slippage Optimization
Addresses execution price improvement and cost reduction strategies
"""

import numpy as np
import pandas as pd
import redis
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    MARKETABLE_LIMIT = "marketable_limit"
    TWAP = "twap"
    ADAPTIVE = "adaptive"


class Venue(Enum):
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    ALPACA = "alpaca"


@dataclass
class MarketData:
    """Real-time market data structure"""

    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    timestamp: datetime
    venue: str
    spread_bps: float = 0.0

    def __post_init__(self):
        if self.bid > 0 and self.ask > 0:
            mid_price = (self.bid + self.ask) / 2
            self.spread_bps = ((self.ask - self.bid) / mid_price) * 10000


@dataclass
class ExecutionConfig:
    """Configuration for execution optimization"""

    max_spread_bps: float = 10.0  # Don't trade if spread > 10 bps
    target_latency_ms: float = 100.0  # Target execution latency
    max_slippage_bps: float = 5.0  # Alert if slippage > 5 bps
    min_edge_threshold_bps: float = 15.0  # Minimum edge after costs
    twap_chunks: int = 5  # Number of TWAP chunks
    adaptive_threshold: float = 0.8  # Confidence threshold for adaptive mode
    venue_preferences: Dict[str, float] = None  # Venue scoring weights

    def __post_init__(self):
        if self.venue_preferences is None:
            self.venue_preferences = {
                "binance": 1.0,
                "coinbase": 0.9,
                "kraken": 0.8,
                "alpaca": 1.0,
            }


class SlippageTracker:
    """Real-time slippage monitoring and analysis"""

    def __init__(self):
        self.slippage_history = []
        self.execution_stats = {}

    def record_execution(
        self,
        symbol: str,
        intended_price: float,
        executed_price: float,
        side: str,
        venue: str,
        timestamp: datetime,
    ) -> Dict[str, Any]:
        """Record execution for slippage analysis"""
        slippage_bps = ((executed_price - intended_price) / intended_price) * 10000
        if side.lower() == "sell":
            slippage_bps = -slippage_bps  # Adjust for sell side

        execution_record = {
            "symbol": symbol,
            "intended_price": intended_price,
            "executed_price": executed_price,
            "slippage_bps": slippage_bps,
            "side": side,
            "venue": venue,
            "timestamp": timestamp,
        }

        self.slippage_history.append(execution_record)

        # Keep only last 1000 executions
        if len(self.slippage_history) > 1000:
            self.slippage_history = self.slippage_history[-1000:]

        return execution_record

    def get_slippage_stats(
        self, symbol: str = None, lookback_minutes: int = 60
    ) -> Dict[str, Any]:
        """Calculate slippage statistics"""
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)

        relevant_records = [
            r
            for r in self.slippage_history
            if r["timestamp"] >= cutoff_time
            and (symbol is None or r["symbol"] == symbol)
        ]

        if not relevant_records:
            return {"count": 0, "avg_slippage_bps": 0.0}

        slippages = [r["slippage_bps"] for r in relevant_records]

        return {
            "count": len(relevant_records),
            "avg_slippage_bps": np.mean(slippages),
            "median_slippage_bps": np.median(slippages),
            "std_slippage_bps": np.std(slippages),
            "max_slippage_bps": np.max(slippages),
            "min_slippage_bps": np.min(slippages),
            "recent_trend": (
                np.mean(slippages[-5:]) if len(slippages) >= 5 else np.mean(slippages)
            ),
        }


class VenueManager:
    """Manage multiple trading venues and routing logic"""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.venue_data = {}
        self.venue_performance = {}

    def update_venue_data(self, venue: str, market_data: MarketData):
        """Update real-time data for venue"""
        self.venue_data[venue] = market_data

    def get_best_venue(
        self, symbol: str, side: str, quantity: float
    ) -> Tuple[str, float]:
        """Find best venue for execution"""
        available_venues = [
            venue
            for venue, data in self.venue_data.items()
            if data.symbol == symbol and data.spread_bps <= self.config.max_spread_bps
        ]

        if not available_venues:
            return None, None

        venue_scores = {}
        for venue in available_venues:
            data = self.venue_data[venue]

            # Calculate venue score based on:
            # 1. Price (bid for sells, ask for buys)
            # 2. Spread
            # 3. Size availability
            # 4. Historical performance

            if side.lower() == "buy":
                price = data.ask
                size_score = min(data.ask_size / quantity, 1.0) if quantity > 0 else 1.0
            else:
                price = data.bid
                size_score = min(data.bid_size / quantity, 1.0) if quantity > 0 else 1.0

            spread_score = max(0, 1 - (data.spread_bps / 20.0))  # Penalize wide spreads
            venue_preference = self.config.venue_preferences.get(venue, 0.5)

            # Historical performance (if available)
            perf_score = self.venue_performance.get(venue, {}).get(
                "avg_slippage_score", 0.5
            )

            venue_scores[venue] = {
                "price": price,
                "total_score": (
                    size_score * 0.3
                    + spread_score * 0.3
                    + venue_preference * 0.2
                    + perf_score * 0.2
                ),
                "size_score": size_score,
                "spread_score": spread_score,
            }

        # Select best venue
        best_venue = max(
            venue_scores.keys(), key=lambda v: venue_scores[v]["total_score"]
        )
        best_price = venue_scores[best_venue]["price"]

        logger.info(
            f"🎯 Best venue for {symbol} {side}: {best_venue} @ {best_price:.4f}"
        )
        return best_venue, best_price


class SmartOrderRouter:
    """Enhanced Smart Order Router with slippage optimization"""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.slippage_tracker = SlippageTracker()
        self.venue_manager = VenueManager(self.config)

        # Redis connection for caching
        try:
            self.redis_client = redis.Redis(
                host="localhost", port=6379, db=0, decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected for order routing")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None

    def calculate_execution_cost(
        self, symbol: str, side: str, quantity: float, market_data: MarketData
    ) -> Dict[str, float]:
        """Calculate expected execution costs"""
        mid_price = (market_data.bid + market_data.ask) / 2

        # Spread cost
        if side.lower() == "buy":
            spread_cost_bps = ((market_data.ask - mid_price) / mid_price) * 10000
        else:
            spread_cost_bps = ((mid_price - market_data.bid) / mid_price) * 10000

        # Expected slippage based on historical data
        slippage_stats = self.slippage_tracker.get_slippage_stats(symbol, 30)
        expected_slippage_bps = slippage_stats.get("avg_slippage_bps", 3.0)

        # Market impact (simplified model)
        if quantity > 0 and market_data.ask_size > 0:
            impact_ratio = quantity / (
                market_data.ask_size if side.lower() == "buy" else market_data.bid_size
            )
            market_impact_bps = min(impact_ratio * 5.0, 10.0)  # Cap at 10 bps
        else:
            market_impact_bps = 1.0

        total_cost_bps = (
            spread_cost_bps + abs(expected_slippage_bps) + market_impact_bps
        )

        return {
            "spread_cost_bps": spread_cost_bps,
            "expected_slippage_bps": expected_slippage_bps,
            "market_impact_bps": market_impact_bps,
            "total_cost_bps": total_cost_bps,
        }

    def should_trade(
        self, predicted_edge_bps: float, execution_costs: Dict[str, float]
    ) -> bool:
        """Determine if trade should proceed based on cost-adjusted edge"""
        net_edge_bps = predicted_edge_bps - execution_costs["total_cost_bps"]

        should_proceed = net_edge_bps >= self.config.min_edge_threshold_bps

        logger.info(
            f"📊 Trade decision: Edge={predicted_edge_bps:.1f}bps, "
            f"Cost={execution_costs['total_cost_bps']:.1f}bps, "
            f"Net={net_edge_bps:.1f}bps, Proceed={should_proceed}"
        )

        return should_proceed

    def select_order_type(
        self, symbol: str, side: str, confidence: float, market_data: MarketData
    ) -> OrderType:
        """Select optimal order type based on market conditions"""

        # High confidence and tight spread -> Market order for speed
        if confidence >= 0.9 and market_data.spread_bps <= 3.0:
            return OrderType.MARKET

        # Medium confidence or wider spread -> Marketable limit
        elif confidence >= 0.7 and market_data.spread_bps <= 8.0:
            return OrderType.MARKETABLE_LIMIT

        # Lower confidence or very wide spread -> Limit order
        elif market_data.spread_bps > 8.0:
            return OrderType.LIMIT

        # Adaptive for uncertain conditions
        else:
            return OrderType.ADAPTIVE

    def calculate_limit_price(
        self, market_data: MarketData, side: str, aggressiveness: float = 0.5
    ) -> float:
        """Calculate optimal limit price"""
        mid_price = (market_data.bid + market_data.ask) / 2
        half_spread = (market_data.ask - market_data.bid) / 2

        if side.lower() == "buy":
            # Buy limit: between mid and ask, based on aggressiveness
            limit_price = mid_price + (half_spread * aggressiveness)
        else:
            # Sell limit: between mid and bid, based on aggressiveness
            limit_price = mid_price - (half_spread * aggressiveness)

        return round(limit_price, 6)

    async def execute_twap_order(
        self, symbol: str, side: str, total_quantity: float, duration_seconds: int = 300
    ) -> List[Dict[str, Any]]:
        """Execute TWAP (Time-Weighted Average Price) order"""
        chunk_size = total_quantity / self.config.twap_chunks
        interval = duration_seconds / self.config.twap_chunks

        executions = []

        for i in range(self.config.twap_chunks):
            # Get fresh market data
            venue, price = self.venue_manager.get_best_venue(symbol, side, chunk_size)

            if venue and price:
                # Simulate execution (in production, call actual API)
                execution = {
                    "chunk": i + 1,
                    "quantity": chunk_size,
                    "price": price,
                    "venue": venue,
                    "timestamp": datetime.now(),
                }
                executions.append(execution)

                logger.info(
                    f"📦 TWAP chunk {i+1}/{self.config.twap_chunks}: "
                    f"{chunk_size:.4f} @ {price:.4f} on {venue}"
                )

            if i < self.config.twap_chunks - 1:
                await asyncio.sleep(interval)

        return executions

    def optimize_execution_strategy(
        self,
        symbol: str,
        side: str,
        quantity: float,
        predicted_edge_bps: float,
        confidence: float,
    ) -> Dict[str, Any]:
        """Main execution optimization logic"""

        # Get best venue and market data
        venue, best_price = self.venue_manager.get_best_venue(symbol, side, quantity)

        if not venue:
            return {"success": False, "reason": "No suitable venue found"}

        market_data = self.venue_manager.venue_data.get(venue)
        if not market_data:
            return {"success": False, "reason": "No market data available"}

        # Calculate execution costs
        execution_costs = self.calculate_execution_cost(
            symbol, side, quantity, market_data
        )

        # Check if trade should proceed
        if not self.should_trade(predicted_edge_bps, execution_costs):
            return {
                "success": False,
                "reason": "Insufficient edge after costs",
                "costs": execution_costs,
            }

        # Select order type
        order_type = self.select_order_type(symbol, side, confidence, market_data)

        # Calculate execution parameters
        if order_type == OrderType.LIMIT:
            execution_price = self.calculate_limit_price(market_data, side, 0.3)
        elif order_type == OrderType.MARKETABLE_LIMIT:
            execution_price = self.calculate_limit_price(market_data, side, 0.8)
        else:
            execution_price = best_price

        execution_plan = {
            "success": True,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "venue": venue,
            "order_type": order_type.value,
            "execution_price": execution_price,
            "market_price": best_price,
            "confidence": confidence,
            "predicted_edge_bps": predicted_edge_bps,
            "execution_costs": execution_costs,
            "spread_bps": market_data.spread_bps,
            "timestamp": datetime.now(),
        }

        return execution_plan

    def store_execution_analytics(self, execution_plan: Dict[str, Any]):
        """Store execution analytics in Redis"""
        if not self.redis_client:
            return

        try:
            # Store individual execution
            key = f"execution:{execution_plan['symbol']}:{int(datetime.now().timestamp())}"
            self.redis_client.setex(key, 3600, json.dumps(execution_plan, default=str))

            # Update aggregated stats
            stats_key = f"execution_stats:{execution_plan['symbol']}"
            current_stats = self.redis_client.get(stats_key)

            if current_stats:
                stats = json.loads(current_stats)
                stats["total_executions"] += 1
                stats["total_costs_bps"] += execution_plan["execution_costs"][
                    "total_cost_bps"
                ]
                stats["avg_cost_bps"] = (
                    stats["total_costs_bps"] / stats["total_executions"]
                )
            else:
                stats = {
                    "total_executions": 1,
                    "total_costs_bps": execution_plan["execution_costs"][
                        "total_cost_bps"
                    ],
                    "avg_cost_bps": execution_plan["execution_costs"]["total_cost_bps"],
                }

            stats["last_updated"] = datetime.now().isoformat()
            self.redis_client.setex(stats_key, 86400, json.dumps(stats))

            logger.info("💾 Execution analytics stored in Redis")

        except Exception as e:
            logger.error(f"Error storing execution analytics: {e}")


def generate_demo_market_data():
    """Generate demo market data for testing"""
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    venues = ["binance", "coinbase", "kraken"]

    market_data = []

    for symbol in symbols:
        base_price = {"BTCUSDT": 45000, "ETHUSDT": 3000, "ADAUSDT": 0.5}[symbol]

        for venue in venues:
            # Add some variation between venues
            price_variation = np.random.normal(0, 0.001)
            mid_price = base_price * (1 + price_variation)

            spread_bps = np.random.uniform(2, 8)  # 2-8 bps spread
            spread = mid_price * (spread_bps / 10000)

            data = MarketData(
                symbol=symbol,
                bid=mid_price - spread / 2,
                ask=mid_price + spread / 2,
                bid_size=np.random.uniform(1, 10),
                ask_size=np.random.uniform(1, 10),
                last_price=mid_price,
                volume=np.random.uniform(100, 1000),
                timestamp=datetime.now(),
                venue=venue,
            )

            market_data.append(data)

    return market_data


async def main():
    """Demo function for Smart Order Routing"""
    print("🚀 Enhanced Smart Order Routing System")
    print("=" * 80)

    # Initialize router
    config = ExecutionConfig(
        max_spread_bps=8.0, min_edge_threshold_bps=12.0, target_latency_ms=100.0
    )

    router = SmartOrderRouter(config)

    # Generate demo market data
    market_data_list = generate_demo_market_data()

    # Update venue manager with market data
    for data in market_data_list:
        router.venue_manager.update_venue_data(data.venue, data)

    print("📊 Market Data Updated:")
    for data in market_data_list[:3]:  # Show first 3
        print(
            f"   {data.symbol} on {data.venue}: "
            f"Bid={data.bid:.4f}, Ask={data.ask:.4f}, "
            f"Spread={data.spread_bps:.1f}bps"
        )

    # Test execution optimization
    print("\n🎯 Testing Execution Optimization:")

    test_cases = [
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 0.1,
            "predicted_edge_bps": 25.0,
            "confidence": 0.85,
        },
        {
            "symbol": "ETHUSDT",
            "side": "sell",
            "quantity": 2.0,
            "predicted_edge_bps": 15.0,
            "confidence": 0.72,
        },
        {
            "symbol": "ADAUSDT",
            "side": "buy",
            "quantity": 1000.0,
            "predicted_edge_bps": 8.0,
            "confidence": 0.60,
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}:")
        print(f"   {case['symbol']} {case['side']} {case['quantity']} units")
        print(
            f"   Predicted edge: {case['predicted_edge_bps']:.1f}bps, Confidence: {case['confidence']:.2f}"
        )

        execution_plan = router.optimize_execution_strategy(**case)

        if execution_plan["success"]:
            print(f"   ✅ Execution approved:")
            print(f"      Venue: {execution_plan['venue']}")
            print(f"      Order type: {execution_plan['order_type']}")
            print(f"      Price: {execution_plan['execution_price']:.4f}")
            print(
                f"      Total cost: {execution_plan['execution_costs']['total_cost_bps']:.1f}bps"
            )

            # Store analytics
            router.store_execution_analytics(execution_plan)
        else:
            print(f"   ❌ Execution rejected: {execution_plan['reason']}")

    # Test TWAP execution
    print(f"\n📦 Testing TWAP Execution:")
    twap_executions = await router.execute_twap_order("BTCUSDT", "buy", 1.0, 60)

    if twap_executions:
        avg_price = np.mean([ex["price"] for ex in twap_executions])
        print(
            f"   ✅ TWAP completed: {len(twap_executions)} chunks, avg price: {avg_price:.4f}"
        )

    print("\n🎉 Smart Order Routing Demo Complete!")
    print("✅ Slippage optimization implemented")
    print("✅ Cost-aware execution logic")
    print("✅ Multi-venue routing")
    print("✅ Adaptive order types")
    print("✅ Real-time analytics")


if __name__ == "__main__":
    asyncio.run(main())
