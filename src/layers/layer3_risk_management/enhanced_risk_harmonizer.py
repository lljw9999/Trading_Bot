#!/usr/bin/env python3
"""
Enhanced Risk Harmonizer with Cost-Aware Position Sizing
Incorporates slippage costs and execution optimization into position sizing
"""

import numpy as np
import pandas as pd
import redis
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CostModel:
    """Model for transaction costs and slippage"""

    spread_cost_bps: float = 3.0  # Average spread cost
    commission_bps: float = 1.0  # Commission per trade
    slippage_bps: float = 2.0  # Expected slippage
    market_impact_factor: float = 0.5  # Market impact coefficient
    latency_cost_bps: float = 0.5  # Cost due to latency

    def total_cost_bps(self, position_size_usd: float = 0) -> float:
        """Calculate total transaction cost in basis points"""
        # Market impact increases with position size (simplified model)
        market_impact = self.market_impact_factor * np.log(
            1 + position_size_usd / 10000
        )

        return (
            self.spread_cost_bps
            + self.commission_bps
            + self.slippage_bps
            + market_impact
            + self.latency_cost_bps
        )


@dataclass
class EnhancedEdge:
    """Enhanced edge calculation with cost adjustment"""

    raw_edge_bps: float
    confidence: float
    model_weight: float
    cost_adjusted_edge_bps: float = 0.0
    net_edge_bps: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EnhancedRiskHarmonizer:
    """Enhanced Risk Harmonizer with cost-aware position sizing"""

    def __init__(self, base_cost_model: Optional[CostModel] = None):
        self.cost_model = base_cost_model or CostModel()
        self.edge_history = []
        self.position_history = []
        self.performance_tracker = {}

        # Kelly Criterion parameters
        self.kelly_multiplier = 0.25  # Conservative Kelly (25%)
        self.max_position_pct = 0.20  # Maximum 20% position
        self.min_edge_threshold_bps = 10.0  # Minimum edge after costs

        # Risk limits
        self.max_portfolio_var = 0.02  # 2% portfolio VaR
        self.max_leverage = 3.0  # Maximum leverage

        # Redis connection
        try:
            self.redis_client = redis.Redis(
                host="localhost", port=6379, db=0, decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected for enhanced risk harmonizer")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None

    def update_cost_model(self, symbol: str, recent_executions: List[Dict[str, Any]]):
        """Update cost model based on recent execution data"""
        if not recent_executions:
            return

        # Calculate actual slippage from recent executions
        slippages = []
        spreads = []

        for execution in recent_executions:
            if "slippage_bps" in execution:
                slippages.append(abs(execution["slippage_bps"]))
            if "spread_bps" in execution:
                spreads.append(execution["spread_bps"])

        if slippages:
            # Use exponential moving average for slippage
            new_slippage = np.mean(slippages)
            self.cost_model.slippage_bps = (
                0.7 * self.cost_model.slippage_bps + 0.3 * new_slippage
            )

        if spreads:
            new_spread = np.mean(spreads)
            self.cost_model.spread_cost_bps = (
                0.7 * self.cost_model.spread_cost_bps + 0.3 * new_spread
            )

        logger.info(
            f"📊 Updated cost model for {symbol}: "
            f"Slippage={self.cost_model.slippage_bps:.1f}bps, "
            f"Spread={self.cost_model.spread_cost_bps:.1f}bps"
        )

    def calculate_edge_with_costs(
        self, raw_edges: List[Dict[str, Any]], position_size_usd: float = 0
    ) -> EnhancedEdge:
        """Calculate cost-adjusted blended edge"""

        if not raw_edges:
            return EnhancedEdge(0.0, 0.0, 0.0, 0.0, 0.0)

        # Traditional edge blending with confidence weighting
        total_weighted_edge = 0.0
        total_weight = 0.0

        for edge_data in raw_edges:
            edge_bps = edge_data.get("edge_bps", 0.0)
            confidence = edge_data.get("confidence", 0.5)
            model_weight = edge_data.get("weight", 1.0)

            # Apply time decay (newer edges get higher weight)
            time_decay = 0.95 ** edge_data.get("age_seconds", 0)

            effective_weight = confidence * model_weight * time_decay
            total_weighted_edge += edge_bps * effective_weight
            total_weight += effective_weight

        # Calculate blended edge
        if total_weight > 0:
            blended_edge_bps = total_weighted_edge / total_weight
            avg_confidence = total_weight / len(raw_edges)
            avg_weight = total_weight / len(raw_edges)
        else:
            blended_edge_bps = 0.0
            avg_confidence = 0.0
            avg_weight = 0.0

        # Calculate transaction costs
        total_cost_bps = self.cost_model.total_cost_bps(position_size_usd)

        # Cost-adjusted edge
        cost_adjusted_edge_bps = blended_edge_bps - total_cost_bps

        # Net edge after all adjustments
        net_edge_bps = cost_adjusted_edge_bps

        enhanced_edge = EnhancedEdge(
            raw_edge_bps=blended_edge_bps,
            confidence=avg_confidence,
            model_weight=avg_weight,
            cost_adjusted_edge_bps=cost_adjusted_edge_bps,
            net_edge_bps=net_edge_bps,
        )

        logger.info(
            f"📈 Edge calculation: Raw={blended_edge_bps:.1f}bps, "
            f"Costs={total_cost_bps:.1f}bps, Net={net_edge_bps:.1f}bps"
        )

        return enhanced_edge

    def calculate_optimal_position_size(
        self,
        enhanced_edge: EnhancedEdge,
        symbol: str,
        current_price: float,
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """Calculate optimal position size using enhanced Kelly criterion"""

        # Check minimum edge threshold
        if enhanced_edge.net_edge_bps < self.min_edge_threshold_bps:
            return {
                "position_size_usd": 0.0,
                "position_size_units": 0.0,
                "position_pct": 0.0,
                "reason": f"Net edge {enhanced_edge.net_edge_bps:.1f}bps below threshold {self.min_edge_threshold_bps}bps",
                "kelly_fraction": 0.0,
            }

        # Estimate variance for Kelly calculation
        # Use historical edge volatility or default
        edge_variance = self.estimate_edge_variance(symbol)

        # Kelly fraction calculation
        if edge_variance > 0:
            kelly_fraction = (enhanced_edge.net_edge_bps / 10000) / edge_variance
        else:
            kelly_fraction = 0.0

        # Apply Kelly multiplier for conservative sizing
        kelly_fraction *= self.kelly_multiplier

        # Confidence adjustment
        confidence_adjustment = enhanced_edge.confidence
        kelly_fraction *= confidence_adjustment

        # Portfolio percentage
        position_pct = min(kelly_fraction, self.max_position_pct)

        # Position size in USD
        position_size_usd = portfolio_value * position_pct

        # Position size in units
        position_size_units = (
            position_size_usd / current_price if current_price > 0 else 0
        )

        # Risk checks
        risk_adjustments = self.apply_risk_limits(
            position_size_usd, portfolio_value, symbol
        )

        if risk_adjustments["adjusted"]:
            position_size_usd = risk_adjustments["new_position_size_usd"]
            position_size_units = (
                position_size_usd / current_price if current_price > 0 else 0
            )
            position_pct = (
                position_size_usd / portfolio_value if portfolio_value > 0 else 0
            )

        return {
            "position_size_usd": position_size_usd,
            "position_size_units": position_size_units,
            "position_pct": position_pct,
            "kelly_fraction": kelly_fraction,
            "confidence_adjustment": confidence_adjustment,
            "edge_bps": enhanced_edge.net_edge_bps,
            "estimated_variance": edge_variance,
            "risk_adjustments": risk_adjustments,
            "timestamp": datetime.now(),
        }

    def estimate_edge_variance(self, symbol: str, lookback_hours: int = 24) -> float:
        """Estimate edge variance for Kelly calculation"""

        # Get recent edge history for symbol
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)

        recent_edges = [
            edge.net_edge_bps
            for edge in self.edge_history
            if edge.timestamp >= cutoff_time
        ]

        if len(recent_edges) >= 5:
            variance = np.var(recent_edges) / (10000**2)  # Convert bps to decimal
        else:
            # Default variance estimate based on asset type
            default_variances = {
                "BTC": 0.0004,  # 2% daily vol squared
                "ETH": 0.0009,  # 3% daily vol squared
                "stocks": 0.0001,  # 1% daily vol squared
            }

            if "BTC" in symbol.upper():
                variance = default_variances["BTC"]
            elif "ETH" in symbol.upper():
                variance = default_variances["ETH"]
            else:
                variance = default_variances["stocks"]

        return variance

    def apply_risk_limits(
        self, position_size_usd: float, portfolio_value: float, symbol: str
    ) -> Dict[str, Any]:
        """Apply portfolio-level risk limits"""

        risk_adjustments = {
            "adjusted": False,
            "new_position_size_usd": position_size_usd,
            "adjustments_applied": [],
        }

        # Maximum position size check
        max_position_usd = portfolio_value * self.max_position_pct
        if position_size_usd > max_position_usd:
            risk_adjustments["new_position_size_usd"] = max_position_usd
            risk_adjustments["adjusted"] = True
            risk_adjustments["adjustments_applied"].append("max_position_limit")

        # Portfolio VaR check (simplified)
        current_var = self.estimate_portfolio_var()
        if current_var > self.max_portfolio_var:
            var_adjustment = max(
                0.5, 1 - (current_var - self.max_portfolio_var) / self.max_portfolio_var
            )
            risk_adjustments["new_position_size_usd"] *= var_adjustment
            risk_adjustments["adjusted"] = True
            risk_adjustments["adjustments_applied"].append("portfolio_var_limit")

        # Leverage check
        total_exposure = (
            self.calculate_total_exposure() + risk_adjustments["new_position_size_usd"]
        )
        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0

        if leverage > self.max_leverage:
            leverage_adjustment = self.max_leverage / leverage
            risk_adjustments["new_position_size_usd"] *= leverage_adjustment
            risk_adjustments["adjusted"] = True
            risk_adjustments["adjustments_applied"].append("leverage_limit")

        return risk_adjustments

    def estimate_portfolio_var(self, confidence_level: float = 0.95) -> float:
        """Estimate current portfolio VaR"""
        # Simplified VaR calculation
        # In production, this would use proper portfolio VaR model
        return 0.015  # 1.5% placeholder

    def calculate_total_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        # Simplified calculation
        # In production, this would sum all current positions
        return 50000.0  # $50k placeholder

    def run_enhanced_harmonization(
        self,
        symbol: str,
        raw_edges: List[Dict[str, Any]],
        current_price: float,
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """Main enhanced harmonization process"""

        start_time = datetime.now()

        # Step 1: Calculate enhanced edge with costs
        enhanced_edge = self.calculate_edge_with_costs(raw_edges)

        # Step 2: Calculate optimal position size
        position_result = self.calculate_optimal_position_size(
            enhanced_edge, symbol, current_price, portfolio_value
        )

        # Step 3: Store results for analytics
        harmonization_result = {
            "symbol": symbol,
            "raw_edge_bps": enhanced_edge.raw_edge_bps,
            "cost_adjusted_edge_bps": enhanced_edge.cost_adjusted_edge_bps,
            "net_edge_bps": enhanced_edge.net_edge_bps,
            "total_cost_bps": self.cost_model.total_cost_bps(
                position_result["position_size_usd"]
            ),
            "position_size_usd": position_result["position_size_usd"],
            "position_size_units": position_result["position_size_units"],
            "position_pct": position_result["position_pct"],
            "kelly_fraction": position_result["kelly_fraction"],
            "confidence": enhanced_edge.confidence,
            "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "timestamp": datetime.now().isoformat(),
        }

        # Store in Redis for monitoring
        self.store_harmonization_result(harmonization_result)

        # Update history
        self.edge_history.append(enhanced_edge)
        if len(self.edge_history) > 1000:
            self.edge_history = self.edge_history[-1000:]

        logger.info(
            f"🎯 Enhanced harmonization for {symbol}: "
            f"Net edge={enhanced_edge.net_edge_bps:.1f}bps, "
            f"Position=${position_result['position_size_usd']:.0f} "
            f"({position_result['position_pct']:.1%})"
        )

        return harmonization_result

    def store_harmonization_result(self, result: Dict[str, Any]):
        """Store harmonization result in Redis"""
        if not self.redis_client:
            return

        try:
            # Store individual result
            key = f"enhanced_harmonization:{result['symbol']}:{int(datetime.now().timestamp())}"
            self.redis_client.setex(key, 3600, json.dumps(result, default=str))

            # Update aggregated metrics
            metrics_key = f"harmonization_metrics:{result['symbol']}"
            current_metrics = self.redis_client.get(metrics_key)

            if current_metrics:
                metrics = json.loads(current_metrics)
                metrics["total_harmonizations"] += 1
                metrics["avg_processing_time_ms"] = (
                    metrics["avg_processing_time_ms"]
                    * (metrics["total_harmonizations"] - 1)
                    + result["processing_time_ms"]
                ) / metrics["total_harmonizations"]
            else:
                metrics = {
                    "total_harmonizations": 1,
                    "avg_processing_time_ms": result["processing_time_ms"],
                }

            metrics["last_edge_bps"] = result["net_edge_bps"]
            metrics["last_position_pct"] = result["position_pct"]
            metrics["last_updated"] = datetime.now().isoformat()

            self.redis_client.setex(metrics_key, 86400, json.dumps(metrics))

        except Exception as e:
            logger.error(f"Error storing harmonization result: {e}")


def generate_demo_edges():
    """Generate demo edge data for testing"""
    return [
        {
            "model": "order_book_pressure",
            "edge_bps": 15.3,
            "confidence": 0.73,
            "weight": 1.0,
            "age_seconds": 2,
        },
        {
            "model": "momentum_alpha",
            "edge_bps": 22.1,
            "confidence": 0.81,
            "weight": 1.2,
            "age_seconds": 5,
        },
        {
            "model": "news_sentiment",
            "edge_bps": 8.7,
            "confidence": 0.65,
            "weight": 0.8,
            "age_seconds": 10,
        },
        {
            "model": "tft_prediction",
            "edge_bps": 31.2,
            "confidence": 0.89,
            "weight": 1.5,
            "age_seconds": 1,
        },
    ]


def main():
    """Demo function for Enhanced Risk Harmonizer"""
    print("🚀 Enhanced Risk Harmonizer with Cost-Aware Position Sizing")
    print("=" * 80)

    # Initialize harmonizer
    cost_model = CostModel(
        spread_cost_bps=2.5,
        commission_bps=1.0,
        slippage_bps=3.2,
        market_impact_factor=0.3,
    )

    harmonizer = EnhancedRiskHarmonizer(cost_model)

    # Demo edge data
    raw_edges = generate_demo_edges()

    print("📊 Input Edge Data:")
    for edge in raw_edges:
        print(
            f"   {edge['model']}: {edge['edge_bps']:.1f}bps "
            f"(confidence={edge['confidence']:.2f}, weight={edge['weight']:.1f})"
        )

    # Test enhanced harmonization
    test_cases = [
        {"symbol": "BTCUSDT", "price": 45000, "portfolio": 100000},
        {"symbol": "ETHUSDT", "price": 3000, "portfolio": 100000},
        {"symbol": "NVDA", "price": 450, "portfolio": 100000},
    ]

    print(f"\n🎯 Enhanced Harmonization Results:")

    for case in test_cases:
        print(f"\n📋 {case['symbol']} @ ${case['price']:.0f}:")

        result = harmonizer.run_enhanced_harmonization(
            case["symbol"], raw_edges, case["price"], case["portfolio"]
        )

        print(f"   Raw Edge: {result['raw_edge_bps']:.1f}bps")
        print(f"   Transaction Costs: {result['total_cost_bps']:.1f}bps")
        print(f"   Net Edge: {result['net_edge_bps']:.1f}bps")
        print(
            f"   Position Size: ${result['position_size_usd']:.0f} ({result['position_pct']:.1%})"
        )
        print(f"   Kelly Fraction: {result['kelly_fraction']:.3f}")
        print(f"   Processing Time: {result['processing_time_ms']:.1f}ms")

    print("\n🎉 Enhanced Risk Harmonizer Demo Complete!")
    print("✅ Cost-aware edge calculation")
    print("✅ Kelly criterion with transaction costs")
    print("✅ Portfolio-level risk limits")
    print("✅ Real-time performance tracking")
    print("✅ Adaptive cost modeling")


if __name__ == "__main__":
    main()
