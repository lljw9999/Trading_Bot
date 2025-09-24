#!/usr/bin/env python3
"""
Smart Routing Integrator

Integrates the Smart Order Router with the existing trading system,
providing a unified interface for multi-exchange execution.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
import json

from .smart_order_router import SmartOrderRouter, OrderUrgency, ExecutionPlan
from .market_order_executor import MarketOrderExecutor, Order, OrderSide
from ...utils.logger import get_logger


class SmartRoutingIntegrator:
    """
    Integrator that connects Smart Order Router with existing execution infrastructure.

    Provides intelligent routing while maintaining compatibility with the existing
    MarketOrderExecutor interface.
    """

    def __init__(
        self,
        market_executor: Optional[MarketOrderExecutor] = None,
        enable_smart_routing: bool = True,
        min_order_size_for_routing: float = 100.0,
    ):
        """
        Initialize the Smart Routing Integrator.

        Args:
            market_executor: Existing market order executor (fallback)
            enable_smart_routing: Whether to use smart routing
            min_order_size_for_routing: Minimum order size to trigger smart routing
        """
        self.logger = get_logger("execution.smart_routing_integrator")

        # Initialize components
        self.smart_router = SmartOrderRouter() if enable_smart_routing else None
        self.market_executor = market_executor or MarketOrderExecutor()
        self.enable_smart_routing = enable_smart_routing
        self.min_order_size_for_routing = min_order_size_for_routing

        # Execution tracking
        self.active_smart_orders = {}  # {order_id: ExecutionPlan}
        self.order_routing_decisions = []

        # Performance metrics
        self.smart_routing_stats = {
            "total_orders": 0,
            "smart_routed_orders": 0,
            "traditional_orders": 0,
            "avg_cost_savings_bps": 0.0,
            "avg_fill_rate_improvement": 0.0,
        }

        self.logger.info(
            f"Smart Routing Integrator initialized: "
            f"smart_routing={'enabled' if enable_smart_routing else 'disabled'}, "
            f"min_routing_size=${min_order_size_for_routing}"
        )

    async def execute_order(
        self,
        symbol: str,
        target_position: Decimal,
        current_price: Decimal,
        urgency: Optional[OrderUrgency] = None,
        max_impact_bps: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Execute order with optimal routing strategy.

        Args:
            symbol: Trading symbol
            target_position: Target position in dollars
            current_price: Current market price
            urgency: Execution urgency (auto-detected if None)
            max_impact_bps: Maximum acceptable market impact

        Returns:
            Execution result with routing information
        """
        try:
            # Calculate order details
            current_shares = self.market_executor.get_position(symbol)
            current_position_dollars = current_shares * current_price
            position_delta = target_position - current_position_dollars

            order_size_dollars = abs(position_delta)

            # Determine if we should use smart routing
            use_smart_routing = self._should_use_smart_routing(
                symbol, order_size_dollars, urgency
            )

            self.smart_routing_stats["total_orders"] += 1

            if use_smart_routing:
                return await self._execute_with_smart_routing(
                    symbol, position_delta, current_price, urgency, max_impact_bps
                )
            else:
                return await self._execute_with_traditional_method(
                    symbol, target_position, current_price
                )

        except Exception as e:
            self.logger.error(f"Error in integrated execution: {e}")
            raise

    def _should_use_smart_routing(
        self, symbol: str, order_size_dollars: float, urgency: Optional[OrderUrgency]
    ) -> bool:
        """Determine whether to use smart routing for this order."""
        if not self.enable_smart_routing or not self.smart_router:
            return False

        # Check minimum order size
        if order_size_dollars < self.min_order_size_for_routing:
            self.logger.debug(
                f"Order size ${order_size_dollars:.0f} below routing threshold"
            )
            return False

        # Check if symbol is supported by multiple venues
        venue_status = self.smart_router.get_venue_status()
        supporting_venues = sum(
            1
            for venue_data in venue_status.values()
            if symbol in venue_data["supported_symbols"]
        )

        if supporting_venues < 2:
            self.logger.debug(
                f"Symbol {symbol} supported by only {supporting_venues} venues"
            )
            return False

        # For very urgent orders on small amounts, traditional execution might be faster
        if urgency == OrderUrgency.IMMEDIATE and order_size_dollars < 500:
            return False

        return True

    async def _execute_with_smart_routing(
        self,
        symbol: str,
        position_delta: Decimal,
        current_price: Decimal,
        urgency: Optional[OrderUrgency],
        max_impact_bps: float,
    ) -> Dict[str, Any]:
        """Execute order using smart routing."""
        try:
            # Determine order parameters
            side = "buy" if position_delta > 0 else "sell"
            size = abs(position_delta)

            # Auto-detect urgency if not provided
            if urgency is None:
                urgency = self._auto_detect_urgency(symbol, float(size))

            self.logger.info(
                f"Smart routing execution: {symbol} {side} ${size:.2f} ({urgency.value})"
            )

            # Create execution plan
            plan = await self.smart_router.route_order(
                symbol=symbol,
                side=side,
                size=size,
                urgency=urgency,
                max_impact_bps=max_impact_bps,
            )

            # Store plan for tracking
            self.active_smart_orders[plan.original_order_id] = plan

            # Execute the plan
            execution_results = await self.smart_router.execute_plan(plan)

            # Update position in market executor
            if execution_results["total_filled"] > 0:
                filled_shares = execution_results["total_filled"] / current_price
                if side == "sell":
                    filled_shares = -filled_shares

                # Update market executor's position tracking
                current_position = self.market_executor.get_position(symbol)
                new_position = current_position + filled_shares
                self.market_executor.positions[symbol] = new_position

            # Record routing decision
            routing_decision = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "order_size_dollars": float(size),
                "routing_method": "smart",
                "urgency": urgency.value,
                "fragments_count": len(plan.fragments),
                "estimated_cost_bps": plan.estimated_cost_bps,
                "actual_cost_bps": execution_results.get("total_cost_bps", 0.0),
                "fill_rate": execution_results.get("success_rate", 0.0),
            }
            self.order_routing_decisions.append(routing_decision)

            # Update statistics
            self.smart_routing_stats["smart_routed_orders"] += 1
            self._update_routing_stats(routing_decision)

            return {
                "routing_method": "smart",
                "execution_plan": plan,
                "execution_results": execution_results,
                "routing_decision": routing_decision,
                "success": execution_results.get("success_rate", 0.0) > 0.5,
            }

        except Exception as e:
            self.logger.error(f"Smart routing execution failed: {e}")
            # Fallback to traditional execution
            return await self._execute_with_traditional_method(
                symbol,
                position_delta
                + self.market_executor.get_position(symbol) * current_price,
                current_price,
            )

    async def _execute_with_traditional_method(
        self, symbol: str, target_position: Decimal, current_price: Decimal
    ) -> Dict[str, Any]:
        """Execute order using traditional market order executor."""
        try:
            self.logger.info(
                f"Traditional execution: {symbol} target=${target_position:.2f}"
            )

            # Use existing market order executor
            order = await self.market_executor.execute_order(
                symbol=symbol,
                target_position=target_position,
                current_price=current_price,
            )

            # Record routing decision
            position_delta = target_position - self.market_executor.get_position_value(
                symbol, current_price
            )
            routing_decision = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "order_size_dollars": float(abs(position_delta)),
                "routing_method": "traditional",
                "urgency": "immediate",
                "fragments_count": 1,
                "estimated_cost_bps": 2.0,  # Typical market order cost
                "actual_cost_bps": 2.0,
                "fill_rate": 1.0 if order else 0.0,
            }
            self.order_routing_decisions.append(routing_decision)

            # Update statistics
            self.smart_routing_stats["traditional_orders"] += 1

            return {
                "routing_method": "traditional",
                "order": order,
                "routing_decision": routing_decision,
                "success": order is not None,
            }

        except Exception as e:
            self.logger.error(f"Traditional execution failed: {e}")
            raise

    def _auto_detect_urgency(
        self, symbol: str, order_size_dollars: float
    ) -> OrderUrgency:
        """Auto-detect appropriate urgency level based on order characteristics."""
        # Large orders should be patient to minimize impact
        if order_size_dollars > 10000:
            return OrderUrgency.PATIENT

        # Medium orders can be moderate
        elif order_size_dollars > 1000:
            return OrderUrgency.MODERATE

        # Small orders can be immediate
        else:
            return OrderUrgency.IMMEDIATE

    def _update_routing_stats(self, routing_decision: Dict[str, Any]):
        """Update routing performance statistics."""
        # Calculate cost savings compared to traditional execution
        traditional_cost_bps = 2.0  # Assumed traditional cost
        smart_cost_bps = routing_decision.get("actual_cost_bps", 0.0)
        cost_savings = traditional_cost_bps - smart_cost_bps

        # Update running averages
        total_smart = self.smart_routing_stats["smart_routed_orders"]
        if total_smart > 1:
            alpha = 1.0 / total_smart
            self.smart_routing_stats["avg_cost_savings_bps"] = (
                self.smart_routing_stats["avg_cost_savings_bps"] * (1 - alpha)
                + cost_savings * alpha
            )

            fill_rate_improvement = (
                routing_decision.get("fill_rate", 0.0) - 0.95
            )  # vs traditional
            self.smart_routing_stats["avg_fill_rate_improvement"] = (
                self.smart_routing_stats["avg_fill_rate_improvement"] * (1 - alpha)
                + fill_rate_improvement * alpha
            )
        else:
            self.smart_routing_stats["avg_cost_savings_bps"] = cost_savings
            self.smart_routing_stats["avg_fill_rate_improvement"] = (
                routing_decision.get("fill_rate", 0.0) - 0.95
            )

    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive routing analytics."""
        total_orders = self.smart_routing_stats["total_orders"]
        smart_orders = self.smart_routing_stats["smart_routed_orders"]

        analytics = {
            "routing_overview": {
                "total_orders": total_orders,
                "smart_routed": smart_orders,
                "traditional": self.smart_routing_stats["traditional_orders"],
                "smart_routing_rate": (
                    f"{(smart_orders / total_orders * 100):.1f}%"
                    if total_orders > 0
                    else "0.0%"
                ),
            },
            "performance_metrics": {
                "avg_cost_savings_bps": f"{self.smart_routing_stats['avg_cost_savings_bps']:.2f}",
                "avg_fill_rate_improvement": f"{self.smart_routing_stats['avg_fill_rate_improvement']:.2%}",
                "active_smart_orders": len(self.active_smart_orders),
            },
            "configuration": {
                "smart_routing_enabled": self.enable_smart_routing,
                "min_routing_size": self.min_order_size_for_routing,
                "supported_venues": (
                    len(self.smart_router.venues) if self.smart_router else 0
                ),
            },
        }

        # Recent routing decisions
        recent_decisions = (
            self.order_routing_decisions[-10:] if self.order_routing_decisions else []
        )
        analytics["recent_decisions"] = recent_decisions

        # Venue status if smart routing is enabled
        if self.smart_router:
            analytics["venue_status"] = self.smart_router.get_venue_status()
            analytics["routing_stats"] = self.smart_router.get_routing_stats()

        return analytics

    def get_position(self, symbol: str) -> Decimal:
        """Get current position for a symbol."""
        return self.market_executor.get_position(symbol)

    def get_position_value(self, symbol: str, current_price: Decimal) -> Decimal:
        """Get current position value in dollars."""
        return self.market_executor.get_position_value(symbol, current_price)

    def get_portfolio_value(self, current_prices: Dict[str, Decimal]) -> Decimal:
        """Calculate total portfolio value."""
        return self.market_executor.get_portfolio_value(current_prices)

    async def cancel_smart_order(self, plan_id: str) -> bool:
        """Cancel an active smart order plan."""
        if plan_id in self.active_smart_orders:
            # In a real implementation, this would cancel individual fragments
            self.logger.info(f"Cancelling smart order plan: {plan_id}")
            del self.active_smart_orders[plan_id]
            return True
        return False

    def optimize_routing_parameters(self):
        """Optimize routing parameters based on historical performance."""
        if len(self.order_routing_decisions) < 10:
            return

        # Analyze recent performance
        recent_decisions = self.order_routing_decisions[-50:]
        smart_decisions = [
            d for d in recent_decisions if d["routing_method"] == "smart"
        ]

        if len(smart_decisions) < 5:
            return

        # Calculate average performance metrics
        avg_cost = sum(d["actual_cost_bps"] for d in smart_decisions) / len(
            smart_decisions
        )
        avg_fill_rate = sum(d["fill_rate"] for d in smart_decisions) / len(
            smart_decisions
        )

        # Adjust minimum routing size if performance is poor
        if avg_cost > 5.0 or avg_fill_rate < 0.8:
            self.min_order_size_for_routing *= 1.2  # Increase threshold
            self.logger.info(
                f"Increased min routing size to ${self.min_order_size_for_routing:.0f}"
            )
        elif avg_cost < 1.0 and avg_fill_rate > 0.95:
            self.min_order_size_for_routing *= 0.9  # Decrease threshold
            self.logger.info(
                f"Decreased min routing size to ${self.min_order_size_for_routing:.0f}"
            )


async def main():
    """Test the Smart Routing Integrator."""
    print("🚀 Testing Smart Routing Integrator...")

    # Initialize integrator
    integrator = SmartRoutingIntegrator(
        enable_smart_routing=True, min_order_size_for_routing=100.0
    )

    # Test different order scenarios
    test_orders = [
        ("BTCUSDT", Decimal("50"), Decimal("50000")),  # Small order - traditional
        ("BTCUSDT", Decimal("500"), Decimal("50000")),  # Medium order - smart routing
        ("ETHUSDT", Decimal("5000"), Decimal("3500")),  # Large order - smart routing
    ]

    for i, (symbol, target_position, current_price) in enumerate(test_orders):
        print(
            f"\n📈 Test Order {i+1}: {symbol} target=${target_position} @ ${current_price}"
        )

        try:
            result = await integrator.execute_order(
                symbol=symbol,
                target_position=target_position,
                current_price=current_price,
            )

            print(f"  ✅ Routing method: {result['routing_method']}")
            print(f"  📊 Success: {result['success']}")

            if result["routing_method"] == "smart":
                exec_results = result["execution_results"]
                print(f"  💰 Cost: {exec_results.get('total_cost_bps', 0):.1f}bps")
                print(f"  📈 Fill rate: {exec_results.get('success_rate', 0):.1%}")
                print(
                    f"  ⏱️  Time: {exec_results.get('execution_time_seconds', 0):.1f}s"
                )

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Show analytics
    print(f"\n📊 Routing Analytics:")
    analytics = integrator.get_routing_analytics()

    print(f"  Overview: {analytics['routing_overview']}")
    print(f"  Performance: {analytics['performance_metrics']}")
    print(f"  Configuration: {analytics['configuration']}")

    if analytics.get("recent_decisions"):
        print(f"  Recent decisions: {len(analytics['recent_decisions'])}")
        for decision in analytics["recent_decisions"][-3:]:
            print(
                f"    {decision['symbol']} ${decision['order_size_dollars']:.0f} "
                f"({decision['routing_method']}) - {decision['actual_cost_bps']:.1f}bps"
            )

    print("🎉 Smart Routing Integrator test completed!")


if __name__ == "__main__":
    asyncio.run(main())
