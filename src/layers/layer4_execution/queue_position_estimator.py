#!/usr/bin/env python3
"""
Queue Position Estimator
Simple queue alpha to estimate position in order book queue and avoid adverse selection.
"""
import numpy as np
import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import redis


@dataclass
class OrderBookSnapshot:
    """Order book snapshot."""

    timestamp: datetime.datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    bid_queue_size: float  # Total size at bid level
    ask_queue_size: float  # Total size at ask level


@dataclass
class QueueEstimate:
    """Queue position estimate."""

    estimated_position: int  # Position in queue (1 = front)
    queue_ahead: float  # Size ahead in queue
    queue_total: float  # Total queue size
    fill_probability: float  # Probability of fill
    estimated_wait_time_ms: float  # Expected wait time


class QueuePositionEstimator:
    def __init__(self, redis_client=None):
        self.r = redis_client or redis.Redis(decode_responses=True)

        # Historical snapshots for flow estimation
        self.book_history = deque(maxlen=100)  # Last 100 snapshots

        # Configuration
        self.flow_estimation_window = 60  # Seconds for flow estimation
        self.min_flow_samples = 5

    def add_book_snapshot(self, snapshot: OrderBookSnapshot):
        """Add new order book snapshot to history."""
        self.book_history.append(snapshot)

    def estimate_flow_rate(self, side: str) -> float:
        """Estimate order flow rate (orders/second) for given side."""
        if len(self.book_history) < self.min_flow_samples:
            # Default flow rates by asset class (mock)
            default_flows = {
                "crypto": 2.5,  # 2.5 orders/second
                "equity": 1.8,  # 1.8 orders/second
                "forex": 3.2,  # 3.2 orders/second
            }
            return default_flows.get("crypto", 2.0)

        # Analyze recent flow from book changes
        recent_snapshots = list(self.book_history)[-self.min_flow_samples :]

        flow_events = 0
        time_window = 0

        for i in range(1, len(recent_snapshots)):
            prev_snap = recent_snapshots[i - 1]
            curr_snap = recent_snapshots[i]

            time_diff = (curr_snap.timestamp - prev_snap.timestamp).total_seconds()
            time_window += time_diff

            # Detect flow events from size changes
            if side == "buy":
                # Size increase at bid = new buy order
                size_change = curr_snap.bid_size - prev_snap.bid_size
                if size_change > 0:
                    flow_events += 1
                # Size decrease = execution/cancellation
                elif size_change < -100:  # Significant decrease
                    flow_events += 0.5  # Partial credit for activity
            else:
                # Size increase at ask = new sell order
                size_change = curr_snap.ask_size - prev_snap.ask_size
                if size_change > 0:
                    flow_events += 1
                elif size_change < -100:
                    flow_events += 0.5

        if time_window > 0:
            estimated_flow = flow_events / time_window
            return max(0.1, min(10.0, estimated_flow))  # Reasonable bounds

        return 2.0  # Default fallback

    def estimate_execution_rate(self, side: str) -> float:
        """Estimate execution/consumption rate at best level."""
        if len(self.book_history) < self.min_flow_samples:
            # Default execution rates
            return 1.2  # executions/second

        recent_snapshots = list(self.book_history)[-self.min_flow_samples :]

        executions = 0
        time_window = 0

        for i in range(1, len(recent_snapshots)):
            prev_snap = recent_snapshots[i - 1]
            curr_snap = recent_snapshots[i]

            time_diff = (curr_snap.timestamp - prev_snap.timestamp).total_seconds()
            time_window += time_diff

            # Detect executions from size decreases
            if side == "buy":
                # Execution at ask (buy order consumes ask liquidity)
                size_decrease = prev_snap.ask_size - curr_snap.ask_size
                if size_decrease > 50:  # Meaningful execution
                    executions += 1
            else:
                # Execution at bid (sell order consumes bid liquidity)
                size_decrease = prev_snap.bid_size - curr_snap.bid_size
                if size_decrease > 50:
                    executions += 1

        if time_window > 0:
            execution_rate = executions / time_window
            return max(0.1, min(5.0, execution_rate))

        return 1.2  # Default

    def estimate_queue_position(
        self, order_size: float, order_side: str, current_snapshot: OrderBookSnapshot
    ) -> QueueEstimate:
        """Estimate position in queue for a new order."""

        # Add current snapshot to history
        self.add_book_snapshot(current_snapshot)

        # Get relevant queue info
        if order_side == "buy":
            # Buy order goes to bid queue
            total_queue = current_snapshot.bid_queue_size
            price_level = current_snapshot.bid_price
        else:
            # Sell order goes to ask queue
            total_queue = current_snapshot.ask_queue_size
            price_level = current_snapshot.ask_price

        # Estimate position based on queue dynamics

        # Simple model: assume we join at back of queue
        # In practice, would use more sophisticated models with:
        # - Order arrival patterns
        # - Price-time priority rules
        # - Hidden order estimation

        estimated_position = int(total_queue / 100) + 1  # Rough position estimate
        queue_ahead = total_queue * 0.7  # Assume 70% of queue ahead

        # Estimate fill probability based on flow rates
        flow_rate = self.estimate_flow_rate(order_side)
        execution_rate = self.estimate_execution_rate(order_side)

        # Queue processing rate (how fast queue moves)
        queue_processing_rate = execution_rate * 100  # orders processed per second

        if queue_processing_rate > 0:
            # Time to reach front of queue
            estimated_wait_time_ms = (queue_ahead / queue_processing_rate) * 1000

            # Fill probability based on typical queue dynamics
            # Higher for smaller orders and more liquid markets
            base_fill_prob = 0.6

            # Size penalty - larger orders less likely to fill
            size_penalty = min(0.4, order_size / 1000)

            # Queue position penalty - back of queue less likely
            position_penalty = min(0.3, estimated_position / 20)

            fill_probability = max(
                0.1, base_fill_prob - size_penalty - position_penalty
            )

        else:
            estimated_wait_time_ms = 60000  # 1 minute fallback
            fill_probability = 0.3  # Low default probability

        return QueueEstimate(
            estimated_position=estimated_position,
            queue_ahead=queue_ahead,
            queue_total=total_queue,
            fill_probability=fill_probability,
            estimated_wait_time_ms=min(estimated_wait_time_ms, 300000),  # Max 5 minutes
        )

    def should_join_queue(
        self,
        queue_estimate: QueueEstimate,
        urgency: float = 0.5,  # 0 = patient, 1 = urgent
        min_fill_prob: float = 0.4,
    ) -> Dict[str, Any]:
        """Decide whether to join queue or cross spread."""

        decision_factors = {
            "fill_probability": queue_estimate.fill_probability,
            "wait_time_ms": queue_estimate.estimated_wait_time_ms,
            "queue_position": queue_estimate.estimated_position,
            "urgency": urgency,
        }

        # Decision logic
        recommendations = []

        # Fill probability check
        if queue_estimate.fill_probability < min_fill_prob:
            recommendations.append("LOW_FILL_PROB")
            should_join = False

        # Wait time vs urgency
        max_acceptable_wait = (1 - urgency) * 120000  # Max wait decreases with urgency
        if queue_estimate.estimated_wait_time_ms > max_acceptable_wait:
            recommendations.append("TOO_SLOW")
            should_join = False

        # Queue position check
        if queue_estimate.estimated_position > 50:
            recommendations.append("BACK_OF_QUEUE")
            should_join = False

        # If no negative factors, recommend joining queue
        if not recommendations:
            recommendations.append("FAVORABLE_QUEUE")
            should_join = True
        else:
            should_join = False

        # Generate reasoning
        if should_join:
            reasoning = (
                f"Good queue conditions: {queue_estimate.fill_probability:.1%} fill probability, "
                f"{queue_estimate.estimated_wait_time_ms/1000:.1f}s expected wait"
            )
        else:
            issues = []
            if queue_estimate.fill_probability < min_fill_prob:
                issues.append(
                    f"low fill probability ({queue_estimate.fill_probability:.1%})"
                )
            if queue_estimate.estimated_wait_time_ms > max_acceptable_wait:
                issues.append(
                    f"long wait time ({queue_estimate.estimated_wait_time_ms/1000:.1f}s)"
                )
            if queue_estimate.estimated_position > 20:
                issues.append(
                    f"poor queue position (#{queue_estimate.estimated_position})"
                )

            reasoning = f"Poor queue conditions: {', '.join(issues)}"

        return {
            "should_join_queue": should_join,
            "recommendation": "JOIN_QUEUE" if should_join else "CROSS_SPREAD",
            "confidence": min(1.0, queue_estimate.fill_probability + 0.3),
            "decision_factors": decision_factors,
            "reasoning": reasoning,
            "alternative_strategy": "IOC" if urgency > 0.7 else "POST_ONLY",
        }

    def get_optimal_posting_strategy(
        self,
        order_size: float,
        order_side: str,
        current_snapshot: OrderBookSnapshot,
        urgency: float = 0.5,
    ) -> Dict[str, Any]:
        """Get optimal posting strategy considering queue dynamics."""

        # Estimate queue position
        queue_estimate = self.estimate_queue_position(
            order_size, order_side, current_snapshot
        )

        # Decide on queue joining
        queue_decision = self.should_join_queue(queue_estimate, urgency)

        # Additional strategy refinements
        spread_bps = (
            (current_snapshot.ask_price - current_snapshot.bid_price)
            / current_snapshot.bid_price
        ) * 10000

        strategy_recommendations = []

        if queue_decision["should_join_queue"]:
            if spread_bps > 15:  # Wide spread
                strategy_recommendations.append(
                    "Consider inside bid/ask for better queue position"
                )

            if queue_estimate.estimated_wait_time_ms < 30000:  # < 30 seconds
                strategy_recommendations.append(
                    "Quick fill expected - good for posting"
                )

            strategy = "POST_AT_TOUCH"

        else:
            if urgency > 0.8:
                strategy = "MARKET_ORDER"
                strategy_recommendations.append(
                    "High urgency - cross spread immediately"
                )
            elif spread_bps < 5:  # Tight spread
                strategy = "IOC_AT_MID"
                strategy_recommendations.append(
                    "Try midpoint first, then cross if no fill"
                )
            else:
                strategy = "IOC_CROSS"
                strategy_recommendations.append(
                    "Cross spread with IOC to limit adverse selection"
                )

        return {
            "strategy": strategy,
            "queue_estimate": {
                "position": queue_estimate.estimated_position,
                "fill_probability": queue_estimate.fill_probability,
                "wait_time_ms": queue_estimate.estimated_wait_time_ms,
            },
            "queue_decision": queue_decision,
            "spread_bps": spread_bps,
            "recommendations": strategy_recommendations,
            "expected_execution_time_ms": (
                queue_estimate.estimated_wait_time_ms
                if queue_decision["should_join_queue"]
                else 100
            ),
        }


# Example usage functions
def create_example_snapshot() -> OrderBookSnapshot:
    """Create example order book snapshot."""
    return OrderBookSnapshot(
        timestamp=datetime.datetime.now(),
        bid_price=50000.0,
        ask_price=50008.5,
        bid_size=1500.0,
        ask_size=1200.0,
        bid_queue_size=3500.0,  # Total at bid level
        ask_queue_size=2800.0,  # Total at ask level
    )


def run_queue_estimation_example():
    """Run example of queue position estimation."""
    estimator = QueuePositionEstimator()
    snapshot = create_example_snapshot()
    order_size = 500.0
    order_side = "buy"
    urgency = 0.6  # Moderate urgency

    # Get optimal strategy
    strategy = estimator.get_optimal_posting_strategy(
        order_size, order_side, snapshot, urgency
    )

    print("Queue Position Analysis:")
    print(f"  Order Size: {order_size}")
    print(f"  Side: {order_side}")
    print(f"  Urgency: {urgency:.1f}")
    print(f"  Strategy: {strategy['strategy']}")
    print(f"  Queue Position: #{strategy['queue_estimate']['position']}")
    print(f"  Fill Probability: {strategy['queue_estimate']['fill_probability']:.1%}")
    print(f"  Expected Wait: {strategy['queue_estimate']['wait_time_ms']/1000:.1f}s")
    print(f"  Spread: {strategy['spread_bps']:.1f} bps")
    print(f"  Decision: {strategy['queue_decision']['reasoning']}")

    if strategy["recommendations"]:
        print("  Recommendations:")
        for rec in strategy["recommendations"]:
            print(f"    - {rec}")


if __name__ == "__main__":
    run_queue_estimation_example()
