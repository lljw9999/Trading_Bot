"""
Smart Order Router (SOR) with Adaptive Venue Weights
Routes orders to optimal venues based on latency and liquidity
"""

import redis
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderRequest:
    """Order request structure."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: str = "market"
    price: Optional[float] = None


@dataclass
class VenueQuote:
    """Quote from a trading venue."""

    venue: str
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp: float
    latency_ms: float


class SmartOrderRouter:
    """
    Smart Order Router with latency-aware venue selection.

    Features:
    - Real-time latency weighting
    - Liquidity-aware routing
    - Slippage minimization
    - Queue loss reduction
    """

    VENUE_APIS = {
        "binance": "https://api.binance.com",
        "coinbase": "https://api.exchange.coinbase.com",
        "kraken": "https://api.kraken.com",
        "bybit": "https://api.bybit.com",
        "okx": "https://www.okx.com",
    }

    def __init__(self):
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.logger = logging.getLogger("smart_router")

        # Default liquidity scores (would be updated from real market data)
        self.default_liquidity_scores = {
            "binance": 1.0,  # Highest liquidity
            "coinbase": 0.8,  # Good liquidity
            "kraken": 0.6,  # Medium liquidity
            "bybit": 0.7,  # Good liquidity
            "okx": 0.75,  # Good liquidity
        }

        self.logger.info("🎯 Smart Order Router initialized")

    def get_venue_latencies(self) -> Dict[str, float]:
        """Get current venue latencies from Redis."""
        try:
            latency_data = self.redis.hgetall("latency:venue")
            latencies = {}

            for venue, latency_str in latency_data.items():
                try:
                    latencies[venue] = float(latency_str)
                except (ValueError, TypeError):
                    latencies[venue] = 100.0  # Default high latency

            return latencies

        except Exception as e:
            self.logger.error(f"Error getting venue latencies: {e}")
            # Return default latencies
            return {venue: 50.0 for venue in self.VENUE_APIS.keys()}

    def calculate_venue_weights(
        self, liquidity_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate venue weights based on latency and liquidity."""
        latencies = self.get_venue_latencies()
        liquidity_scores = liquidity_scores or self.default_liquidity_scores

        weights = {}

        for venue in self.VENUE_APIS.keys():
            latency_ms = latencies.get(venue, 100.0)
            liquidity = liquidity_scores.get(venue, 1.0)

            # weight = 1 / (latency_ms + tiny) as specified in task brief
            tiny = 1.0  # Prevent division by zero
            latency_weight = 1.0 / (latency_ms + tiny)

            # Combined score: max(weight × liquidity_score) as specified
            composite_score = latency_weight * liquidity
            weights[venue] = composite_score

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {venue: w / total_weight for venue, w in weights.items()}
        else:
            # Fallback to equal weights
            weights = {
                venue: 1.0 / len(self.VENUE_APIS) for venue in self.VENUE_APIS.keys()
            }

        return weights

    def get_optimal_venue(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        liquidity_scores: Optional[Dict[str, float]] = None,
    ) -> str:
        """Get optimal venue for order execution."""
        try:
            weights = self.calculate_venue_weights(liquidity_scores)
            latencies = self.get_venue_latencies()

            # Log routing decision factors
            self.logger.debug(f"Routing {side.value} {quantity} {symbol}")
            self.logger.debug(
                f"Latencies: {', '.join(f'{v}={l:.1f}ms' for v, l in sorted(latencies.items()))}"
            )

            # Select venue with highest composite score
            optimal_venue = max(weights, key=weights.get)
            optimal_score = weights[optimal_venue]
            optimal_latency = latencies.get(optimal_venue, 100.0)

            self.logger.info(
                f"📍 Optimal venue: {optimal_venue} (score: {optimal_score:.3f}, latency: {optimal_latency:.1f}ms)"
            )

            # Store routing decision in Redis for analytics
            routing_data = {
                "venue": optimal_venue,
                "timestamp": int(time.time()),
                "symbol": symbol,
                "side": side.value,
                "quantity": quantity,
                "latency_ms": optimal_latency,
                "composite_score": optimal_score,
            }

            self.redis.lpush("routing:decisions", str(routing_data))
            self.redis.ltrim("routing:decisions", 0, 999)  # Keep last 1000 decisions

            return optimal_venue

        except Exception as e:
            self.logger.error(f"Error selecting optimal venue: {e}")
            return "binance"  # Safe default

    def estimate_execution_cost(
        self, venue: str, symbol: str, side: OrderSide, quantity: float
    ) -> Dict[str, float]:
        """Estimate execution costs for venue comparison."""
        try:
            latencies = self.get_venue_latencies()
            venue_latency = latencies.get(venue, 50.0)

            # Base trading fees by venue (simplified)
            fee_schedules = {
                "binance": 0.001,  # 0.1%
                "coinbase": 0.005,  # 0.5%
                "kraken": 0.0026,  # 0.26%
                "bybit": 0.001,  # 0.1%
                "okx": 0.001,  # 0.1%
            }

            base_fee = fee_schedules.get(venue, 0.001)

            # Latency impact on slippage (higher latency = more slippage risk)
            latency_slippage = venue_latency * 0.00001  # 1bp per 100ms latency

            # Market impact (simplified model based on quantity)
            market_impact = min(0.005, quantity * 0.000001)  # Cap at 50bp

            total_cost = base_fee + latency_slippage + market_impact

            return {
                "base_fee": base_fee,
                "latency_slippage": latency_slippage,
                "market_impact": market_impact,
                "total_cost": total_cost,
                "venue_latency_ms": venue_latency,
            }

        except Exception as e:
            self.logger.error(f"Error estimating execution cost: {e}")
            return {"total_cost": 0.001, "venue_latency_ms": 50.0}

    def route_order(
        self,
        order: OrderRequest,
        liquidity_scores: Optional[Dict[str, float]] = None,
        model_price: Optional[float] = None,
        account_equity: float = 100000.0,
    ) -> Dict[str, any]:
        """Route order to optimal venue with Kelly position sizing."""
        start_time = time.perf_counter()

        try:
            # Get optimal venue
            optimal_venue = self.get_optimal_venue(
                order.symbol, order.side, order.quantity, liquidity_scores
            )

            # Apply Kelly position sizing if model price provided
            if model_price is not None:
                from src.risk.kelly_vol_sizer import compute_size

                # Get current market prices (simplified - would use real market data)
                best_bid = model_price * 0.9995  # Mock bid
                best_ask = model_price * 1.0005  # Mock ask

                # Calculate edge as specified in task brief
                if order.side == OrderSide.BUY:
                    edge = model_price - best_bid
                else:
                    edge = best_ask - model_price

                # Get Kelly-optimal size fraction
                size_frac = compute_size(order.symbol, edge)

                # Calculate quantity: size_frac * account_equity / price
                kelly_qty = abs(size_frac * account_equity / model_price)

                # Update order quantity with Kelly sizing
                original_qty = order.quantity
                order.quantity = kelly_qty

                self.logger.info(
                    f"🎯 Kelly sizing: {original_qty:.6f} → {kelly_qty:.6f} (edge: {edge:.4f}, size_frac: {size_frac:.4f})"
                )

            # Apply liquidity-aware spread optimization if model price provided
            limit_price = None
            if model_price is not None:
                from execution.spread_optimizer import optimal_offset

                # Get market data (simplified - would use real order book)
                mid_px = model_price
                raw_spread_bp = 5.0  # 5 basis points typical crypto spread
                book_depth_bp = 10.0  # Book depth penalty

                # Calculate optimal offset as specified in task brief
                offset_bp = optimal_offset(raw_spread_bp, book_depth_bp)

                # Calculate limit price: mid_px * (1 + offset_bp * 1e-4 * (1 if side=="BUY" else -1))
                side_multiplier = 1 if order.side == OrderSide.BUY else -1
                limit_price = mid_px * (1 + offset_bp * 1e-4 * side_multiplier)

                self.logger.info(
                    f"📊 Spread optimization: offset {offset_bp:.2f}bp, limit ${limit_price:.2f}"
                )

            # Estimate execution costs
            execution_cost = self.estimate_execution_cost(
                optimal_venue, order.symbol, order.side, order.quantity
            )

            routing_time_ms = (time.perf_counter() - start_time) * 1000

            routing_result = {
                "venue": optimal_venue,
                "order": order,
                "estimated_cost": execution_cost,
                "routing_time_ms": routing_time_ms,
                "timestamp": time.time(),
                "limit_price": limit_price,
                "kelly_sized": model_price is not None,
            }

            self.logger.info(
                f"✅ Order routed to {optimal_venue} in {routing_time_ms:.1f}ms"
            )

            # Log order to immutable audit trail as specified in task brief
            try:
                from audit.ledger import log_order

                order_payload = {
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": order.quantity,
                    "venue": optimal_venue,
                    "timestamp": routing_result["timestamp"],
                    "limit_price": limit_price,
                    "kelly_sized": routing_result["kelly_sized"],
                }

                cid = log_order(order_payload)
                routing_result["audit_cid"] = cid
                self.logger.info(f"📝 Order logged to IPFS: {cid}")

            except Exception as e:
                self.logger.warning(f"⚠️ Audit logging failed: {e}")
                routing_result["audit_error"] = str(e)

            return routing_result

        except Exception as e:
            self.logger.error(f"Error routing order: {e}")
            # Return fallback routing
            return {
                "venue": "binance",
                "order": order,
                "estimated_cost": {"total_cost": 0.001},
                "routing_time_ms": (time.perf_counter() - start_time) * 1000,
                "error": str(e),
            }

    def get_routing_analytics(self) -> Dict[str, any]:
        """Get routing performance analytics."""
        try:
            # Get recent routing decisions
            recent_decisions = self.redis.lrange("routing:decisions", 0, 99)

            if not recent_decisions:
                return {"message": "No routing decisions recorded"}

            venue_counts = {}
            total_decisions = len(recent_decisions)

            for decision_str in recent_decisions:
                try:
                    decision = eval(decision_str)  # Note: Use json.loads in production
                    venue = decision.get("venue", "unknown")
                    venue_counts[venue] = venue_counts.get(venue, 0) + 1
                except:
                    continue

            venue_percentages = {
                venue: (count / total_decisions) * 100
                for venue, count in venue_counts.items()
            }

            current_weights = self.calculate_venue_weights()
            current_latencies = self.get_venue_latencies()

            return {
                "total_decisions": total_decisions,
                "venue_distribution": venue_percentages,
                "current_weights": current_weights,
                "current_latencies": current_latencies,
                "best_venue": max(current_weights, key=current_weights.get),
            }

        except Exception as e:
            self.logger.error(f"Error getting routing analytics: {e}")
            return {"error": str(e)}


def create_smart_router() -> SmartOrderRouter:
    """Factory function for creating smart router."""
    return SmartOrderRouter()


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    router = create_smart_router()

    # Example order routing
    test_order = OrderRequest(symbol="BTCUSDT", side=OrderSide.BUY, quantity=0.1)

    result = router.route_order(test_order)
    print(f"Routing result: {result}")

    # Show analytics
    analytics = router.get_routing_analytics()
    print(f"Routing analytics: {analytics}")
