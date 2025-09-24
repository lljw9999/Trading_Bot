#!/usr/bin/env python3
"""
Execution TTL & Reprice Watchdog
Prevents stuck orders via auto-cancel/reprice using latency & spread context
"""

import os
import sys
import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

import redis
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("ttl_watchdog")


class ExecutionTTLWatchdog:
    """Execution TTL and repricing watchdog."""

    def __init__(self):
        """Initialize TTL watchdog."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # TTL and repricing configuration
        self.config = {
            "ttl_seconds": 3.0,  # Maximum order age in seconds
            "latency_threshold_ms": 150,  # Maximum queue latency in ms
            "check_interval": 0.2,  # Check every 200ms
            "reprice_enabled": True,  # Enable repricing (from ParamServer)
            "max_reprices": 5,  # Max reprices per order
            "min_spread_bps": 2,  # Minimum spread for repricing
            "aggressive_timeout": 5.0,  # Fallback to aggressive after this time
            "ioc_fallback": True,  # Use IOC when repricing fails
            "market_fallback": True,  # Use market orders as last resort
        }

        # Supported symbols and their configs
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.symbol_configs = {
            "BTCUSDT": {"tick_size": 0.01, "min_qty": 1e-6},
            "ETHUSDT": {"tick_size": 0.01, "min_qty": 1e-6},
            "SOLUSDT": {"tick_size": 0.001, "min_qty": 1e-3},
        }

        # Track watchdog state
        self.active_orders = {}  # order_id -> order_info
        self.reprice_history = []
        self.total_checks = 0
        self.total_reprices = 0
        self.total_cancellations = 0
        self.total_timeouts = 0

        # Prometheus metrics
        self.metrics = {
            "exec_order_age_max": 0.0,
            "exec_reprices": 0,
            "exec_timeouts": 0,
            "exec_cancelled": 0,
            "exec_fallback_ioc": 0,
            "exec_fallback_market": 0,
        }

        logger.info("⏰ Execution TTL Watchdog initialized")
        logger.info(f"   TTL: {self.config['ttl_seconds']}s")
        logger.info(f"   Latency threshold: {self.config['latency_threshold_ms']}ms")
        logger.info(f"   Check interval: {self.config['check_interval']}s")
        logger.info(f"   Symbols: {self.symbols}")

    def get_param_server_flags(self) -> Dict[str, bool]:
        """Get flags from parameter server."""
        try:
            flags = {}

            # Get reprice enabled flag
            reprice_flag = self.redis.hget("features:flags", "REPRICE_ENABLED")
            flags["reprice_enabled"] = bool(int(reprice_flag)) if reprice_flag else True

            # Get per-symbol flags
            for symbol in self.symbols:
                symbol_flag = self.redis.hget("features:flags", f"REPRICE_{symbol}")
                flags[f"reprice_{symbol}"] = (
                    bool(int(symbol_flag)) if symbol_flag else True
                )

            return flags

        except Exception as e:
            logger.error(f"Error getting param server flags: {e}")
            return {"reprice_enabled": True}

    def get_live_orders(self) -> Dict[str, Dict]:
        """Get live orders from Redis."""
        try:
            orders = {}

            # Get orders from exec:orders:live stream
            order_stream = self.redis.xrevrange("exec:orders:live", "+", "-", count=100)

            for stream_id, order_data in order_stream:
                try:
                    order_id = order_data.get("order_id", stream_id)
                    symbol = order_data.get("symbol", "")
                    side = order_data.get("side", "")
                    order_type = order_data.get("type", "limit")
                    quantity = float(order_data.get("quantity", 0))
                    price = float(order_data.get("price", 0))
                    status = order_data.get("status", "open")
                    created_time = float(order_data.get("created_time", time.time()))

                    # Only process open/working orders
                    if status in ["open", "new", "working", "partially_filled"]:
                        orders[order_id] = {
                            "order_id": order_id,
                            "symbol": symbol,
                            "side": side,
                            "type": order_type,
                            "quantity": quantity,
                            "price": price,
                            "status": status,
                            "created_time": created_time,
                            "age": time.time() - created_time,
                            "reprice_count": 0,
                            "stream_id": stream_id,
                        }

                except Exception as e:
                    logger.warning(f"Error parsing order data: {e}")

            # If no orders from stream, create mock orders for testing
            if not orders:
                current_time = time.time()
                orders = {
                    "mock_order_1": {
                        "order_id": "mock_order_1",
                        "symbol": "BTCUSDT",
                        "side": "buy",
                        "type": "limit",
                        "quantity": 0.001,
                        "price": 97450.0,
                        "status": "working",
                        "created_time": current_time
                        - 4.5,  # 4.5 seconds old (exceeds TTL)
                        "age": 4.5,
                        "reprice_count": 1,
                        "stream_id": "mock_stream_1",
                    },
                    "mock_order_2": {
                        "order_id": "mock_order_2",
                        "symbol": "ETHUSDT",
                        "side": "sell",
                        "type": "limit",
                        "quantity": 0.1,
                        "price": 3505.0,
                        "status": "working",
                        "created_time": current_time
                        - 1.2,  # 1.2 seconds old (within TTL)
                        "age": 1.2,
                        "reprice_count": 0,
                        "stream_id": "mock_stream_2",
                    },
                }
                logger.debug("Using mock orders for testing")

            return orders

        except Exception as e:
            logger.error(f"Error getting live orders: {e}")
            return {}

    def get_queue_latency(self, symbol: str) -> float:
        """Get current queue latency for symbol."""
        try:
            # Try to get from Redis latency tracking
            latency_key = f"latency:{symbol.lower()}:queue_ms"
            latency = self.redis.get(latency_key)

            if latency:
                return float(latency)

            # Fallback to hop latency
            hop_latency = self.redis.hget("lat_hops", symbol.lower())
            if hop_latency:
                return float(hop_latency)

            # Mock realistic latencies for demo
            import random

            base_latency = {"BTCUSDT": 45, "ETHUSDT": 55, "SOLUSDT": 85}
            mock_latency = base_latency.get(symbol, 60) + random.uniform(-15, 30)

            return max(0, mock_latency)

        except Exception as e:
            logger.warning(f"Error getting queue latency for {symbol}: {e}")
            return 60.0  # Default latency

    def get_optimal_offset(self, symbol: str, side: str, current_price: float) -> float:
        """Calculate optimal price offset for repricing."""
        try:
            # Get current market data
            spread_bps = self.get_current_spread_bps(symbol)

            # Basic spread optimizer - target middle of spread
            if side.lower() == "buy":
                # For buys, improve price (bid up)
                offset_bps = min(spread_bps * 0.4, 5.0)  # 40% of spread, max 5bps
                return current_price * (1 + offset_bps / 10000)
            else:
                # For sells, improve price (offer down)
                offset_bps = min(spread_bps * 0.4, 5.0)
                return current_price * (1 - offset_bps / 10000)

        except Exception as e:
            logger.warning(f"Error calculating optimal offset: {e}")
            # Fallback: small improvement
            if side.lower() == "buy":
                return current_price * 1.0001  # 1bp improvement
            else:
                return current_price * 0.9999

    def get_current_spread_bps(self, symbol: str) -> float:
        """Get current bid-ask spread in bps."""
        try:
            # Try to get spread from Redis
            spread_key = f"spread:{symbol.lower()}:bps"
            spread = self.redis.get(spread_key)

            if spread:
                return float(spread)

            # Mock realistic spreads
            spreads = {"BTCUSDT": 2.5, "ETHUSDT": 3.0, "SOLUSDT": 8.0}
            return spreads.get(symbol, 5.0)

        except Exception as e:
            logger.warning(f"Error getting spread for {symbol}: {e}")
            return 5.0

    async def should_reprice_order(self, order: Dict) -> Tuple[bool, str]:
        """Check if order should be repriced."""
        try:
            order_id = order["order_id"]
            symbol = order["symbol"]
            age = order["age"]
            reprice_count = order["reprice_count"]

            # Check param server flags
            flags = self.get_param_server_flags()
            if not flags.get("reprice_enabled", True):
                return False, "repricing_disabled_global"

            symbol_flag = flags.get(f"reprice_{symbol}", True)
            if not symbol_flag:
                return False, f"repricing_disabled_{symbol}"

            # Check max reprices
            if reprice_count >= self.config["max_reprices"]:
                return True, "max_reprices_exceeded"  # Should cancel/fallback

            # Check TTL
            if age > self.config["ttl_seconds"]:
                return True, "ttl_exceeded"

            # Check queue latency
            queue_latency = self.get_queue_latency(symbol)
            if queue_latency > self.config["latency_threshold_ms"]:
                return True, "latency_exceeded"

            # Check if spread is wide enough for repricing
            spread_bps = self.get_current_spread_bps(symbol)
            if spread_bps < self.config["min_spread_bps"]:
                return False, "spread_too_tight"

            return False, "no_reprice_needed"

        except Exception as e:
            logger.error(f"Error checking reprice condition: {e}")
            return False, "check_error"

    async def reprice_order(self, order: Dict) -> Dict[str, Any]:
        """Reprice an order."""
        try:
            order_id = order["order_id"]
            symbol = order["symbol"]
            side = order["side"]
            current_price = order["price"]
            quantity = order["quantity"]

            # Calculate new price
            new_price = self.get_optimal_offset(symbol, side, current_price)

            # Round to tick size
            tick_size = self.symbol_configs.get(symbol, {}).get("tick_size", 0.01)
            new_price = round(new_price / tick_size) * tick_size

            # Simulate order repricing (would call real exchange API)
            reprice_result = {
                "success": True,
                "old_order_id": order_id,
                "new_order_id": f"{order_id}_reprice_{int(time.time())}",
                "old_price": current_price,
                "new_price": new_price,
                "price_improvement_bps": abs(
                    (new_price - current_price) / current_price * 10000
                ),
                "timestamp": time.time(),
                "method": "reprice",
            }

            # Log reprice event
            event_data = {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "old_price": current_price,
                "new_price": new_price,
                "quantity": quantity,
                "reason": "ttl_reprice",
                "timestamp": time.time(),
            }

            self.redis.xadd("exec:reprice_events", event_data)

            # Update metrics
            self.total_reprices += 1
            self.metrics["exec_reprices"] = self.total_reprices

            # Track reprice history
            self.reprice_history.append(
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "price_improvement_bps": reprice_result["price_improvement_bps"],
                    "timestamp": time.time(),
                }
            )

            if len(self.reprice_history) > 1000:
                self.reprice_history = self.reprice_history[-800:]

            logger.info(
                f"📈 Repriced {symbol} {side} order: "
                f"{current_price:.2f} → {new_price:.2f} "
                f"({reprice_result['price_improvement_bps']:.1f}bps)"
            )

            return reprice_result

        except Exception as e:
            logger.error(f"Error repricing order {order.get('order_id')}: {e}")
            return {"success": False, "error": str(e)}

    async def cancel_order(self, order: Dict, reason: str) -> Dict[str, Any]:
        """Cancel an order."""
        try:
            order_id = order["order_id"]
            symbol = order["symbol"]

            # Simulate order cancellation
            cancel_result = {
                "success": True,
                "order_id": order_id,
                "symbol": symbol,
                "reason": reason,
                "timestamp": time.time(),
                "method": "cancel",
            }

            # Log cancellation event
            event_data = {
                "order_id": order_id,
                "symbol": symbol,
                "side": order["side"],
                "price": order["price"],
                "quantity": order["quantity"],
                "reason": reason,
                "timestamp": time.time(),
                "action": "cancel",
            }

            self.redis.xadd("exec:reprice_events", event_data)

            # Update metrics
            self.total_cancellations += 1
            self.metrics["exec_cancelled"] = self.total_cancellations

            logger.info(f"❌ Cancelled {symbol} order {order_id}: {reason}")

            return cancel_result

        except Exception as e:
            logger.error(f"Error cancelling order {order.get('order_id')}: {e}")
            return {"success": False, "error": str(e)}

    async def fallback_to_ioc(self, order: Dict) -> Dict[str, Any]:
        """Fallback to IOC order."""
        try:
            # Simulate IOC order submission
            ioc_result = {
                "success": True,
                "original_order_id": order["order_id"],
                "ioc_order_id": f"IOC_{order['order_id']}_{int(time.time())}",
                "symbol": order["symbol"],
                "side": order["side"],
                "quantity": order["quantity"],
                "timestamp": time.time(),
                "method": "ioc_fallback",
            }

            # Update metrics
            self.metrics["exec_fallback_ioc"] += 1

            logger.info(
                f"⚡ IOC fallback for {order['symbol']} order {order['order_id']}"
            )

            return ioc_result

        except Exception as e:
            logger.error(f"Error with IOC fallback: {e}")
            return {"success": False, "error": str(e)}

    async def fallback_to_market(self, order: Dict) -> Dict[str, Any]:
        """Fallback to market order."""
        try:
            # Simulate market order submission
            market_result = {
                "success": True,
                "original_order_id": order["order_id"],
                "market_order_id": f"MKT_{order['order_id']}_{int(time.time())}",
                "symbol": order["symbol"],
                "side": order["side"],
                "quantity": order["quantity"],
                "timestamp": time.time(),
                "method": "market_fallback",
            }

            # Update metrics
            self.metrics["exec_fallback_market"] += 1

            logger.warning(
                f"🚨 Market fallback for {order['symbol']} order {order['order_id']}"
            )

            return market_result

        except Exception as e:
            logger.error(f"Error with market fallback: {e}")
            return {"success": False, "error": str(e)}

    async def process_order(self, order: Dict) -> Dict[str, Any]:
        """Process a single order for TTL/reprice logic."""
        try:
            order_id = order["order_id"]
            symbol = order["symbol"]
            age = order["age"]
            reprice_count = order["reprice_count"]

            # Check if order needs repricing
            should_reprice, reason = await self.should_reprice_order(order)

            if not should_reprice:
                return {"action": "none", "reason": reason}

            # Handle different reprice scenarios
            if reason == "max_reprices_exceeded":
                # Too many reprices, cancel and potentially fallback
                await self.cancel_order(order, "max_reprices_exceeded")

                if age > self.config["aggressive_timeout"]:
                    if self.config["market_fallback"]:
                        return await self.fallback_to_market(order)
                    elif self.config["ioc_fallback"]:
                        return await self.fallback_to_ioc(order)

                return {"action": "cancel", "reason": "max_reprices_exceeded"}

            elif reason in ["ttl_exceeded", "latency_exceeded"]:
                # Try repricing first
                if reprice_count < self.config["max_reprices"]:
                    reprice_result = await self.reprice_order(order)
                    if reprice_result["success"]:
                        return {"action": "reprice", "result": reprice_result}

                # Repricing failed or not allowed, try fallbacks
                if age > self.config["aggressive_timeout"]:
                    if self.config["ioc_fallback"]:
                        return await self.fallback_to_ioc(order)
                    elif self.config["market_fallback"]:
                        return await self.fallback_to_market(order)

                # Last resort: cancel
                await self.cancel_order(order, reason)
                return {"action": "cancel", "reason": reason}

            return {"action": "none", "reason": "no_action_needed"}

        except Exception as e:
            logger.error(f"Error processing order {order.get('order_id')}: {e}")
            return {"action": "error", "error": str(e)}

    async def run_watchdog_cycle(self) -> Dict[str, Any]:
        """Run one watchdog cycle."""
        try:
            cycle_start = time.time()
            self.total_checks += 1

            # Get live orders
            orders = self.get_live_orders()

            if not orders:
                return {
                    "timestamp": cycle_start,
                    "status": "no_orders",
                    "orders_processed": 0,
                    "cycle_duration": time.time() - cycle_start,
                }

            # Process each order
            actions_taken = {
                "reprice": 0,
                "cancel": 0,
                "ioc_fallback": 0,
                "market_fallback": 0,
                "none": 0,
                "error": 0,
            }

            max_order_age = 0

            for order_id, order in orders.items():
                try:
                    # Update order age
                    order["age"] = time.time() - order["created_time"]
                    max_order_age = max(max_order_age, order["age"])

                    # Process order
                    result = await self.process_order(order)
                    action = result.get("action", "none")
                    actions_taken[action] = actions_taken.get(action, 0) + 1

                    # Update active orders tracking
                    if action in ["cancel", "market_fallback"]:
                        # Order is no longer active
                        if order_id in self.active_orders:
                            del self.active_orders[order_id]
                    else:
                        # Update order info
                        self.active_orders[order_id] = order

                except Exception as e:
                    logger.error(f"Error processing order {order_id}: {e}")
                    actions_taken["error"] += 1

            # Update metrics
            self.metrics["exec_order_age_max"] = max_order_age

            cycle_duration = time.time() - cycle_start

            # Log summary periodically
            if self.total_checks % 50 == 0:  # Every 50 cycles
                logger.info(
                    f"🔄 Cycle #{self.total_checks}: "
                    f"{len(orders)} orders, "
                    f"max_age={max_order_age:.1f}s, "
                    f"actions={sum(actions_taken.values())}"
                )

            return {
                "timestamp": cycle_start,
                "status": "completed",
                "orders_processed": len(orders),
                "max_order_age": max_order_age,
                "actions_taken": actions_taken,
                "total_reprices": self.total_reprices,
                "total_cancellations": self.total_cancellations,
                "cycle_duration": cycle_duration,
            }

        except Exception as e:
            logger.error(f"Error in watchdog cycle: {e}")
            return {"timestamp": time.time(), "status": "error", "error": str(e)}

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        try:
            # Get current orders
            orders = self.get_live_orders()

            # Calculate statistics
            if orders:
                ages = [order["age"] for order in orders.values()]
                max_age = max(ages)
                avg_age = sum(ages) / len(ages)
            else:
                max_age = 0
                avg_age = 0

            status = {
                "service": "ttl_watchdog",
                "timestamp": time.time(),
                "config": self.config,
                "status": "active",
                "current_orders": len(orders),
                "active_orders": len(self.active_orders),
                "order_age_stats": {
                    "max_age": max_age,
                    "avg_age": avg_age,
                    "ttl_seconds": self.config["ttl_seconds"],
                },
                "metrics": self.metrics.copy(),
                "totals": {
                    "total_checks": self.total_checks,
                    "total_reprices": self.total_reprices,
                    "total_cancellations": self.total_cancellations,
                    "total_timeouts": self.total_timeouts,
                },
                "recent_reprices": (
                    self.reprice_history[-5:] if self.reprice_history else []
                ),
            }

            # Store metrics in Redis for Prometheus
            for metric_name, value in self.metrics.items():
                self.redis.set(f"metric:{metric_name}", value)

            return status

        except Exception as e:
            return {
                "service": "ttl_watchdog",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }

    async def run_continuous_watchdog(self):
        """Run continuous TTL watchdog."""
        logger.info("⏰ Starting continuous TTL watchdog")

        try:
            while True:
                try:
                    # Run watchdog cycle
                    result = await self.run_watchdog_cycle()

                    if result["status"] == "completed":
                        actions_count = sum(result["actions_taken"].values())
                        if actions_count > 0:
                            logger.debug(
                                f"📊 Cycle #{self.total_checks}: {actions_count} actions, "
                                f"max_age={result.get('max_order_age', 0):.1f}s"
                            )

                    # Wait for next cycle
                    await asyncio.sleep(self.config["check_interval"])

                except Exception as e:
                    logger.error(f"Error in TTL watchdog loop: {e}")
                    await asyncio.sleep(1)  # Short delay on error

        except asyncio.CancelledError:
            logger.info("TTL watchdog loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in TTL watchdog loop: {e}")


async def main():
    """Main entry point for TTL watchdog."""
    import argparse

    parser = argparse.ArgumentParser(description="Execution TTL & Reprice Watchdog")
    parser.add_argument("--run", action="store_true", help="Run continuous watchdog")
    parser.add_argument(
        "--cycle", action="store_true", help="Run single watchdog cycle"
    )
    parser.add_argument("--status", action="store_true", help="Show status report")

    args = parser.parse_args()

    # Create watchdog
    watchdog = ExecutionTTLWatchdog()

    if args.status:
        # Show status report
        status = watchdog.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.cycle:
        # Run single cycle
        result = await watchdog.run_watchdog_cycle()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.run:
        # Run continuous watchdog
        try:
            await watchdog.run_continuous_watchdog()
        except KeyboardInterrupt:
            logger.info("TTL watchdog stopped by user")
        return

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
