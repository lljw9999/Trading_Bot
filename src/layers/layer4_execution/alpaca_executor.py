#!/usr/bin/env python3
"""
Alpaca Paper Trading Executor (Layer 4)

Implements order execution for stocks using Alpaca Trade API.
Supports both paper trading and live trading modes via environment variables.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST
from alpaca_trade_api.common import URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    """Standard order request format"""

    symbol: str
    side: str  # 'buy' or 'sell'
    qty: float
    order_type: str = "market"  # 'market' or 'limit'
    time_in_force: str = "day"  # 'day', 'gtc', 'ioc', 'fok'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None


@dataclass
class OrderResponse:
    """Standard order response format"""

    order_id: str
    symbol: str
    side: str
    qty: float
    status: str
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    timestamp: str = None
    client_order_id: Optional[str] = None


class AlpacaExecutor:
    """
    Alpaca Paper Trading Executor

    Handles order execution for stocks using Alpaca Trade API.
    Supports both paper and live trading modes.
    """

    def __init__(
        self,
        *,
        dry_run: Optional[
            bool
        ] = None,  # Changed to Optional to detect explicit setting
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        self.base_url = base_url or os.getenv(
            "ALPACA_PAPER_BASE", "https://paper-api.alpaca.markets"
        )
        self.exec_mode = os.getenv("EXEC_MODE", "paper")

        # Executor safety audit as per Future_instruction.txt
        dry_run_env = os.getenv("DRY_RUN", "1")  # Default to "1" (dry run) for safety

        if dry_run is None:
            self.dry_run = dry_run_env != "0"
        else:
            self.dry_run = dry_run

        if not self.dry_run and dry_run_env != "0":
            raise RuntimeError(
                "Refusing live trading: set DRY_RUN=0 explicitly for production."
            )

        if self.exec_mode == "live" and not self.dry_run:
            logger.warning(
                "🚨 LIVE TRADING MODE ENABLED - Real money orders will be placed!"
            )
            logger.warning(
                "🚨 Ensure DRY_RUN=0 was set intentionally for production trading"
            )

        # Initialize API client
        self.api = None

        if not self.dry_run and self.api_key and self.api_secret:
            try:
                self.api = REST(
                    key_id=self.api_key,
                    secret_key=self.api_secret,
                    base_url=URL(self.base_url),
                    api_version="v2",
                )
                logger.info(f"✅ Alpaca API initialized - Mode: {self.exec_mode}")
                logger.info(f"📡 Base URL: {self.base_url}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Alpaca API: {e}")
                self.dry_run = True
        else:
            if not self.dry_run:
                logger.warning("⚠️  Alpaca API keys not found - running in dry-run mode")
                self.dry_run = True

        # Order tracking
        self.orders: Dict[str, OrderResponse] = {}
        self.order_counter = 0

    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"alpaca_order_{int(time.time())}_{self.order_counter}"

    def _validate_order(self, order: OrderRequest) -> bool:
        """Validate order parameters"""
        if not order.symbol:
            logger.error("❌ Order validation failed: Symbol is required")
            return False

        if order.side not in ["buy", "sell"]:
            logger.error(f"❌ Order validation failed: Invalid side '{order.side}'")
            return False

        if order.qty <= 0:
            logger.error(f"❌ Order validation failed: Invalid quantity {order.qty}")
            return False

        if order.order_type not in ["market", "limit", "stop", "stop_limit"]:
            logger.error(
                f"❌ Order validation failed: Invalid order type '{order.order_type}'"
            )
            return False

        if order.order_type == "limit" and not order.limit_price:
            logger.error(
                "❌ Order validation failed: Limit price required for limit orders"
            )
            return False

        # Add IOC support validation
        if order.time_in_force not in ["day", "gtc", "ioc", "fok"]:
            logger.error(
                f"❌ Order validation failed: Invalid time_in_force '{order.time_in_force}'"
            )
            return False

        # Lot-size and min-notional checks for equities
        min_notional = 2.0  # $2 minimum notional
        estimated_notional = order.qty * (order.limit_price or 100.0)  # rough estimate

        if estimated_notional < min_notional:
            logger.error(
                f"❌ Order validation failed: Notional ${estimated_notional:.2f} < ${min_notional} minimum"
            )
            return False

        # US stocks typically trade in whole shares (lot size = 1)
        if order.qty != int(order.qty):
            logger.error(
                f"❌ Order validation failed: Fractional shares not supported, qty={order.qty}"
            )
            return False

        return True

    def submit_order(self, order: OrderRequest) -> OrderResponse:
        """
        Submit order to Alpaca

        Args:
            order: OrderRequest object

        Returns:
            OrderResponse object
        """
        if not self._validate_order(order):
            raise ValueError("Order validation failed")

        order_id = self._generate_order_id()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create order response
        response = OrderResponse(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            status="pending_new",
            timestamp=timestamp,
            client_order_id=order.client_order_id,
        )

        if self.dry_run or self.exec_mode == "paper":
            # Dry run mode - just log the order
            order_json = {
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.qty,
                "type": order.order_type,
                "time_in_force": order.time_in_force,
                "limit_price": order.limit_price,
                "stop_price": order.stop_price,
                "client_order_id": order.client_order_id,
            }

            logger.info(f"📋 DRY RUN ORDER: {json.dumps(order_json, indent=2)}")
            logger.info("✅ ORDER ACCEPTED (dry-run mode)")

            # Simulate immediate fill for paper trading
            response.status = "filled"
            response.filled_qty = order.qty
            response.avg_fill_price = 100.0  # Mock price

        else:
            # Live trading mode
            try:
                # Submit order via Alpaca API
                alpaca_order = self.api.submit_order(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=order.side,
                    type=order.order_type,
                    time_in_force=order.time_in_force,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    client_order_id=order.client_order_id,
                )

                # Update response with actual order data
                response.order_id = alpaca_order.id
                response.status = alpaca_order.status
                response.filled_qty = float(alpaca_order.filled_qty or 0)
                response.avg_fill_price = (
                    float(alpaca_order.filled_avg_price)
                    if alpaca_order.filled_avg_price
                    else None
                )

                logger.info(f"✅ Order submitted successfully: {response.order_id}")

            except Exception as e:
                logger.error(f"❌ Order submission failed: {e}")
                response.status = "rejected"
                raise

        # Store order for tracking
        self.orders[order_id] = response
        return response

    def execute(
        self,
        *,
        symbol: str,
        notional: float,
        price: float,
        timestamp: Optional[str] = None,
    ) -> OrderResponse:
        """Compatibility wrapper used by higher-level pipeline tests."""

        side = "buy" if notional >= 0 else "sell"
        abs_notional = abs(notional)
        qty = max(int(round(abs_notional / max(price, 1e-6))), 1)

        order = OrderRequest(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="market",
            client_order_id=(
                f"legacy_{symbol}_{int(time.time())}"
                if timestamp is None
                else f"legacy_{symbol}_{timestamp}"
            ),
        )

        return self.submit_order(order)

    def get_order(self, order_id: str) -> Optional[OrderResponse]:
        """Get order status"""
        if order_id in self.orders:
            return self.orders[order_id]

        if not self.dry_run and self.api:
            try:
                alpaca_order = self.api.get_order(order_id)
                response = OrderResponse(
                    order_id=alpaca_order.id,
                    symbol=alpaca_order.symbol,
                    side=alpaca_order.side,
                    qty=float(alpaca_order.qty),
                    status=alpaca_order.status,
                    filled_qty=float(alpaca_order.filled_qty or 0),
                    avg_fill_price=(
                        float(alpaca_order.filled_avg_price)
                        if alpaca_order.filled_avg_price
                        else None
                    ),
                    timestamp=alpaca_order.created_at.isoformat(),
                    client_order_id=alpaca_order.client_order_id,
                )
                self.orders[order_id] = response
                return response
            except Exception as e:
                logger.error(f"❌ Failed to get order {order_id}: {e}")

        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        if self.dry_run:
            logger.info(f"📋 DRY RUN: Canceling order {order_id}")
            if order_id in self.orders:
                self.orders[order_id].status = "canceled"
            return True

        if self.api:
            try:
                self.api.cancel_order(order_id)
                logger.info(f"✅ Order canceled: {order_id}")
                if order_id in self.orders:
                    self.orders[order_id].status = "canceled"
                return True
            except Exception as e:
                logger.error(f"❌ Failed to cancel order {order_id}: {e}")

        return False

    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        if self.dry_run:
            return {
                "account_id": "paper_account",
                "equity": 100000.0,
                "cash": 50000.0,
                "buying_power": 200000.0,
                "status": "ACTIVE",
            }

        if self.api:
            try:
                account = self.api.get_account()
                return {
                    "account_id": account.id,
                    "equity": float(account.equity),
                    "cash": float(account.cash),
                    "buying_power": float(account.buying_power),
                    "status": account.status,
                }
            except Exception as e:
                logger.error(f"❌ Failed to get account info: {e}")

        return {}

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        if self.dry_run:
            return [
                {
                    "symbol": "AAPL",
                    "qty": 10,
                    "market_value": 1500.0,
                    "avg_entry_price": 150.0,
                }
            ]

        if self.api:
            try:
                positions = self.api.list_positions()
                return [
                    {
                        "symbol": pos.symbol,
                        "qty": float(pos.qty),
                        "market_value": float(pos.market_value),
                        "avg_entry_price": float(pos.avg_entry_price),
                    }
                    for pos in positions
                ]
            except Exception as e:
                logger.error(f"❌ Failed to get positions: {e}")

        return []


def main():
    """Test the Alpaca executor"""
    logger.info("🚀 Testing Alpaca Executor...")

    # Initialize executor
    executor = AlpacaExecutor()

    # Test account info
    account = executor.get_account()
    logger.info(f"📊 Account Info: {json.dumps(account, indent=2)}")

    # Test order submission
    test_order = OrderRequest(
        symbol="AAPL",
        side="buy",
        qty=1,
        order_type="market",
        client_order_id="test_order_001",
    )

    try:
        response = executor.submit_order(test_order)
        logger.info(f"✅ Order Response: {json.dumps(asdict(response), indent=2)}")

        # Test order status
        order_status = executor.get_order(response.order_id)
        if order_status:
            logger.info(
                f"📋 Order Status: {json.dumps(asdict(order_status), indent=2)}"
            )

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

    # Test positions
    positions = executor.get_positions()
    logger.info(f"📈 Positions: {json.dumps(positions, indent=2)}")

    logger.info("🎉 Alpaca Executor test completed!")


if __name__ == "__main__":
    main()
