#!/usr/bin/env python3
"""
Coinbase Executor (Layer 4)

Implements order execution for crypto using Coinbase Advanced Trade API.
Supports both paper trading and live trading modes via environment variables.
"""

import os
import json
import logging
import time
import jwt
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict

import requests

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
    time_in_force: str = "IOC"  # 'IOC', 'GTC', 'GTD'
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


class CoinbaseExecutor:
    """
    Coinbase Advanced Trade Executor

    Handles order execution for crypto using Coinbase Advanced Trade API.
    Supports both paper and live trading modes.
    """

    def __init__(self):
        self.api_key = os.getenv("COINBASE_API_KEY")
        self.api_secret = os.getenv("COINBASE_API_SECRET")
        self.passphrase = os.getenv("COINBASE_PASSPHRASE", "")
        self.base_url = "https://api.coinbase.com"
        self.exec_mode = os.getenv("EXEC_MODE", "paper")

        # Initialize API client
        self.dry_run = False

        if self.api_key and self.api_secret:
            try:
                logger.info(f"✅ Coinbase API initialized - Mode: {self.exec_mode}")
                logger.info(f"📡 Base URL: {self.base_url}")

                # Test connection
                if self.exec_mode == "live":
                    self._test_connection()

            except Exception as e:
                logger.error(f"❌ Failed to initialize Coinbase API: {e}")
                self.dry_run = True
        else:
            logger.warning("⚠️  Coinbase API keys not found - running in dry-run mode")
            self.dry_run = True

        # Order tracking
        self.orders: Dict[str, OrderResponse] = {}
        self.order_counter = 0

    def _generate_jwt_token(self, method: str, path: str) -> str:
        """Generate JWT token for Coinbase Advanced Trade API"""

        # Handle the secret key format - use correct secret from JSON
        private_key = self.api_secret.strip()

        # Fix corrupted secret from .env (missing 'M' at start, extra 'n')
        if private_key.startswith("nMHc"):
            private_key = "MHc" + private_key[3:]  # Remove 'nM', replace with 'M'

        # Create PEM format if we have base64 content
        if not private_key.startswith("-----BEGIN"):
            private_key = f"""-----BEGIN EC PRIVATE KEY-----
{private_key}
-----END EC PRIVATE KEY-----"""

        # Create JWT payload
        now = datetime.now(timezone.utc)
        payload = {
            "sub": self.api_key,  # API key name
            "iss": "coinbase-cloud",
            "nbf": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=1)).timestamp()),  # 1 minute expiry
            "aud": ["public"],
            "uri": f"{method} api.coinbase.com{path}",
        }

        # Generate JWT token
        token = jwt.encode(
            payload, private_key, algorithm="ES256", headers={"kid": self.api_key}
        )
        return token

    def _get_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate API request headers using JWT"""

        headers = {
            "Authorization": f"Bearer {self._generate_jwt_token(method, path)}",
        }

        # Add Content-Type for requests with body
        if body:
            headers["Content-Type"] = "application/json"

        return headers

    def _test_connection(self):
        """Test API connection"""
        try:
            path = "/api/v3/brokerage/accounts"
            headers = self._get_headers("GET", path)

            logger.info(f"🔍 Testing connection to {self.base_url}{path}")
            logger.info(
                f"🔑 Headers: {json.dumps({k: v[:20]+'...' if len(v) > 20 else v for k, v in headers.items()}, indent=2)}"
            )

            response = requests.get(
                f"{self.base_url}{path}", headers=headers, timeout=10
            )

            logger.info(f"📡 Response status: {response.status_code}")
            logger.info(f"📡 Response body: {response.text}")

            if response.status_code == 200:
                logger.info("✅ Coinbase API connection successful")
            else:
                logger.warning(f"⚠️  Coinbase API test failed: {response.status_code}")
                logger.warning(f"⚠️  Response: {response.text}")
                logger.warning(f"⚠️  Headers: {dict(response.headers)}")
                self.dry_run = True

        except Exception as e:
            logger.error(f"❌ Coinbase API connection test failed: {e}")
            self.dry_run = True

    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"coinbase_order_{int(time.time())}_{self.order_counter}"

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

        if order.order_type not in ["market", "limit"]:
            logger.error(
                f"❌ Order validation failed: Invalid order type '{order.order_type}'"
            )
            return False

        if order.order_type == "limit" and not order.limit_price:
            logger.error(
                "❌ Order validation failed: Limit price required for limit orders"
            )
            return False

        return True

    def submit_order(self, order: OrderRequest) -> OrderResponse:
        """
        Submit order to Coinbase

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
                "product_id": order.symbol,
                "side": order.side,
                "order_configuration": {
                    (
                        "market_market_ioc"
                        if order.order_type == "market"
                        else "limit_limit_gtc"
                    ): {
                        "quote_size" if order.side == "buy" else "base_size": str(
                            order.qty
                        )
                    }
                },
                "client_order_id": order.client_order_id,
            }

            if order.order_type == "limit":
                order_json["order_configuration"]["limit_limit_gtc"]["limit_price"] = (
                    str(order.limit_price)
                )

            logger.info(f"📋 DRY RUN ORDER: {json.dumps(order_json, indent=2)}")
            logger.info("✅ ORDER ACCEPTED (dry-run mode)")

            # Simulate immediate fill for paper trading
            response.status = "filled"
            response.filled_qty = order.qty
            response.avg_fill_price = order.limit_price or 50000.0  # Mock price

        else:
            # Live trading mode
            try:
                # Submit order via Coinbase Advanced Trade API
                path = "/api/v3/brokerage/orders"

                order_data = {
                    "product_id": order.symbol,
                    "side": order.side.upper(),  # Coinbase requires uppercase side
                    "client_order_id": order.client_order_id or order_id,
                }

                if order.order_type == "market":
                    order_data["order_configuration"] = {
                        "market_market_ioc": {
                            "quote_size" if order.side == "buy" else "base_size": str(
                                order.qty
                            )
                        }
                    }
                else:  # limit order
                    order_data["order_configuration"] = {
                        "limit_limit_gtc": {
                            "base_size": str(order.qty),
                            "limit_price": str(order.limit_price),
                        }
                    }

                body = json.dumps(order_data)
                headers = self._get_headers("POST", path, body)

                cb_response = requests.post(
                    f"{self.base_url}{path}", headers=headers, data=body, timeout=10
                )

                if cb_response.status_code == 200:
                    result = cb_response.json()

                    # Update response with actual order data
                    response.order_id = result.get("order_id", order_id)
                    response.status = "submitted"

                    logger.info(f"✅ Order submitted successfully: {response.order_id}")

                else:
                    logger.error(
                        f"❌ Order submission failed: {cb_response.status_code} - {cb_response.text}"
                    )
                    response.status = "rejected"
                    raise Exception(
                        f"Order submission failed: {cb_response.status_code}"
                    )

            except Exception as e:
                logger.error(f"❌ Order submission failed: {e}")
                response.status = "rejected"
                raise

        # Store order for tracking
        self.orders[order_id] = response
        return response

    def get_order(self, order_id: str) -> Optional[OrderResponse]:
        """Get order status"""
        if order_id in self.orders:
            return self.orders[order_id]

        if not self.dry_run and self.exec_mode == "live":
            try:
                path = f"/api/v3/brokerage/orders/historical/{order_id}"
                headers = self._get_headers("GET", path)

                response = requests.get(
                    f"{self.base_url}{path}", headers=headers, timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    order_data = result.get("order", {})

                    order_response = OrderResponse(
                        order_id=order_data.get("order_id", order_id),
                        symbol=order_data.get("product_id", ""),
                        side=order_data.get("side", ""),
                        qty=float(
                            order_data.get("order_configuration", {})
                            .get("market_market_ioc", {})
                            .get("quote_size", 0)
                        ),
                        status=order_data.get("status", "unknown"),
                        filled_qty=float(order_data.get("filled_size", 0)),
                        avg_fill_price=(
                            float(order_data.get("average_filled_price", 0))
                            if order_data.get("average_filled_price")
                            else None
                        ),
                        timestamp=order_data.get("created_time", ""),
                        client_order_id=order_data.get("client_order_id"),
                    )

                    self.orders[order_id] = order_response
                    return order_response

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

        if self.exec_mode == "live":
            try:
                path = f"/api/v3/brokerage/orders/batch_cancel"

                cancel_data = {"order_ids": [order_id]}

                body = json.dumps(cancel_data)
                headers = self._get_headers("POST", path, body)

                response = requests.post(
                    f"{self.base_url}{path}", headers=headers, data=body, timeout=10
                )

                if response.status_code == 200:
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
                "available_balance": {"value": "100000.00", "currency": "USD"},
                "status": "ACTIVE",
            }

        if self.exec_mode == "live":
            try:
                path = "/api/v3/brokerage/accounts"
                headers = self._get_headers("GET", path)

                response = requests.get(
                    f"{self.base_url}{path}", headers=headers, timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    accounts = result.get("accounts", [])

                    # Find USD account
                    usd_account = next(
                        (acc for acc in accounts if acc.get("currency") == "USD"), {}
                    )

                    return {
                        "account_id": usd_account.get("uuid", ""),
                        "available_balance": usd_account.get("available_balance", {}),
                        "status": "ACTIVE",
                    }

            except Exception as e:
                logger.error(f"❌ Failed to get account info: {e}")

        return {}

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        if self.dry_run:
            return [
                {
                    "symbol": "BTC-USD",
                    "qty": 0.1,
                    "market_value": 5000.0,
                    "avg_entry_price": 50000.0,
                }
            ]

        if self.exec_mode == "live":
            try:
                path = "/api/v3/brokerage/accounts"
                headers = self._get_headers("GET", path)

                response = requests.get(
                    f"{self.base_url}{path}", headers=headers, timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    accounts = result.get("accounts", [])

                    positions = []
                    for account in accounts:
                        if account.get("currency") != "USD":
                            balance = float(
                                account.get("available_balance", {}).get("value", 0)
                            )
                            if balance > 0:
                                positions.append(
                                    {
                                        "symbol": f"{account.get('currency')}-USD",
                                        "qty": balance,
                                        "market_value": balance * 50000,  # Mock price
                                        "avg_entry_price": 50000.0,
                                    }
                                )

                    return positions

            except Exception as e:
                logger.error(f"❌ Failed to get positions: {e}")

        return []


def main():
    """Test the Coinbase executor"""
    logger.info("🚀 Testing Coinbase Executor...")

    # Initialize executor
    executor = CoinbaseExecutor()

    # Test account info
    account = executor.get_account()
    logger.info(f"📊 Account Info: {json.dumps(account, indent=2)}")

    # Test order submission
    test_order = OrderRequest(
        symbol="BTC-USD",
        side="buy",
        qty=100.0,  # $100 worth
        order_type="market",
        client_order_id="test_crypto_order_001",
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

    logger.info("🎉 Coinbase Executor test completed!")


if __name__ == "__main__":
    main()
