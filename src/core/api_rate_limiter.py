#!/usr/bin/env python3
"""
API Rate Limiter Integration

Provides easy integration for trading connectors to manage API quotas:
- Automatic rate limiting with adaptive backoff
- Pre-request quota checking
- Response header parsing
- WebSocket reconnection tracking
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.api_quota_monitor import APIQuotaMonitor

logger = logging.getLogger("api_rate_limiter")


class APIRateLimiter:
    """
    Rate limiter that integrates with APIQuotaMonitor.
    Use this in trading connectors to prevent hitting rate limits.
    """

    def __init__(self, exchange: str):
        """
        Initialize rate limiter for specific exchange.

        Args:
            exchange: Exchange name (binance, coinbase, etc.)
        """
        self.exchange = exchange
        self.quota_monitor = APIQuotaMonitor()
        self.last_request_times = {}  # Track last request time per endpoint

        logger.info(f"Initialized rate limiter for {exchange}")

    async def acquire_permit(self, endpoint_type: str = "default") -> bool:
        """
        Acquire permit to make API request with automatic backoff.

        Args:
            endpoint_type: Type of endpoint (default, orders, account, etc.)

        Returns:
            True when safe to proceed with request
        """
        try:
            # Check current quota status
            status = self.quota_monitor.get_exchange_quota_status(
                self.exchange, endpoint_type
            )

            # If backoff needed, calculate delay
            if status.get("backoff_needed", False):
                backoff_ms = self.quota_monitor.calculate_backoff_delay(
                    self.exchange, endpoint_type, "exponential"
                )

                if backoff_ms > 0:
                    logger.warning(
                        f"Rate limiting {self.exchange}:{endpoint_type} for {backoff_ms:.0f}ms"
                    )
                    await asyncio.sleep(backoff_ms / 1000.0)

            # Enforce minimum delay between requests if configured
            min_delay = self._get_minimum_delay(endpoint_type)
            if min_delay > 0:
                last_time = self.last_request_times.get(endpoint_type, 0)
                time_since_last = time.time() - last_time

                if time_since_last < min_delay:
                    sleep_time = min_delay - time_since_last
                    await asyncio.sleep(sleep_time)

            # Update last request time
            self.last_request_times[endpoint_type] = time.time()

            return True

        except Exception as e:
            logger.error(f"Error acquiring API permit: {e}")
            # Fail-safe: small delay and continue
            await asyncio.sleep(0.1)
            return True

    def _get_minimum_delay(self, endpoint_type: str) -> float:
        """Get minimum delay between requests for endpoint type."""
        # Some exchanges have strict per-second limits
        if self.exchange == "coinbase" and endpoint_type == "orders":
            return 0.2  # 200ms between order requests (5/second max)
        elif self.exchange == "deribit" and endpoint_type == "orders":
            return 0.2  # 200ms between order requests (5/second max)
        else:
            return 0.0  # No minimum delay

    def record_request(
        self,
        endpoint_type: str = "default",
        response_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, any]:
        """
        Record API request for quota tracking.

        Args:
            endpoint_type: Type of endpoint called
            response_headers: HTTP response headers

        Returns:
            Current quota status
        """
        return self.quota_monitor.record_api_call(
            self.exchange, endpoint_type, response_headers
        )

    def record_websocket_reconnect(self, reason: str = "unknown") -> Dict[str, any]:
        """
        Record WebSocket reconnection.

        Args:
            reason: Reason for reconnection

        Returns:
            Reconnection status
        """
        return self.quota_monitor.record_websocket_reconnect(self.exchange, reason)

    def get_quota_status(self, endpoint_type: str = "default") -> Dict[str, any]:
        """Get current quota status for endpoint."""
        return self.quota_monitor.get_exchange_quota_status(
            self.exchange, endpoint_type
        )

    def is_quota_healthy(self, endpoint_type: str = "default") -> bool:
        """Check if quota is healthy (not in warning/critical state)."""
        status = self.get_quota_status(endpoint_type)
        warning_level = status.get("warning_level", "OK")
        return warning_level == "OK"


class RateLimitedHTTPClient:
    """
    HTTP client wrapper with automatic rate limiting.
    Use this instead of direct requests to ensure quota compliance.
    """

    def __init__(self, exchange: str):
        """
        Initialize rate-limited HTTP client.

        Args:
            exchange: Exchange name
        """
        self.rate_limiter = APIRateLimiter(exchange)

        # Import requests here to avoid dependency issues
        try:
            import requests

            self.session = requests.Session()
        except ImportError:
            self.session = None
            logger.warning("requests library not available")

    async def get(
        self, url: str, endpoint_type: str = "default", **kwargs
    ) -> Tuple[any, Dict[str, any]]:
        """
        Make rate-limited GET request.

        Args:
            url: Request URL
            endpoint_type: API endpoint type for quota tracking
            **kwargs: Additional requests parameters

        Returns:
            (response, quota_status) tuple
        """
        return await self._make_request("GET", url, endpoint_type, **kwargs)

    async def post(
        self, url: str, endpoint_type: str = "orders", **kwargs
    ) -> Tuple[any, Dict[str, any]]:
        """
        Make rate-limited POST request.

        Args:
            url: Request URL
            endpoint_type: API endpoint type (defaults to 'orders' for POST)
            **kwargs: Additional requests parameters

        Returns:
            (response, quota_status) tuple
        """
        return await self._make_request("POST", url, endpoint_type, **kwargs)

    async def delete(
        self, url: str, endpoint_type: str = "orders", **kwargs
    ) -> Tuple[any, Dict[str, any]]:
        """
        Make rate-limited DELETE request.

        Args:
            url: Request URL
            endpoint_type: API endpoint type (defaults to 'orders' for DELETE)
            **kwargs: Additional requests parameters

        Returns:
            (response, quota_status) tuple
        """
        return await self._make_request("DELETE", url, endpoint_type, **kwargs)

    async def _make_request(
        self, method: str, url: str, endpoint_type: str, **kwargs
    ) -> Tuple[any, Dict[str, any]]:
        """
        Make rate-limited HTTP request.

        Args:
            method: HTTP method
            url: Request URL
            endpoint_type: API endpoint type
            **kwargs: Additional requests parameters

        Returns:
            (response, quota_status) tuple
        """
        try:
            if not self.session:
                raise Exception("HTTP session not available")

            # Acquire rate limit permit
            await self.rate_limiter.acquire_permit(endpoint_type)

            # Make request
            if method == "GET":
                response = self.session.get(url, **kwargs)
            elif method == "POST":
                response = self.session.post(url, **kwargs)
            elif method == "DELETE":
                response = self.session.delete(url, **kwargs)
            else:
                raise Exception(f"Unsupported HTTP method: {method}")

            # Record request for quota tracking
            quota_status = self.rate_limiter.record_request(
                endpoint_type, dict(response.headers)
            )

            return response, quota_status

        except Exception as e:
            logger.error(f"Error making rate-limited request: {e}")
            # Return error info
            error_status = {"error": str(e)}
            return None, error_status


# Example usage functions for integration


def create_rate_limiter(exchange: str) -> APIRateLimiter:
    """
    Create rate limiter for exchange.

    Usage:
        limiter = create_rate_limiter("binance")
        await limiter.acquire_permit("orders")
        # ... make API request ...
        limiter.record_request("orders", response.headers)
    """
    return APIRateLimiter(exchange)


def create_http_client(exchange: str) -> RateLimitedHTTPClient:
    """
    Create rate-limited HTTP client for exchange.

    Usage:
        client = create_http_client("binance")
        response, quota = await client.post("/api/v3/order", endpoint_type="orders", json=order_data)
    """
    return RateLimitedHTTPClient(exchange)
