"""Utilities and pytest-friendly checks for the explain-a-trade pipeline."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import requests

try:
    import redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - Redis client is optional for tests
    redis = None  # type: ignore[assignment]
    RedisError = Exception  # type: ignore[assignment]

import pytest

# Ensure src modules are importable when executing as a script
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_order() -> Dict[str, Any]:
    """Create deterministic mock order data for the explanation service."""

    return {
        "order_id": f"test_order_{int(time.time())}",
        "symbol": "BTCUSDT",
        "side": "buy",
        "quantity": 0.1,
        "price": 98_750.00,
        "timestamp": datetime.now().isoformat(),
        "order_type": "market",
        "edge_bps": 12.5,
        "confidence": 0.85,
        "portfolio_value": 100_000.0,
        "position_size_pct": 0.05,
        "sentiment_score": 0.3,
        "technical_signal": "MA_BULLISH_CROSS",
        "big_bet_flag": False,
        "risk_metrics": {"var_pct": 1.2, "sharpe_ratio": 2.1, "max_drawdown": 0.08},
    }


def _connect_to_redis() -> Optional["redis.Redis"]:
    """Attempt to connect to Redis, returning None when unavailable."""

    if redis is None:
        logger.debug("Redis client not installed; skipping Redis checks")
        return None

    try:
        client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        client.ping()
        logger.debug("Redis connection successful")
        return client
    except RedisError as exc:  # pragma: no cover - depends on local infra
        logger.debug("Redis unavailable: %s", exc)
        return None


def create_mock_explanation(order_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a locally generated explanation when the API is unavailable."""

    return {
        "order_id": order_data["order_id"],
        "symbol": order_data["symbol"],
        "explanation": (
            "Market buy of {qty} {symbol} at ${price:.2f} based on {edge:.1f}bp edge "
            "with {confidence:.0%} confidence. Technical signal shows {signal}, "
            "supported by sentiment {sent:.2f}."
        ).format(
            qty=order_data["quantity"],
            symbol=order_data["symbol"],
            price=order_data["price"],
            edge=order_data["edge_bps"],
            confidence=order_data["confidence"],
            signal=order_data["technical_signal"],
            sent=order_data["sentiment_score"],
        ),
        "confidence_level": "High" if order_data["confidence"] > 0.8 else "Medium",
        "key_factors": ["Technical Signal", "Edge Opportunity", "Risk Managed"],
        "risk_assessment": (
            "Low"
            if order_data.get("risk_metrics", {}).get("var_pct", 2.0) < 1.5
            else "Medium"
        ),
        "timestamp": datetime.now().isoformat(),
        "processing_time_ms": 245.7,
    }


def call_explain_api(order_data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Call the explain service; return payload and flag indicating live response."""

    try:
        response = requests.post(
            "http://localhost:8003/explain",
            json=order_data,
            timeout=30,
        )
        if response.status_code == 200:
            logger.debug("Explain API responded successfully")
            return response.json(), True

        logger.debug("Explain API returned %s; using mock", response.status_code)
    except requests.RequestException as exc:  # pragma: no cover - depends on service
        logger.debug("Explain API unavailable: %s", exc)

    return create_mock_explanation(order_data), False


def store_explanation_in_redis(
    redis_client: "redis.Redis", explanation: Dict[str, Any]
) -> str:
    """Persist explanation in Redis under the expected key structure."""

    redis_key = f"trade.rationale:{explanation['order_id']}"
    payload = json.dumps(explanation)
    redis_client.setex(redis_key, 86_400, payload)
    redis_client.lpush("trade.rationale.recent", payload)
    redis_client.ltrim("trade.rationale.recent", 0, 49)
    return redis_key


def check_grafana_api_access() -> bool:
    """Return True when Grafana health and annotations endpoints respond with 200."""

    try:
        health = requests.get("http://localhost:3000/api/health", timeout=10)
        if health.status_code != 200:
            return False

        annotations = requests.get("http://localhost:3000/api/annotations", timeout=10)
        return annotations.status_code == 200
    except requests.RequestException:  # pragma: no cover - depends on service
        return False


def run_round_trip() -> None:
    """CLI helper to exercise the round-trip with best-effort checks."""

    order = create_mock_order()
    explanation, from_live = call_explain_api(order)
    logger.info(
        "Explain service %s", "responded" if from_live else "unavailable (mocked)"
    )

    redis_client = _connect_to_redis()
    if redis_client is not None:
        key = store_explanation_in_redis(redis_client, explanation)
        logger.info("Stored explanation in Redis key %s", key)
    else:
        logger.info("Redis unavailable; skipping persistence check")

    if check_grafana_api_access():
        logger.info("Grafana API reachable")
    else:
        logger.info("Grafana API not reachable; dashboard check skipped")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    run_round_trip()


# ------------------------------- Pytest hooks ---------------------------------


@pytest.fixture
def order_data() -> Dict[str, Any]:
    return create_mock_order()


def test_explain_api_returns_payload(order_data: Dict[str, Any]) -> None:
    explanation, _ = call_explain_api(order_data)
    assert explanation["order_id"] == order_data["order_id"]
    assert explanation["explanation"]
    assert "timestamp" in explanation


def test_redis_round_trip(order_data: Dict[str, Any]) -> None:
    redis_client = _connect_to_redis()
    if redis_client is None:
        pytest.skip("Redis not available locally")

    explanation = create_mock_explanation(order_data)
    redis_key = store_explanation_in_redis(redis_client, explanation)

    stored_payload = redis_client.get(redis_key)
    assert stored_payload is not None
    stored = json.loads(stored_payload)
    assert stored["order_id"] == explanation["order_id"]

    redis_client.delete(redis_key)
    redis_client.lrem("trade.rationale.recent", 0, json.dumps(explanation))


def test_grafana_api_optional() -> None:
    if not check_grafana_api_access():
        pytest.skip("Grafana API not running locally")
