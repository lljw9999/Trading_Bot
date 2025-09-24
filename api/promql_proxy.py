#!/usr/bin/env python3
"""
PromQL Proxy API
GET /api/promql?q=<promql> (whitelisted queries)
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional

import redis
import requests
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("promql_proxy")

# Initialize FastAPI app
app = FastAPI(
    title="Trading System PromQL Proxy",
    description="Secure proxy for Prometheus queries with whitelisted queries",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
    ],  # React dev server
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


class PromQLProxy:
    """Secure PromQL proxy with query whitelisting."""

    def __init__(self):
        """Initialize PromQL proxy."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Prometheus configuration
        self.prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
        self.grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
        self.grafana_token = os.getenv("GRAFANA_TOKEN", "")

        # Whitelisted queries for security
        self.whitelisted_queries = {
            "ab_gate_status": {
                "query": "ab_gate_consecutive_passes",
                "description": "A/B testing gate consecutive passes",
            },
            "ab_gate_total": {
                "query": "ab_gate_total_tests",
                "description": "Total A/B tests run",
            },
            "recon_breaches": {
                "query": "recon_breaches_24h",
                "description": "Reconciliation breaches in last 24h",
            },
            "position_mismatches": {
                "query": "recon_position_mismatches",
                "description": "Position mismatches from reconciliation",
            },
            "capital_effective": {
                "query": "risk_capital_effective",
                "description": "Current effective capital allocation",
            },
            "capital_cap": {
                "query": "risk_capital_cap",
                "description": "Maximum capital cap",
            },
            "capital_staged": {
                "query": "risk_capital_staged",
                "description": "Staged capital allocation",
            },
            "hedge_enabled": {
                "query": "hedge_enabled",
                "description": "Hedge system enabled status",
            },
            "hedge_ratio_btc": {
                "query": 'hedge_ratio{symbol="BTC"}',
                "description": "BTC hedge ratio",
            },
            "hedge_ratio_eth": {
                "query": 'hedge_ratio{symbol="ETH"}',
                "description": "ETH hedge ratio",
            },
            "system_mode": {
                "query": "system_mode",
                "description": "System operating mode (0=halt, 1=auto)",
            },
            "feature_flags": {
                "query": "feature_flag_enabled",
                "description": "Feature flag states",
            },
            "venue_health_binance": {
                "query": 'venue_health{venue="binance"}',
                "description": "Binance venue health score",
            },
            "venue_health_coinbase": {
                "query": 'venue_health{venue="coinbase"}',
                "description": "Coinbase venue health score",
            },
            "pnl_total": {"query": "pnl_total_usd", "description": "Total P&L in USD"},
            "pnl_daily": {"query": "pnl_daily_usd", "description": "Daily P&L in USD"},
            "orders_active": {
                "query": "orders_active_count",
                "description": "Active orders count",
            },
            "orders_stale": {
                "query": "orders_stale_count",
                "description": "Stale orders count",
            },
        }

        logger.info("🔌 PromQL Proxy initialized")

    def validate_query(self, query_name: str) -> Optional[str]:
        """Validate and return whitelisted query."""
        if query_name not in self.whitelisted_queries:
            return None
        return self.whitelisted_queries[query_name]["query"]

    def query_prometheus(self, promql_query: str) -> Dict[str, Any]:
        """Execute PromQL query against Prometheus."""
        try:
            url = f"{self.prometheus_url}/api/v1/query"
            params = {"query": promql_query}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data["status"] != "success":
                raise Exception(
                    f"Prometheus query failed: {data.get('error', 'unknown error')}"
                )

            return {
                "status": "success",
                "query": promql_query,
                "data": data["data"],
                "timestamp": time.time(),
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying Prometheus: {e}")
            return {
                "status": "error",
                "error": f"Prometheus connection error: {str(e)}",
                "query": promql_query,
            }
        except Exception as e:
            logger.error(f"Error in Prometheus query: {e}")
            return {"status": "error", "error": str(e), "query": promql_query}

    def get_redis_metrics(self) -> Dict[str, Any]:
        """Get key metrics from Redis as fallback."""
        try:
            metrics = {
                "ab_gate_consecutive": int(self.redis.get("ab:last4:exec") or 0),
                "ab_gate_total": int(self.redis.get("ab:total_tests") or 0),
                "recon_breaches": int(self.redis.get("recon:breaches_24h") or 0),
                "position_mismatches": int(
                    self.redis.get("recon:position_mismatches") or 0
                ),
                "capital_effective": float(
                    self.redis.get("risk:capital_effective") or 0.0
                ),
                "capital_cap": float(
                    self.redis.get("risk:capital_cap_next_week") or 0.0
                ),
                "capital_staged": self.redis.get("risk:capital_stage_request"),
                "hedge_enabled": bool(int(self.redis.get("HEDGE_ENABLED") or 1)),
                "system_mode": self.redis.get("mode") or "unknown",
                "orders_stale": int(self.redis.get("orders:stale_count") or 0),
            }

            # Convert staged capital to float if present
            if metrics["capital_staged"]:
                metrics["capital_staged"] = float(metrics["capital_staged"])

            return {
                "status": "success",
                "source": "redis",
                "metrics": metrics,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error getting Redis metrics: {e}")
            return {"status": "error", "error": str(e), "source": "redis"}


# Initialize proxy instance
proxy = PromQLProxy()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Trading System PromQL Proxy",
        "version": "1.0.0",
        "endpoints": {
            "/api/promql": "Execute whitelisted PromQL queries",
            "/api/queries": "List available queries",
            "/api/metrics": "Get key metrics from Redis",
            "/api/health": "Health check",
        },
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test Redis connection
        redis_ping = proxy.redis.ping()

        # Test Prometheus connection
        prom_response = requests.get(
            f"{proxy.prometheus_url}/api/v1/query", params={"query": "up"}, timeout=5
        )
        prom_healthy = prom_response.status_code == 200

        return {
            "status": "healthy" if (redis_ping and prom_healthy) else "degraded",
            "redis": "connected" if redis_ping else "disconnected",
            "prometheus": "connected" if prom_healthy else "disconnected",
            "timestamp": time.time(),
        }

    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e), "timestamp": time.time()},
        )


@app.get("/api/queries")
async def list_queries():
    """List available whitelisted queries."""
    return {
        "queries": proxy.whitelisted_queries,
        "total": len(proxy.whitelisted_queries),
    }


@app.get("/api/promql")
async def execute_promql(q: str = Query(..., description="Query name from whitelist")):
    """Execute whitelisted PromQL query."""
    try:
        # Validate query
        promql_query = proxy.validate_query(q)
        if not promql_query:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Query '{q}' not in whitelist",
                    "available_queries": list(proxy.whitelisted_queries.keys()),
                },
            )

        # Execute query
        result = proxy.query_prometheus(promql_query)

        if result["status"] == "error":
            # Try Redis fallback for key metrics
            if q in ["ab_gate_status", "recon_breaches", "capital_effective"]:
                redis_result = proxy.get_redis_metrics()
                if redis_result["status"] == "success":
                    # Map specific metric from Redis
                    metric_mapping = {
                        "ab_gate_status": redis_result["metrics"][
                            "ab_gate_consecutive"
                        ],
                        "recon_breaches": redis_result["metrics"]["recon_breaches"],
                        "capital_effective": redis_result["metrics"][
                            "capital_effective"
                        ],
                    }

                    return {
                        "status": "success",
                        "query": promql_query,
                        "source": "redis_fallback",
                        "data": {
                            "result": [
                                {
                                    "metric": {"__name__": q},
                                    "value": [
                                        time.time(),
                                        str(metric_mapping.get(q, 0)),
                                    ],
                                }
                            ]
                        },
                        "timestamp": time.time(),
                    }

            raise HTTPException(status_code=500, detail=result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in promql endpoint: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.get("/api/metrics")
async def get_metrics():
    """Get key operational metrics from Redis."""
    try:
        result = proxy.get_redis_metrics()

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in metrics endpoint: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.get("/api/ops-dashboard")
async def ops_dashboard():
    """Get all data needed for ops dashboard."""
    try:
        # Get comprehensive metrics
        redis_metrics = proxy.get_redis_metrics()

        if redis_metrics["status"] != "success":
            raise HTTPException(status_code=500, detail=redis_metrics)

        metrics = redis_metrics["metrics"]

        # Structure data for frontend dashboard
        dashboard_data = {
            "timestamp": time.time(),
            "status": "success",
            "sections": {
                "ab_gate": {
                    "title": "A/B Testing Gate",
                    "status": "PASS" if metrics["ab_gate_consecutive"] >= 4 else "FAIL",
                    "data": {
                        "consecutive_passes": metrics["ab_gate_consecutive"],
                        "total_tests": metrics["ab_gate_total"],
                        "required_passes": 4,
                    },
                },
                "reconciliation": {
                    "title": "Reconciliation & Feature Gates",
                    "status": (
                        "CLEAN"
                        if (
                            metrics["recon_breaches"] == 0
                            and metrics["position_mismatches"] == 0
                        )
                        else "ISSUES"
                    ),
                    "data": {
                        "breaches": metrics["recon_breaches"],
                        "position_mismatches": metrics["position_mismatches"],
                        "total_issues": metrics["recon_breaches"]
                        + metrics["position_mismatches"],
                    },
                },
                "capital": {
                    "title": "Capital Management",
                    "status": (
                        "ACTIVE" if metrics["capital_effective"] > 0 else "INACTIVE"
                    ),
                    "data": {
                        "effective": metrics["capital_effective"],
                        "cap": metrics["capital_cap"],
                        "staged": metrics["capital_staged"],
                        "utilization": (
                            (metrics["capital_effective"] / metrics["capital_cap"])
                            if metrics["capital_cap"] > 0
                            else 0
                        ),
                    },
                },
                "hedge": {
                    "title": "Hedge System",
                    "status": "ENABLED" if metrics["hedge_enabled"] else "DISABLED",
                    "data": {
                        "enabled": metrics["hedge_enabled"],
                        "system_mode": metrics["system_mode"],
                    },
                },
                "orders": {
                    "title": "Order Management",
                    "status": (
                        "CLEAN" if metrics["orders_stale"] == 0 else "STALE_ORDERS"
                    ),
                    "data": {"stale_count": metrics["orders_stale"]},
                },
            },
        }

        return dashboard_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ops-dashboard endpoint: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
