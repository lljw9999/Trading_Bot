#!/usr/bin/env python3
"""
API Quota and Rate Limit Budget Monitor

Monitors API usage across exchanges to prevent rate limiting:
- Tracks API calls per exchange and endpoint 
- Implements adaptive backoff when approaching limits
- Alerts if 15-min quota usage > 80%
- Monitors WebSocket reconnections > 3/hour
- Provides rate limiting budget management
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import math

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import redis

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger("api_quota_monitor")


class APIQuotaMonitor:
    """
    Monitors API quotas and rate limits across trading venues.
    Prevents hitting rate limits that could disrupt trading operations.
    """

    def __init__(self):
        """Initialize API quota monitor."""
        self.redis_client = None
        if DEPS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        # Exchange-specific rate limit configurations
        self.config = {
            "exchanges": {
                "binance": {
                    "name": "Binance",
                    "rest_limits": {
                        "default": {
                            "requests": 1200,
                            "window_seconds": 60,
                        },  # 1200 requests/minute
                        "orders": {
                            "requests": 10,
                            "window_seconds": 1,
                        },  # 10 orders/second
                        "account": {
                            "requests": 10,
                            "window_seconds": 60,
                        },  # Account info limits
                    },
                    "ws_reconnect_threshold": 3,  # Max 3 reconnects/hour
                    "warning_threshold": 0.8,  # Alert at 80% usage
                    "backoff_threshold": 0.9,  # Start backoff at 90% usage
                },
                "coinbase": {
                    "name": "Coinbase Pro",
                    "rest_limits": {
                        "default": {
                            "requests": 10,
                            "window_seconds": 1,
                        },  # 10 requests/second
                        "orders": {
                            "requests": 5,
                            "window_seconds": 1,
                        },  # 5 orders/second
                        "private": {
                            "requests": 5,
                            "window_seconds": 1,
                        },  # Private endpoints
                    },
                    "ws_reconnect_threshold": 3,
                    "warning_threshold": 0.8,
                    "backoff_threshold": 0.9,
                },
                "alpaca": {
                    "name": "Alpaca Markets",
                    "rest_limits": {
                        "default": {
                            "requests": 200,
                            "window_seconds": 60,
                        },  # 200 requests/minute
                        "orders": {
                            "requests": 200,
                            "window_seconds": 60,
                        },  # Orders included in default
                        "data": {"requests": 200, "window_seconds": 60},  # Market data
                    },
                    "ws_reconnect_threshold": 3,
                    "warning_threshold": 0.8,
                    "backoff_threshold": 0.9,
                },
                "deribit": {
                    "name": "Deribit",
                    "rest_limits": {
                        "default": {
                            "requests": 20,
                            "window_seconds": 1,
                        },  # 20 requests/second
                        "orders": {
                            "requests": 5,
                            "window_seconds": 1,
                        },  # Order management
                        "private": {"requests": 20, "window_seconds": 1},  # Private API
                    },
                    "ws_reconnect_threshold": 3,
                    "warning_threshold": 0.8,
                    "backoff_threshold": 0.9,
                },
            },
            "monitoring_window_minutes": 15,  # Monitor 15-minute windows
            "backoff_curves": {
                "linear": lambda usage: usage * 1000,  # Linear backoff in ms
                "exponential": lambda usage: min(
                    5000, math.exp(usage * 5) - 1
                ),  # Exponential with cap
                "polynomial": lambda usage: (usage**2) * 2000,  # Quadratic backoff
            },
            "alert_retention_hours": 24,
            "metrics_retention_days": 7,
        }

        logger.info("Initialized API quota monitor")

    def record_api_call(
        self,
        exchange: str,
        endpoint_type: str = "default",
        response_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, any]:
        """
        Record an API call for quota tracking.

        Args:
            exchange: Exchange name (binance, coinbase, etc.)
            endpoint_type: Type of endpoint (default, orders, account, etc.)
            response_headers: HTTP response headers containing rate limit info

        Returns:
            Current quota status after recording the call
        """
        try:
            if not self.redis_client:
                return {"error": "Redis unavailable"}

            current_time = time.time()
            timestamp = datetime.now().isoformat()

            # Record the API call
            call_key = f"api_calls:{exchange}:{endpoint_type}"
            self.redis_client.zadd(call_key, {timestamp: current_time})

            # Clean old entries (keep only current window)
            window_seconds = (
                self.config["exchanges"]
                .get(exchange, {})
                .get("rest_limits", {})
                .get(endpoint_type, {})
                .get("window_seconds", 60)
            )
            cutoff_time = current_time - window_seconds
            self.redis_client.zremrangebyscore(call_key, 0, cutoff_time)

            # Set expiration
            self.redis_client.expire(call_key, window_seconds * 2)

            # Parse rate limit headers if provided
            rate_limit_info = {}
            if response_headers:
                rate_limit_info = self._parse_rate_limit_headers(
                    exchange, response_headers
                )

            # Get current usage status
            usage_status = self.get_exchange_quota_status(exchange, endpoint_type)
            usage_status.update(rate_limit_info)

            # Check for warnings or backoff needed
            self._check_quota_alerts(exchange, endpoint_type, usage_status)

            return usage_status

        except Exception as e:
            logger.error(f"Error recording API call: {e}")
            return {"error": str(e)}

    def _parse_rate_limit_headers(
        self, exchange: str, headers: Dict[str, str]
    ) -> Dict[str, any]:
        """Parse rate limit information from response headers."""
        rate_limit_info = {}

        try:
            if exchange == "binance":
                if "x-mbx-used-weight-1m" in headers:
                    rate_limit_info["weight_used"] = int(
                        headers["x-mbx-used-weight-1m"]
                    )
                if "x-mbx-order-count-1s" in headers:
                    rate_limit_info["order_count"] = int(
                        headers["x-mbx-order-count-1s"]
                    )

            elif exchange == "coinbase":
                if "cb-before" in headers:
                    rate_limit_info["pagination_before"] = headers["cb-before"]
                if "cb-after" in headers:
                    rate_limit_info["pagination_after"] = headers["cb-after"]

            elif exchange == "alpaca":
                if "x-ratelimit-limit" in headers:
                    rate_limit_info["limit"] = int(headers["x-ratelimit-limit"])
                if "x-ratelimit-remaining" in headers:
                    rate_limit_info["remaining"] = int(headers["x-ratelimit-remaining"])
                if "x-ratelimit-reset" in headers:
                    rate_limit_info["reset_time"] = headers["x-ratelimit-reset"]

            elif exchange == "deribit":
                if "rate-limit" in headers:
                    rate_limit_info["rate_limit"] = headers["rate-limit"]

        except Exception as e:
            logger.debug(f"Error parsing rate limit headers for {exchange}: {e}")

        return rate_limit_info

    def get_exchange_quota_status(
        self, exchange: str, endpoint_type: str = "default"
    ) -> Dict[str, any]:
        """
        Get current quota usage status for an exchange endpoint.

        Args:
            exchange: Exchange name
            endpoint_type: Endpoint type to check

        Returns:
            Current quota usage status
        """
        try:
            if not self.redis_client:
                return {"error": "Redis unavailable"}

            # Get exchange config
            exchange_config = self.config["exchanges"].get(exchange, {})
            limits = exchange_config.get("rest_limits", {}).get(
                endpoint_type, {"requests": 100, "window_seconds": 60}
            )

            # Count calls in current window
            current_time = time.time()
            window_seconds = limits["window_seconds"]
            cutoff_time = current_time - window_seconds

            call_key = f"api_calls:{exchange}:{endpoint_type}"
            current_calls = self.redis_client.zcount(
                call_key, cutoff_time, current_time
            )

            # Calculate usage metrics
            limit = limits["requests"]
            usage_rate = current_calls / limit if limit > 0 else 0
            remaining = max(0, limit - current_calls)

            status = {
                "exchange": exchange,
                "endpoint_type": endpoint_type,
                "timestamp": datetime.now().isoformat(),
                "current_calls": current_calls,
                "limit": limit,
                "remaining": remaining,
                "usage_rate": usage_rate,
                "window_seconds": window_seconds,
                "backoff_needed": usage_rate
                >= exchange_config.get("backoff_threshold", 0.9),
                "warning_level": self._determine_warning_level(
                    usage_rate, exchange_config
                ),
            }

            return status

        except Exception as e:
            logger.error(f"Error getting quota status: {e}")
            return {"error": str(e)}

    def _determine_warning_level(self, usage_rate: float, exchange_config: Dict) -> str:
        """Determine warning level based on usage rate."""
        backoff_threshold = exchange_config.get("backoff_threshold", 0.9)
        warning_threshold = exchange_config.get("warning_threshold", 0.8)

        if usage_rate >= backoff_threshold:
            return "CRITICAL"
        elif usage_rate >= warning_threshold:
            return "WARNING"
        else:
            return "OK"

    def calculate_backoff_delay(
        self, exchange: str, endpoint_type: str, backoff_type: str = "exponential"
    ) -> float:
        """
        Calculate recommended backoff delay based on current usage.

        Args:
            exchange: Exchange name
            endpoint_type: Endpoint type
            backoff_type: Backoff curve type (linear, exponential, polynomial)

        Returns:
            Recommended delay in milliseconds
        """
        try:
            status = self.get_exchange_quota_status(exchange, endpoint_type)
            usage_rate = status.get("usage_rate", 0)

            if usage_rate < 0.8:  # No backoff needed below 80%
                return 0

            # Apply backoff curve
            backoff_func = self.config["backoff_curves"].get(
                backoff_type, self.config["backoff_curves"]["exponential"]
            )

            # Scale usage rate to 0-1 range for backoff calculation
            scaled_usage = max(0, (usage_rate - 0.8) / 0.2)  # 0.8-1.0 -> 0-1
            delay_ms = backoff_func(scaled_usage)

            logger.debug(
                f"Backoff delay for {exchange}:{endpoint_type}: {delay_ms}ms (usage: {usage_rate:.1%})"
            )

            return delay_ms

        except Exception as e:
            logger.error(f"Error calculating backoff delay: {e}")
            return 1000  # Default 1 second delay on error

    def record_websocket_reconnect(
        self, exchange: str, reason: str = "unknown"
    ) -> Dict[str, any]:
        """
        Record a WebSocket reconnection event.

        Args:
            exchange: Exchange name
            reason: Reason for reconnection

        Returns:
            Current reconnection status
        """
        try:
            if not self.redis_client:
                return {"error": "Redis unavailable"}

            current_time = time.time()
            timestamp = datetime.now().isoformat()

            # Record reconnection event
            reconnect_event = {
                "timestamp": timestamp,
                "reason": reason,
                "exchange": exchange,
            }

            reconnect_key = f"ws_reconnects:{exchange}"
            self.redis_client.zadd(
                reconnect_key, {json.dumps(reconnect_event): current_time}
            )

            # Clean old entries (keep only last hour)
            cutoff_time = current_time - 3600  # 1 hour
            self.redis_client.zremrangebyscore(reconnect_key, 0, cutoff_time)
            self.redis_client.expire(reconnect_key, 7200)  # 2 hour TTL

            # Check reconnection rate
            hourly_reconnects = self.redis_client.zcount(
                reconnect_key, cutoff_time, current_time
            )
            threshold = (
                self.config["exchanges"]
                .get(exchange, {})
                .get("ws_reconnect_threshold", 3)
            )

            status = {
                "exchange": exchange,
                "timestamp": timestamp,
                "hourly_reconnects": hourly_reconnects,
                "threshold": threshold,
                "exceeds_threshold": hourly_reconnects > threshold,
                "latest_reason": reason,
            }

            # Generate alert if threshold exceeded
            if status["exceeds_threshold"]:
                self._generate_reconnect_alert(exchange, status)

            logger.info(
                f"WebSocket reconnect recorded: {exchange} ({hourly_reconnects}/{threshold})"
            )

            return status

        except Exception as e:
            logger.error(f"Error recording WebSocket reconnect: {e}")
            return {"error": str(e)}

    def _check_quota_alerts(
        self, exchange: str, endpoint_type: str, usage_status: Dict[str, any]
    ):
        """Check if quota alerts need to be generated."""
        try:
            usage_rate = usage_status.get("usage_rate", 0)
            warning_level = usage_status.get("warning_level", "OK")

            if warning_level in ["WARNING", "CRITICAL"]:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "alert_type": "api_quota_usage",
                    "exchange": exchange,
                    "endpoint_type": endpoint_type,
                    "level": warning_level,
                    "usage_rate": usage_rate,
                    "current_calls": usage_status.get("current_calls", 0),
                    "limit": usage_status.get("limit", 0),
                    "message": f"{warning_level}: {exchange} {endpoint_type} usage at {usage_rate:.1%}",
                }

                if self.redis_client:
                    self.redis_client.lpush("alerts:api_quota", json.dumps(alert))
                    self.redis_client.ltrim(
                        "alerts:api_quota", 0, 99
                    )  # Keep last 100 alerts

                logger.warning(f"⚠️ API QUOTA ALERT: {alert['message']}")

        except Exception as e:
            logger.error(f"Error checking quota alerts: {e}")

    def _generate_reconnect_alert(self, exchange: str, status: Dict[str, any]):
        """Generate WebSocket reconnection alert."""
        try:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "alert_type": "websocket_reconnects",
                "exchange": exchange,
                "level": "WARNING",
                "hourly_reconnects": status.get("hourly_reconnects", 0),
                "threshold": status.get("threshold", 3),
                "latest_reason": status.get("latest_reason", "unknown"),
                "message": f"WebSocket reconnects exceeded threshold: {status.get('hourly_reconnects', 0)}/{status.get('threshold', 3)}",
            }

            if self.redis_client:
                self.redis_client.lpush(
                    "alerts:websocket_reconnects", json.dumps(alert)
                )
                self.redis_client.ltrim("alerts:websocket_reconnects", 0, 99)

            logger.warning(f"⚠️ WEBSOCKET ALERT: {alert['message']}")

        except Exception as e:
            logger.error(f"Error generating reconnect alert: {e}")

    def get_all_exchange_status(self) -> Dict[str, any]:
        """Get quota status for all exchanges and endpoint types."""
        try:
            all_status = {
                "timestamp": datetime.now().isoformat(),
                "exchanges": {},
                "summary": {},
            }

            total_warnings = 0
            total_critical = 0

            for exchange in self.config["exchanges"].keys():
                exchange_status = {"endpoints": {}, "ws_reconnects": {}}

                # Check all endpoint types for this exchange
                endpoint_types = self.config["exchanges"][exchange][
                    "rest_limits"
                ].keys()
                for endpoint_type in endpoint_types:
                    endpoint_status = self.get_exchange_quota_status(
                        exchange, endpoint_type
                    )
                    exchange_status["endpoints"][endpoint_type] = endpoint_status

                    # Count warnings/critical
                    warning_level = endpoint_status.get("warning_level", "OK")
                    if warning_level == "WARNING":
                        total_warnings += 1
                    elif warning_level == "CRITICAL":
                        total_critical += 1

                # Get WebSocket reconnection status
                if self.redis_client:
                    reconnect_key = f"ws_reconnects:{exchange}"
                    current_time = time.time()
                    cutoff_time = current_time - 3600
                    hourly_reconnects = self.redis_client.zcount(
                        reconnect_key, cutoff_time, current_time
                    )
                    threshold = self.config["exchanges"][exchange].get(
                        "ws_reconnect_threshold", 3
                    )

                    exchange_status["ws_reconnects"] = {
                        "hourly_reconnects": hourly_reconnects,
                        "threshold": threshold,
                        "exceeds_threshold": hourly_reconnects > threshold,
                    }

                all_status["exchanges"][exchange] = exchange_status

            # Summary statistics
            all_status["summary"] = {
                "total_exchanges": len(self.config["exchanges"]),
                "warning_alerts": total_warnings,
                "critical_alerts": total_critical,
                "overall_healthy": total_critical == 0,
            }

            return all_status

        except Exception as e:
            logger.error(f"Error getting all exchange status: {e}")
            return {"error": str(e)}

    def run_monitoring_cycle(self) -> Dict[str, any]:
        """Run single API quota monitoring cycle."""
        try:
            logger.debug("📊 Running API quota monitoring cycle")

            cycle_results = {
                "timestamp": datetime.now().isoformat(),
                "cycle_type": "api_quota_monitoring",
            }

            # Get status for all exchanges
            all_status = self.get_all_exchange_status()
            cycle_results["all_status"] = all_status

            # Log critical issues
            summary = all_status.get("summary", {})
            if summary.get("critical_alerts", 0) > 0:
                logger.warning(
                    f"API quota critical alerts: {summary['critical_alerts']}"
                )

            return cycle_results

        except Exception as e:
            logger.error(f"Error in API quota monitoring cycle: {e}")
            return {"error": str(e)}

    def run_monitoring_daemon(self):
        """Run API quota monitor as continuous daemon."""
        logger.info("📊 Starting API quota monitoring daemon")

        try:
            while True:
                cycle_results = self.run_monitoring_cycle()

                # Log status summary
                all_status = cycle_results.get("all_status", {})
                summary = all_status.get("summary", {})

                if summary.get("critical_alerts", 0) > 0:
                    logger.warning(
                        f"Critical quota alerts: {summary['critical_alerts']}"
                    )
                elif summary.get("warning_alerts", 0) > 0:
                    logger.info(f"Quota warnings: {summary['warning_alerts']}")

                # Wait before next cycle
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("API quota monitoring daemon stopped by user")
        except Exception as e:
            logger.error(f"API quota monitoring daemon error: {e}")

    def export_prometheus_metrics(self) -> str:
        """Export API quota metrics in Prometheus format."""
        try:
            all_status = self.get_all_exchange_status()
            metrics = []

            for exchange, exchange_data in all_status.get("exchanges", {}).items():
                # REST API quota metrics
                for endpoint_type, endpoint_data in exchange_data.get(
                    "endpoints", {}
                ).items():
                    usage_rate = endpoint_data.get("usage_rate", 0)
                    current_calls = endpoint_data.get("current_calls", 0)
                    limit = endpoint_data.get("limit", 0)
                    remaining = endpoint_data.get("remaining", 0)

                    labels = f'exchange="{exchange}",endpoint="{endpoint_type}"'

                    metrics.append(f"api_quota_usage_rate{{{labels}}} {usage_rate}")
                    metrics.append(
                        f"api_quota_current_calls{{{labels}}} {current_calls}"
                    )
                    metrics.append(f"api_quota_limit{{{labels}}} {limit}")
                    metrics.append(f"api_quota_remaining{{{labels}}} {remaining}")

                # WebSocket reconnection metrics
                ws_data = exchange_data.get("ws_reconnects", {})
                hourly_reconnects = ws_data.get("hourly_reconnects", 0)
                threshold = ws_data.get("threshold", 3)
                exceeds = int(ws_data.get("exceeds_threshold", False))

                exchange_label = f'exchange="{exchange}"'
                metrics.append(
                    f"websocket_hourly_reconnects{{{exchange_label}}} {hourly_reconnects}"
                )
                metrics.append(
                    f"websocket_reconnect_threshold{{{exchange_label}}} {threshold}"
                )
                metrics.append(
                    f"websocket_reconnects_exceeds_threshold{{{exchange_label}}} {exceeds}"
                )

            # Summary metrics
            summary = all_status.get("summary", {})
            metrics.append(
                f"api_quota_warning_alerts {summary.get('warning_alerts', 0)}"
            )
            metrics.append(
                f"api_quota_critical_alerts {summary.get('critical_alerts', 0)}"
            )
            metrics.append(
                f"api_quota_overall_healthy {int(summary.get('overall_healthy', False))}"
            )

            return "\n".join(metrics) + "\n"

        except Exception as e:
            logger.error(f"Error exporting Prometheus metrics: {e}")
            return f"# Error: {e}\n"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="API Quota Monitor")

    parser.add_argument(
        "--mode",
        choices=["status", "record", "daemon", "metrics"],
        default="status",
        help="Monitor mode",
    )
    parser.add_argument("--exchange", type=str, help="Exchange to check/record for")
    parser.add_argument("--endpoint", type=str, default="default", help="Endpoint type")
    parser.add_argument(
        "--reconnect-reason", type=str, help="Record WebSocket reconnect with reason"
    )
    parser.add_argument("--output", type=str, help="Output file for results")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info("📊 Starting API Quota Monitor")

    try:
        monitor = APIQuotaMonitor()

        if args.mode == "status":
            if args.exchange:
                results = monitor.get_exchange_quota_status(
                    args.exchange, args.endpoint
                )
                print(f"\n📊 QUOTA STATUS ({args.exchange}):")
            else:
                results = monitor.get_all_exchange_status()
                print(f"\n📊 ALL EXCHANGE QUOTA STATUS:")
            print(json.dumps(results, indent=2))

        elif args.mode == "record":
            if not args.exchange:
                print("--exchange required for record mode")
                return 1

            if args.reconnect_reason:
                results = monitor.record_websocket_reconnect(
                    args.exchange, args.reconnect_reason
                )
                print(f"\n📡 WEBSOCKET RECONNECT RECORDED:")
            else:
                results = monitor.record_api_call(args.exchange, args.endpoint)
                print(f"\n📊 API CALL RECORDED:")
            print(json.dumps(results, indent=2))

        elif args.mode == "daemon":
            monitor.run_monitoring_daemon()
            return 0

        elif args.mode == "metrics":
            metrics = monitor.export_prometheus_metrics()
            print(metrics)
            return 0

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Error in API quota monitor: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
