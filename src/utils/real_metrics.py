#!/usr/bin/env python3
"""
Real Prometheus Metrics Implementation

Implements the real prometheus_client with proper metric types, buckets, 
and SLO monitoring as specified in Future_instruction.txt.

Features:
- Real prometheus_client library
- Proper histogram buckets for latency
- VaR and risk metrics gauges  
- SLO monitoring and alerting
- Exposing on :9090 for Prometheus scraping
"""

import threading
import time
from typing import Dict, Any, List, Optional

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    start_http_server,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY,
)


class RealMetricsExporter:
    """
    Real Prometheus metrics exporter with comprehensive trading system metrics.

    Implements Future_instruction.txt metrics requirements:
    - REQ_LAT histogram with proper buckets
    - VAR_GAUGE for risk metrics
    - Expose on :9090
    """

    def __init__(self, port: int = 9090):
        """
        Initialize real prometheus metrics exporter.

        Args:
            port: Port to serve metrics on (default 9090 as per Future_instruction.txt)
        """
        self.port = port
        self._server_started = False
        self._lock = threading.RLock()

        # Define metric buckets as specified in Future_instruction.txt
        self.latency_buckets = [10, 25, 50, 100, 200, 500, 1000]  # milliseconds

        # Core trading system metrics
        self.request_latency = Histogram(
            "req_latency_ms",
            "Request latency in milliseconds",
            buckets=self.latency_buckets,
        )

        self.var_gauge = Gauge(
            "risk_var", "Daily Value-at-Risk", ["symbol", "confidence_level"]
        )

        self.cvar_gauge = Gauge(
            "risk_cvar", "Conditional Value-at-Risk", ["symbol", "confidence_level"]
        )

        # Trading performance metrics
        self.crypto_ticks_total = Counter(
            "crypto_ticks_total",
            "Total crypto ticks processed",
            ["symbol", "source", "exchange"],
        )

        self.crypto_price_usd = Gauge(
            "crypto_price_usd", "Current crypto price in USD", ["symbol", "exchange"]
        )

        self.alpha_signal_edge_bps = Gauge(
            "alpha_signal_edge_bps",
            "Alpha signal edge in basis points",
            ["symbol", "model", "strategy"],
        )

        self.alpha_signal_confidence = Gauge(
            "alpha_signal_confidence",
            "Alpha signal confidence score",
            ["symbol", "model", "strategy"],
        )

        # Position and portfolio metrics
        self.portfolio_value_usd = Gauge(
            "portfolio_value_usd",
            "Current portfolio value in USD",
            ["account", "currency"],
        )

        self.position_size_usd = Gauge(
            "position_size_usd",
            "Current position size in USD",
            ["symbol", "side", "exchange"],
        )

        self.pnl_realized_usd = Counter(
            "pnl_realized_usd_total",
            "Total realized PnL in USD",
            ["symbol", "strategy", "exchange"],
        )

        self.pnl_unrealized_usd = Gauge(
            "pnl_unrealized_usd",
            "Current unrealized PnL in USD",
            ["symbol", "strategy", "exchange"],
        )

        # Execution metrics
        self.order_fill_latency = Histogram(
            "order_fill_latency_ms",
            "Order fill latency in milliseconds",
            ["exchange", "order_type"],
            buckets=self.latency_buckets,
        )

        self.slippage_bps = Histogram(
            "slippage_bps",
            "Execution slippage in basis points",
            ["symbol", "exchange", "order_type"],
            buckets=[0.5, 1, 2, 5, 10, 20, 50, 100],
        )

        # Risk management metrics
        self.risk_limit_breaches_total = Counter(
            "risk_limit_breaches_total",
            "Total risk limit breaches",
            ["limit_type", "symbol", "severity"],
        )

        self.kill_switch_activations_total = Counter(
            "kill_switch_activations_total",
            "Total kill switch activations",
            ["reason", "trigger"],
        )

        # System health metrics
        self.system_uptime_seconds = Gauge(
            "system_uptime_seconds", "System uptime in seconds"
        )

        self.memory_usage_bytes = Gauge(
            "memory_usage_bytes", "Memory usage in bytes", ["component"]
        )

        self.cpu_usage_percent = Gauge(
            "cpu_usage_percent", "CPU usage percentage", ["component"]
        )

        # Data quality metrics
        self.data_staleness_seconds = Gauge(
            "data_staleness_seconds", "Data staleness in seconds", ["feed", "symbol"]
        )

        self.data_gaps_total = Counter(
            "data_gaps_total", "Total data gaps detected", ["feed", "symbol"]
        )

        # SLO metrics (Service Level Objectives)
        self.slo_latency_target = Gauge(
            "slo_latency_target_ms", "SLO latency target in milliseconds", ["service"]
        )

        self.slo_availability_target = Gauge(
            "slo_availability_target_percent",
            "SLO availability target percentage",
            ["service"],
        )

        self.slo_error_budget_remaining = Gauge(
            "slo_error_budget_remaining_percent",
            "Remaining error budget percentage",
            ["service"],
        )

        self.slo_violations_total = Counter(
            "slo_violations_total", "Total SLO violations", ["service", "slo_type"]
        )

        # Initialize start time for uptime calculation
        self.start_time = time.time()

        # Start metrics server
        self.start_server()

    def start_server(self):
        """Start the Prometheus metrics HTTP server on specified port."""
        try:
            with self._lock:
                if not self._server_started:
                    start_http_server(self.port)
                    self._server_started = True
                    print(
                        f"✅ Real Prometheus metrics server started on port {self.port}"
                    )
                    print(
                        f"📊 Metrics available at: http://localhost:{self.port}/metrics"
                    )
        except Exception as e:
            print(f"❌ Error starting metrics server on port {self.port}: {e}")

    def record_market_tick(self, symbol: str, exchange: str, source: str = "live"):
        """Record a market tick."""
        self.crypto_ticks_total.labels(
            symbol=symbol, source=source, exchange=exchange
        ).inc()

    def update_crypto_price(self, symbol: str, price: float, exchange: str = "unknown"):
        """Update crypto price gauge."""
        self.crypto_price_usd.labels(symbol=symbol, exchange=exchange).set(price)

    def record_alpha_signal(
        self,
        symbol: str,
        edge_bps: float,
        confidence: float,
        model: str = "unknown",
        strategy: str = "unknown",
    ):
        """Record alpha signal metrics."""
        self.alpha_signal_edge_bps.labels(
            symbol=symbol, model=model, strategy=strategy
        ).set(edge_bps)

        self.alpha_signal_confidence.labels(
            symbol=symbol, model=model, strategy=strategy
        ).set(confidence)

    def update_var_metrics(
        self, symbol: str, var_95: float, var_99: float, cvar_95: float, cvar_99: float
    ):
        """Update VaR and CVaR metrics as per Future_instruction.txt."""
        self.var_gauge.labels(symbol=symbol, confidence_level="95").set(var_95)
        self.var_gauge.labels(symbol=symbol, confidence_level="99").set(var_99)
        self.cvar_gauge.labels(symbol=symbol, confidence_level="95").set(cvar_95)
        self.cvar_gauge.labels(symbol=symbol, confidence_level="99").set(cvar_99)

    def record_request_latency(self, latency_ms: float):
        """Record request latency in the histogram."""
        self.request_latency.observe(latency_ms)

    def record_order_fill(self, exchange: str, order_type: str, latency_ms: float):
        """Record order fill latency."""
        self.order_fill_latency.labels(
            exchange=exchange, order_type=order_type
        ).observe(latency_ms)

    def record_slippage(
        self, symbol: str, exchange: str, order_type: str, slippage_bps: float
    ):
        """Record execution slippage."""
        self.slippage_bps.labels(
            symbol=symbol, exchange=exchange, order_type=order_type
        ).observe(slippage_bps)

    def record_risk_breach(
        self, limit_type: str, symbol: str, severity: str = "medium"
    ):
        """Record risk limit breach."""
        self.risk_limit_breaches_total.labels(
            limit_type=limit_type, symbol=symbol, severity=severity
        ).inc()

    def record_kill_switch_activation(self, reason: str, trigger: str = "automatic"):
        """Record kill switch activation."""
        self.kill_switch_activations_total.labels(reason=reason, trigger=trigger).inc()

    def update_portfolio_value(
        self, value_usd: float, account: str = "main", currency: str = "USD"
    ):
        """Update portfolio value."""
        self.portfolio_value_usd.labels(account=account, currency=currency).set(
            value_usd
        )

    def update_position_size(
        self, symbol: str, size_usd: float, side: str, exchange: str
    ):
        """Update position size."""
        self.position_size_usd.labels(symbol=symbol, side=side, exchange=exchange).set(
            size_usd
        )

    def record_realized_pnl(
        self, symbol: str, pnl_usd: float, strategy: str, exchange: str
    ):
        """Record realized PnL."""
        # For counter, we add the PnL (can be negative)
        if pnl_usd > 0:
            self.pnl_realized_usd.labels(
                symbol=symbol, strategy=strategy, exchange=exchange
            ).inc(pnl_usd)

    def update_unrealized_pnl(
        self, symbol: str, pnl_usd: float, strategy: str, exchange: str
    ):
        """Update unrealized PnL."""
        self.pnl_unrealized_usd.labels(
            symbol=symbol, strategy=strategy, exchange=exchange
        ).set(pnl_usd)

    def update_system_health(
        self, memory_bytes: float, cpu_percent: float, component: str = "main"
    ):
        """Update system health metrics."""
        self.system_uptime_seconds.set(time.time() - self.start_time)
        self.memory_usage_bytes.labels(component=component).set(memory_bytes)
        self.cpu_usage_percent.labels(component=component).set(cpu_percent)

    def update_data_quality(self, feed: str, symbol: str, staleness_seconds: float):
        """Update data quality metrics."""
        self.data_staleness_seconds.labels(feed=feed, symbol=symbol).set(
            staleness_seconds
        )

    def record_data_gap(self, feed: str, symbol: str):
        """Record data gap."""
        self.data_gaps_total.labels(feed=feed, symbol=symbol).inc()

    def update_slo_metrics(
        self,
        service: str,
        latency_target_ms: float,
        availability_target_percent: float,
        error_budget_remaining_percent: float,
    ):
        """Update SLO metrics."""
        self.slo_latency_target.labels(service=service).set(latency_target_ms)
        self.slo_availability_target.labels(service=service).set(
            availability_target_percent
        )
        self.slo_error_budget_remaining.labels(service=service).set(
            error_budget_remaining_percent
        )

    def record_slo_violation(self, service: str, slo_type: str):
        """Record SLO violation."""
        self.slo_violations_total.labels(service=service, slo_type=slo_type).inc()

    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        return generate_latest(REGISTRY).decode("utf-8")

    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST


# Global metrics exporter instance
_metrics_exporter = None


def get_real_metrics_exporter(port: int = 9090) -> RealMetricsExporter:
    """Get or create the global real metrics exporter instance."""
    global _metrics_exporter
    if _metrics_exporter is None:
        _metrics_exporter = RealMetricsExporter(port)
    return _metrics_exporter


# Convenience functions for common metrics
def record_market_tick(symbol: str, exchange: str, source: str = "live"):
    """Record market tick."""
    get_real_metrics_exporter().record_market_tick(symbol, exchange, source)


def update_crypto_price(symbol: str, price: float, exchange: str = "unknown"):
    """Update crypto price."""
    get_real_metrics_exporter().update_crypto_price(symbol, price, exchange)


def record_alpha_signal(
    symbol: str,
    edge_bps: float,
    confidence: float,
    model: str = "unknown",
    strategy: str = "unknown",
):
    """Record alpha signal."""
    get_real_metrics_exporter().record_alpha_signal(
        symbol, edge_bps, confidence, model, strategy
    )


def update_var_metrics(
    symbol: str, var_95: float, var_99: float, cvar_95: float, cvar_99: float
):
    """Update VaR/CVaR metrics."""
    get_real_metrics_exporter().update_var_metrics(
        symbol, var_95, var_99, cvar_95, cvar_99
    )


def record_request_latency(latency_ms: float):
    """Record request latency."""
    get_real_metrics_exporter().record_request_latency(latency_ms)
