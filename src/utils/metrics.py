"""
Prometheus Metrics for Trading System

Provides standardized metrics collection for monitoring trading system
performance, latencies, and business metrics.
"""

import time
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    generate_latest,
    start_http_server,
)
import threading

from .logger import get_logger


class TradingMetrics:
    """
    Centralized metrics collection for the trading system.

    Provides Prometheus metrics for monitoring all aspects of the
    trading system performance.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize trading metrics."""
        self.registry = registry or CollectorRegistry()
        self.logger = get_logger("metrics")

        # Data ingestion metrics
        self.market_ticks_received = Counter(
            "trading_market_ticks_total",
            "Total market ticks received",
            ["symbol", "exchange", "asset_type"],
            registry=self.registry,
        )

        self.market_tick_latency = Histogram(
            "trading_market_tick_latency_seconds",
            "Market tick processing latency",
            ["symbol", "exchange"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry,
        )

        # Feature computation metrics
        self.feature_computation_latency = Summary(
            "trading_feature_computation_latency_microseconds",
            "Feature computation latency in microseconds",
            ["symbol"],
            registry=self.registry,
        )

        self.features_computed = Counter(
            "trading_features_computed_total",
            "Total features computed",
            ["symbol"],
            registry=self.registry,
        )

        # Soft Information Metrics (Task 9)
        self.soft_docs_total = Counter(
            "soft_docs_total",
            "Total sentiment documents processed",
            ["status", "source"],
            registry=self.registry,
        )

        self.sent_score_latest = Gauge(
            "sent_score_latest",
            "Latest sentiment score for symbol",
            ["symbol"],
            registry=self.registry,
        )

        self.sentiment_processing_latency = Histogram(
            "sentiment_processing_latency_seconds",
            "Sentiment analysis processing latency",
            ["source"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry,
        )

        self.big_bet_flags_total = Counter(
            "big_bet_flags_total",
            "Total big bet flags generated",
            ["symbol", "flag_status"],
            registry=self.registry,
        )

        self.explanation_requests_total = Counter(
            "explanation_requests_total",
            "Total explanation requests processed",
            ["status"],
            registry=self.registry,
        )

        # Alpha model metrics
        self.alpha_predictions = Counter(
            "trading_alpha_predictions_total",
            "Total alpha predictions made",
            ["model_name", "symbol"],
            registry=self.registry,
        )

        self.alpha_edge_bps = Histogram(
            "trading_alpha_edge_bps",
            "Alpha edge predictions in basis points",
            ["model_name", "symbol"],
            buckets=[-50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50],
            registry=self.registry,
        )

        self.alpha_confidence = Histogram(
            "trading_alpha_confidence",
            "Alpha model confidence scores",
            ["model_name", "symbol"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        # Ensemble metrics
        self.ensemble_predictions = Counter(
            "trading_ensemble_predictions_total",
            "Total ensemble predictions",
            ["symbol"],
            registry=self.registry,
        )

        self.ensemble_edge_bps = Histogram(
            "trading_ensemble_edge_bps",
            "Ensemble edge predictions in basis points",
            ["symbol"],
            buckets=[-50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50],
            registry=self.registry,
        )

        # Position sizing metrics
        self.position_sizes_calculated = Counter(
            "trading_position_sizes_calculated_total",
            "Total position sizes calculated",
            ["symbol"],
            registry=self.registry,
        )

        self.kelly_fraction = Histogram(
            "trading_kelly_fraction",
            "Kelly fraction calculated",
            ["symbol"],
            buckets=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
            registry=self.registry,
        )

        # Execution metrics
        self.orders_submitted = Counter(
            "trading_orders_submitted_total",
            "Total orders submitted",
            ["symbol", "side", "order_type"],
            registry=self.registry,
        )

        self.orders_filled = Counter(
            "trading_orders_filled_total",
            "Total orders filled",
            ["symbol", "side"],
            registry=self.registry,
        )

        self.order_fill_latency = Histogram(
            "trading_order_fill_latency_seconds",
            "Order fill latency",
            ["symbol", "side"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

        self.slippage_bps = Histogram(
            "trading_slippage_bps",
            "Execution slippage in basis points",
            ["symbol", "side"],
            buckets=[0, 1, 2, 5, 10, 20, 50, 100],
            registry=self.registry,
        )

        # Portfolio metrics
        self.portfolio_value = Gauge(
            "trading_portfolio_value_dollars",
            "Current portfolio value in dollars",
            registry=self.registry,
        )

        self.cash_balance = Gauge(
            "trading_cash_balance_dollars",
            "Current cash balance",
            registry=self.registry,
        )

        self.position_value = Gauge(
            "trading_position_value_dollars",
            "Position value by symbol",
            ["symbol"],
            registry=self.registry,
        )

        self.pnl_realized = Counter(
            "trading_pnl_realized_dollars",
            "Realized P&L in dollars",
            ["symbol"],
            registry=self.registry,
        )

        self.pnl_unrealized = Gauge(
            "trading_pnl_unrealized_dollars",
            "Unrealized P&L in dollars",
            ["symbol"],
            registry=self.registry,
        )

        # Risk metrics
        self.risk_score = Gauge(
            "trading_risk_score",
            "Overall portfolio risk score (0-100)",
            registry=self.registry,
        )

        self.drawdown = Gauge(
            "trading_drawdown_fraction",
            "Current drawdown as fraction",
            registry=self.registry,
        )

        self.exposure = Gauge(
            "trading_exposure_fraction",
            "Portfolio exposure as fraction",
            registry=self.registry,
        )

        self.risk_violations = Counter(
            "trading_risk_violations_total",
            "Risk limit violations",
            ["violation_type"],
            registry=self.registry,
        )

        # System health metrics
        self.system_uptime = Gauge(
            "trading_system_uptime_seconds",
            "System uptime in seconds",
            registry=self.registry,
        )

        self.memory_usage = Gauge(
            "trading_memory_usage_bytes",
            "Memory usage in bytes",
            registry=self.registry,
        )

        self.cpu_usage = Gauge(
            "trading_cpu_usage_percent", "CPU usage percentage", registry=self.registry
        )

        # Component health
        self.component_health = Gauge(
            "trading_component_health",
            "Component health status (1=healthy, 0=unhealthy)",
            ["component"],
            registry=self.registry,
        )

        # Initialize system start time
        self.start_time = time.time()

        self.logger.info("Trading metrics initialized")

    # Data ingestion methods
    def record_market_tick(
        self, symbol: str, exchange: str, asset_type: str, latency: float
    ):
        """Record a market tick with latency."""
        self.market_ticks_received.labels(
            symbol=symbol, exchange=exchange, asset_type=asset_type
        ).inc()
        self.market_tick_latency.labels(symbol=symbol, exchange=exchange).observe(
            latency
        )

    def record_feature_computation(self, symbol: str, latency_us: float):
        """Record feature computation latency."""
        self.features_computed.labels(symbol=symbol).inc()
        self.feature_computation_latency.labels(symbol=symbol).observe(latency_us)

    # Alpha model methods
    def record_alpha_prediction(
        self, model_name: str, symbol: str, edge_bps: float, confidence: float
    ):
        """Record an alpha model prediction."""
        self.alpha_predictions.labels(model_name=model_name, symbol=symbol).inc()
        self.alpha_edge_bps.labels(model_name=model_name, symbol=symbol).observe(
            edge_bps
        )
        self.alpha_confidence.labels(model_name=model_name, symbol=symbol).observe(
            confidence
        )

    # Ensemble methods
    def record_ensemble_prediction(self, symbol: str, edge_bps: float):
        """Record an ensemble prediction."""
        self.ensemble_predictions.labels(symbol=symbol).inc()
        self.ensemble_edge_bps.labels(symbol=symbol).observe(edge_bps)

    # Position sizing methods
    def record_position_sizing(self, symbol: str, kelly_frac: float):
        """Record position sizing calculation."""
        self.position_sizes_calculated.labels(symbol=symbol).inc()
        self.kelly_fraction.labels(symbol=symbol).observe(kelly_frac)

    # Execution methods
    def record_order_submitted(
        self, symbol: str, side: str, order_type: str = "market"
    ):
        """Record order submission."""
        self.orders_submitted.labels(
            symbol=symbol, side=side, order_type=order_type
        ).inc()

    def record_order_filled(
        self, symbol: str, side: str, fill_latency: float, slippage_bps: float
    ):
        """Record order fill with execution metrics."""
        self.orders_filled.labels(symbol=symbol, side=side).inc()
        self.order_fill_latency.labels(symbol=symbol, side=side).observe(fill_latency)
        self.slippage_bps.labels(symbol=symbol, side=side).observe(slippage_bps)

    # Portfolio methods
    def update_portfolio_metrics(self, portfolio_value: float, cash_balance: float):
        """Update portfolio-level metrics."""
        self.portfolio_value.set(portfolio_value)
        self.cash_balance.set(cash_balance)

    def update_position_value(self, symbol: str, value: float):
        """Update position value for a symbol."""
        self.position_value.labels(symbol=symbol).set(value)

    def record_realized_pnl(self, symbol: str, pnl: float):
        """Record realized P&L."""
        self.pnl_realized.labels(symbol=symbol).inc(pnl)

    def update_unrealized_pnl(self, symbol: str, pnl: float):
        """Update unrealized P&L."""
        self.pnl_unrealized.labels(symbol=symbol).set(pnl)

    # Risk methods
    def update_risk_metrics(self, risk_score: float, drawdown: float, exposure: float):
        """Update risk metrics."""
        self.risk_score.set(risk_score)
        self.drawdown.set(drawdown)
        self.exposure.set(exposure)

    def record_risk_violation(self, violation_type: str):
        """Record a risk violation."""
        self.risk_violations.labels(violation_type=violation_type).inc()

    # System health methods
    def update_system_health(self, memory_bytes: float, cpu_percent: float):
        """Update system health metrics."""
        uptime = time.time() - self.start_time
        self.system_uptime.set(uptime)
        self.memory_usage.set(memory_bytes)
        self.cpu_usage.set(cpu_percent)

    def set_component_health(self, component: str, healthy: bool):
        """Set component health status."""
        self.component_health.labels(component=component).set(1 if healthy else 0)

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode("utf-8")

    def register_histogram(
        self, name: str, documentation: str, buckets: tuple[float, ...]
    ):
        """Register and return a custom histogram metric.

        Prevents re-registering the same metric multiple times by checking
        the collector registry first. Useful for ad-hoc layer-specific
        metrics like exchange-specific latency histograms.
        """
        try:
            return Histogram(  # type: ignore[call-arg]
                name,
                documentation,
                buckets=buckets,
                registry=self.registry,
            )
        except ValueError:
            # Metric already registered – return the existing one
            return self.registry._names_to_collectors[name]  # type: ignore[attr-defined]


# Global metrics instance
_metrics_instance: Optional[TradingMetrics] = None
_metrics_lock = threading.Lock()


def get_metrics() -> TradingMetrics:
    """Get global metrics instance (singleton)."""
    global _metrics_instance

    with _metrics_lock:
        if _metrics_instance is None:
            _metrics_instance = TradingMetrics()
        return _metrics_instance


def start_metrics_server(port: int = 8000) -> None:
    """Start Prometheus metrics HTTP server."""
    try:
        start_http_server(port, registry=get_metrics().registry)
        logger = get_logger("metrics_server")
        logger.info(f"Metrics server started on port {port}")
    except Exception as e:
        logger = get_logger("metrics_server")
        logger.error(f"Failed to start metrics server: {e}")


# Convenience decorators
def time_operation(metric_name: str):
    """Decorator to time an operation and record to metrics."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start
                # This would need to be customized based on the specific metric
                # For now, just log
                logger = get_logger("metrics_timing")
                logger.debug(f"{metric_name}: {latency:.4f}s")
                return result
            except Exception as e:
                latency = time.time() - start
                logger = get_logger("metrics_timing")
                logger.error(f"{metric_name} failed after {latency:.4f}s: {e}")
                raise

        return wrapper

    return decorator
