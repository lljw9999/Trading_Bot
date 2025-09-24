#!/usr/bin/env python3
"""
FastAPI Metrics Server for Trading System

Exposes Prometheus metrics on /metrics endpoint for monitoring
trading system performance, signals, and execution.
"""

import argparse
import asyncio
import logging
import time
from typing import Dict, Any
from datetime import datetime, timezone

from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
import uvicorn
from fastapi import Request


# Prometheus metrics storage
class MetricsCollector:
    """Simple in-memory metrics collector for Prometheus format."""

    def __init__(self):
        self.metrics = {}
        self.counters = {}
        self.gauges = {}
        self.histograms = {}
        self.start_time = time.time()

    def counter(self, name: str, help_text: str = "", labels: Dict[str, str] = None):
        """Increment a counter metric."""
        labels = labels or {}
        key = f"{name}_{self._labels_to_string(labels)}"

        if name not in self.counters:
            self.counters[name] = {"help": help_text, "type": "counter", "values": {}}

        if key not in self.counters[name]["values"]:
            self.counters[name]["values"][key] = 0

        self.counters[name]["values"][key] += 1

    def gauge(
        self,
        name: str,
        value: float,
        help_text: str = "",
        labels: Dict[str, str] = None,
    ):
        """Set a gauge metric value."""
        labels = labels or {}
        key = f"{name}_{self._labels_to_string(labels)}"

        if name not in self.gauges:
            self.gauges[name] = {"help": help_text, "type": "gauge", "values": {}}

        self.gauges[name]["values"][key] = value

    def histogram_observe(
        self,
        name: str,
        value: float,
        help_text: str = "",
        labels: Dict[str, str] = None,
    ):
        """Add observation to histogram."""
        labels = labels or {}
        key = f"{name}_{self._labels_to_string(labels)}"

        if name not in self.histograms:
            self.histograms[name] = {
                "help": help_text,
                "type": "histogram",
                "observations": {},
            }

        if key not in self.histograms[name]["observations"]:
            self.histograms[name]["observations"][key] = []

        self.histograms[name]["observations"][key].append(value)

    def _labels_to_string(self, labels: Dict[str, str]) -> str:
        """Convert labels dict to string representation."""
        if not labels:
            return ""

        items = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(items) + "}"

    def generate_prometheus_format(self) -> str:
        """Generate Prometheus format metrics."""
        lines = []

        # Add standard metrics
        lines.append(
            "# HELP trading_system_uptime_seconds Total uptime of the trading system"
        )
        lines.append("# TYPE trading_system_uptime_seconds gauge")
        lines.append(
            f"trading_system_uptime_seconds {time.time() - self.start_time:.2f}"
        )
        lines.append("")

        # Add counters
        for name, data in self.counters.items():
            if data["help"]:
                lines.append(f"# HELP {name} {data['help']}")
            lines.append(f"# TYPE {name} {data['type']}")

            for key, value in data["values"].items():
                metric_name = key.split("_", 1)[0] if "_" in key else name
                labels_part = key[len(name) + 1 :] if len(key) > len(name) else ""
                lines.append(f"{name}{labels_part} {value}")
            lines.append("")

        # Add gauges
        for name, data in self.gauges.items():
            if data["help"]:
                lines.append(f"# HELP {name} {data['help']}")
            lines.append(f"# TYPE {name} {data['type']}")

            for key, value in data["values"].items():
                metric_name = key.split("_", 1)[0] if "_" in key else name
                labels_part = key[len(name) + 1 :] if len(key) > len(name) else ""
                lines.append(f"{name}{labels_part} {value}")
            lines.append("")

        # Add histograms (simplified)
        for name, data in self.histograms.items():
            if data["help"]:
                lines.append(f"# HELP {name} {data['help']}")
            lines.append(f"# TYPE {name} histogram")

            for key, observations in data["observations"].items():
                if observations:
                    count = len(observations)
                    total = sum(observations)
                    labels_part = key[len(name) + 1 :] if len(key) > len(name) else ""

                    lines.append(f"{name}_count{labels_part} {count}")
                    lines.append(f"{name}_sum{labels_part} {total:.4f}")
            lines.append("")

        return "\n".join(lines)


# Global metrics collector
metrics_collector = MetricsCollector()

# FastAPI app
app = FastAPI(title="Trading System Metrics", version="1.0.0")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "trading-system-metrics",
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": time.time() - metrics_collector.start_time,
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Prometheus metrics endpoint."""

    # Update some basic metrics
    metrics_collector.gauge(
        "trading_system_timestamp", time.time(), "Current timestamp"
    )

    # Add some sample crypto metrics
    import random

    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        # Simulate tick counts
        metrics_collector.counter(
            "crypto_ticks_total",
            "Total number of crypto ticks processed",
            {"symbol": symbol, "source": "demo"},
        )

        # Simulate price data
        base_prices = {"BTCUSDT": 45000, "ETHUSDT": 2500, "SOLUSDT": 100}
        price = base_prices[symbol] * (1 + random.uniform(-0.01, 0.01))
        metrics_collector.gauge(
            "crypto_price_usd", price, "Current crypto price in USD", {"symbol": symbol}
        )

        # Simulate alpha signals
        edge_bps = random.uniform(-20, 20)
        confidence = random.uniform(0.5, 1.0)

        metrics_collector.gauge(
            "alpha_signal_edge_bps",
            edge_bps,
            "Alpha signal edge in basis points",
            {"symbol": symbol, "model": "ensemble"},
        )

        metrics_collector.gauge(
            "alpha_signal_confidence",
            confidence,
            "Alpha signal confidence",
            {"symbol": symbol, "model": "ensemble"},
        )

    return metrics_collector.generate_prometheus_format()


@app.post("/metrics/counter/{name}")
async def increment_counter(name: str, request: Request):
    """API endpoint to increment counters."""
    try:
        body = await request.json()
        labels = body.get("labels", {})
    except:
        labels = {}

    metrics_collector.counter(name, labels=labels)
    return {"status": "ok", "metric": name}


@app.post("/metrics/gauge/{name}")
async def set_gauge(name: str, request: Request):
    """API endpoint to set gauge values."""
    try:
        body = await request.json()
        value = body.get("value")
        labels = body.get("labels", {})
    except:
        return {"status": "error", "message": "Invalid JSON body"}

    if value is not None:
        metrics_collector.gauge(name, value, labels=labels)
        return {"status": "ok", "metric": name, "value": value}
    else:
        return {"status": "error", "message": "No value provided"}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Trading System Metrics Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting Trading System Metrics Server on {args.host}:{args.port}")

    # Run the server
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
