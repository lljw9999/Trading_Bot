"""Stub for `prometheus_client` when package is unavailable.

Defines minimal classes/functions used in the trading system for counters,
histograms, gauges, summaries, etc. All operations are no-ops so metrics are
silently ignored during offline testing.
"""

from __future__ import annotations

from typing import Any


class _Metric:
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def labels(self, *args: Any, **kwargs: Any):
        return self

    def inc(self, amount: float = 1.0):
        pass

    def dec(self, amount: float = 1.0):
        pass

    def observe(self, value: float):
        pass


Counter = Histogram = Gauge = Summary = _Metric  # type: ignore


class CollectorRegistry:  # Dummy placeholder
    def __init__(self, *args: Any, **kwargs: Any):
        pass


def generate_latest(*args: Any, **kwargs: Any) -> bytes:  # noqa: D401
    return b""


def start_http_server(*args: Any, **kwargs: Any):  # noqa: D401
    # Simply ignore – no real server in stub mode
    pass 