#!/usr/bin/env python3
"""
RL Redis Prometheus Exporter
Exports RL policy metrics from Redis for Prometheus scraping
"""
import os
import time
import redis
from wsgiref.simple_server import make_server
from datetime import datetime, timezone


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PORT = int(os.getenv("EXPORTER_PORT", "9108"))


def safe_float(value, default=0.0):
    """Convert value to float, return default if conversion fails."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def app(environ, start_response):
    """WSGI application serving Prometheus metrics."""
    try:
        r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        now = time.time()

        # Fetch metrics from Redis
        last_update = r.get("policy:last_update_ts")
        influence_pct = r.get("policy:allowed_influence_pct")

        # Try both policy:stats and policy:current for metrics
        entropy = r.hget("policy:current", "entropy") or r.hget(
            "policy:stats", "entropy"
        )
        qspread = r.hget("policy:current", "qspread") or r.hget(
            "policy:stats", "q_spread"
        )
        collapse_risk = r.hget("policy:current", "collapse_risk") or "UNKNOWN"

        # Convert to safe floats
        last_update_ts = safe_float(last_update, 0)
        entropy_val = safe_float(entropy, float("nan"))
        qspread_val = safe_float(qspread, float("nan"))
        influence_val = safe_float(influence_pct, 0)

        # Calculate heartbeat age
        if last_update_ts > 0:
            heartbeat_age = now - last_update_ts
        else:
            heartbeat_age = float("inf")

        # Build Prometheus metrics
        lines = []

        # Last update timestamp
        lines += [
            "# HELP rl_policy_last_update_seconds Unix timestamp of last policy heartbeat",
            "# TYPE rl_policy_last_update_seconds gauge",
        ]
        lines += [f"rl_policy_last_update_seconds {last_update_ts}"]

        # Policy entropy
        lines += [
            "# HELP rl_policy_entropy Current policy entropy (higher is better exploration)",
            "# TYPE rl_policy_entropy gauge",
        ]
        lines += [f"rl_policy_entropy {entropy_val}"]

        # Q-value spread
        lines += [
            "# HELP rl_policy_q_spread Current policy Q-value spread",
            "# TYPE rl_policy_q_spread gauge",
        ]
        lines += [f"rl_policy_q_spread {qspread_val}"]

        # Heartbeat age
        lines += [
            "# HELP rl_policy_heartbeat_age_seconds Age of last policy heartbeat in seconds",
            "# TYPE rl_policy_heartbeat_age_seconds gauge",
        ]
        lines += [f"rl_policy_heartbeat_age_seconds {heartbeat_age}"]

        # Collapse risk as numeric (0=LOW, 1=MEDIUM, 2=HIGH)
        risk_numeric = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(collapse_risk, 3)
        lines += [
            "# HELP rl_policy_collapse_risk Policy collapse risk level (0=LOW, 1=MEDIUM, 2=HIGH, 3=UNKNOWN)",
            "# TYPE rl_policy_collapse_risk gauge",
        ]
        lines += [f"rl_policy_collapse_risk {risk_numeric}"]

        # Exporter health
        lines += [
            "# HELP rl_exporter_up Exporter health status",
            "# TYPE rl_exporter_up gauge",
        ]
        lines += ["rl_exporter_up 1"]

        # Policy influence percentage
        lines += [
            "# HELP rl_policy_influence_pct Current policy influence percentage (0-100)",
            "# TYPE rl_policy_influence_pct gauge",
        ]
        lines += [f"rl_policy_influence_pct {influence_val}"]

        # Influence weight (0-1 for calculations)
        lines += [
            "# HELP rl_policy_influence_weight Current policy influence weight (0.0-1.0)",
            "# TYPE rl_policy_influence_weight gauge",
        ]
        lines += [f"rl_policy_influence_weight {influence_val / 100.0}"]

        # Influence TTL remaining
        try:
            influence_ttl = r.ttl("policy:allowed_influence_pct")
            if influence_ttl == -2:  # Key doesn't exist
                influence_ttl = 0
            elif influence_ttl == -1:  # No expiration
                influence_ttl = -1
        except:
            influence_ttl = 0

        lines += [
            "# HELP rl_policy_influence_ttl_seconds TTL remaining for influence key (seconds)",
            "# TYPE rl_policy_influence_ttl_seconds gauge",
        ]
        lines += [f"rl_policy_influence_ttl_seconds {influence_ttl}"]

        # Exporter scrape timestamp
        lines += [
            "# HELP rl_exporter_last_scrape_timestamp Unix timestamp of last scrape",
            "# TYPE rl_exporter_last_scrape_timestamp gauge",
        ]
        lines += [f"rl_exporter_last_scrape_timestamp {now}"]

        body = ("\n".join(lines) + "\n").encode("utf-8")
        start_response("200 OK", [("Content-Type", "text/plain; version=0.0.4")])
        return [body]

    except Exception as e:
        # Return error metrics on failure
        error_lines = [
            "# HELP rl_exporter_up Exporter health status",
            "# TYPE rl_exporter_up gauge",
            "rl_exporter_up 0",
            f"# Redis connection error: {str(e)}",
        ]
        body = ("\n".join(error_lines) + "\n").encode("utf-8")
        start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
        return [body]


if __name__ == "__main__":
    try:
        httpd = make_server("", PORT, app)
        print(f"🚀 RL Redis Prometheus Exporter listening on port :{PORT}")
        print(f"📊 Metrics endpoint: http://localhost:{PORT}/metrics")
        print(f"🔗 Redis: {REDIS_URL}")
        print(f"📡 Ready for Prometheus scraping...")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n🛑 Exporter shutting down...")
    except Exception as e:
        print(f"❌ Failed to start exporter: {e}")
        exit(1)
