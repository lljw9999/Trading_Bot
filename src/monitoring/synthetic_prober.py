#!/usr/bin/env python3
"""
Synthetic Prober - Continuous Health Monitoring
Probes critical system components and exports metrics for Prometheus
"""
import os
import sys
import time
import redis
import requests
import threading
import subprocess
from datetime import datetime
from wsgiref.simple_server import make_server


class SyntheticProber:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.exporter_url = os.getenv(
            "RL_EXPORTER_URL", "http://localhost:9108/metrics"
        )
        self.state = {
            "redis_ok": 1,
            "redis_latency": 0.01,
            "exporter_ok": 1,
            "exporter_latency": 0.02,
            "eval_smoke_ok": 1,
            "eval_latency": 0.5,
            "last_probe_ts": time.time(),
            "probe_count": 0,
        }
        self.running = True

    def probe_redis(self):
        """Probe Redis connectivity and key existence."""
        start_time = time.time()
        try:
            r = redis.Redis.from_url(self.redis_url, decode_responses=True)

            # Test basic connectivity
            r.ping()

            # Check critical keys
            policy_update = r.get("policy:last_update_ts")
            policy_stats = r.hgetall("policy:stats") or r.hgetall("policy:current")

            # Verify key freshness
            if policy_update:
                age = time.time() - float(policy_update)
                if age > 1800:  # 30 minutes
                    raise Exception(f"Policy heartbeat stale: {age}s")

            latency = time.time() - start_time
            self.state.update({"redis_ok": 1, "redis_latency": latency})
            return True

        except Exception as e:
            latency = time.time() - start_time
            self.state.update({"redis_ok": 0, "redis_latency": latency})
            print(f"❌ Redis probe failed: {e}")
            return False

    def probe_exporter(self):
        """Probe RL Redis exporter availability."""
        start_time = time.time()
        try:
            response = requests.get(self.exporter_url, timeout=10)
            latency = time.time() - start_time

            if response.status_code == 200:
                # Verify essential metrics are present
                content = response.text
                required_metrics = [
                    "rl_policy_heartbeat_age_seconds",
                    "rl_policy_influence_pct",
                    "rl_exporter_up",
                ]

                for metric in required_metrics:
                    if metric not in content:
                        raise Exception(f"Missing metric: {metric}")

                self.state.update({"exporter_ok": 1, "exporter_latency": latency})
                return True
            else:
                raise Exception(f"HTTP {response.status_code}")

        except Exception as e:
            latency = time.time() - start_time
            self.state.update({"exporter_ok": 0, "exporter_latency": latency})
            print(f"❌ Exporter probe failed: {e}")
            return False

    def probe_eval_smoke(self):
        """Probe RL environment with quick smoke test."""
        start_time = time.time()
        try:
            # Quick dry-run test of evaluation environment
            # This is a lightweight check, not a full evaluation
            result = subprocess.run(
                [
                    "python",
                    "-c",
                    """
import sys
sys.path.insert(0, 'src')
try:
    from rl.influence_controller import InfluenceController
    ic = InfluenceController()
    status = ic.get_status()
    if 'weight' not in status:
        raise Exception('Controller status missing weight')
    print('SMOKE_OK')
except Exception as e:
    print(f'SMOKE_FAIL: {e}')
    sys.exit(1)
                """,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=".",
            )

            latency = time.time() - start_time

            if result.returncode == 0 and "SMOKE_OK" in result.stdout:
                self.state.update({"eval_smoke_ok": 1, "eval_latency": latency})
                return True
            else:
                raise Exception(f"Smoke test failed: {result.stderr}")

        except Exception as e:
            latency = time.time() - start_time
            self.state.update({"eval_smoke_ok": 0, "eval_latency": latency})
            print(f"❌ Eval smoke probe failed: {e}")
            return False

    def run_probes(self):
        """Run all probes and update state."""
        try:
            redis_ok = self.probe_redis()
            exporter_ok = self.probe_exporter()
            eval_ok = self.probe_eval_smoke()

            self.state.update(
                {
                    "last_probe_ts": time.time(),
                    "probe_count": self.state["probe_count"] + 1,
                }
            )

            # Overall probe success
            overall_success = redis_ok and exporter_ok and eval_ok
            self.state["probe_success"] = 1 if overall_success else 0

            status = "✅" if overall_success else "❌"
            print(
                f"{status} Probe #{self.state['probe_count']}: Redis={redis_ok}, Exporter={exporter_ok}, Eval={eval_ok}"
            )

        except Exception as e:
            print(f"❌ Probe execution failed: {e}")
            self.state["probe_success"] = 0

    def probe_loop(self):
        """Continuous probing loop."""
        while self.running:
            try:
                self.run_probes()
                time.sleep(60)  # Probe every minute
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Probe loop error: {e}")
                time.sleep(60)

    def metrics_app(self, environ, start_response):
        """WSGI app serving Prometheus metrics."""
        try:
            state = self.state
            now = time.time()

            metrics = [
                "# HELP probe_success Overall probe success (1=success, 0=failure)",
                "# TYPE probe_success gauge",
                f"probe_success {state.get('probe_success', 0)}",
                "",
                "# HELP probe_redis_ok Redis probe success",
                "# TYPE probe_redis_ok gauge",
                f"probe_redis_ok {state.get('redis_ok', 0)}",
                "",
                "# HELP probe_redis_latency_seconds Redis probe latency",
                "# TYPE probe_redis_latency_seconds gauge",
                f"probe_redis_latency_seconds {state.get('redis_latency', 0)}",
                "",
                "# HELP probe_exporter_ok Exporter probe success",
                "# TYPE probe_exporter_ok gauge",
                f"probe_exporter_ok {state.get('exporter_ok', 0)}",
                "",
                "# HELP probe_exporter_latency_seconds Exporter probe latency",
                "# TYPE probe_exporter_latency_seconds gauge",
                f"probe_exporter_latency_seconds {state.get('exporter_latency', 0)}",
                "",
                "# HELP probe_eval_smoke_ok Evaluation smoke test success",
                "# TYPE probe_eval_smoke_ok gauge",
                f"probe_eval_smoke_ok {state.get('eval_smoke_ok', 0)}",
                "",
                "# HELP probe_eval_latency_seconds Evaluation smoke test latency",
                "# TYPE probe_eval_latency_seconds gauge",
                f"probe_eval_latency_seconds {state.get('eval_latency', 0)}",
                "",
                "# HELP probe_last_run_timestamp Unix timestamp of last probe run",
                "# TYPE probe_last_run_timestamp gauge",
                f"probe_last_run_timestamp {state.get('last_probe_ts', 0)}",
                "",
                "# HELP probe_count_total Total number of probes executed",
                "# TYPE probe_count_total counter",
                f"probe_count_total {state.get('probe_count', 0)}",
                "",
                "# HELP prober_up Prober service health",
                "# TYPE prober_up gauge",
                "prober_up 1",
                "",
                "# HELP prober_scrape_timestamp Current scrape timestamp",
                "# TYPE prober_scrape_timestamp gauge",
                f"prober_scrape_timestamp {now}",
            ]

            body = ("\n".join(metrics) + "\n").encode("utf-8")
            start_response("200 OK", [("Content-Type", "text/plain; version=0.0.4")])
            return [body]

        except Exception as e:
            error_body = f"# Prober metrics error: {e}\nprober_up 0\n".encode("utf-8")
            start_response(
                "500 Internal Server Error", [("Content-Type", "text/plain")]
            )
            return [error_body]

    def start(self, port=9110):
        """Start the synthetic prober service."""
        print(f"🚀 Starting Synthetic Prober on port {port}")
        print(f"📊 Metrics endpoint: http://localhost:{port}/metrics")
        print(f"🔗 Redis: {self.redis_url}")
        print(f"📡 Exporter: {self.exporter_url}")

        # Start probe loop in background thread
        probe_thread = threading.Thread(target=self.probe_loop, daemon=True)
        probe_thread.start()

        # Start metrics server
        try:
            httpd = make_server("", port, self.metrics_app)
            print(f"🎯 Prober ready - probing every 60 seconds")
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n🛑 Prober shutting down...")
            self.running = False
        except Exception as e:
            print(f"❌ Failed to start prober: {e}")
            sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic Prober")
    parser.add_argument("--port", type=int, default=9110, help="Metrics port")
    parser.add_argument("--once", action="store_true", help="Run probes once and exit")
    args = parser.parse_args()

    prober = SyntheticProber()

    if args.once:
        print("🧪 Running probes once...")
        prober.run_probes()
        print(f"📊 Results: {prober.state}")
    else:
        prober.start(port=args.port)


if __name__ == "__main__":
    main()
