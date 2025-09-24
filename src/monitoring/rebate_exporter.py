#!/usr/bin/env python3
"""
Rebate Exporter
Export Prometheus metrics for maker/taker performance and rebate tracking.
"""
import os
import sys
import time
import json
import datetime
import threading
from pathlib import Path
from typing import Dict, List, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import numpy as np


class RebateMetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests for metrics."""
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()

            metrics = self.server.exporter.generate_metrics()
            self.wfile.write(metrics.encode())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class RebateExporter:
    def __init__(self, port: int = 9111):
        self.port = port
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")
        self.assets = ["SOL-USD", "BTC-USD", "ETH-USD", "NVDA"]
        self.venues = ["coinbase", "binance", "alpaca"]

        # Metrics state
        self.metrics_state = {
            "rebate_bps_realized": {},  # {asset_venue: value}
            "maker_fill_ratio": {},  # {asset_venue: ratio}
            "post_only_blocks": {},  # {asset_venue: count}
            "net_fee_bps": {},  # {asset_venue: net_fee}
            "total_fills": {},  # {asset_venue: count}
            "maker_fills": {},  # {asset_venue: count}
            "taker_fills": {},  # {asset_venue: count}
        }

        self.lock = threading.Lock()
        self.last_update = datetime.datetime.now()

    def load_fill_statistics(self) -> Dict[str, Any]:
        """Load fill statistics from audit files."""
        audit_dir = self.base_dir / "artifacts" / "audit"

        stats = {
            "fills_today": 0,
            "maker_fills_today": 0,
            "total_rebate_usd": 0.0,
            "avg_fill_size": 0.0,
            "fills_by_asset_venue": {},
        }

        if not audit_dir.exists():
            return stats

        # Load today's maker/taker fill logs
        today = datetime.datetime.now().strftime("%Y%m%d")
        fill_file = audit_dir / f"maker_taker_fills_{today}.jsonl"

        if fill_file.exists():
            try:
                with open(fill_file, "r") as f:
                    for line in f:
                        fill_data = json.loads(line.strip())
                        asset = fill_data.get("asset", "")
                        venue = fill_data.get("venue", "")
                        was_maker = fill_data.get("was_maker", False)
                        rebate_bps = fill_data.get("rebate_bps", 0.0)

                        key = f"{asset}_{venue}"

                        if key not in stats["fills_by_asset_venue"]:
                            stats["fills_by_asset_venue"][key] = {
                                "total_fills": 0,
                                "maker_fills": 0,
                                "total_rebate_bps": 0.0,
                            }

                        stats["fills_by_asset_venue"][key]["total_fills"] += 1
                        if was_maker:
                            stats["fills_by_asset_venue"][key]["maker_fills"] += 1
                        stats["fills_by_asset_venue"][key][
                            "total_rebate_bps"
                        ] += rebate_bps

                        stats["fills_today"] += 1
                        if was_maker:
                            stats["maker_fills_today"] += 1
                        stats["total_rebate_usd"] += (
                            rebate_bps / 10000 * 1000
                        )  # Estimate

            except Exception as e:
                print(f"⚠️ Error loading fill statistics: {e}")

        return stats

    def simulate_realtime_data(self) -> Dict[str, Any]:
        """Generate simulated realtime rebate data."""

        # Simulate some fills with varying maker/taker ratios
        simulated_data = {}

        for asset in self.assets:
            for venue in self.venues:
                # Skip invalid combinations
                if venue == "binance" and asset == "NVDA":
                    continue
                if venue == "alpaca" and asset != "NVDA":
                    continue

                key = f"{asset}_{venue}"

                # Simulate varying maker ratios based on conditions
                base_maker_ratio = np.random.uniform(0.4, 0.8)

                # Crypto generally has better maker ratios
                if asset != "NVDA":
                    base_maker_ratio += 0.1

                # Binance typically has good maker ratios due to rebates
                if venue == "binance":
                    base_maker_ratio += 0.15

                maker_ratio = min(0.95, base_maker_ratio)

                # Simulate realized rebates (negative = earning rebates)
                if asset != "NVDA" and venue in ["coinbase", "binance"]:
                    realized_rebate = np.random.uniform(-1.5, -0.3)  # Earning rebates
                elif asset == "NVDA" and venue == "alpaca":
                    realized_rebate = np.random.uniform(-0.3, 0.1)  # Small rebates
                else:
                    realized_rebate = np.random.uniform(0.0, 2.0)  # Paying fees

                # Post-only blocks (attempts to post)
                post_only_blocks = int(np.random.poisson(50))  # ~50 attempts per period

                # Net fees (including rebates and taker fees)
                net_fee = realized_rebate * maker_ratio + np.random.uniform(
                    1.0, 4.0
                ) * (1 - maker_ratio)

                simulated_data[key] = {
                    "maker_fill_ratio": maker_ratio,
                    "rebate_bps_realized": realized_rebate,
                    "post_only_blocks": post_only_blocks,
                    "net_fee_bps": net_fee,
                    "total_fills": int(np.random.poisson(100)),
                    "maker_fills": int(np.random.poisson(100) * maker_ratio),
                    "taker_fills": int(np.random.poisson(100) * (1 - maker_ratio)),
                }

        return simulated_data

    def update_metrics(self):
        """Update metrics from latest data."""
        with self.lock:
            # In production, this would query Redis or database
            # For demo, we'll use simulated data
            simulated_data = self.simulate_realtime_data()

            for key, data in simulated_data.items():
                self.metrics_state["maker_fill_ratio"][key] = data["maker_fill_ratio"]
                self.metrics_state["rebate_bps_realized"][key] = data[
                    "rebate_bps_realized"
                ]
                self.metrics_state["post_only_blocks"][key] = data["post_only_blocks"]
                self.metrics_state["net_fee_bps"][key] = data["net_fee_bps"]
                self.metrics_state["total_fills"][key] = data["total_fills"]
                self.metrics_state["maker_fills"][key] = data["maker_fills"]
                self.metrics_state["taker_fills"][key] = data["taker_fills"]

            self.last_update = datetime.datetime.now()

    def generate_metrics(self) -> str:
        """Generate Prometheus metrics format."""

        # Update metrics if stale
        if (datetime.datetime.now() - self.last_update).seconds > 30:
            self.update_metrics()

        metrics = []

        # Help text
        metrics.append(
            "# HELP rebate_bps_realized Realized rebate in basis points (negative = earning)"
        )
        metrics.append("# TYPE rebate_bps_realized gauge")

        with self.lock:
            # Rebate BPS realized
            for key, value in self.metrics_state["rebate_bps_realized"].items():
                if "_" in key:
                    asset, venue = key.split("_", 1)
                    metrics.append(
                        f'rebate_bps_realized{{asset="{asset}",venue="{venue}"}} {value:.2f}'
                    )

            # Maker fill ratio
            metrics.append(
                "# HELP maker_fill_ratio Ratio of maker fills to total fills"
            )
            metrics.append("# TYPE maker_fill_ratio gauge")
            for key, value in self.metrics_state["maker_fill_ratio"].items():
                if "_" in key:
                    asset, venue = key.split("_", 1)
                    metrics.append(
                        f'maker_fill_ratio{{asset="{asset}",venue="{venue}"}} {value:.3f}'
                    )

            # Post-only blocks
            metrics.append("# HELP post_only_blocks Number of post-only order attempts")
            metrics.append("# TYPE post_only_blocks counter")
            for key, value in self.metrics_state["post_only_blocks"].items():
                if "_" in key:
                    asset, venue = key.split("_", 1)
                    metrics.append(
                        f'post_only_blocks{{asset="{asset}",venue="{venue}"}} {value}'
                    )

            # Net fee BPS
            metrics.append(
                "# HELP net_fee_bps Net fees in basis points (positive = cost)"
            )
            metrics.append("# TYPE net_fee_bps gauge")
            for key, value in self.metrics_state["net_fee_bps"].items():
                if "_" in key:
                    asset, venue = key.split("_", 1)
                    metrics.append(
                        f'net_fee_bps{{asset="{asset}",venue="{venue}"}} {value:.2f}'
                    )

            # Total fills
            metrics.append("# HELP total_fills_count Total number of fills")
            metrics.append("# TYPE total_fills_count counter")
            for key, value in self.metrics_state["total_fills"].items():
                if "_" in key:
                    asset, venue = key.split("_", 1)
                    metrics.append(
                        f'total_fills_count{{asset="{asset}",venue="{venue}"}} {value}'
                    )

            # Maker fills
            metrics.append("# HELP maker_fills_count Number of maker fills")
            metrics.append("# TYPE maker_fills_count counter")
            for key, value in self.metrics_state["maker_fills"].items():
                if "_" in key:
                    asset, venue = key.split("_", 1)
                    metrics.append(
                        f'maker_fills_count{{asset="{asset}",venue="{venue}"}} {value}'
                    )

        # Add timestamp
        metrics.append(f"# Last updated: {self.last_update.isoformat()}")

        return "\n".join(metrics) + "\n"

    def run(self):
        """Run the rebate exporter server."""
        print(f"🚀 Starting rebate exporter on port {self.port}")
        print(f"📊 Metrics endpoint: http://localhost:{self.port}/metrics")

        # Initial metrics update
        self.update_metrics()

        # Start HTTP server
        server = HTTPServer(("localhost", self.port), RebateMetricsHandler)
        server.exporter = self

        try:
            print(f"✅ Rebate exporter running on http://localhost:{self.port}")
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down rebate exporter...")
            server.shutdown()


def main():
    """Main rebate exporter function."""
    import argparse

    parser = argparse.ArgumentParser(description="Rebate Performance Exporter")
    parser.add_argument("--port", type=int, default=9111, help="HTTP server port")
    parser.add_argument(
        "--test", action="store_true", help="Test mode - print metrics and exit"
    )
    args = parser.parse_args()

    exporter = RebateExporter(port=args.port)

    if args.test:
        print("📊 Testing rebate exporter...")
        exporter.update_metrics()
        metrics = exporter.generate_metrics()
        print(metrics)
        print("✅ Test complete")
        return 0

    try:
        exporter.run()
        return 0
    except Exception as e:
        print(f"❌ Rebate exporter failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
