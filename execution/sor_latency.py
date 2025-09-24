"""
Smart Order Router (SOR) Latency Mapping
Pings multiple venues every 250ms to build real-time latency heat-map
"""

import asyncio
import aiohttp
import redis
import time
import statistics
from typing import Dict, List, Optional
import logging
from prometheus_client import Gauge


# Prometheus metrics for venue latency
venue_latency_gauge = Gauge(
    "venue_latency_ms", "Venue latency in milliseconds", ["venue"]
)


class VenueLatencyTracker:
    """
    Tracks latency to multiple crypto venues for smart order routing.
    Measures RTT every 250ms and maintains rolling averages.
    """

    VENUES = {
        "binance": "https://api.binance.com/api/v3/time",
        "coinbase": "https://api.exchange.coinbase.com/time",
        "kraken": "https://api.kraken.com/0/public/Time",
        "bybit": "https://api.bybit.com/v5/market/time",
        "okx": "https://www.okx.com/api/v5/public/time",
    }

    def __init__(self, measurement_interval: float = 0.25):  # 250ms as specified
        self.measurement_interval = measurement_interval
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.logger = logging.getLogger("sor_latency")

        # Rolling latency windows (last 5 seconds as specified)
        self.latency_windows = {venue: [] for venue in self.VENUES.keys()}
        self.window_size = int(
            5.0 / measurement_interval
        )  # ~20 measurements for 5s window

        self.logger.info(
            f"🎯 SOR Latency Tracker initialized - {len(self.VENUES)} venues, {measurement_interval*1000}ms interval"
        )

    async def measure_venue_latency(
        self, session: aiohttp.ClientSession, venue: str, url: str
    ) -> Optional[float]:
        """Measure RTT to a single venue."""
        try:
            start_time = time.perf_counter()

            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=2.0)
            ) as response:
                await response.read()  # Ensure full response is received

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            return latency_ms

        except Exception as e:
            self.logger.warning(f"Failed to measure {venue} latency: {e}")
            return None

    async def measure_all_venues(
        self, session: aiohttp.ClientSession
    ) -> Dict[str, Optional[float]]:
        """Measure latency to all venues concurrently."""
        tasks = []

        for venue, url in self.VENUES.items():
            task = self.measure_venue_latency(session, venue, url)
            tasks.append((venue, task))

        results = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in tasks], return_exceptions=True
        )

        for (venue, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                results[venue] = None
            else:
                results[venue] = result

        return results

    def update_latency_windows(self, measurements: Dict[str, Optional[float]]):
        """Update rolling latency windows and calculate medians."""
        current_time = int(time.time())

        for venue, latency_ms in measurements.items():
            if latency_ms is not None:
                # Add to rolling window
                window = self.latency_windows[venue]
                window.append(latency_ms)

                # Keep only last N measurements (5 second window)
                if len(window) > self.window_size:
                    window.pop(0)

                # Calculate median RTT for last 5s as specified in task brief
                if len(window) >= 3:  # Need at least 3 points for stable median
                    median_latency = statistics.median(window)

                    # Store in Redis hash latency:venue as specified
                    self.redis.hset("latency:venue", venue, f"{median_latency:.2f}")

                    # Update Prometheus metrics
                    venue_latency_gauge.labels(venue=venue).set(median_latency)

                    self.logger.debug(
                        f"{venue}: {median_latency:.1f}ms (window: {len(window)})"
                    )
                else:
                    # Not enough data points yet
                    self.redis.hset("latency:venue", venue, f"{latency_ms:.2f}")
                    venue_latency_gauge.labels(venue=venue).set(latency_ms)

    def get_venue_weights(self) -> Dict[str, float]:
        """Calculate venue weights based on latency for router."""
        try:
            latencies = self.redis.hgetall("latency:venue")
            weights = {}

            for venue, latency_str in latencies.items():
                latency_ms = float(latency_str)

                # Weight = 1 / (latency_ms + tiny) as specified in task brief
                tiny = 1.0  # Prevent division by zero
                weight = 1.0 / (latency_ms + tiny)
                weights[venue] = weight

            # Normalize weights to sum to 1.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {venue: w / total_weight for venue, w in weights.items()}

            return weights

        except Exception as e:
            self.logger.error(f"Error calculating venue weights: {e}")
            # Default equal weights
            return {venue: 1.0 / len(self.VENUES) for venue in self.VENUES.keys()}

    def get_best_venue(
        self, liquidity_scores: Optional[Dict[str, float]] = None
    ) -> str:
        """Get best venue based on latency and liquidity."""
        try:
            weights = self.get_venue_weights()

            if liquidity_scores:
                # Choose venue with max(weight × liquidity_score) as specified
                scores = {}
                for venue in self.VENUES.keys():
                    latency_weight = weights.get(venue, 0)
                    liquidity = liquidity_scores.get(venue, 1.0)
                    scores[venue] = latency_weight * liquidity

                best_venue = max(scores, key=scores.get)
                self.logger.debug(
                    f"Best venue by composite score: {best_venue} ({scores[best_venue]:.3f})"
                )

            else:
                # Choose venue with lowest latency (highest weight)
                best_venue = max(weights, key=weights.get)
                self.logger.debug(
                    f"Best venue by latency: {best_venue} ({weights[best_venue]:.3f})"
                )

            return best_venue

        except Exception as e:
            self.logger.error(f"Error selecting best venue: {e}")
            return "binance"  # Default fallback

    async def run_latency_monitoring(self):
        """Main monitoring loop."""
        self.logger.info("🚀 Starting venue latency monitoring")

        connector = aiohttp.TCPConnector(limit=20, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=2.0)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            measurement_count = 0

            while True:
                try:
                    # Measure all venues
                    measurements = await self.measure_all_venues(session)

                    # Update rolling windows and Redis
                    self.update_latency_windows(measurements)

                    measurement_count += 1

                    # Log summary every 20 measurements (~5 seconds)
                    if measurement_count % 20 == 0:
                        weights = self.get_venue_weights()
                        best_venue = self.get_best_venue()
                        self.logger.info(
                            f"📊 Best venue: {best_venue} | Weights: {', '.join(f'{v}={w:.2f}' for v, w in sorted(weights.items()))}"
                        )

                    # Wait for next measurement cycle
                    await asyncio.sleep(self.measurement_interval)

                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(1.0)  # Brief pause before retry


async def main():
    """Main entry point for SOR latency tracker."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    tracker = VenueLatencyTracker(measurement_interval=0.25)

    try:
        await tracker.run_latency_monitoring()
    except KeyboardInterrupt:
        logging.getLogger("sor_latency").info("🛑 Latency monitoring stopped")


if __name__ == "__main__":
    asyncio.run(main())
