#!/usr/bin/env python3
"""
M18: Venue Saturation Guard
Monitors participation rate vs venue capacity to prevent market disruption.
Dynamically reduces slice sizes and post-only ratios when approaching saturation.
"""
import os
import json
import time
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import redis
from dataclasses import dataclass


@dataclass
class VenueMetrics:
    """Venue capacity and utilization metrics."""

    timestamp: datetime.datetime
    venue: str
    asset: str
    estimated_volume_1h: float  # Estimated 1h volume in USD
    our_volume_5m: float  # Our volume in last 5 minutes
    participation_pct: float  # Our participation % of venue volume
    queue_position_p80: int  # 80th percentile queue position
    queue_ahead_estimate: float  # Estimated orders ahead in queue
    book_depth_bps: Dict[int, float]  # Depth at various bps levels


class VenueSaturationGuard:
    """Monitor and prevent venue saturation."""

    def __init__(self):
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

        # Redis for live metrics
        try:
            self.redis_client = redis.Redis(decode_responses=True)
            self.redis_client.ping()
        except Exception:
            self.redis_client = None

        # Saturation guard configuration
        self.config = {
            # Participation thresholds
            "target_pov_5m": 8.0,  # Target participation (8% of volume)
            "warning_pov_5m": 10.0,  # Warning threshold (10%)
            "critical_pov_5m": 12.0,  # Critical threshold (12%)
            # Queue position thresholds
            "target_queue_p80": 15,  # Target 80th percentile queue position
            "warning_queue_p80": 25,  # Warning threshold
            "critical_queue_p80": 40,  # Critical threshold
            # Book depth requirements (minimum depth in USD at bps levels)
            "min_depth_requirements": {
                5: 50000,  # $50k within 5 bps
                10: 100000,  # $100k within 10 bps
                20: 200000,  # $200k within 20 bps
            },
            # Adjustment parameters
            "slice_reduction_step": 0.15,  # Reduce slice by 15% when triggered
            "post_only_increase_step": 0.05,  # Increase post-only by 5% when triggered
            "min_slice_pct": 0.3,  # Minimum slice percentage
            "max_post_only_ratio": 0.95,  # Maximum post-only ratio
            # Measurement windows
            "pov_window_minutes": 5,  # POV measurement window
            "queue_window_minutes": 10,  # Queue metrics window
            "min_measurements": 5,  # Minimum measurements for decision
        }

        # Venue metrics history
        self.venue_metrics: List[VenueMetrics] = []
        self.current_adjustments: Dict[str, Dict[str, float]] = {}

        # Load historical data
        self.load_venue_history()

    def load_venue_history(self) -> None:
        """Load historical venue metrics."""

        history_file = self.base_dir / "artifacts" / "venue" / "venue_metrics.json"

        try:
            if history_file.exists():
                with open(history_file, "r") as f:
                    history_data = json.load(f)

                for entry in history_data.get("metrics", []):
                    metrics = VenueMetrics(
                        timestamp=datetime.datetime.fromisoformat(entry["timestamp"]),
                        venue=entry["venue"],
                        asset=entry["asset"],
                        estimated_volume_1h=entry["estimated_volume_1h"],
                        our_volume_5m=entry["our_volume_5m"],
                        participation_pct=entry["participation_pct"],
                        queue_position_p80=entry["queue_position_p80"],
                        queue_ahead_estimate=entry["queue_ahead_estimate"],
                        book_depth_bps=entry["book_depth_bps"],
                    )
                    self.venue_metrics.append(metrics)

                print(f"📊 Loaded {len(self.venue_metrics)} venue metrics")
        except Exception as e:
            print(f"⚠️ Error loading venue history: {e}")

    def save_venue_history(self) -> None:
        """Save venue metrics to disk."""

        venue_dir = self.base_dir / "artifacts" / "venue"
        venue_dir.mkdir(parents=True, exist_ok=True)

        history_file = venue_dir / "venue_metrics.json"

        try:
            # Keep only last 24 hours
            cutoff = datetime.datetime.now() - datetime.timedelta(hours=24)
            recent_metrics = [m for m in self.venue_metrics if m.timestamp >= cutoff]

            history_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "config": self.config,
                "metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "venue": m.venue,
                        "asset": m.asset,
                        "estimated_volume_1h": m.estimated_volume_1h,
                        "our_volume_5m": m.our_volume_5m,
                        "participation_pct": m.participation_pct,
                        "queue_position_p80": m.queue_position_p80,
                        "queue_ahead_estimate": m.queue_ahead_estimate,
                        "book_depth_bps": m.book_depth_bps,
                    }
                    for m in recent_metrics
                ],
            }

            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2)

            self.venue_metrics = recent_metrics

        except Exception as e:
            print(f"⚠️ Error saving venue history: {e}")

    def record_venue_metrics(
        self,
        venue: str,
        asset: str,
        estimated_volume_1h: float,
        our_volume_5m: float,
        queue_position_p80: int,
        queue_ahead_estimate: float,
        book_depth_bps: Dict[int, float],
    ) -> None:
        """Record new venue metrics."""

        # Calculate participation percentage
        participation_pct = 0.0
        if estimated_volume_1h > 0:
            # Convert our 5-minute volume to hourly equivalent for comparison
            our_volume_1h_equivalent = our_volume_5m * 12  # 12 x 5-minute periods
            participation_pct = (our_volume_1h_equivalent / estimated_volume_1h) * 100

        metrics = VenueMetrics(
            timestamp=datetime.datetime.now(),
            venue=venue,
            asset=asset,
            estimated_volume_1h=estimated_volume_1h,
            our_volume_5m=our_volume_5m,
            participation_pct=participation_pct,
            queue_position_p80=queue_position_p80,
            queue_ahead_estimate=queue_ahead_estimate,
            book_depth_bps=book_depth_bps,
        )

        self.venue_metrics.append(metrics)

        # Trim old metrics
        cutoff = datetime.datetime.now() - datetime.timedelta(hours=6)
        self.venue_metrics = [m for m in self.venue_metrics if m.timestamp >= cutoff]

        # Update adjustments based on new metrics
        self.update_venue_adjustments()

    def compute_venue_status(self, venue: str, asset: str) -> Dict[str, Any]:
        """Compute current venue saturation status."""

        # Get recent metrics for this venue/asset
        cutoff = datetime.datetime.now() - datetime.timedelta(
            minutes=max(
                self.config["pov_window_minutes"], self.config["queue_window_minutes"]
            )
        )

        recent_metrics = [
            m
            for m in self.venue_metrics
            if (m.timestamp >= cutoff and m.venue == venue and m.asset == asset)
        ]

        if len(recent_metrics) < self.config["min_measurements"]:
            return {
                "status": "INSUFFICIENT_DATA",
                "measurements": len(recent_metrics),
                "required": self.config["min_measurements"],
            }

        # Compute POV statistics
        pov_values = [m.participation_pct for m in recent_metrics]
        current_pov = (
            np.mean(pov_values[-3:]) if len(pov_values) >= 3 else np.mean(pov_values)
        )
        max_pov_5m = np.max(pov_values)

        # Compute queue statistics
        queue_values = [m.queue_position_p80 for m in recent_metrics]
        current_queue_p80 = (
            np.mean(queue_values[-3:])
            if len(queue_values) >= 3
            else np.mean(queue_values)
        )
        max_queue_p80 = np.max(queue_values)

        # Check book depth adequacy
        latest_metrics = recent_metrics[-1]
        depth_status = {}
        depth_adequate = True

        for bps_level, min_depth in self.config["min_depth_requirements"].items():
            current_depth = latest_metrics.book_depth_bps.get(bps_level, 0)
            adequate = current_depth >= min_depth
            depth_status[f"depth_{bps_level}bps"] = {
                "current": current_depth,
                "required": min_depth,
                "adequate": adequate,
            }
            if not adequate:
                depth_adequate = False

        # Determine overall status
        status_factors = []

        # POV assessment
        if current_pov <= self.config["target_pov_5m"]:
            pov_status = "GOOD"
            pov_score = 1.0
        elif current_pov <= self.config["warning_pov_5m"]:
            pov_status = "WARNING"
            pov_score = 0.7
        else:
            pov_status = "CRITICAL"
            pov_score = 0.3

        # Queue assessment
        if current_queue_p80 <= self.config["target_queue_p80"]:
            queue_status = "GOOD"
            queue_score = 1.0
        elif current_queue_p80 <= self.config["warning_queue_p80"]:
            queue_status = "WARNING"
            queue_score = 0.7
        else:
            queue_status = "CRITICAL"
            queue_score = 0.3

        # Depth assessment
        depth_score = 1.0 if depth_adequate else 0.5

        # Overall score and status
        overall_score = np.mean([pov_score, queue_score, depth_score])

        if overall_score >= 0.8:
            overall_status = "HEALTHY"
        elif overall_score >= 0.6:
            overall_status = "WARNING"
        else:
            overall_status = "SATURATED"

        return {
            "status": overall_status,
            "score": overall_score,
            "participation": {
                "current_pov_5m": current_pov,
                "max_pov_5m": max_pov_5m,
                "status": pov_status,
                "target": self.config["target_pov_5m"],
            },
            "queue": {
                "current_p80": current_queue_p80,
                "max_p80": max_queue_p80,
                "status": queue_status,
                "target": self.config["target_queue_p80"],
            },
            "depth": depth_status,
            "depth_adequate": depth_adequate,
            "measurements": len(recent_metrics),
            "latest_metrics": {
                "our_volume_5m": latest_metrics.our_volume_5m,
                "estimated_volume_1h": latest_metrics.estimated_volume_1h,
                "queue_ahead": latest_metrics.queue_ahead_estimate,
            },
        }

    def calculate_required_adjustments(
        self, venue: str, asset: str
    ) -> Dict[str, float]:
        """Calculate required parameter adjustments for venue/asset."""

        status = self.compute_venue_status(venue, asset)

        adjustments = {
            "slice_pct_multiplier": 1.0,  # Multiplier for slice percentage
            "post_only_adjustment": 0.0,  # Absolute adjustment to post-only ratio
            "max_escalations": 3,  # Maximum escalation attempts
        }

        if status.get("status") == "INSUFFICIENT_DATA":
            # Conservative defaults when insufficient data
            adjustments["slice_pct_multiplier"] = 0.8
            adjustments["post_only_adjustment"] = 0.05
            return adjustments

        # POV-based adjustments
        participation = status["participation"]
        if participation["status"] == "CRITICAL":
            adjustments["slice_pct_multiplier"] *= (
                1 - self.config["slice_reduction_step"]
            )
            adjustments["post_only_adjustment"] += self.config[
                "post_only_increase_step"
            ]
            adjustments["max_escalations"] = 1
        elif participation["status"] == "WARNING":
            adjustments["slice_pct_multiplier"] *= (
                1 - self.config["slice_reduction_step"] * 0.5
            )
            adjustments["post_only_adjustment"] += (
                self.config["post_only_increase_step"] * 0.5
            )
            adjustments["max_escalations"] = 2

        # Queue-based adjustments
        queue = status["queue"]
        if queue["status"] == "CRITICAL":
            adjustments["slice_pct_multiplier"] *= 0.7  # Further reduction
            adjustments["post_only_adjustment"] += 0.1  # More aggressive post-only
            adjustments["max_escalations"] = 1
        elif queue["status"] == "WARNING":
            adjustments["slice_pct_multiplier"] *= 0.85
            adjustments["post_only_adjustment"] += 0.05

        # Depth-based adjustments
        if not status["depth_adequate"]:
            adjustments["slice_pct_multiplier"] *= 0.8
            adjustments["max_escalations"] = min(adjustments["max_escalations"], 2)

        # Apply limits
        adjustments["slice_pct_multiplier"] = max(
            0.3, min(1.5, adjustments["slice_pct_multiplier"])
        )
        adjustments["post_only_adjustment"] = max(
            0.0, min(0.2, adjustments["post_only_adjustment"])
        )
        adjustments["max_escalations"] = max(1, min(3, adjustments["max_escalations"]))

        return adjustments

    def update_venue_adjustments(self) -> None:
        """Update venue adjustments for all active venue/asset pairs."""

        # Get unique venue/asset combinations from recent metrics
        recent_cutoff = datetime.datetime.now() - datetime.timedelta(hours=1)
        recent_pairs = set(
            (m.venue, m.asset)
            for m in self.venue_metrics
            if m.timestamp >= recent_cutoff
        )

        for venue, asset in recent_pairs:
            adjustments = self.calculate_required_adjustments(venue, asset)

            key = f"{venue}:{asset}"
            self.current_adjustments[key] = adjustments

            # Publish to Redis if available
            if self.redis_client:
                for adj_type, value in adjustments.items():
                    redis_key = f"venue_saturation:{venue}:{asset}:{adj_type}"
                    self.redis_client.set(redis_key, value, ex=1800)  # 30-minute expiry

    def get_venue_adjustments(self, venue: str, asset: str) -> Dict[str, float]:
        """Get current venue adjustments for specific venue/asset."""

        # Check Redis first
        if self.redis_client:
            try:
                adjustments = {}
                for adj_type in [
                    "slice_pct_multiplier",
                    "post_only_adjustment",
                    "max_escalations",
                ]:
                    redis_key = f"venue_saturation:{venue}:{asset}:{adj_type}"
                    value = self.redis_client.get(redis_key)
                    if value:
                        adjustments[adj_type] = float(value)

                if len(adjustments) == 3:
                    return adjustments
            except Exception:
                pass

        # Fall back to computed adjustments
        key = f"{venue}:{asset}"
        if key in self.current_adjustments:
            return self.current_adjustments[key]

        # Conservative defaults
        return {
            "slice_pct_multiplier": 0.8,
            "post_only_adjustment": 0.05,
            "max_escalations": 2,
        }

    def get_saturation_status(self) -> Dict[str, Any]:
        """Get overall venue saturation status."""

        status = {
            "timestamp": datetime.datetime.now().isoformat(),
            "config": self.config,
            "total_metrics": len(self.venue_metrics),
            "venue_asset_status": {},
            "current_adjustments": self.current_adjustments.copy(),
        }

        # Get unique venue/asset combinations from recent metrics
        recent_cutoff = datetime.datetime.now() - datetime.timedelta(hours=1)
        recent_pairs = set(
            (m.venue, m.asset)
            for m in self.venue_metrics
            if m.timestamp >= recent_cutoff
        )

        for venue, asset in recent_pairs:
            venue_status = self.compute_venue_status(venue, asset)
            adjustments = self.get_venue_adjustments(venue, asset)

            combined_status = {
                "venue_status": venue_status,
                "adjustments": adjustments,
                "effective_slice_reduction": (1 - adjustments["slice_pct_multiplier"])
                * 100,
                "effective_post_only_increase": adjustments["post_only_adjustment"]
                * 100,
            }

            status["venue_asset_status"][f"{venue}:{asset}"] = combined_status

        return status

    def simulate_venue_metrics(self, count: int = 30) -> None:
        """Simulate venue metrics for testing."""

        venues = ["coinbase", "binance"]
        assets = ["BTC-USD", "ETH-USD", "SOL-USD"]

        base_time = datetime.datetime.now() - datetime.timedelta(hours=1)

        for i in range(count):
            venue = np.random.choice(venues)
            asset = np.random.choice(assets)

            # Simulate volume data
            estimated_volume_1h = np.random.uniform(
                1000000, 5000000
            )  # $1-5M hourly volume
            our_volume_5m = np.random.uniform(5000, 25000)  # $5-25k in 5 minutes

            # Simulate queue metrics
            base_queue = 20
            queue_noise = np.random.normal(0, 8)
            queue_position_p80 = max(5, int(base_queue + queue_noise))
            queue_ahead_estimate = queue_position_p80 * np.random.uniform(0.7, 1.3)

            # Simulate book depth
            book_depth_bps = {
                5: np.random.uniform(30000, 80000),
                10: np.random.uniform(70000, 150000),
                20: np.random.uniform(150000, 300000),
            }

            # Record metrics
            timestamp = base_time + datetime.timedelta(minutes=i * 2)
            metrics = VenueMetrics(
                timestamp=timestamp,
                venue=venue,
                asset=asset,
                estimated_volume_1h=estimated_volume_1h,
                our_volume_5m=our_volume_5m,
                participation_pct=0,  # Will be calculated
                queue_position_p80=queue_position_p80,
                queue_ahead_estimate=queue_ahead_estimate,
                book_depth_bps=book_depth_bps,
            )

            # Calculate participation
            our_volume_1h_equivalent = our_volume_5m * 12
            metrics.participation_pct = (
                our_volume_1h_equivalent / estimated_volume_1h
            ) * 100

            self.venue_metrics.append(metrics)

        # Update adjustments
        self.update_venue_adjustments()
        print(f"📊 Simulated {count} venue metrics")


def main():
    """Test the venue saturation guard."""

    print("🛡️ M18: Venue Saturation Guard Test")
    print("=" * 40)

    # Initialize guard
    guard = VenueSaturationGuard()

    # Simulate metrics
    print("📊 Simulating venue metrics...")
    guard.simulate_venue_metrics(60)

    # Get status
    print("\n📈 Venue Saturation Status:")
    status = guard.get_saturation_status()

    for venue_asset, va_status in status["venue_asset_status"].items():
        venue, asset = venue_asset.split(":")
        vs = va_status["venue_status"]
        adj = va_status["adjustments"]

        print(f"  {venue}:{asset}:")
        print(f"    Status: {vs.get('status', 'UNKNOWN')}")
        print(f"    POV: {vs.get('participation', {}).get('current_pov_5m', 0):.1f}%")
        print(f"    Queue P80: {vs.get('queue', {}).get('current_p80', 0):.0f}")
        print(f"    Slice Reduction: {va_status['effective_slice_reduction']:.1f}%")
        print(
            f"    Post-Only Increase: {va_status['effective_post_only_increase']:.1f}%"
        )

    # Test specific adjustments
    print(f"\n🔧 Example Adjustments:")
    for venue in ["coinbase", "binance"]:
        for asset in ["BTC-USD", "ETH-USD"]:
            adj = guard.get_venue_adjustments(venue, asset)
            print(
                f"  {venue}:{asset} -> Slice: {adj['slice_pct_multiplier']:.2f}x, "
                f"Post-only: +{adj['post_only_adjustment']:.1%}"
            )

    # Save metrics
    guard.save_venue_history()
    print(f"\n✅ Venue saturation guard test complete")


if __name__ == "__main__":
    main()
