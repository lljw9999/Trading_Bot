#!/usr/bin/env python3
"""
M18: Impact Budget System
Dynamic slice sizing based on rolling market impact measurements.
Prevents excessive market impact during 20% ramp by adjusting order sizes.
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
class ImpactMeasurement:
    """Single impact measurement."""

    timestamp: datetime.datetime
    asset: str
    venue: str
    route: str
    notional_usd: float
    impact_bps: float
    slice_pct: float
    market_conditions: Dict[str, Any]


class ImpactBudget:
    """Dynamic impact budget and slice sizing controller."""

    def __init__(self):
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

        # Redis for live metrics
        try:
            self.redis_client = redis.Redis(decode_responses=True)
            self.redis_client.ping()
        except Exception:
            self.redis_client = None

        # Impact budget configuration
        self.config = {
            # Target impact levels (bp per $1k notional)
            "target_impact_p95": 6.0,  # P95 target
            "hard_cap_impact_p95": 8.0,  # Hard cap - triggers immediate reduction
            "warning_impact_p95": 7.0,  # Warning threshold
            # Measurement windows
            "measurement_window_minutes": 30,  # Rolling window for impact calculation
            "min_samples_for_decision": 10,  # Minimum samples needed
            # Slice adjustment parameters
            "default_slice_pct_cap": 1.2,  # Default max slice (1.2% of ADV)
            "min_slice_pct_cap": 0.3,  # Minimum allowed slice
            "max_slice_pct_cap": 2.0,  # Maximum allowed slice
            "adjustment_step_pct": 0.1,  # Adjustment increment
            # Route-specific multipliers
            "route_multipliers": {
                "post_only": 0.8,  # Post-only has lower impact
                "mid_point": 1.0,  # Mid-point baseline
                "cross_spread": 1.5,  # Cross-spread higher impact
            },
        }

        # Impact history
        self.impact_history: List[ImpactMeasurement] = []
        self.current_caps: Dict[str, float] = {}  # Asset -> slice_pct_cap

        # Load historical data if available
        self.load_impact_history()

    def load_impact_history(self) -> None:
        """Load historical impact measurements."""

        history_file = self.base_dir / "artifacts" / "impact" / "impact_history.json"

        try:
            if history_file.exists():
                with open(history_file, "r") as f:
                    history_data = json.load(f)

                for entry in history_data.get("measurements", []):
                    measurement = ImpactMeasurement(
                        timestamp=datetime.datetime.fromisoformat(entry["timestamp"]),
                        asset=entry["asset"],
                        venue=entry["venue"],
                        route=entry["route"],
                        notional_usd=entry["notional_usd"],
                        impact_bps=entry["impact_bps"],
                        slice_pct=entry["slice_pct"],
                        market_conditions=entry.get("market_conditions", {}),
                    )
                    self.impact_history.append(measurement)

                print(f"📊 Loaded {len(self.impact_history)} impact measurements")
        except Exception as e:
            print(f"⚠️ Error loading impact history: {e}")

    def save_impact_history(self) -> None:
        """Save impact history to disk."""

        history_dir = self.base_dir / "artifacts" / "impact"
        history_dir.mkdir(parents=True, exist_ok=True)

        history_file = history_dir / "impact_history.json"

        try:
            # Keep only last 7 days
            cutoff = datetime.datetime.now() - datetime.timedelta(days=7)
            recent_history = [m for m in self.impact_history if m.timestamp >= cutoff]

            history_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "config": self.config,
                "measurements": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "asset": m.asset,
                        "venue": m.venue,
                        "route": m.route,
                        "notional_usd": m.notional_usd,
                        "impact_bps": m.impact_bps,
                        "slice_pct": m.slice_pct,
                        "market_conditions": m.market_conditions,
                    }
                    for m in recent_history
                ],
            }

            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2)

            self.impact_history = recent_history

        except Exception as e:
            print(f"⚠️ Error saving impact history: {e}")

    def record_impact_measurement(
        self,
        asset: str,
        venue: str,
        route: str,
        notional_usd: float,
        impact_bps: float,
        slice_pct: float,
        market_conditions: Dict[str, Any] = None,
    ) -> None:
        """Record a new impact measurement."""

        measurement = ImpactMeasurement(
            timestamp=datetime.datetime.now(),
            asset=asset,
            venue=venue,
            route=route,
            notional_usd=notional_usd,
            impact_bps=impact_bps,
            slice_pct=slice_pct,
            market_conditions=market_conditions or {},
        )

        self.impact_history.append(measurement)

        # Trim to keep only recent measurements
        cutoff = datetime.datetime.now() - datetime.timedelta(hours=24)
        self.impact_history = [m for m in self.impact_history if m.timestamp >= cutoff]

        # Update caps based on new measurement
        self.update_slice_caps()

    def compute_rolling_impact_stats(
        self, asset: str, venue: str = None, route: str = None
    ) -> Dict[str, Any]:
        """Compute rolling impact statistics for given filters."""

        window_start = datetime.datetime.now() - datetime.timedelta(
            minutes=self.config["measurement_window_minutes"]
        )

        # Filter measurements
        filtered_measurements = [
            m
            for m in self.impact_history
            if (
                m.timestamp >= window_start
                and m.asset == asset
                and (venue is None or m.venue == venue)
                and (route is None or m.route == route)
            )
        ]

        if len(filtered_measurements) < self.config["min_samples_for_decision"]:
            return {
                "insufficient_data": True,
                "sample_count": len(filtered_measurements),
                "min_required": self.config["min_samples_for_decision"],
            }

        # Compute statistics
        impact_values = [m.impact_bps for m in filtered_measurements]
        notional_values = [m.notional_usd for m in filtered_measurements]

        stats = {
            "sample_count": len(filtered_measurements),
            "window_minutes": self.config["measurement_window_minutes"],
            "impact_mean_bps": np.mean(impact_values),
            "impact_p95_bps": np.percentile(impact_values, 95),
            "impact_max_bps": np.max(impact_values),
            "notional_total": np.sum(notional_values),
            "notional_mean": np.mean(notional_values),
            "measurements_per_hour": len(filtered_measurements)
            / (self.config["measurement_window_minutes"] / 60),
        }

        # Compute impact per $1k
        if stats["notional_total"] > 0:
            stats["impact_per_1k_p95"] = stats["impact_p95_bps"] * (
                1000 / stats["notional_mean"]
            )
        else:
            stats["impact_per_1k_p95"] = 0

        return stats

    def determine_slice_cap(self, asset: str, venue: str, route: str) -> float:
        """Determine appropriate slice cap for given asset/venue/route."""

        # Get rolling impact stats
        stats = self.compute_rolling_impact_stats(asset, venue, route)

        if stats.get("insufficient_data", False):
            # Not enough data - use conservative default
            return self.config["default_slice_pct_cap"] * 0.8

        # Get route multiplier
        route_multiplier = self.config["route_multipliers"].get(route, 1.0)

        # Determine cap based on impact levels
        impact_p95 = stats["impact_per_1k_p95"]
        target_impact = self.config["target_impact_p95"]
        hard_cap_impact = self.config["hard_cap_impact_p95"]

        if impact_p95 <= target_impact:
            # Below target - can use default or slightly higher
            base_cap = self.config["default_slice_pct_cap"]
            adjustment = min(0.3, (target_impact - impact_p95) / target_impact * 0.5)
            proposed_cap = base_cap + adjustment

        elif impact_p95 <= hard_cap_impact:
            # Between target and hard cap - reduce proportionally
            base_cap = self.config["default_slice_pct_cap"]
            excess_ratio = (impact_p95 - target_impact) / (
                hard_cap_impact - target_impact
            )
            reduction = excess_ratio * 0.5  # Up to 50% reduction
            proposed_cap = base_cap * (1 - reduction)

        else:
            # Above hard cap - aggressive reduction
            proposed_cap = self.config["min_slice_pct_cap"]

        # Apply route multiplier (inverse - higher impact routes get lower caps)
        final_cap = proposed_cap / route_multiplier

        # Clamp to limits
        final_cap = max(
            self.config["min_slice_pct_cap"],
            min(self.config["max_slice_pct_cap"], final_cap),
        )

        return final_cap

    def update_slice_caps(self) -> None:
        """Update slice caps for all assets based on recent impact data."""

        # Get unique assets from recent measurements
        recent_cutoff = datetime.datetime.now() - datetime.timedelta(hours=2)
        recent_assets = set(
            m.asset for m in self.impact_history if m.timestamp >= recent_cutoff
        )

        for asset in recent_assets:
            # Compute overall asset cap (most restrictive across venues/routes)
            asset_caps = []

            for venue in ["coinbase", "binance"]:  # Major venues
                for route in ["post_only", "mid_point", "cross_spread"]:
                    cap = self.determine_slice_cap(asset, venue, route)
                    asset_caps.append(cap)

            if asset_caps:
                # Use most restrictive cap
                self.current_caps[asset] = min(asset_caps)

        # Publish to Redis if available
        if self.redis_client:
            for asset, cap in self.current_caps.items():
                key = f"impact_budget:slice_cap:{asset}"
                self.redis_client.set(key, cap, ex=1800)  # 30-minute expiry

    def get_allowed_slice_pct_cap(
        self, asset: str, venue: str = None, route: str = None
    ) -> float:
        """Get current allowed slice percentage cap for asset."""

        # Check Redis first for live updates
        if self.redis_client:
            key = f"impact_budget:slice_cap:{asset}"
            try:
                redis_cap = self.redis_client.get(key)
                if redis_cap:
                    return float(redis_cap)
            except Exception:
                pass

        # Fall back to computed caps
        if asset in self.current_caps:
            return self.current_caps[asset]

        # Default conservative cap
        return self.config["default_slice_pct_cap"] * 0.8

    def get_impact_status(self, asset: str = None) -> Dict[str, Any]:
        """Get current impact budget status."""

        status = {
            "timestamp": datetime.datetime.now().isoformat(),
            "config": self.config,
            "measurement_count": len(self.impact_history),
            "current_caps": self.current_caps.copy(),
            "asset_status": {},
        }

        # Filter by asset if specified
        assets_to_check = (
            [asset] if asset else list(set(m.asset for m in self.impact_history))
        )

        for check_asset in assets_to_check:
            stats = self.compute_rolling_impact_stats(check_asset)

            asset_status = {
                "current_cap": self.get_allowed_slice_pct_cap(check_asset),
                "stats": stats,
            }

            # Determine status level
            if stats.get("insufficient_data", False):
                asset_status["status"] = "INSUFFICIENT_DATA"
                asset_status["status_color"] = "gray"
            else:
                impact_p95 = stats["impact_per_1k_p95"]
                if impact_p95 <= self.config["target_impact_p95"]:
                    asset_status["status"] = "GOOD"
                    asset_status["status_color"] = "green"
                elif impact_p95 <= self.config["warning_impact_p95"]:
                    asset_status["status"] = "WARNING"
                    asset_status["status_color"] = "yellow"
                else:
                    asset_status["status"] = "CRITICAL"
                    asset_status["status_color"] = "red"

            status["asset_status"][check_asset] = asset_status

        return status

    def simulate_impact_measurements(self, count: int = 50) -> None:
        """Simulate impact measurements for testing."""

        assets = ["BTC-USD", "ETH-USD", "SOL-USD", "NVDA"]
        venues = ["coinbase", "binance"]
        routes = ["post_only", "mid_point", "cross_spread"]

        base_time = datetime.datetime.now() - datetime.timedelta(hours=2)

        for i in range(count):
            asset = np.random.choice(assets)
            venue = np.random.choice(venues)
            route = np.random.choice(routes)

            # Simulate realistic impact based on route
            if route == "post_only":
                base_impact = 4.0  # Lower impact
            elif route == "mid_point":
                base_impact = 6.5  # Medium impact
            else:  # cross_spread
                base_impact = 12.0  # Higher impact

            # Add noise and market conditions
            impact_bps = max(1.0, base_impact + np.random.normal(0, 2.0))

            # Simulate notional and slice
            notional_usd = np.random.uniform(2000, 8000)
            slice_pct = np.random.uniform(0.5, 1.5)

            # Create measurement
            measurement_time = base_time + datetime.timedelta(minutes=i * 2)
            measurement = ImpactMeasurement(
                timestamp=measurement_time,
                asset=asset,
                venue=venue,
                route=route,
                notional_usd=notional_usd,
                impact_bps=impact_bps,
                slice_pct=slice_pct,
                market_conditions={
                    "spread_bps": np.random.uniform(1, 5),
                    "book_depth_$1k": np.random.uniform(50, 200),
                    "volatility_1h": np.random.uniform(0.5, 3.0),
                },
            )

            self.impact_history.append(measurement)

        # Update caps based on simulated data
        self.update_slice_caps()
        print(f"📊 Simulated {count} impact measurements")


def main():
    """Test the impact budget system."""

    print("🎯 M18: Impact Budget System Test")
    print("=" * 40)

    # Initialize system
    impact_budget = ImpactBudget()

    # Simulate some measurements
    print("📊 Simulating impact measurements...")
    impact_budget.simulate_impact_measurements(100)

    # Get status
    print("\n📈 Impact Budget Status:")
    status = impact_budget.get_impact_status()

    for asset, asset_status in status["asset_status"].items():
        print(f"  {asset}:")
        print(f"    Status: {asset_status['status']} ({asset_status['status_color']})")
        print(f"    Current Cap: {asset_status['current_cap']:.2f}%")

        if not asset_status["stats"].get("insufficient_data", False):
            stats = asset_status["stats"]
            print(f"    Impact P95: {stats['impact_per_1k_p95']:.1f}bp/$1k")
            print(f"    Samples: {stats['sample_count']}")

    # Test specific caps
    print(f"\n🔍 Slice Cap Examples:")
    for asset in ["BTC-USD", "ETH-USD"]:
        cap = impact_budget.get_allowed_slice_pct_cap(asset)
        print(f"  {asset}: {cap:.2f}% max slice")

    # Save history
    impact_budget.save_impact_history()
    print(f"\n✅ Impact budget system test complete")


if __name__ == "__main__":
    main()
