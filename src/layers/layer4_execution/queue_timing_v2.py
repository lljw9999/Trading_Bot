#!/usr/bin/env python3
"""
Queue-Timing v2: Time-to-Fill Prediction for Post-Only Orders
Predicts ETA for order fills based on queue position and market conditions.
"""
import os
import sys
import json
import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QueueTimingConfig:
    """Configuration for queue timing predictions."""

    base_fill_rate_crypto: float = 0.15  # Fill rate per second for crypto
    base_fill_rate_equity: float = 0.08  # Fill rate per second for equities
    volatility_multiplier: float = 2.0  # Volatility increases fill rate
    imbalance_factor: float = 1.5  # Order book imbalance effect
    spread_penalty: float = 0.5  # Wide spreads slow fills

    # Maker book toxicity filter (M16.1 optimization)
    toxicity_mid_change_p80: float = 0.8  # P80 threshold for mid price change rate
    toxicity_queue_reset_p80: float = 0.8  # P80 threshold for queue resets
    toxicity_defer_min_ms: float = 300  # Min defer time when toxic
    toxicity_defer_max_ms: float = 600  # Max defer time when toxic


class QueueTimingV2:
    """Advanced queue timing predictor with market microstructure awareness."""

    def __init__(self, config: QueueTimingConfig = None):
        self.config = config or QueueTimingConfig()
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

    def detect_maker_book_toxicity(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if maker book conditions are toxic (M16.1 optimization)."""

        # Get microstructure metrics with defaults
        mid_change_rate = market_data.get("mid_change_rate_500ms", 0.0)
        queue_resets = market_data.get("queue_resets_500ms", 0.0)

        # Calculate percentile-based thresholds (simulated for now)
        # In production, these would be historical P80 values
        mid_change_p80 = self.config.toxicity_mid_change_p80
        queue_reset_p80 = self.config.toxicity_queue_reset_p80

        # Check toxicity conditions
        is_mid_toxic = mid_change_rate > mid_change_p80
        is_queue_toxic = queue_resets > queue_reset_p80
        is_toxic = is_mid_toxic or is_queue_toxic

        # Calculate defer time if toxic
        defer_ms = 0.0
        if is_toxic:
            defer_ms = np.random.uniform(
                self.config.toxicity_defer_min_ms, self.config.toxicity_defer_max_ms
            )

        return {
            "is_toxic": is_toxic,
            "mid_change_toxic": is_mid_toxic,
            "queue_reset_toxic": is_queue_toxic,
            "mid_change_rate": mid_change_rate,
            "queue_resets": queue_resets,
            "defer_seconds": defer_ms / 1000.0,
            "reason": (
                "mid_volatility"
                if is_mid_toxic
                else "queue_instability" if is_queue_toxic else None
            ),
        }

    def estimate_queue_position(
        self, order_size: float, side: str, market_data: Dict[str, Any]
    ) -> float:
        """Estimate queue position for post-only order."""

        # Get order book depth at best level
        if side.lower() == "buy":
            bid_size = market_data.get("bid_size", 1000)
            our_queue_pos = min(order_size / bid_size, 1.0)
        else:  # sell
            ask_size = market_data.get("ask_size", 1000)
            our_queue_pos = min(order_size / ask_size, 1.0)

        # Adjust for typical queue dynamics
        # Larger orders tend to be further back in queue
        queue_position = our_queue_pos * (1 + np.log10(max(1, order_size / 100)))

        return min(queue_position, 0.95)  # Cap at 95% back in queue

    def calculate_fill_rate(self, asset: str, market_data: Dict[str, Any]) -> float:
        """Calculate expected fill rate (orders per second) for current conditions."""

        # Base fill rate by asset type
        is_crypto = asset.endswith("-USD") and asset != "NVDA"
        base_rate = (
            self.config.base_fill_rate_crypto
            if is_crypto
            else self.config.base_fill_rate_equity
        )

        # Market condition adjustments
        spread_bps = market_data.get("spread_bps", 10)
        volatility = market_data.get("volatility_1m", 1.0)
        ob_imbalance = market_data.get("ob_imbalance", 0.0)
        volume_ratio = market_data.get("volume_ratio_5m", 1.0)

        # Volatility increases activity (more fills)
        vol_adjustment = 1 + (volatility - 1) * self.config.volatility_multiplier

        # Volume increases activity
        volume_adjustment = 1 + (volume_ratio - 1) * 0.5

        # Order book imbalance affects fill probability
        # If we're on the heavy side, fills come faster
        imbalance_adjustment = 1 + abs(ob_imbalance) * self.config.imbalance_factor

        # Wide spreads slow down fills (less activity at best levels)
        spread_adjustment = max(
            0.2, 1 - (spread_bps - 5) * self.config.spread_penalty / 100
        )

        # Time of day adjustment (market hours vs off hours)
        hour = datetime.datetime.now().hour
        if not is_crypto:
            # Equity market hours effect
            if 9 <= hour <= 16:
                time_adjustment = 1.2  # Active market hours
            elif 4 <= hour <= 9 or 16 <= hour <= 20:
                time_adjustment = 0.4  # Pre/post market
            else:
                time_adjustment = 0.1  # After hours
        else:
            # Crypto is 24/7 but has some patterns
            if 13 <= hour <= 21:  # US/Europe overlap
                time_adjustment = 1.1
            elif 1 <= hour <= 6:  # Asian off hours
                time_adjustment = 0.7
            else:
                time_adjustment = 1.0

        # Combine all factors
        adjusted_rate = (
            base_rate
            * vol_adjustment
            * volume_adjustment
            * imbalance_adjustment
            * spread_adjustment
            * time_adjustment
        )

        return max(0.001, adjusted_rate)  # Minimum rate to avoid division by zero

    def predict_time_to_fill(
        self, asset: str, order_size: float, side: str, market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Predict time to fill for post-only order."""

        # Check for maker book toxicity (M16.1 optimization)
        toxicity_analysis = self.detect_maker_book_toxicity(market_data)

        # Estimate our position in queue
        queue_position = self.estimate_queue_position(order_size, side, market_data)

        # Calculate market fill rate
        fill_rate = self.calculate_fill_rate(asset, market_data)

        # Convert queue position to expected fill time
        # If we're 50% back in queue and market processes 0.1 orders/sec,
        # we expect ~5 seconds to our turn
        base_eta_seconds = queue_position / fill_rate

        # Add size impact - larger orders take longer even after reaching front
        size_penalty = max(1.0, np.log10(order_size / 100)) * 10  # seconds

        # Add spread impact - tight spreads fill faster
        spread_bps = market_data.get("spread_bps", 10)
        spread_penalty = max(0, (spread_bps - 2) * 2)  # seconds

        # Add toxicity defer time if conditions are toxic
        toxicity_defer = (
            toxicity_analysis["defer_seconds"] if toxicity_analysis["is_toxic"] else 0.0
        )

        # Total expected time
        eta_seconds = base_eta_seconds + size_penalty + spread_penalty + toxicity_defer

        # Calculate confidence intervals
        # More volatile markets have higher uncertainty
        volatility = market_data.get("volatility_1m", 1.0)
        uncertainty_factor = 1 + volatility * 0.5

        eta_p50 = eta_seconds
        eta_p75 = eta_seconds * uncertainty_factor * 1.3
        eta_p90 = eta_seconds * uncertainty_factor * 2.0
        eta_p95 = eta_seconds * uncertainty_factor * 2.5

        return {
            "eta_seconds_p50": float(eta_p50),
            "eta_seconds_p75": float(eta_p75),
            "eta_seconds_p90": float(eta_p90),
            "eta_seconds_p95": float(eta_p95),
            "queue_position": float(queue_position),
            "fill_rate": float(fill_rate),
            "confidence": (
                "high"
                if uncertainty_factor < 1.5
                else "medium" if uncertainty_factor < 2.0 else "low"
            ),
            "toxicity_analysis": toxicity_analysis,  # M16.1 optimization
        }

    def should_escalate_urgency(
        self,
        eta_prediction: Dict[str, float],
        sla_seconds: float,
        confidence_level: str = "p90",
    ) -> bool:
        """Determine if order should escalate from post-only due to timing."""

        eta_key = f"eta_seconds_{confidence_level}"
        predicted_eta = eta_prediction.get(eta_key, eta_prediction["eta_seconds_p50"])

        # Escalate if predicted ETA exceeds SLA
        return predicted_eta > sla_seconds

    def get_recommended_slice_size(
        self,
        total_size: float,
        eta_prediction: Dict[str, float],
        market_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """Recommend child order slice size based on queue timing."""

        # Base slice as percentage of depth
        depth_1 = market_data.get("depth_1", 1000)
        depth_5 = market_data.get("depth_5", 5000)

        # Start with depth-aware sizing
        conservative_slice = min(total_size, depth_1 * 0.1)  # 10% of top level
        aggressive_slice = min(total_size, depth_5 * 0.05)  # 5% of 5-level depth

        # Adjust based on ETA prediction
        queue_position = eta_prediction.get("queue_position", 0.5)

        if queue_position < 0.2:
            # We're near front of queue, can be more aggressive
            recommended_slice = aggressive_slice
        elif queue_position > 0.7:
            # We're far back, be more conservative
            recommended_slice = conservative_slice * 0.5
        else:
            # Moderate position
            recommended_slice = conservative_slice

        # Ensure minimum and maximum bounds
        min_slice = max(1, total_size * 0.001)  # At least 0.1% of order
        max_slice = min(total_size, total_size * 0.1)  # At most 10% of order

        final_slice = max(min_slice, min(max_slice, recommended_slice))

        return {
            "recommended_slice": float(final_slice),
            "slice_pct_of_order": float(final_slice / total_size),
            "slice_pct_of_depth": float(final_slice / depth_1),
            "reasoning": f"queue_pos_{queue_position:.1%}_depth_{depth_1:.0f}",
        }

    def create_timing_report(
        self, asset: str, scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create comprehensive timing analysis report."""

        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "asset": asset,
            "scenarios": [],
            "summary": {},
        }

        all_etas = []

        for scenario in scenarios:
            market_data = scenario["market_data"]
            order_size = scenario["order_size"]
            side = scenario.get("side", "buy")

            # Get timing prediction
            eta_prediction = self.predict_time_to_fill(
                asset, order_size, side, market_data
            )

            # Get slice recommendation
            slice_rec = self.get_recommended_slice_size(
                order_size, eta_prediction, market_data
            )

            # Check escalation
            sla_seconds = scenario.get("sla_seconds", 120)
            should_escalate = self.should_escalate_urgency(eta_prediction, sla_seconds)

            scenario_result = {
                "scenario_name": scenario.get("name", "unnamed"),
                "market_conditions": market_data,
                "order_size": order_size,
                "side": side,
                "sla_seconds": sla_seconds,
                "eta_prediction": eta_prediction,
                "slice_recommendation": slice_rec,
                "should_escalate": should_escalate,
                "escalation_reason": (
                    "eta_exceeds_sla" if should_escalate else "within_sla"
                ),
            }

            report["scenarios"].append(scenario_result)
            all_etas.append(eta_prediction["eta_seconds_p50"])

        # Summary statistics
        if all_etas:
            report["summary"] = {
                "avg_eta_seconds": float(np.mean(all_etas)),
                "min_eta_seconds": float(np.min(all_etas)),
                "max_eta_seconds": float(np.max(all_etas)),
                "escalation_rate": sum(
                    s["should_escalate"] for s in report["scenarios"]
                )
                / len(scenarios),
            }

        return report


def test_queue_timing_v2():
    """Test queue timing predictions with various scenarios."""

    print("🕐 Testing Queue-Timing v2 Predictions")
    print("=" * 45)

    timing = QueueTimingV2()

    # Test scenarios
    test_scenarios = [
        {
            "name": "crypto_normal_conditions",
            "market_data": {
                "spread_bps": 5.0,
                "volatility_1m": 1.2,
                "ob_imbalance": 0.1,
                "volume_ratio_5m": 1.1,
                "bid_size": 5000,
                "ask_size": 4500,
                "depth_1": 8000,
                "depth_5": 25000,
            },
            "order_size": 1000,
            "side": "buy",
            "sla_seconds": 120,
        },
        {
            "name": "equity_volatile_conditions",
            "market_data": {
                "spread_bps": 15.0,
                "volatility_1m": 2.5,
                "ob_imbalance": -0.4,
                "volume_ratio_5m": 3.2,
                "bid_size": 2000,
                "ask_size": 1800,
                "depth_1": 3000,
                "depth_5": 12000,
            },
            "order_size": 500,
            "side": "sell",
            "sla_seconds": 180,
        },
        {
            "name": "crypto_wide_spread",
            "market_data": {
                "spread_bps": 25.0,
                "volatility_1m": 0.8,
                "ob_imbalance": 0.0,
                "volume_ratio_5m": 0.7,
                "bid_size": 1000,
                "ask_size": 1200,
                "depth_1": 2000,
                "depth_5": 8000,
            },
            "order_size": 2000,
            "side": "buy",
            "sla_seconds": 120,
        },
    ]

    # Test crypto asset
    print("\n🔍 Testing BTC-USD scenarios:")
    btc_report = timing.create_timing_report("BTC-USD", test_scenarios)

    for scenario in btc_report["scenarios"]:
        eta_p50 = scenario["eta_prediction"]["eta_seconds_p50"]
        eta_p90 = scenario["eta_prediction"]["eta_seconds_p90"]
        queue_pos = scenario["eta_prediction"]["queue_position"]
        should_escalate = scenario["should_escalate"]
        slice_pct = scenario["slice_recommendation"]["slice_pct_of_order"]

        print(f"  {scenario['scenario_name']}:")
        print(f"    ETA: {eta_p50:.0f}s (p50), {eta_p90:.0f}s (p90)")
        print(f"    Queue position: {queue_pos:.1%}")
        print(f"    Recommended slice: {slice_pct:.1%} of order")
        print(f"    Escalate: {'✅ YES' if should_escalate else '❌ NO'}")

    print(f"\n📊 Summary:")
    print(f"  Average ETA: {btc_report['summary']['avg_eta_seconds']:.0f}s")
    print(f"  Escalation rate: {btc_report['summary']['escalation_rate']:.1%}")

    # Test equity asset
    print(f"\n🔍 Testing NVDA scenarios:")
    nvda_report = timing.create_timing_report("NVDA", test_scenarios)

    print(f"  Average ETA: {nvda_report['summary']['avg_eta_seconds']:.0f}s")
    print(f"  Escalation rate: {nvda_report['summary']['escalation_rate']:.1%}")

    print(f"\n✅ Queue-Timing v2 test complete!")

    return btc_report, nvda_report


if __name__ == "__main__":
    test_queue_timing_v2()
