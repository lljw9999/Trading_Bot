#!/usr/bin/env python3
"""
Child Sizer v2: Volume/Depth Aware Order Sizing
Smart order slicing based on market microstructure, volatility, and KRIs.
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
class ChildSizerConfig:
    """Configuration for child order sizing."""

    # Size bounds by asset type
    crypto_min_slice_pct: float = 0.002  # 0.2% of 5-min volume
    crypto_max_slice_pct: float = 0.03  # 3% of 5-min volume
    equity_min_slice_pct: float = 0.005  # 0.5% of ADV
    equity_max_slice_pct: float = 0.06  # 6% of ADV

    # Depth-based sizing
    min_depth_ratio: float = 0.05  # Min 5% of depth at best
    max_depth_ratio: float = 0.15  # Max 15% of depth at best

    # POV (Participation of Volume) limits
    max_pov_normal: float = 0.25  # Max 25% of recent volume
    max_pov_urgent: float = 0.40  # Max 40% for urgent orders

    # Risk controls
    max_single_slice_usd: float = 50000  # Max $50k per slice
    min_single_slice_usd: float = 100  # Min $100 per slice

    # Timing parameters
    min_rest_ms: float = 500  # Min 500ms between slices
    max_rest_ms: float = 30000  # Max 30s between slices

    # Post-only ratio bounds
    min_post_only_ratio: float = 0.4  # Min 40% post-only
    max_post_only_ratio: float = 0.9  # Max 90% post-only


class ChildSizerV2:
    """Advanced child order sizer with market microstructure awareness."""

    def __init__(self, config: ChildSizerConfig = None):
        self.config = config or ChildSizerConfig()
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

        # Live knobs integration
        try:
            sys.path.insert(0, str(self.base_dir))
            from scripts.exec_knobs import ExecutionKnobs

            self.knobs = ExecutionKnobs()
        except ImportError:
            self.knobs = None

    def get_volume_metrics(
        self, asset: str, market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get volume-based sizing metrics."""

        # Get volume data
        vol_5m = market_data.get("volume_5m", 10000)  # 5-minute volume
        vol_1h = market_data.get("volume_1h", 120000)  # 1-hour volume
        vol_24h = market_data.get("volume_24h", 2000000)  # 24-hour volume

        # Calculate derived metrics
        adv = vol_24h  # Average Daily Volume approximation
        avg_5m_vol = adv / (24 * 12)  # Expected 5-min volume

        # Volume ratio (current vs expected)
        vol_ratio_5m = vol_5m / avg_5m_vol if avg_5m_vol > 0 else 1.0
        vol_ratio_1h = (vol_1h / 12) / avg_5m_vol if avg_5m_vol > 0 else 1.0

        return {
            "volume_5m": vol_5m,
            "volume_1h": vol_1h,
            "volume_24h": vol_24h,
            "adv_estimate": adv,
            "avg_5m_volume": avg_5m_vol,
            "vol_ratio_5m": vol_ratio_5m,
            "vol_ratio_1h": vol_ratio_1h,
            "is_high_volume": vol_ratio_5m > 2.0,
        }

    def get_depth_metrics(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Get order book depth metrics."""

        # Order book depth at various levels
        depth_1 = market_data.get("depth_1", 1000)
        depth_5 = market_data.get("depth_5", 5000)
        depth_10 = market_data.get("depth_10", 10000)

        # Depth quality metrics
        depth_concentration = depth_1 / depth_5 if depth_5 > 0 else 0.2
        depth_slope = (depth_10 - depth_1) / 9 if depth_10 > depth_1 else 0

        return {
            "depth_1": depth_1,
            "depth_5": depth_5,
            "depth_10": depth_10,
            "depth_concentration": depth_concentration,
            "depth_slope": depth_slope,
            "is_thin_book": depth_1 < 500,
            "is_deep_book": depth_10 > 20000,
        }

    def get_volatility_adjustment(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate volatility-based sizing adjustments."""

        vol_1m = market_data.get("volatility_1m", 1.0)
        vol_5m = market_data.get("volatility_5m", 1.0)
        vol_1h = market_data.get("volatility_1h", 1.0)

        # Volatility regime classification
        is_low_vol = vol_5m < 0.8
        is_high_vol = vol_5m > 2.0
        is_extreme_vol = vol_5m > 4.0

        # Size adjustment factors
        if is_extreme_vol:
            size_multiplier = 0.3  # Much smaller slices in extreme vol
            rest_multiplier = 2.0  # Longer waits
        elif is_high_vol:
            size_multiplier = 0.5  # Smaller slices in high vol
            rest_multiplier = 1.5  # Slightly longer waits
        elif is_low_vol:
            size_multiplier = 1.3  # Larger slices in low vol
            rest_multiplier = 0.7  # Shorter waits
        else:
            size_multiplier = 1.0  # Normal sizing
            rest_multiplier = 1.0  # Normal timing

        return {
            "volatility_1m": vol_1m,
            "volatility_5m": vol_5m,
            "volatility_1h": vol_1h,
            "is_low_vol": is_low_vol,
            "is_high_vol": is_high_vol,
            "is_extreme_vol": is_extreme_vol,
            "size_multiplier": size_multiplier,
            "rest_multiplier": rest_multiplier,
        }

    def get_spread_adjustment(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate spread-based adjustments."""

        spread_bps = market_data.get("spread_bps", 10)

        # Spread regime classification
        is_tight = spread_bps < 5
        is_wide = spread_bps > 20
        is_very_wide = spread_bps > 50

        # Adjustment factors
        if is_very_wide:
            post_only_bias = 0.3  # Prefer aggressive in very wide spreads
            size_penalty = 0.7  # Smaller slices
        elif is_wide:
            post_only_bias = 0.5  # Balanced approach
            size_penalty = 0.85  # Slightly smaller slices
        elif is_tight:
            post_only_bias = 0.9  # Strong post-only preference
            size_penalty = 1.1  # Slightly larger slices okay
        else:
            post_only_bias = 0.7  # Normal post-only preference
            size_penalty = 1.0  # Normal sizing

        return {
            "spread_bps": spread_bps,
            "is_tight": is_tight,
            "is_wide": is_wide,
            "is_very_wide": is_very_wide,
            "post_only_bias": post_only_bias,
            "size_penalty": size_penalty,
        }

    def get_kri_adjustments(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Get Key Risk Indicator adjustments."""

        # Simulated KRI data (in practice, would come from risk systems)
        cancel_ratio = market_data.get("cancel_ratio_1h", 0.2)
        adverse_fill_ratio = market_data.get("adverse_fill_ratio", 0.1)
        latency_p95 = market_data.get("latency_p95_ms", 50)

        # Risk-based adjustments
        if cancel_ratio > 0.4:
            # High cancel ratio - be more conservative
            risk_size_multiplier = 0.7
            risk_post_only_bias = 0.8
        elif adverse_fill_ratio > 0.3:
            # High adverse selection - be more careful
            risk_size_multiplier = 0.8
            risk_post_only_bias = 0.9
        elif latency_p95 > 200:
            # High latency - prefer smaller, faster orders
            risk_size_multiplier = 0.6
            risk_post_only_bias = 0.5
        else:
            # Normal risk environment
            risk_size_multiplier = 1.0
            risk_post_only_bias = 0.7

        return {
            "cancel_ratio": cancel_ratio,
            "adverse_fill_ratio": adverse_fill_ratio,
            "latency_p95_ms": latency_p95,
            "risk_size_multiplier": risk_size_multiplier,
            "risk_post_only_bias": risk_post_only_bias,
            "is_high_risk": cancel_ratio > 0.4
            or adverse_fill_ratio > 0.3
            or latency_p95 > 200,
        }

    def calculate_base_slice_size(
        self, total_size: float, asset: str, market_data: Dict[str, Any]
    ) -> float:
        """Calculate base slice size before adjustments."""

        # Get metrics
        volume_metrics = self.get_volume_metrics(asset, market_data)
        depth_metrics = self.get_depth_metrics(market_data)

        # Determine if crypto or equity
        is_crypto = asset.endswith("-USD") and asset != "NVDA"

        # Volume-based sizing
        if is_crypto:
            # Size as percentage of 5-minute volume
            vol_base = volume_metrics["volume_5m"]
            min_pct = self.config.crypto_min_slice_pct
            max_pct = self.config.crypto_max_slice_pct
        else:
            # Size as percentage of ADV
            vol_base = volume_metrics["adv_estimate"] / (24 * 12)  # 5-min equivalent
            min_pct = self.config.equity_min_slice_pct
            max_pct = self.config.equity_max_slice_pct

        vol_based_size = vol_base * ((min_pct + max_pct) / 2)  # Start with midpoint

        # Depth-based sizing
        depth_1 = depth_metrics["depth_1"]
        depth_based_size = depth_1 * (
            (self.config.min_depth_ratio + self.config.max_depth_ratio) / 2
        )

        # Take the more conservative of volume or depth sizing
        base_size = min(vol_based_size, depth_based_size)

        # Ensure it's a reasonable fraction of total order
        max_fraction = 0.1  # Max 10% of total order per slice
        base_size = min(base_size, total_size * max_fraction)

        return max(1, base_size)  # At least 1 unit

    def check_micro_halt_conditions(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check for micro-halt conditions based on M16.1 policy."""

        halt_conditions = {"should_halt": False, "halt_duration_ms": 0, "reasons": []}

        # Get live knob values or defaults
        spread_widen_threshold = (
            self.knobs.get_knob_value("micro_halt.spread_widen_threshold", 1.5)
            if self.knobs
            else 1.5
        )
        spread_halt_duration_ms = (
            self.knobs.get_knob_value("micro_halt.spread_halt_duration_ms", 600)
            if self.knobs
            else 600
        )
        vol_spike_zscore = (
            self.knobs.get_knob_value("micro_halt.vol_spike_zscore", 3.0)
            if self.knobs
            else 3.0
        )
        vol_halt_duration_ms = (
            self.knobs.get_knob_value("micro_halt.vol_halt_duration_ms", 500)
            if self.knobs
            else 500
        )

        # Spread widening check
        current_spread = market_data.get("spread_bps", 10)
        baseline_spread = market_data.get("baseline_spread_bps", current_spread)

        if (
            baseline_spread > 0
            and current_spread > baseline_spread * spread_widen_threshold
        ):
            halt_conditions["should_halt"] = True
            halt_conditions["halt_duration_ms"] = max(
                halt_conditions["halt_duration_ms"], spread_halt_duration_ms
            )
            halt_conditions["reasons"].append(
                f"spread_widening_{current_spread:.1f}bp_vs_{baseline_spread:.1f}bp"
            )

        # Volatility spike check
        vol_1m = market_data.get("volatility_1m", 1.0)
        vol_zscore = market_data.get("volatility_zscore", 0.0)  # Assume provided

        if vol_zscore > vol_spike_zscore:
            halt_conditions["should_halt"] = True
            halt_conditions["halt_duration_ms"] = max(
                halt_conditions["halt_duration_ms"], vol_halt_duration_ms
            )
            halt_conditions["reasons"].append(f"vol_spike_zscore_{vol_zscore:.2f}")

        # Decision latency check
        decision_latency_ms = market_data.get("decision_latency_ms", 0)
        latency_budget_ms = (
            self.knobs.get_knob_value("micro_halt.latency_p95_budget_ms", 120)
            if self.knobs
            else 120
        )

        if decision_latency_ms > latency_budget_ms:
            halt_conditions["should_halt"] = True
            halt_conditions["halt_duration_ms"] = max(
                halt_conditions["halt_duration_ms"], 800
            )
            halt_conditions["reasons"].append(
                f"latency_{decision_latency_ms:.0f}ms_exceeds_{latency_budget_ms}ms"
            )

        return halt_conditions

    def calculate_child_slice(
        self,
        total_size: float,
        asset: str,
        market_data: Dict[str, Any],
        urgency: str = "normal",
    ) -> Dict[str, Any]:
        """Calculate optimal child order slice parameters."""

        # Check micro-halt conditions first
        halt_check = self.check_micro_halt_conditions(market_data)
        if halt_check["should_halt"]:
            return {
                "slice_size": 0.0,
                "action": "MICRO_HALT",
                "halt_duration_ms": halt_check["halt_duration_ms"],
                "halt_reasons": halt_check["reasons"],
                "recommendation": f"Skip trading for {halt_check['halt_duration_ms']}ms due to: {', '.join(halt_check['reasons'])}",
            }

        # Get all metrics
        volume_metrics = self.get_volume_metrics(asset, market_data)
        depth_metrics = self.get_depth_metrics(market_data)
        volatility_metrics = self.get_volatility_adjustment(market_data)
        spread_metrics = self.get_spread_adjustment(market_data)
        kri_metrics = self.get_kri_adjustments(market_data)

        # Calculate base slice size
        base_slice_size = self.calculate_base_slice_size(total_size, asset, market_data)

        # Apply adjustments with live knob integration
        adjusted_size = base_slice_size

        # Get live knob values
        slice_pct_max = (
            self.knobs.get_knob_value("sizer_v2.slice_pct_max", 1.2)
            if self.knobs
            else 1.2
        )
        pov_cap = (
            self.knobs.get_knob_value("sizer_v2.pov_cap", 0.12) if self.knobs else 0.12
        )
        thin_spread_bp = (
            self.knobs.get_knob_value("sizer_v2.thin_spread_bp", 6) if self.knobs else 6
        )
        post_only_base = (
            self.knobs.get_knob_value("sizer_v2.post_only_base", 0.70)
            if self.knobs
            else 0.70
        )

        # Volatility adjustment
        adjusted_size *= volatility_metrics["size_multiplier"]

        # M16.1 Regime-based adjustments
        spread_bps = market_data.get("spread_bps", 10)

        # Thin spread regime: increase maker bias, reduce size
        if spread_bps <= thin_spread_bp:
            adjusted_size *= 0.6  # M16.1: halve slice in tight spreads
            maker_bias_bonus = (
                self.knobs.get_knob_value("sizer_v2.post_only_thin_bonus", 0.15)
                if self.knobs
                else 0.15
            )
        else:
            maker_bias_bonus = 0.0

        # Spread adjustment
        adjusted_size *= spread_metrics["size_penalty"]

        # KRI adjustment
        adjusted_size *= kri_metrics["risk_size_multiplier"]

        # Volume adjustment (increase size in high volume)
        if volume_metrics["is_high_volume"]:
            adjusted_size *= 1.2

        # M16.1 Depth-based halting for thin markets
        depth_p20_threshold = (
            self.knobs.get_knob_value("sizer_v2.depth_p20_threshold", 500)
            if self.knobs
            else 500
        )
        if depth_metrics["depth_1"] < depth_p20_threshold:
            thin_mult = (
                self.knobs.get_knob_value("sizer_v2.thin_market_slice_mult", 0.6)
                if self.knobs
                else 0.6
            )
            adjusted_size *= thin_mult
        elif depth_metrics["is_deep_book"]:
            adjusted_size *= 1.1

        # Apply hard bounds
        is_crypto = asset.endswith("-USD") and asset != "NVDA"

        if is_crypto:
            min_bound = volume_metrics["volume_5m"] * self.config.crypto_min_slice_pct
            max_bound = volume_metrics["volume_5m"] * self.config.crypto_max_slice_pct
        else:
            adv_5m = volume_metrics["adv_estimate"] / (24 * 12)
            min_bound = adv_5m * self.config.equity_min_slice_pct
            max_bound = adv_5m * self.config.equity_max_slice_pct

        # Depth bounds
        min_bound = max(
            min_bound, depth_metrics["depth_1"] * self.config.min_depth_ratio
        )
        max_bound = min(
            max_bound, depth_metrics["depth_1"] * self.config.max_depth_ratio
        )

        # USD bounds
        mid_price = market_data.get("last_price", 100)
        min_bound = max(min_bound, self.config.min_single_slice_usd / mid_price)
        max_bound = min(max_bound, self.config.max_single_slice_usd / mid_price)

        # POV bounds
        max_pov = (
            self.config.max_pov_urgent
            if urgency == "urgent"
            else self.config.max_pov_normal
        )
        pov_bound = volume_metrics["volume_5m"] * max_pov
        max_bound = min(max_bound, pov_bound)

        # Final size
        final_slice_size = np.clip(adjusted_size, min_bound, min(max_bound, total_size))

        # Calculate rest times
        base_rest_ms = (self.config.min_rest_ms + self.config.max_rest_ms) / 2
        rest_adjustment = volatility_metrics["rest_multiplier"]

        if urgency == "urgent":
            rest_adjustment *= 0.5  # Faster execution for urgent orders

        min_rest_ms = max(self.config.min_rest_ms, base_rest_ms * 0.5 * rest_adjustment)
        max_rest_ms = min(self.config.max_rest_ms, base_rest_ms * 1.5 * rest_adjustment)

        # Calculate post-only ratio with M16.1 optimizations
        base_post_only = post_only_base  # Use live knob value

        # M16.1: Thin spread regime gets maker bonus
        if spread_bps <= thin_spread_bp:
            base_post_only = min(0.9, base_post_only + maker_bias_bonus)

        # Adjust based on spread
        post_only_ratio = base_post_only * spread_metrics["post_only_bias"]

        # Adjust based on KRI
        post_only_ratio = (post_only_ratio + kri_metrics["risk_post_only_bias"]) / 2

        # Urgency adjustment
        if urgency == "urgent":
            post_only_ratio *= 0.7  # Less post-only for urgent orders

        # Apply bounds with live knob values
        min_post_only = (
            self.knobs.get_knob_value(
                "sizer_v2.post_only_min", self.config.min_post_only_ratio
            )
            if self.knobs
            else self.config.min_post_only_ratio
        )
        max_post_only = (
            self.knobs.get_knob_value("sizer_v2.post_only_max", 0.90)
            if self.knobs
            else 0.90
        )

        post_only_ratio = np.clip(post_only_ratio, min_post_only, max_post_only)

        return {
            "slice_size": float(final_slice_size),
            "slice_pct_of_order": float(final_slice_size / total_size),
            "slice_pct_of_volume": (
                float(final_slice_size / volume_metrics["volume_5m"])
                if volume_metrics["volume_5m"] > 0
                else 0
            ),
            "slice_pct_of_depth": (
                float(final_slice_size / depth_metrics["depth_1"])
                if depth_metrics["depth_1"] > 0
                else 0
            ),
            "min_rest_ms": float(min_rest_ms),
            "max_rest_ms": float(max_rest_ms),
            "post_only_ratio": float(post_only_ratio),
            "usd_notional": float(final_slice_size * mid_price),
            "adjustments": {
                "volatility_multiplier": volatility_metrics["size_multiplier"],
                "spread_penalty": spread_metrics["size_penalty"],
                "risk_multiplier": kri_metrics["risk_size_multiplier"],
                "volume_boost": 1.2 if volume_metrics["is_high_volume"] else 1.0,
                "depth_adjustment": (
                    0.7
                    if depth_metrics["is_thin_book"]
                    else 1.1 if depth_metrics["is_deep_book"] else 1.0
                ),
            },
            "market_regime": {
                "volatility_regime": (
                    "extreme"
                    if volatility_metrics["is_extreme_vol"]
                    else (
                        "high"
                        if volatility_metrics["is_high_vol"]
                        else "low" if volatility_metrics["is_low_vol"] else "normal"
                    )
                ),
                "spread_regime": (
                    "very_wide"
                    if spread_metrics["is_very_wide"]
                    else (
                        "wide"
                        if spread_metrics["is_wide"]
                        else "tight" if spread_metrics["is_tight"] else "normal"
                    )
                ),
                "volume_regime": (
                    "high" if volume_metrics["is_high_volume"] else "normal"
                ),
                "depth_regime": (
                    "thin"
                    if depth_metrics["is_thin_book"]
                    else "deep" if depth_metrics["is_deep_book"] else "normal"
                ),
                "risk_regime": "high" if kri_metrics["is_high_risk"] else "normal",
            },
        }

    def create_sizing_report(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive sizing analysis report."""

        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "config": {
                "crypto_slice_range": f"{self.config.crypto_min_slice_pct:.1%}-{self.config.crypto_max_slice_pct:.1%}",
                "equity_slice_range": f"{self.config.equity_min_slice_pct:.1%}-{self.config.equity_max_slice_pct:.1%}",
                "depth_range": f"{self.config.min_depth_ratio:.1%}-{self.config.max_depth_ratio:.1%}",
                "pov_limits": f"{self.config.max_pov_normal:.1%} normal, {self.config.max_pov_urgent:.1%} urgent",
            },
            "test_cases": [],
            "summary": {},
        }

        all_slice_pcts = []
        all_post_only_ratios = []
        all_rest_times = []

        for test_case in test_cases:
            total_size = test_case["total_size"]
            asset = test_case["asset"]
            market_data = test_case["market_data"]
            urgency = test_case.get("urgency", "normal")

            sizing_result = self.calculate_child_slice(
                total_size, asset, market_data, urgency
            )

            case_result = {
                "case_name": test_case.get("name", "unnamed"),
                "asset": asset,
                "total_size": total_size,
                "urgency": urgency,
                "market_data": market_data,
                "sizing_result": sizing_result,
            }

            report["test_cases"].append(case_result)

            all_slice_pcts.append(sizing_result["slice_pct_of_order"])
            all_post_only_ratios.append(sizing_result["post_only_ratio"])
            all_rest_times.append(
                (sizing_result["min_rest_ms"] + sizing_result["max_rest_ms"]) / 2
            )

        # Summary statistics
        if all_slice_pcts:
            report["summary"] = {
                "avg_slice_pct": float(np.mean(all_slice_pcts)),
                "min_slice_pct": float(np.min(all_slice_pcts)),
                "max_slice_pct": float(np.max(all_slice_pcts)),
                "avg_post_only_ratio": float(np.mean(all_post_only_ratios)),
                "avg_rest_time_ms": float(np.mean(all_rest_times)),
                "regime_distribution": {},
            }

            # Count regime occurrences
            regimes = [
                "volatility_regime",
                "spread_regime",
                "volume_regime",
                "depth_regime",
                "risk_regime",
            ]
            for regime in regimes:
                regime_counts = {}
                for case in report["test_cases"]:
                    regime_value = case["sizing_result"]["market_regime"][regime]
                    regime_counts[regime_value] = regime_counts.get(regime_value, 0) + 1
                report["summary"]["regime_distribution"][regime] = regime_counts

        return report


def test_child_sizer_v2():
    """Test child sizer v2 with various market conditions."""

    print("📐 Testing Child Sizer v2: Volume/Depth Aware")
    print("=" * 50)

    sizer = ChildSizerV2()

    # Test cases covering different market regimes
    test_cases = [
        {
            "name": "crypto_normal_conditions",
            "asset": "BTC-USD",
            "total_size": 1.0,  # 1 BTC
            "urgency": "normal",
            "market_data": {
                "last_price": 50000,
                "spread_bps": 5.0,
                "volatility_1m": 1.2,
                "volatility_5m": 1.1,
                "volume_5m": 100,
                "volume_1h": 1200,
                "volume_24h": 28800,
                "depth_1": 10,
                "depth_5": 50,
                "depth_10": 100,
            },
        },
        {
            "name": "equity_high_volatility",
            "asset": "NVDA",
            "total_size": 1000,  # 1000 shares
            "urgency": "urgent",
            "market_data": {
                "last_price": 500,
                "spread_bps": 15.0,
                "volatility_1m": 3.5,
                "volatility_5m": 3.2,
                "volume_5m": 50000,
                "volume_1h": 600000,
                "volume_24h": 14400000,
                "depth_1": 2000,
                "depth_5": 8000,
                "depth_10": 15000,
                "cancel_ratio_1h": 0.5,
                "latency_p95_ms": 250,
            },
        },
        {
            "name": "crypto_thin_book",
            "asset": "SOL-USD",
            "total_size": 100,  # 100 SOL
            "urgency": "normal",
            "market_data": {
                "last_price": 100,
                "spread_bps": 35.0,
                "volatility_1m": 2.0,
                "volatility_5m": 1.8,
                "volume_5m": 500,
                "volume_1h": 6000,
                "volume_24h": 144000,
                "depth_1": 200,  # Thin book
                "depth_5": 800,
                "depth_10": 1500,
            },
        },
        {
            "name": "equity_high_volume",
            "asset": "NVDA",
            "total_size": 5000,  # Large order
            "urgency": "normal",
            "market_data": {
                "last_price": 500,
                "spread_bps": 8.0,
                "volatility_1m": 0.8,
                "volatility_5m": 0.9,
                "volume_5m": 200000,  # High volume
                "volume_1h": 2000000,
                "volume_24h": 40000000,
                "depth_1": 5000,
                "depth_5": 25000,
                "depth_10": 50000,
            },
        },
    ]

    # Generate sizing report
    report = sizer.create_sizing_report(test_cases)

    print("\n📊 Child Sizing Results:")
    for case in report["test_cases"]:
        result = case["sizing_result"]

        print(f"\n  {case['case_name']}:")
        print(f"    Asset: {case['asset']} (size: {case['total_size']:,})")
        print(
            f"    Slice: {result['slice_size']:.2f} ({result['slice_pct_of_order']:.1%} of order)"
        )
        print(f"    Volume impact: {result['slice_pct_of_volume']:.1%}")
        print(f"    Depth impact: {result['slice_pct_of_depth']:.1%}")
        print(f"    Post-only ratio: {result['post_only_ratio']:.1%}")
        print(
            f"    Rest time: {result['min_rest_ms']:.0f}-{result['max_rest_ms']:.0f}ms"
        )
        print(f"    USD notional: ${result['usd_notional']:,.0f}")

        regime = result["market_regime"]
        print(
            f"    Regimes: {regime['volatility_regime']} vol, {regime['spread_regime']} spread, {regime['volume_regime']} volume"
        )

    print(f"\n📈 Summary Statistics:")
    summary = report["summary"]
    print(f"  Average slice: {summary['avg_slice_pct']:.1%} of order")
    print(
        f"  Slice range: {summary['min_slice_pct']:.1%} - {summary['max_slice_pct']:.1%}"
    )
    print(f"  Average post-only: {summary['avg_post_only_ratio']:.1%}")
    print(f"  Average rest time: {summary['avg_rest_time_ms']:.0f}ms")

    print(f"\n✅ Child Sizer v2 test complete!")

    return report


if __name__ == "__main__":
    test_child_sizer_v2()
