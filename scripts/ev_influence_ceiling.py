#!/usr/bin/env python3
"""
M18: EV Influence Ceiling Mapper
Maps EV band scores to maximum allowable influence levels for 20% ramp.
Ensures we only increase influence during high-confidence profitable windows.
"""
import os
import sys
import json
import datetime
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


class EVInfluenceCeiling:
    """Map EV bands to influence ceilings for safe 20% ramp."""

    def __init__(self):
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

        # EV band to influence ceiling mapping
        self.ceiling_config = {
            # High-confidence green windows
            "green_hi": {
                "max_influence_pct": 20.0,  # Full 20% allowed
                "ev_threshold_min": 15.0,  # $15+/hour EV
                "confidence_min": 0.85,  # 85%+ confidence
                "description": "High-confidence profitable windows",
            },
            # Standard green windows
            "green_lo": {
                "max_influence_pct": 17.0,  # Moderate 17%
                "ev_threshold_min": 9.4,  # $9.4+/hour EV (current threshold)
                "confidence_min": 0.70,  # 70%+ confidence
                "description": "Standard profitable windows",
            },
            # Marginal/amber windows
            "amber": {
                "max_influence_pct": 12.0,  # Conservative 12%
                "ev_threshold_min": 5.0,  # $5+/hour EV
                "confidence_min": 0.60,  # 60%+ confidence
                "description": "Marginal profitability windows",
            },
            # Unprofitable/red windows
            "red": {
                "max_influence_pct": 0.0,  # No influence
                "ev_threshold_min": -999.0,  # Any negative EV
                "confidence_min": 0.0,  # Any confidence
                "description": "Unprofitable windows - no trading",
            },
        }

        # Dynamic adjustment factors
        self.adjustment_factors = {
            "market_volatility": {
                "low": 1.0,  # No adjustment
                "medium": 0.9,  # 10% reduction
                "high": 0.8,  # 20% reduction
            },
            "execution_quality": {
                "excellent": 1.1,  # 10% increase
                "good": 1.0,  # No adjustment
                "poor": 0.8,  # 20% reduction
            },
            "risk_budget": {
                "ample": 1.0,  # No adjustment
                "moderate": 0.9,  # 10% reduction
                "limited": 0.7,  # 30% reduction
            },
        }

    def classify_ev_window(
        self, ev_usd_per_hour: float, confidence: float, asset: str = None
    ) -> str:
        """Classify EV window into band based on value and confidence."""

        # Check each band in order of preference
        for band, config in self.ceiling_config.items():
            if (
                ev_usd_per_hour >= config["ev_threshold_min"]
                and confidence >= config["confidence_min"]
            ):
                return band

        # Default to red if no criteria met
        return "red"

    def get_base_influence_ceiling(self, ev_band: str) -> float:
        """Get base influence ceiling for EV band."""

        if ev_band in self.ceiling_config:
            return self.ceiling_config[ev_band]["max_influence_pct"]
        else:
            # Unknown band - conservative default
            return 0.0

    def assess_market_conditions(self) -> Dict[str, str]:
        """Assess current market conditions for ceiling adjustments."""

        # In production, this would query real market data
        # For now, simulate realistic conditions

        conditions = {
            "market_volatility": "medium",  # Could be low/medium/high
            "execution_quality": "good",  # Could be excellent/good/poor
            "risk_budget": "ample",  # Could be ample/moderate/limited
        }

        # Try to load from recent execution metrics
        try:
            # Check recent slippage performance
            exec_dir = self.base_dir / "artifacts" / "exec"
            slip_gate_file = exec_dir / "slip_gate_ok"

            if slip_gate_file.exists():
                with open(slip_gate_file, "r") as f:
                    slip_data = json.load(f)

                p95_slip = slip_data.get("p95_slippage_bps", 15)
                maker_ratio = slip_data.get("maker_ratio", 0.75)

                # Classify execution quality
                if p95_slip <= 10 and maker_ratio >= 0.85:
                    conditions["execution_quality"] = "excellent"
                elif p95_slip <= 15 and maker_ratio >= 0.75:
                    conditions["execution_quality"] = "good"
                else:
                    conditions["execution_quality"] = "poor"

        except Exception:
            pass

        # Simulate volatility based on asset
        volatility_sim = np.random.choice(["low", "medium", "high"], p=[0.3, 0.5, 0.2])
        conditions["market_volatility"] = volatility_sim

        return conditions

    def calculate_adjusted_ceiling(
        self, base_ceiling: float, market_conditions: Dict[str, str]
    ) -> float:
        """Calculate adjusted influence ceiling based on market conditions."""

        if base_ceiling == 0.0:
            return 0.0  # Never adjust red windows

        adjustment_factor = 1.0

        # Apply each adjustment factor
        for condition_type, condition_value in market_conditions.items():
            if condition_type in self.adjustment_factors:
                factor_map = self.adjustment_factors[condition_type]
                if condition_value in factor_map:
                    adjustment_factor *= factor_map[condition_value]

        adjusted_ceiling = base_ceiling * adjustment_factor

        # Ensure adjusted ceiling doesn't exceed original base
        return min(base_ceiling, max(0.0, adjusted_ceiling))

    def get_influence_ceiling(
        self,
        ev_usd_per_hour: float,
        confidence: float,
        asset: str = None,
        apply_adjustments: bool = True,
    ) -> Dict[str, Any]:
        """Get influence ceiling for given EV window with full context."""

        # Classify EV window
        ev_band = self.classify_ev_window(ev_usd_per_hour, confidence, asset)

        # Get base ceiling
        base_ceiling = self.get_base_influence_ceiling(ev_band)

        # Apply market condition adjustments if requested
        adjusted_ceiling = base_ceiling
        market_conditions = {}

        if apply_adjustments and base_ceiling > 0:
            market_conditions = self.assess_market_conditions()
            adjusted_ceiling = self.calculate_adjusted_ceiling(
                base_ceiling, market_conditions
            )

        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "asset": asset,
            "ev_usd_per_hour": ev_usd_per_hour,
            "confidence": confidence,
            "ev_band": ev_band,
            "base_ceiling_pct": base_ceiling,
            "adjusted_ceiling_pct": adjusted_ceiling,
            "market_conditions": market_conditions,
            "band_config": self.ceiling_config[ev_band],
            "adjustment_applied": adjusted_ceiling != base_ceiling,
        }

    def process_ev_calendar(self, calendar_file: str) -> pd.DataFrame:
        """Process EV calendar and add influence ceilings."""

        try:
            # Load calendar
            calendar_df = pd.read_parquet(calendar_file)
            calendar_df["timestamp"] = pd.to_datetime(
                calendar_df["timestamp"], utc=True
            )

            # Add ceiling columns
            ceiling_results = []

            for _, row in calendar_df.iterrows():
                ev_value = row.get("ev_usd_per_hour", 0)
                confidence = 0.8  # Default confidence if not in calendar
                asset = row.get("asset", "BTC-USD")

                ceiling_info = self.get_influence_ceiling(ev_value, confidence, asset)
                ceiling_results.append(ceiling_info)

            # Add ceiling data to dataframe
            ceiling_df = pd.DataFrame(ceiling_results)

            calendar_with_ceilings = calendar_df.copy()
            calendar_with_ceilings["ev_band"] = ceiling_df["ev_band"]
            calendar_with_ceilings["base_ceiling_pct"] = ceiling_df["base_ceiling_pct"]
            calendar_with_ceilings["adjusted_ceiling_pct"] = ceiling_df[
                "adjusted_ceiling_pct"
            ]
            calendar_with_ceilings["market_conditions"] = ceiling_df[
                "market_conditions"
            ].apply(json.dumps)

            print(f"📊 Processed {len(calendar_with_ceilings)} calendar windows")
            print(
                f"   Band distribution: {calendar_with_ceilings['ev_band'].value_counts().to_dict()}"
            )

            return calendar_with_ceilings

        except Exception as e:
            print(f"❌ Error processing calendar: {e}")
            return pd.DataFrame()

    def get_current_influence_ceiling(
        self, asset: str, calendar_file: str = None
    ) -> Dict[str, Any]:
        """Get current influence ceiling for asset based on live EV assessment."""

        current_time = datetime.datetime.now(datetime.timezone.utc)

        # Try to get from live calendar if provided
        if calendar_file:
            try:
                calendar_df = pd.read_parquet(calendar_file)
                calendar_df["timestamp"] = pd.to_datetime(
                    calendar_df["timestamp"], utc=True
                )

                # Find current or next window for this asset
                asset_windows = calendar_df[calendar_df["asset"] == asset]
                upcoming_windows = asset_windows[
                    asset_windows["timestamp"] >= current_time
                ]

                if not upcoming_windows.empty:
                    next_window = upcoming_windows.iloc[0]
                    ev_value = next_window.get("ev_usd_per_hour", 0)

                    # Get ceiling for this window
                    ceiling_info = self.get_influence_ceiling(ev_value, 0.8, asset)
                    ceiling_info["calendar_source"] = True
                    ceiling_info["next_window_time"] = next_window[
                        "timestamp"
                    ].isoformat()

                    return ceiling_info

            except Exception as e:
                print(f"⚠️ Error reading calendar: {e}")

        # Fallback: conservative ceiling
        return self.get_influence_ceiling(5.0, 0.6, asset)  # Amber level

    def save_ceiling_analysis(self, output_dir: str, calendar_file: str = None) -> str:
        """Save detailed ceiling analysis to output directory."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%SZ")

        # Basic analysis
        analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "config": {
                "ceiling_config": self.ceiling_config,
                "adjustment_factors": self.adjustment_factors,
            },
            "market_conditions": self.assess_market_conditions(),
        }

        # Process calendar if provided
        if calendar_file and Path(calendar_file).exists():
            calendar_with_ceilings = self.process_ev_calendar(calendar_file)

            if not calendar_with_ceilings.empty:
                # Save enhanced calendar
                calendar_output = (
                    output_path / f"calendar_with_ceilings_{timestamp_str}.parquet"
                )
                calendar_with_ceilings.to_parquet(calendar_output, index=False)

                # Add summary statistics
                analysis["calendar_analysis"] = {
                    "total_windows": len(calendar_with_ceilings),
                    "band_distribution": calendar_with_ceilings["ev_band"]
                    .value_counts()
                    .to_dict(),
                    "avg_ceiling_by_band": calendar_with_ceilings.groupby("ev_band")[
                        "adjusted_ceiling_pct"
                    ]
                    .mean()
                    .to_dict(),
                    "max_ceiling_available": calendar_with_ceilings[
                        "adjusted_ceiling_pct"
                    ].max(),
                    "calendar_file": str(calendar_output),
                }

        # Current asset ceilings
        assets = ["BTC-USD", "ETH-USD", "SOL-USD", "NVDA"]
        analysis["current_asset_ceilings"] = {}

        for asset in assets:
            ceiling_info = self.get_current_influence_ceiling(asset, calendar_file)
            analysis["current_asset_ceilings"][asset] = ceiling_info

        # Save analysis
        analysis_file = output_path / f"ev_ceiling_analysis_{timestamp_str}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)

        print(f"💾 Saved ceiling analysis: {analysis_file}")
        return str(analysis_file)


def main():
    """Test EV influence ceiling system."""

    parser = argparse.ArgumentParser(description="M18: EV Influence Ceiling Mapper")
    parser.add_argument("--calendar", help="EV calendar file to process")
    parser.add_argument(
        "--out", default="artifacts/ev_ceiling", help="Output directory"
    )
    parser.add_argument("--asset", default="BTC-USD", help="Asset to check")
    args = parser.parse_args()

    print("🎯 M18: EV Influence Ceiling Test")
    print("=" * 40)

    # Initialize ceiling mapper
    ceiling_mapper = EVInfluenceCeiling()

    # Test individual window classifications
    print("📊 EV Window Classification Examples:")
    test_windows = [
        {"ev": 20.0, "conf": 0.9, "desc": "High EV, high confidence"},
        {"ev": 12.0, "conf": 0.8, "desc": "Medium EV, good confidence"},
        {"ev": 7.0, "conf": 0.7, "desc": "Low EV, moderate confidence"},
        {"ev": -2.0, "conf": 0.9, "desc": "Negative EV, high confidence"},
    ]

    for test in test_windows:
        ceiling_info = ceiling_mapper.get_influence_ceiling(
            test["ev"], test["conf"], args.asset
        )
        print(f"  {test['desc']}:")
        print(f"    EV: ${test['ev']}/h, Conf: {test['conf']:.1%}")
        print(f"    Band: {ceiling_info['ev_band']}")
        print(
            f"    Ceiling: {ceiling_info['base_ceiling_pct']:.1f}% -> {ceiling_info['adjusted_ceiling_pct']:.1f}%"
        )

    # Test current ceiling for specific asset
    print(f"\n🎯 Current Ceiling for {args.asset}:")
    current_ceiling = ceiling_mapper.get_current_influence_ceiling(
        args.asset, args.calendar
    )
    print(f"  Band: {current_ceiling['ev_band']}")
    print(f"  Ceiling: {current_ceiling['adjusted_ceiling_pct']:.1f}%")
    print(f"  Market Conditions: {current_ceiling.get('market_conditions', {})}")

    # Save full analysis
    print(f"\n💾 Saving ceiling analysis...")
    analysis_file = ceiling_mapper.save_ceiling_analysis(args.out, args.calendar)
    print(f"✅ Ceiling analysis complete: {analysis_file}")


if __name__ == "__main__":
    main()
