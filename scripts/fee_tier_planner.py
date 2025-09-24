#!/usr/bin/env python3
"""
Fee-Tier Planner: Venue Fee Structure Optimization
Plan fee-tier upgrades with ROI analysis to minimize transaction costs.
"""
import os
import sys
import json
import datetime
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FeeTier:
    """Venue fee tier structure."""

    tier_name: str
    maker_fee_bps: float
    taker_fee_bps: float
    min_volume_30d: float  # USD volume requirement
    rebate_bps: float  # Negative = earning rebates
    monthly_cost: float  # Fixed monthly fee


@dataclass
class VolumeProjection:
    """Projected trading volume for ROI analysis."""

    asset: str
    venue: str
    current_30d_volume: float
    projected_30d_volume: float
    current_maker_ratio: float
    projected_maker_ratio: float


@dataclass
class TierUpgradeRecommendation:
    """Fee tier upgrade recommendation."""

    venue: str
    current_tier: str
    recommended_tier: str
    monthly_savings: float
    roi_months: float
    volume_requirement: float
    current_volume: float
    meets_volume: bool
    reason: str


class FeeTierPlanner:
    def __init__(self):
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

        # Fee tier structures for major venues
        self.venue_tiers = {
            "coinbase": [
                FeeTier("retail", 0.6, 0.6, 0, 0.0, 0),
                FeeTier("advanced", 0.35, 0.6, 10000, 0.0, 0),
                FeeTier("prime", 0.50, 0.50, 100000, -0.5, 0),  # Small rebate
                FeeTier("prime_plus", 0.25, 0.40, 1000000, -1.0, 0),  # Better rebates
            ],
            "binance": [
                FeeTier("vip0", 1.0, 1.0, 0, 0.0, 0),
                FeeTier("vip1", 0.90, 1.0, 50000, -0.5, 0),
                FeeTier("vip2", 0.80, 0.90, 250000, -1.0, 0),
                FeeTier("vip3", 0.70, 0.80, 1000000, -1.5, 0),
                FeeTier("vip4", 0.50, 0.70, 5000000, -2.0, 0),
            ],
            "alpaca": [
                FeeTier("unlimited", 0.35, 0.35, 0, 0.0, 0),
                FeeTier("pro", 0.25, 0.35, 25000, -0.2, 99),  # Monthly fee
                FeeTier(
                    "pro_plus", 0.15, 0.25, 100000, -0.5, 299
                ),  # Higher monthly fee
            ],
        }

        # Current tier assumptions
        self.current_tiers = {
            "coinbase": "retail",
            "binance": "vip0",
            "alpaca": "unlimited",
        }

    def load_historical_volume(self) -> Dict[str, VolumeProjection]:
        """Load historical volume data and project future volume."""

        # In production, this would query actual execution logs
        # For now, simulate realistic volume patterns

        projections = {}
        assets = ["SOL-USD", "BTC-USD", "ETH-USD", "NVDA"]
        venues = ["coinbase", "binance", "alpaca"]

        for asset in assets:
            for venue in venues:
                # Skip invalid combinations
                if venue == "binance" and asset == "NVDA":
                    continue
                if venue == "alpaca" and asset != "NVDA":
                    continue

                # Simulate current volume based on asset/venue
                if asset == "NVDA" and venue == "alpaca":
                    # Stock trading typically lower volume
                    base_volume = np.random.uniform(50000, 150000)
                elif asset != "NVDA" and venue in ["coinbase", "binance"]:
                    # Crypto trading higher volume
                    base_volume = np.random.uniform(200000, 800000)
                else:
                    base_volume = np.random.uniform(25000, 100000)

                # Project 20-50% growth based on M13 optimizations
                growth_factor = np.random.uniform(1.2, 1.5)
                projected_volume = base_volume * growth_factor

                # Current maker ratio from rebate exporter data
                current_maker_ratio = np.random.uniform(0.4, 0.8)

                # Project improved maker ratio from M13 maker/taker controller
                projected_maker_ratio = min(0.95, current_maker_ratio + 0.1)

                key = f"{asset}_{venue}"
                projections[key] = VolumeProjection(
                    asset=asset,
                    venue=venue,
                    current_30d_volume=base_volume,
                    projected_30d_volume=projected_volume,
                    current_maker_ratio=current_maker_ratio,
                    projected_maker_ratio=projected_maker_ratio,
                )

        return projections

    def calculate_fee_costs(
        self, tier: FeeTier, volume: float, maker_ratio: float
    ) -> float:
        """Calculate monthly fee costs for a tier given volume and maker ratio."""

        maker_volume = volume * maker_ratio
        taker_volume = volume * (1 - maker_ratio)

        # Calculate fees/rebates (rebate_bps is negative for earnings)
        maker_cost = maker_volume * (tier.maker_fee_bps + tier.rebate_bps) / 10000
        taker_cost = taker_volume * tier.taker_fee_bps / 10000

        total_variable_cost = maker_cost + taker_cost
        total_cost = total_variable_cost + tier.monthly_cost

        return total_cost

    def find_optimal_tier(
        self, venue: str, volume: float, maker_ratio: float
    ) -> Tuple[FeeTier, float]:
        """Find the optimal fee tier for given volume and maker ratio."""

        tiers = self.venue_tiers.get(venue, [])
        if not tiers:
            raise ValueError(f"No fee tiers defined for venue: {venue}")

        best_tier = tiers[0]
        best_cost = float("inf")

        for tier in tiers:
            # Check if volume requirement is met
            if volume >= tier.min_volume_30d:
                cost = self.calculate_fee_costs(tier, volume, maker_ratio)
                if cost < best_cost:
                    best_cost = cost
                    best_tier = tier

        return best_tier, best_cost

    def analyze_tier_upgrade(
        self, venue: str, projection: VolumeProjection
    ) -> TierUpgradeRecommendation:
        """Analyze fee tier upgrade opportunity for a venue."""

        current_tier_name = self.current_tiers.get(venue, "")
        tiers = self.venue_tiers.get(venue, [])

        # Find current tier
        current_tier = None
        for tier in tiers:
            if tier.tier_name == current_tier_name:
                current_tier = tier
                break

        if not current_tier:
            current_tier = tiers[0]  # Default to first tier

        # Calculate current costs
        current_cost = self.calculate_fee_costs(
            current_tier,
            projection.projected_30d_volume,
            projection.projected_maker_ratio,
        )

        # Find optimal tier for projected volume
        optimal_tier, optimal_cost = self.find_optimal_tier(
            venue, projection.projected_30d_volume, projection.projected_maker_ratio
        )

        # Calculate savings and ROI
        monthly_savings = current_cost - optimal_cost

        # Calculate upgrade cost (difference in monthly fees)
        upgrade_cost = optimal_tier.monthly_cost - current_tier.monthly_cost

        # ROI calculation
        if monthly_savings > 0 and upgrade_cost > 0:
            roi_months = upgrade_cost / monthly_savings
        elif monthly_savings > 0:
            roi_months = 0  # Immediate savings, no upgrade cost
        else:
            roi_months = float("inf")  # No savings

        # Volume requirement check
        meets_volume = projection.projected_30d_volume >= optimal_tier.min_volume_30d

        # Generate recommendation reason
        reasons = []
        if optimal_tier.tier_name != current_tier.tier_name:
            reasons.append(f"upgrade_to_{optimal_tier.tier_name}")
        if monthly_savings > 100:
            reasons.append(f"savings_${monthly_savings:.0f}/month")
        if not meets_volume:
            reasons.append(
                f"volume_gap_${optimal_tier.min_volume_30d - projection.projected_30d_volume:.0f}"
            )
        if roi_months <= 3:
            reasons.append("fast_roi")

        reason = "; ".join(reasons) if reasons else "no_upgrade_needed"

        return TierUpgradeRecommendation(
            venue=venue,
            current_tier=current_tier.tier_name,
            recommended_tier=optimal_tier.tier_name,
            monthly_savings=monthly_savings,
            roi_months=roi_months,
            volume_requirement=optimal_tier.min_volume_30d,
            current_volume=projection.projected_30d_volume,
            meets_volume=meets_volume,
            reason=reason,
        )

    def generate_fee_plan(self, dry_run: bool = True) -> Dict[str, Any]:
        """Generate comprehensive fee tier optimization plan."""

        print("💰 Fee-Tier Planner: Venue Cost Optimization")
        print("=" * 60)

        # Load volume projections
        projections = self.load_historical_volume()

        # Analyze each venue
        recommendations = {}
        total_potential_savings = 0

        for venue in ["coinbase", "binance", "alpaca"]:
            # Aggregate volume for venue across all assets
            venue_projections = [p for p in projections.values() if p.venue == venue]

            if not venue_projections:
                continue

            # Calculate aggregate metrics
            total_volume = sum(p.projected_30d_volume for p in venue_projections)
            avg_maker_ratio = np.mean(
                [p.projected_maker_ratio for p in venue_projections]
            )

            # Create aggregate projection for venue analysis
            venue_projection = VolumeProjection(
                asset="aggregate",
                venue=venue,
                current_30d_volume=sum(p.current_30d_volume for p in venue_projections),
                projected_30d_volume=total_volume,
                current_maker_ratio=avg_maker_ratio,
                projected_maker_ratio=avg_maker_ratio,
            )

            # Analyze tier upgrade
            recommendation = self.analyze_tier_upgrade(venue, venue_projection)
            recommendations[venue] = recommendation

            if recommendation.monthly_savings > 0:
                total_potential_savings += recommendation.monthly_savings

            # Print venue analysis
            print(f"\n📊 {venue.upper()} Analysis:")
            print(f"  Current Tier: {recommendation.current_tier}")
            print(f"  Recommended: {recommendation.recommended_tier}")
            print(f"  Volume (30d): ${recommendation.current_volume:,.0f}")
            print(f"  Requirement: ${recommendation.volume_requirement:,.0f}")
            print(f"  Meets Volume: {'✅' if recommendation.meets_volume else '❌'}")
            print(f"  Monthly Savings: ${recommendation.monthly_savings:.2f}")
            print(f"  ROI (months): {recommendation.roi_months:.1f}")
            print(f"  Reason: {recommendation.reason}")

        # Generate implementation plan
        implementation_actions = []
        priority_venues = []

        for venue, rec in recommendations.items():
            if rec.monthly_savings > 50 and rec.meets_volume and rec.roi_months <= 6:
                priority_venues.append(venue)
                implementation_actions.append(
                    {
                        "venue": venue,
                        "action": f"upgrade_to_{rec.recommended_tier}",
                        "savings": rec.monthly_savings,
                        "roi_months": rec.roi_months,
                        "priority": "high",
                    }
                )
            elif rec.monthly_savings > 20:
                implementation_actions.append(
                    {
                        "venue": venue,
                        "action": f"plan_upgrade_to_{rec.recommended_tier}",
                        "savings": rec.monthly_savings,
                        "volume_gap": rec.volume_requirement - rec.current_volume,
                        "priority": "medium",
                    }
                )

        # Create audit record
        plan_data = {
            "timestamp": datetime.datetime.now().isoformat() + "Z",
            "analysis_type": "fee_tier_optimization",
            "dry_run": dry_run,
            "total_potential_savings_monthly": total_potential_savings,
            "venue_recommendations": {
                venue: {
                    "current_tier": rec.current_tier,
                    "recommended_tier": rec.recommended_tier,
                    "monthly_savings": rec.monthly_savings,
                    "roi_months": rec.roi_months,
                    "volume_requirement": rec.volume_requirement,
                    "current_volume": rec.current_volume,
                    "meets_volume": rec.meets_volume,
                    "reason": rec.reason,
                }
                for venue, rec in recommendations.items()
            },
            "implementation_actions": implementation_actions,
            "priority_venues": priority_venues,
        }

        # Save plan to artifacts
        output_dir = self.base_dir / "artifacts" / "fee_planning"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plan_file = output_dir / f"fee_tier_plan_{timestamp}.json"

        with open(plan_file, "w") as f:
            json.dump(plan_data, f, indent=2)

        # Create latest symlink
        latest_file = output_dir / "latest.json"
        if latest_file.exists() or latest_file.is_symlink():
            latest_file.unlink()
        latest_file.symlink_to(plan_file.name)

        print(f"\n💰 Fee Tier Optimization Summary:")
        print(f"  Total Potential Savings: ${total_potential_savings:.2f}/month")
        print(
            f"  Priority Venues: {', '.join(priority_venues) if priority_venues else 'None'}"
        )
        print(
            f"  High-ROI Actions: {len([a for a in implementation_actions if a.get('priority') == 'high'])}"
        )
        print(f"  Plan saved: {plan_file}")

        if total_potential_savings > 100:
            print(f"\n💡 Significant savings opportunity identified!")
            print(f"   Annual savings potential: ${total_potential_savings * 12:.2f}")

        return plan_data

    def simulate_cost_scenarios(self) -> Dict[str, Any]:
        """Simulate different cost scenarios for sensitivity analysis."""

        print("\n🧪 Cost Scenario Simulation")
        print("-" * 40)

        scenarios = {
            "conservative": {"volume_growth": 1.1, "maker_improvement": 0.05},
            "base_case": {"volume_growth": 1.3, "maker_improvement": 0.1},
            "optimistic": {"volume_growth": 1.5, "maker_improvement": 0.15},
        }

        scenario_results = {}

        for scenario_name, params in scenarios.items():
            print(f"\n📈 {scenario_name.title()} Scenario:")

            # Modify projections for scenario
            projections = self.load_historical_volume()
            for proj in projections.values():
                proj.projected_30d_volume *= params["volume_growth"]
                proj.projected_maker_ratio = min(
                    0.95, proj.current_maker_ratio + params["maker_improvement"]
                )

            # Calculate savings for scenario
            total_savings = 0
            for venue in ["coinbase", "binance", "alpaca"]:
                venue_projections = [
                    p for p in projections.values() if p.venue == venue
                ]
                if venue_projections:
                    total_volume = sum(
                        p.projected_30d_volume for p in venue_projections
                    )
                    avg_maker_ratio = np.mean(
                        [p.projected_maker_ratio for p in venue_projections]
                    )

                    venue_projection = VolumeProjection(
                        asset="aggregate",
                        venue=venue,
                        current_30d_volume=sum(
                            p.current_30d_volume for p in venue_projections
                        ),
                        projected_30d_volume=total_volume,
                        current_maker_ratio=avg_maker_ratio,
                        projected_maker_ratio=avg_maker_ratio,
                    )

                    recommendation = self.analyze_tier_upgrade(venue, venue_projection)
                    if recommendation.monthly_savings > 0:
                        total_savings += recommendation.monthly_savings

            scenario_results[scenario_name] = {
                "monthly_savings": total_savings,
                "annual_savings": total_savings * 12,
                "volume_growth": params["volume_growth"],
                "maker_improvement": params["maker_improvement"],
            }

            print(f"  Monthly savings: ${total_savings:.2f}")
            print(f"  Annual savings: ${total_savings * 12:.2f}")

        return scenario_results


def main():
    """Main fee tier planner function."""
    parser = argparse.ArgumentParser(
        description="Fee-Tier Planner: Venue Cost Optimization"
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (default)")
    parser.add_argument(
        "--scenarios", action="store_true", help="Run cost scenario analysis"
    )
    args = parser.parse_args()

    try:
        planner = FeeTierPlanner()

        # Generate fee optimization plan
        plan_data = planner.generate_fee_plan(
            dry_run=True
        )  # Always dry run for planning

        # Run scenario analysis if requested
        if args.scenarios:
            scenario_results = planner.simulate_cost_scenarios()
            plan_data["scenarios"] = scenario_results

        print(f"\n✅ Fee tier analysis complete!")
        print(f"📊 View plan: cat artifacts/fee_planning/latest.json")

        if plan_data["total_potential_savings_monthly"] > 100:
            print(
                f"💰 High-impact opportunity: ${plan_data['total_potential_savings_monthly']:.2f}/month savings"
            )
            print(f"🎯 Priority venues: {', '.join(plan_data['priority_venues'])}")

        return 0

    except Exception as e:
        print(f"❌ Fee tier planner failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
