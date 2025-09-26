#!/usr/bin/env python3
"""
Execution Grid Sweep
Grid search over execution parameters to optimize TCA metrics (IS, slippage, fill ratio).
"""
import os
import sys
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import itertools
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import concurrent.futures


@dataclass
class ExecConfig:
    """Execution configuration parameters."""

    strategy: str  # "TWAP", "VWAP", "POV"
    slice_pct: float  # % of parent order per slice
    participation_rate: float  # % of market volume
    post_only_prob: float  # Probability of post-only orders
    max_spread_bps: float  # Max spread to cross
    urgency: float  # 0-1 urgency factor


@dataclass
class TCAResults:
    """Transaction Cost Analysis results."""

    implementation_shortfall_bps: float
    slippage_p95_bps: float
    fill_ratio: float
    num_fills: int
    avg_fill_size: float
    total_volume: float
    execution_time_sec: float


class ExecutionSimulator:
    """Simulate execution with different parameters."""

    def __init__(self):
        # Market simulation parameters
        self.base_spread_bps = 8.5
        self.volatility = 0.015
        self.typical_volume_per_minute = 2500

    def generate_market_data(self, duration_minutes: int) -> pd.DataFrame:
        """Generate synthetic market data for simulation."""
        timestamps = pd.date_range(
            start=datetime.now()
            - datetime.timedelta(minutes=duration_minutes),
            end=datetime.now(),
            freq="1min",
        )

        # Price random walk
        price_changes = np.random.normal(
            0, self.volatility / np.sqrt(1440), len(timestamps)
        )
        prices = 50000 * np.exp(np.cumsum(price_changes))

        # Volume with mean reversion and time-of-day effects
        base_volume = self.typical_volume_per_minute
        volume_noise = np.random.gamma(2, base_volume / 2, len(timestamps))

        # Spread dynamics (wider in low volume, volatile periods)
        vol_factor = np.abs(price_changes) / (self.volatility / np.sqrt(1440))
        volume_factor = base_volume / (volume_noise + 1)
        spread_bps = self.base_spread_bps * (1 + vol_factor * 0.5 + volume_factor * 0.3)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
                "volume": volume_noise,
                "spread_bps": spread_bps,
                "bid": prices * (1 - spread_bps / 20000),
                "ask": prices * (1 + spread_bps / 20000),
            }
        )

    def simulate_twap_execution(
        self,
        config: ExecConfig,
        market_data: pd.DataFrame,
        order_size: float,
        side: str,
    ) -> TCAResults:
        """Simulate TWAP execution strategy."""

        # Calculate slices
        slice_size = order_size * config.slice_pct
        num_slices = int(np.ceil(order_size / slice_size))

        # Execution timing (spread evenly)
        execution_minutes = min(len(market_data), max(num_slices, 10))
        slice_intervals = np.linspace(0, execution_minutes - 1, num_slices).astype(int)

        fills = []
        remaining_size = order_size

        for i, minute_idx in enumerate(slice_intervals):
            if remaining_size <= 0:
                break

            current_slice = min(slice_size, remaining_size)
            market_row = market_data.iloc[minute_idx]

            # Execution logic
            should_post = np.random.random() < config.post_only_prob
            spread_too_wide = market_row["spread_bps"] > config.max_spread_bps

            if should_post and not spread_too_wide:
                # Post-only execution (better price, but fill risk)
                fill_prob = self._calculate_fill_probability(
                    current_slice, market_row["volume"], config
                )

                if np.random.random() < fill_prob:
                    # Successful fill at better price
                    if side == "buy":
                        fill_price = market_row["bid"]
                        execution_cost_bps = 0  # No spread crossing
                    else:
                        fill_price = market_row["ask"]
                        execution_cost_bps = 0

                    slippage_bps = abs(
                        (fill_price - market_row["price"]) / market_row["price"] * 10000
                    )

                    fills.append(
                        {
                            "size": current_slice,
                            "price": fill_price,
                            "timestamp": market_row["timestamp"],
                            "is_bps": execution_cost_bps,
                            "slippage_bps": slippage_bps,
                        }
                    )

                    remaining_size -= current_slice

                # If no fill, size rolls to next interval (partial fill handling)

            else:
                # Cross spread (guaranteed fill but higher cost)
                if side == "buy":
                    fill_price = market_row["ask"]
                    execution_cost_bps = market_row["spread_bps"] / 2
                else:
                    fill_price = market_row["bid"]
                    execution_cost_bps = market_row["spread_bps"] / 2

                # Market impact
                impact_bps = self._calculate_market_impact(
                    current_slice, market_row["volume"], config
                )

                total_cost_bps = execution_cost_bps + impact_bps
                slippage_bps = total_cost_bps + np.random.normal(0, 2)  # Noise

                fills.append(
                    {
                        "size": current_slice,
                        "price": fill_price,
                        "timestamp": market_row["timestamp"],
                        "is_bps": total_cost_bps,
                        "slippage_bps": abs(slippage_bps),
                    }
                )

                remaining_size -= current_slice

        return self._calculate_tca_metrics(fills, order_size, market_data)

    def simulate_vwap_execution(
        self,
        config: ExecConfig,
        market_data: pd.DataFrame,
        order_size: float,
        side: str,
    ) -> TCAResults:
        """Simulate VWAP execution strategy."""

        # VWAP sizing follows volume distribution
        total_market_volume = market_data["volume"].sum()

        fills = []
        remaining_size = order_size

        for _, market_row in market_data.iterrows():
            if remaining_size <= 0:
                break

            # Size based on market volume proportion
            volume_proportion = market_row["volume"] / total_market_volume
            target_slice = (
                order_size * volume_proportion * 10
            )  # Scale up for reasonable execution
            actual_slice = min(target_slice, remaining_size)

            if actual_slice < order_size * 0.01:  # Skip very small slices
                continue

            # Participation rate constraint
            max_participation_slice = market_row["volume"] * config.participation_rate
            actual_slice = min(actual_slice, max_participation_slice)

            if actual_slice <= 0:
                continue

            # Execution similar to TWAP but volume-weighted
            should_post = np.random.random() < config.post_only_prob

            if should_post:
                fill_prob = self._calculate_fill_probability(
                    actual_slice, market_row["volume"], config
                )
                if np.random.random() < fill_prob:
                    if side == "buy":
                        fill_price = market_row["bid"]
                        execution_cost_bps = -market_row["spread_bps"] / 4  # Rebate
                    else:
                        fill_price = market_row["ask"]
                        execution_cost_bps = -market_row["spread_bps"] / 4

                    fills.append(
                        {
                            "size": actual_slice,
                            "price": fill_price,
                            "timestamp": market_row["timestamp"],
                            "is_bps": execution_cost_bps,
                            "slippage_bps": abs(
                                execution_cost_bps + np.random.normal(0, 1.5)
                            ),
                        }
                    )

                    remaining_size -= actual_slice
            else:
                # Market order
                if side == "buy":
                    fill_price = market_row["ask"]
                else:
                    fill_price = market_row["bid"]

                impact_bps = self._calculate_market_impact(
                    actual_slice, market_row["volume"], config
                )
                execution_cost_bps = market_row["spread_bps"] / 2 + impact_bps

                fills.append(
                    {
                        "size": actual_slice,
                        "price": fill_price,
                        "timestamp": market_row["timestamp"],
                        "is_bps": execution_cost_bps,
                        "slippage_bps": execution_cost_bps + np.random.normal(0, 2),
                    }
                )

                remaining_size -= actual_slice

        return self._calculate_tca_metrics(fills, order_size, market_data)

    def simulate_pov_execution(
        self,
        config: ExecConfig,
        market_data: pd.DataFrame,
        order_size: float,
        side: str,
    ) -> TCAResults:
        """Simulate Percentage of Volume (POV) execution strategy."""

        fills = []
        remaining_size = order_size

        for _, market_row in market_data.iterrows():
            if remaining_size <= 0:
                break

            # POV sizing
            target_slice = market_row["volume"] * config.participation_rate
            actual_slice = min(target_slice, remaining_size)

            if actual_slice < order_size * 0.005:  # Skip tiny slices
                continue

            # Urgency affects aggressiveness
            crossing_prob = config.urgency * 0.7 + (1 - config.post_only_prob) * 0.3
            should_cross = np.random.random() < crossing_prob

            if should_cross:
                # Cross spread
                if side == "buy":
                    fill_price = market_row["ask"]
                else:
                    fill_price = market_row["bid"]

                impact_bps = self._calculate_market_impact(
                    actual_slice, market_row["volume"], config
                )
                execution_cost_bps = market_row["spread_bps"] / 2 + impact_bps

                fills.append(
                    {
                        "size": actual_slice,
                        "price": fill_price,
                        "timestamp": market_row["timestamp"],
                        "is_bps": execution_cost_bps,
                        "slippage_bps": execution_cost_bps + np.random.normal(0, 1.8),
                    }
                )

                remaining_size -= actual_slice

            else:
                # Post and wait
                fill_prob = self._calculate_fill_probability(
                    actual_slice, market_row["volume"], config
                )
                if np.random.random() < fill_prob * 1.2:  # POV gets better fill rates
                    if side == "buy":
                        fill_price = market_row["bid"]
                        execution_cost_bps = 0
                    else:
                        fill_price = market_row["ask"]
                        execution_cost_bps = 0

                    fills.append(
                        {
                            "size": actual_slice,
                            "price": fill_price,
                            "timestamp": market_row["timestamp"],
                            "is_bps": execution_cost_bps,
                            "slippage_bps": abs(np.random.normal(0, 1)),
                        }
                    )

                    remaining_size -= actual_slice

        return self._calculate_tca_metrics(fills, order_size, market_data)

    def _calculate_fill_probability(
        self, slice_size: float, market_volume: float, config: ExecConfig
    ) -> float:
        """Calculate probability of order fill."""
        # Base probability
        base_prob = 0.7

        # Size relative to volume
        size_ratio = slice_size / market_volume
        size_penalty = min(0.5, size_ratio * 2)

        # Urgency bonus
        urgency_bonus = config.urgency * 0.2

        return max(0.1, base_prob - size_penalty + urgency_bonus)

    def _calculate_market_impact(
        self, slice_size: float, market_volume: float, config: ExecConfig
    ) -> float:
        """Calculate market impact in bps."""
        # Square root impact model
        volume_ratio = slice_size / market_volume
        base_impact = np.sqrt(volume_ratio) * 3.0  # Base 3 bps impact

        # Urgency increases impact (more aggressive)
        urgency_multiplier = 1 + config.urgency * 0.5

        return base_impact * urgency_multiplier

    def _calculate_tca_metrics(
        self, fills: List[Dict], order_size: float, market_data: pd.DataFrame
    ) -> TCAResults:
        """Calculate TCA metrics from fills."""
        if not fills:
            return TCAResults(
                implementation_shortfall_bps=999.9,
                slippage_p95_bps=999.9,
                fill_ratio=0.0,
                num_fills=0,
                avg_fill_size=0.0,
                total_volume=0.0,
                execution_time_sec=0.0,
            )

        fills_df = pd.DataFrame(fills)

        # Implementation shortfall
        is_bps = fills_df["is_bps"].mean()

        # Slippage P95
        slippage_p95 = np.percentile(fills_df["slippage_bps"], 95)

        # Fill ratio
        total_filled = fills_df["size"].sum()
        fill_ratio = total_filled / order_size

        # Other metrics
        num_fills = len(fills)
        avg_fill_size = fills_df["size"].mean()

        # Execution time
        start_time = fills_df["timestamp"].min()
        end_time = fills_df["timestamp"].max()
        execution_time_sec = (end_time - start_time).total_seconds()

        return TCAResults(
            implementation_shortfall_bps=is_bps,
            slippage_p95_bps=slippage_p95,
            fill_ratio=fill_ratio,
            num_fills=num_fills,
            avg_fill_size=avg_fill_size,
            total_volume=total_filled,
            execution_time_sec=max(1, execution_time_sec),
        )


class ExecutionGridSweeper:
    """Orchestrate grid search over execution parameters."""

    def __init__(self, hours_lookback: int = 24):
        self.hours_lookback = hours_lookback
        self.simulator = ExecutionSimulator()

    def define_parameter_grid(self) -> List[ExecConfig]:
        """Define parameter grid for sweep."""

        # Parameter ranges
        strategies = ["TWAP", "VWAP", "POV"]
        slice_pcts = [0.05, 0.1, 0.2, 0.3]  # 5% to 30% of parent order
        participation_rates = [0.1, 0.15, 0.2, 0.25]  # 10% to 25% of volume
        post_only_probs = [0.3, 0.5, 0.7]  # 30% to 70% post-only
        max_spread_bps = [10, 15, 25]  # Max spread to cross
        urgencies = [0.2, 0.5, 0.8]  # Low, medium, high urgency

        # Generate all combinations
        parameter_combinations = itertools.product(
            strategies,
            slice_pcts,
            participation_rates,
            post_only_probs,
            max_spread_bps,
            urgencies,
        )

        configs = []
        for combo in parameter_combinations:
            config = ExecConfig(
                strategy=combo[0],
                slice_pct=combo[1],
                participation_rate=combo[2],
                post_only_prob=combo[3],
                max_spread_bps=combo[4],
                urgency=combo[5],
            )
            configs.append(config)

        print(f"Generated {len(configs)} parameter combinations")
        return configs

    def evaluate_config(self, config: ExecConfig) -> Dict[str, Any]:
        """Evaluate single configuration."""

        # Generate market data
        market_data = self.simulator.generate_market_data(self.hours_lookback * 60)

        # Test different order sizes
        test_sizes = [1000, 2500, 5000]  # Different order sizes
        test_sides = ["buy", "sell"]

        results = []

        for order_size in test_sizes:
            for side in test_sides:
                # Simulate execution
                if config.strategy == "TWAP":
                    tca_result = self.simulator.simulate_twap_execution(
                        config, market_data, order_size, side
                    )
                elif config.strategy == "VWAP":
                    tca_result = self.simulator.simulate_vwap_execution(
                        config, market_data, order_size, side
                    )
                else:  # POV
                    tca_result = self.simulator.simulate_pov_execution(
                        config, market_data, order_size, side
                    )

                results.append(
                    {
                        "order_size": order_size,
                        "side": side,
                        "is_bps": tca_result.implementation_shortfall_bps,
                        "slippage_p95_bps": tca_result.slippage_p95_bps,
                        "fill_ratio": tca_result.fill_ratio,
                        "num_fills": tca_result.num_fills,
                        "execution_time_sec": tca_result.execution_time_sec,
                    }
                )

        # Aggregate results
        results_df = pd.DataFrame(results)

        aggregated = {
            "config": {
                "strategy": config.strategy,
                "slice_pct": config.slice_pct,
                "participation_rate": config.participation_rate,
                "post_only_prob": config.post_only_prob,
                "max_spread_bps": config.max_spread_bps,
                "urgency": config.urgency,
            },
            "performance": {
                "avg_is_bps": results_df["is_bps"].mean(),
                "avg_slippage_p95_bps": results_df["slippage_p95_bps"].mean(),
                "avg_fill_ratio": results_df["fill_ratio"].mean(),
                "avg_execution_time_sec": results_df["execution_time_sec"].mean(),
                "consistency_score": 1.0 / (1.0 + results_df["is_bps"].std()),
            },
        }

        return aggregated

    def run_grid_sweep(self, max_workers: int = 4) -> List[Dict[str, Any]]:
        """Run parallel grid sweep."""
        print("🔍 Starting execution parameter grid sweep...")

        # Generate parameter grid
        configs = self.define_parameter_grid()

        # Parallel evaluation
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_config = {
                executor.submit(self.evaluate_config, config): config
                for config in configs[:50]  # Limit for demo
            }

            for i, future in enumerate(
                concurrent.futures.as_completed(future_to_config)
            ):
                try:
                    result = future.result()
                    results.append(result)

                    if (i + 1) % 10 == 0:
                        print(
                            f"   Completed {i + 1}/{len(future_to_config)} configurations"
                        )

                except Exception as e:
                    print(f"   Configuration failed: {e}")

        print(f"✅ Grid sweep completed: {len(results)} valid results")
        return results

    def find_winner(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find best configuration based on multi-objective scoring."""

        # Scoring weights
        weights = {
            "is_bps": -1.0,  # Lower IS is better (negative weight)
            "slippage_p95_bps": -0.8,  # Lower slippage is better
            "fill_ratio": 2.0,  # Higher fill ratio is better
            "execution_time": -0.3,  # Faster execution is better
            "consistency": 1.0,  # Higher consistency is better
        }

        best_score = float("-inf")
        winner = None

        for result in results:
            perf = result["performance"]

            # Normalize metrics and apply weights
            score = (
                weights["is_bps"] * min(perf["avg_is_bps"], 50)  # Cap at 50 bps
                + weights["slippage_p95_bps"] * min(perf["avg_slippage_p95_bps"], 100)
                + weights["fill_ratio"] * perf["avg_fill_ratio"] * 100
                + weights["execution_time"]
                * min(1800 / max(perf["avg_execution_time_sec"], 60), 10)
                + weights["consistency"] * perf["consistency_score"] * 100
            )

            if score > best_score:
                best_score = score
                winner = result

        if winner:
            winner["score"] = best_score
            winner["ranking_criteria"] = weights

        return winner


def main():
    """Main execution grid sweep function."""
    parser = argparse.ArgumentParser(description="Execution Parameter Grid Sweep")
    parser.add_argument("--hours", type=int, default=24, help="Hours of market data")
    parser.add_argument(
        "--out", default="artifacts/exec_sweep", help="Output directory"
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    try:
        # Create output directory
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        output_dir = Path(args.out) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run grid sweep
        sweeper = ExecutionGridSweeper(hours_lookback=args.hours)
        results = sweeper.run_grid_sweep(max_workers=args.workers)

        if not results:
            print("❌ No valid results from grid sweep")
            return 1

        # Find winner
        winner = sweeper.find_winner(results)

        # Save results
        all_results = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "sweep_parameters": {
                "hours_lookback": args.hours,
                "total_configs": len(results),
                "max_workers": args.workers,
            },
            "winner": winner,
            "all_results": results,
        }

        results_path = output_dir / "sweep_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        winner_path = output_dir / "winner.json"
        with open(winner_path, "w") as f:
            json.dump(winner, f, indent=2)

        # Create latest symlinks
        latest_results = Path(args.out) / "sweep_results_latest.json"
        latest_winner = Path(args.out) / "winner_latest.json"

        for latest, target in [
            (latest_results, results_path),
            (latest_winner, winner_path),
        ]:
            if latest.exists():
                latest.unlink()
            latest.symlink_to(target)

        # Display results
        print(f"\n🏆 Grid Sweep Results:")
        print(f"  Best Strategy: {winner['config']['strategy']}")
        print(f"  Slice %: {winner['config']['slice_pct']:.1%}")
        print(f"  Participation: {winner['config']['participation_rate']:.1%}")
        print(f"  Post-Only Prob: {winner['config']['post_only_prob']:.1%}")
        print(f"  IS: {winner['performance']['avg_is_bps']:.2f} bps")
        print(
            f"  Slippage P95: {winner['performance']['avg_slippage_p95_bps']:.2f} bps"
        )
        print(f"  Fill Ratio: {winner['performance']['avg_fill_ratio']:.1%}")
        print(f"  Score: {winner['score']:.2f}")

        print(f"\n📄 Results:")
        print(f"  All Results: {results_path}")
        print(f"  Winner Config: {winner_path}")

        return 0

    except Exception as e:
        print(f"❌ Execution grid sweep failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
