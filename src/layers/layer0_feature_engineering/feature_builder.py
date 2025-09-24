#!/usr/bin/env python3
"""
High-Performance Feature Engineering with Polars + DuckDB
10-100x faster than pandas for large datasets
"""

import pandas as pd
import polars as pl
import duckdb
import numpy as np
from typing import Optional, List, Dict


def _to_pl(df: "pd.DataFrame") -> pl.LazyFrame:
    """Convert pandas DataFrame to Polars LazyFrame for faster processing"""
    return pl.from_pandas(df).lazy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build trading features using Polars for 10-100x speedup
    Replaces pandas operations with Polars lazy evaluation
    """
    if len(df) == 0:
        return df

    # Convert to Polars LazyFrame for high-performance operations
    lf = (
        _to_pl(df)
        .with_columns(
            [
                # Price-based features
                (pl.col("mid_px").diff()).alias("ret_1"),
                (pl.col("mid_px").rolling_mean(window_size=20)).alias("sma_20"),
                (pl.col("mid_px").rolling_std(window_size=20)).alias("vol_20"),
                # Technical indicators
                (
                    pl.col("mid_px").rolling_mean(window_size=5)
                    - pl.col("mid_px").rolling_mean(window_size=20)
                ).alias("momentum_5_20"),
                (pl.col("volume").rolling_sum(window_size=10)).alias("volume_sum_10"),
                # Volatility features
                (pl.col("mid_px").diff().abs().rolling_mean(window_size=20)).alias(
                    "avg_true_range_20"
                ),
                # Price position features
                (
                    (pl.col("mid_px") - pl.col("mid_px").rolling_min(window_size=20))
                    / (
                        pl.col("mid_px").rolling_max(window_size=20)
                        - pl.col("mid_px").rolling_min(window_size=20)
                    )
                ).alias("stoch_20"),
            ]
        )
        .collect(streaming=True)
    )  # Streaming for memory efficiency

    return lf.to_pandas()


def build_advanced_features_duckdb(
    df: pd.DataFrame, market_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Use DuckDB for complex joins and aggregations (>1M rows)
    Much faster than pandas merge operations
    """
    if len(df) == 0:
        return df

    # Connect to DuckDB
    conn = duckdb.connect()

    try:
        # Register DataFrames as DuckDB tables
        conn.register("trades", df)

        if market_data is not None and len(market_data) > 1_000_000:
            conn.register("market", market_data)

            # Complex join query using DuckDB SQL (much faster than pandas)
            result = conn.execute(
                """
                SELECT 
                    t.*,
                    m.volume as market_volume,
                    m.volatility as market_vol,
                    AVG(t.mid_px) OVER (
                        PARTITION BY DATE_TRUNC('hour', t.timestamp) 
                        ORDER BY t.timestamp 
                        ROWS BETWEEN 100 PRECEDING AND CURRENT ROW
                    ) as rolling_mean_100,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.mid_px) OVER (
                        ORDER BY t.timestamp 
                        ROWS BETWEEN 50 PRECEDING AND 50 FOLLOWING
                    ) as median_price_101,
                    RANK() OVER (
                        PARTITION BY DATE_TRUNC('minute', t.timestamp) 
                        ORDER BY t.volume DESC
                    ) as volume_rank
                FROM trades t
                LEFT JOIN market m ON DATE_TRUNC('minute', t.timestamp) = DATE_TRUNC('minute', m.timestamp)
                ORDER BY t.timestamp
            """
            ).df()
        else:
            # Simpler aggregations for smaller datasets
            result = conn.execute(
                """
                SELECT *,
                    AVG(mid_px) OVER (
                        ORDER BY timestamp 
                        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
                    ) as sma_21,
                    STDDEV(mid_px) OVER (
                        ORDER BY timestamp 
                        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
                    ) as std_21
                FROM trades
                ORDER BY timestamp
            """
            ).df()

        return result

    finally:
        conn.close()


def build_whale_impact_features(
    df: pd.DataFrame, whale_events: List[Dict]
) -> pd.DataFrame:
    """
    Add whale alert impact features using Polars
    """
    if len(df) == 0 or not whale_events:
        df["whale_impact_1h"] = 0
        df["whale_impact_24h"] = 0
        return df

    # Convert whale events to Polars DataFrame
    whale_df = pl.DataFrame(whale_events)

    # Convert main data to Polars
    lf = _to_pl(df)

    # Add whale impact features with time-based joins
    enhanced_lf = lf.with_columns(
        [
            # Count whale events in last 1 hour
            pl.lit(0).alias(
                "whale_impact_1h"
            ),  # Placeholder - would need proper time-based join
            pl.lit(0).alias(
                "whale_impact_24h"
            ),  # Placeholder - would need proper time-based join
        ]
    )

    return enhanced_lf.collect().to_pandas()


def create_feature_pipeline(
    df: pd.DataFrame, use_advanced: bool = False
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline with Polars acceleration
    """
    # Basic features with Polars
    df_features = build_features(df)

    # Advanced features with DuckDB for large datasets
    if use_advanced and len(df) > 100_000:
        df_features = build_advanced_features_duckdb(df_features)

    # Fill NaN values
    df_features = df_features.fillna(0)

    return df_features


def benchmark_polars_vs_pandas(df: pd.DataFrame, iterations: int = 3) -> Dict:
    """
    Benchmark Polars vs Pandas performance
    """
    import time

    def pandas_features(data):
        data["ret_1"] = data["mid_px"].diff()
        data["sma_20"] = data["mid_px"].rolling(20).mean()
        data["vol_20"] = data["mid_px"].rolling(20).std()
        return data

    def polars_features(data):
        return build_features(data)

    # Pandas benchmark
    pandas_times = []
    for _ in range(iterations):
        df_copy = df.copy()
        start = time.time()
        pandas_features(df_copy)
        pandas_times.append(time.time() - start)

    # Polars benchmark
    polars_times = []
    for _ in range(iterations):
        start = time.time()
        polars_features(df)
        polars_times.append(time.time() - start)

    pandas_avg = np.mean(pandas_times)
    polars_avg = np.mean(polars_times)
    speedup = pandas_avg / polars_avg if polars_avg > 0 else float("inf")

    return {
        "pandas_avg_time": pandas_avg,
        "polars_avg_time": polars_avg,
        "speedup_factor": speedup,
        "rows_processed": len(df),
    }


if __name__ == "__main__":
    # Demo with sample data
    np.random.seed(42)
    n_rows = 100_000

    sample_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n_rows, freq="1s"),
            "mid_px": 100 + np.cumsum(np.random.randn(n_rows) * 0.01),
            "volume": np.random.lognormal(8, 1, n_rows),
            "bid": lambda x: x["mid_px"] - 0.01,
            "ask": lambda x: x["mid_px"] + 0.01,
        }
    )

    print("🚀 Polars + DuckDB Feature Engineering Demo")
    print(f"📊 Processing {n_rows:,} rows")

    # Benchmark
    results = benchmark_polars_vs_pandas(sample_data)
    print(f"\n⚡ Performance Results:")
    print(f"   Pandas time: {results['pandas_avg_time']:.3f}s")
    print(f"   Polars time: {results['polars_avg_time']:.3f}s")
    print(f"   Speedup: {results['speedup_factor']:.1f}x faster")

    # Create features
    featured_data = create_feature_pipeline(sample_data, use_advanced=True)
    print(f"\n✅ Generated {len(featured_data.columns)} features:")
    print(f"   {list(featured_data.columns)}")
