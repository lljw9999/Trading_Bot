#!/usr/bin/env python3
"""
LLM Signal Evaluation Script
Computes rolling correlation between LLM sentiment impact and future returns for A/B testing
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("llm_signal_eval")


class LLMSignalEvaluator:
    """Evaluates LLM sentiment signal correlation with future returns."""

    def __init__(self, window_size: int = 500):
        """Initialize LLM signal evaluator."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.window_size = window_size

        # Storage keys
        self.llm_events_key = "event:news_llm"
        self.price_data_key = "price:btc:1m"  # Assuming BTC minute data
        self.correlation_key = "llm:corr:futret5m"
        self.samples_key = "llm:corr:samples"

        logger.info(f"📊 LLM Signal Evaluator initialized (window: {window_size})")

    def get_llm_events(self, hours_back: int = 24) -> pd.DataFrame:
        """Get LLM sentiment events from Redis streams."""
        try:
            # Calculate timestamp for hours back
            cutoff_ts = int(
                (datetime.now() - timedelta(hours=hours_back)).timestamp() * 1000
            )

            # Get events from Redis stream
            events = self.redis.xrange(self.llm_events_key, min=cutoff_ts, max="+")

            if not events:
                logger.warning("No LLM events found")
                return pd.DataFrame()

            # Parse events into DataFrame
            records = []
            for event_id, fields in events:
                try:
                    # Extract timestamp from Redis stream ID
                    timestamp_ms = int(event_id.split("-")[0])
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000)

                    record = {
                        "timestamp": timestamp,
                        "impact": float(fields.get("impact", 0.0)),
                        "sentiment": fields.get("sentiment", "neutral"),
                        "bull": int(fields.get("bull", 0)),
                        "bear": int(fields.get("bear", 0)),
                        "title": fields.get("title", ""),
                    }
                    records.append(record)

                except Exception as e:
                    logger.debug(f"Error parsing event {event_id}: {e}")
                    continue

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df = df.sort_values("timestamp").reset_index(drop=True)

            logger.info(f"Retrieved {len(df)} LLM events")
            return df

        except Exception as e:
            logger.error(f"Error getting LLM events: {e}")
            return pd.DataFrame()

    def get_price_data(self, hours_back: int = 24) -> pd.DataFrame:
        """Get price data for correlation analysis."""
        try:
            # For demo, generate synthetic price data
            # In production, this would pull from Redis time series or market data API

            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            timestamps = pd.date_range(cutoff_time, datetime.now(), freq="1min")

            # Generate realistic price walk
            n_periods = len(timestamps)
            np.random.seed(42)  # For reproducible synthetic data
            returns = np.random.normal(
                0, 0.001, n_periods
            )  # 0.1% volatility per minute

            # Start at a realistic BTC price
            base_price = 50000.0
            prices = [base_price]

            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "price": prices,
                    "return_1m": np.concatenate([[0], np.diff(np.log(prices))]),
                }
            )

            # Calculate future returns
            df["future_ret_5m"] = df["return_1m"].rolling(5).sum().shift(-5)

            logger.info(f"Generated {len(df)} price points")
            return df

        except Exception as e:
            logger.error(f"Error getting price data: {e}")
            return pd.DataFrame()

    def compute_correlation(self, llm_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
        """Compute correlation between LLM impact and future returns."""
        try:
            if llm_df.empty or price_df.empty:
                return {"correlation": 0.0, "samples": 0, "pvalue": 1.0}

            # Merge LLM events with price data
            merged_df = pd.merge_asof(
                llm_df.sort_values("timestamp"),
                price_df.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("5min"),  # Match within 5 minutes
            )

            # Drop rows where merge failed
            merged_df = merged_df.dropna(subset=["future_ret_5m", "impact"])

            if len(merged_df) < 10:
                logger.warning(
                    f"Insufficient data for correlation: {len(merged_df)} samples"
                )
                return {"correlation": 0.0, "samples": len(merged_df), "pvalue": 1.0}

            # Take recent window
            if len(merged_df) > self.window_size:
                merged_df = merged_df.tail(self.window_size)

            # Compute correlation
            impact_values = merged_df["impact"].values
            future_returns = merged_df["future_ret_5m"].values

            # Handle edge cases
            if np.std(impact_values) < 1e-10 or np.std(future_returns) < 1e-10:
                correlation = 0.0
                pvalue = 1.0
            else:
                from scipy.stats import pearsonr

                correlation, pvalue = pearsonr(impact_values, future_returns)

                # Handle NaN
                if np.isnan(correlation):
                    correlation = 0.0
                    pvalue = 1.0

            result = {
                "correlation": float(correlation),
                "samples": len(merged_df),
                "pvalue": float(pvalue),
                "std_impact": float(np.std(impact_values)),
                "std_returns": float(np.std(future_returns)),
                "mean_impact": float(np.mean(impact_values)),
                "mean_future_ret": float(np.mean(future_returns)),
            }

            logger.info(
                f"Correlation analysis: r={correlation:.4f}, "
                f"p={pvalue:.4f}, n={len(merged_df)}"
            )

            return result

        except Exception as e:
            logger.error(f"Error computing correlation: {e}")
            return {"correlation": 0.0, "samples": 0, "pvalue": 1.0}

    def store_correlation_metrics(self, result: dict) -> None:
        """Store correlation metrics in Redis for A/B gate."""
        try:
            # Store main metrics for A/B evaluation
            self.redis.set(self.correlation_key, result["correlation"])
            self.redis.set(self.samples_key, result["samples"])

            # Store detailed results as JSON
            detailed_key = "llm:corr:detailed"
            detailed_result = {
                **result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "evaluator": "llm_signal_eval",
            }
            self.redis.set(detailed_key, json.dumps(detailed_result))

            # Store history
            history_key = "llm:corr:history"
            self.redis.lpush(history_key, json.dumps(detailed_result))
            self.redis.ltrim(history_key, 0, 999)  # Keep last 1000 evaluations

            # Set expiration for metrics (24 hours)
            for key in [self.correlation_key, self.samples_key, detailed_key]:
                self.redis.expire(key, 86400)

            logger.info(f"✅ Stored correlation metrics: r={result['correlation']:.4f}")

        except Exception as e:
            logger.error(f"Error storing correlation metrics: {e}")

    def run_evaluation(self, hours_back: int = 24) -> dict:
        """Run full LLM signal evaluation."""
        logger.info(f"🔍 Running LLM signal evaluation ({hours_back}h lookback)")

        start_time = time.time()

        try:
            # Get data
            llm_df = self.get_llm_events(hours_back)
            price_df = self.get_price_data(hours_back)

            # Compute correlation
            result = self.compute_correlation(llm_df, price_df)

            # Store metrics
            self.store_correlation_metrics(result)

            elapsed = time.time() - start_time
            logger.info(f"📊 LLM evaluation complete in {elapsed:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return {"correlation": 0.0, "samples": 0, "pvalue": 1.0}

    def get_evaluation_status(self) -> dict:
        """Get current evaluation status."""
        try:
            # Get current metrics
            correlation = float(self.redis.get(self.correlation_key) or 0.0)
            samples = int(self.redis.get(self.samples_key) or 0)

            # Get detailed results
            detailed_key = "llm:corr:detailed"
            detailed_data = self.redis.get(detailed_key)
            detailed = json.loads(detailed_data) if detailed_data else {}

            # Get recent history
            history_data = self.redis.lrange("llm:corr:history", 0, 9)
            history = [json.loads(h) for h in history_data]

            return {
                "current_correlation": correlation,
                "current_samples": samples,
                "detailed_results": detailed,
                "recent_history": history,
                "last_update": detailed.get("timestamp", "Never"),
            }

        except Exception as e:
            logger.error(f"Error getting evaluation status: {e}")
            return {"error": str(e)}


def main():
    """Main entry point for LLM signal evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Signal Evaluator")
    parser.add_argument(
        "--hours", type=int, default=24, help="Hours of data to analyze"
    )
    parser.add_argument("--window", type=int, default=500, help="Rolling window size")
    parser.add_argument("--status", action="store_true", help="Show evaluation status")
    parser.add_argument(
        "--continuous", action="store_true", help="Run continuously (daemon mode)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Interval for continuous mode (seconds)",
    )

    args = parser.parse_args()

    evaluator = LLMSignalEvaluator(window_size=args.window)

    if args.status:
        # Show current status
        status = evaluator.get_evaluation_status()
        print(json.dumps(status, indent=2))
        return

    if args.continuous:
        # Run continuously
        logger.info(f"🔄 Starting continuous evaluation (interval: {args.interval}s)")

        while True:
            try:
                evaluator.run_evaluation(args.hours)
                time.sleep(args.interval)
            except KeyboardInterrupt:
                logger.info("🛑 Stopping continuous evaluation")
                break
            except Exception as e:
                logger.error(f"Error in continuous evaluation: {e}")
                time.sleep(60)  # Wait before retrying
    else:
        # Single evaluation
        result = evaluator.run_evaluation(args.hours)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
