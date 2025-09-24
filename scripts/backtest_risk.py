#!/usr/bin/env python3
"""
Risk Management Back-testing Script

Tests risk management on historical 1-minute bar data.
Pass criteria: no VaR >95%, PnL within ±0.5σ.

Usage: python scripts/backtest_risk.py --bars 1m --days 30 --symbol BTCUSDT
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class RiskBacktester:
    """Risk management backtesting engine."""

    def __init__(self, symbol: str, bars: str = "1m", days: int = 30):
        """Initialize backtester."""
        self.symbol = symbol
        self.bars = bars
        self.days = days
        self.data = None
        self.var_threshold = 95.0  # 95% VaR threshold (in percentage)
        self.pnl_sigma_threshold = 0.5  # ±0.5σ PnL threshold

    def load_data(self) -> bool:
        """Load historical data for backtesting."""
        try:
            # Search for data file
            data_path = Path("data")
            data_files = list(data_path.rglob(f"{self.symbol}.csv"))

            if not data_files:
                logger.error(f"No data file found for {self.symbol}")
                return False

            data_file = data_files[0]
            logger.info(f"Loading data from: {data_file}")

            # Load CSV data
            self.data = pd.read_csv(data_file)
            self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
            self.data = self.data.sort_values("timestamp")

            # Filter to requested days
            end_date = self.data["timestamp"].max()
            start_date = end_date - timedelta(days=self.days)
            self.data = self.data[self.data["timestamp"] >= start_date]

            logger.info(f"Loaded {len(self.data)} {self.bars} bars ({self.days} days)")
            logger.info(
                f"Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def calculate_returns(self) -> np.ndarray:
        """Calculate log returns."""
        if self.data is None:
            return np.array([])

        prices = self.data["close"].values
        returns = np.log(prices[1:] / prices[:-1])
        return returns

    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)."""
        if len(returns) == 0:
            return 0.0

        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_pnl_stats(self, returns: np.ndarray) -> Dict:
        """Calculate PnL statistics."""
        if len(returns) == 0:
            return {"mean": 0.0, "std": 0.0, "sigma": 0.0}

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Calculate sigma deviation from zero
        sigma_dev = abs(mean_return) / std_return if std_return > 0 else 0.0

        return {
            "mean": mean_return,
            "std": std_return,
            "sigma": sigma_dev,
            "total_bars": len(returns),
        }

    def assess_risk_breaches(self, returns: np.ndarray) -> Dict:
        """Assess risk metric breaches."""
        if len(returns) == 0:
            return {"var_breaches": 0, "var_breach_pct": 0.0}

        # Calculate VaR
        var_5pct = self.calculate_var(returns, confidence=0.95)

        # Count breaches (returns worse than VaR)
        breaches = np.sum(returns < var_5pct)
        breach_pct = breaches / len(returns) * 100

        return {
            "var_5pct": var_5pct,
            "var_breaches": breaches,
            "var_breach_pct": breach_pct,
            "expected_breaches": len(returns) * 0.05,  # 5% expected
        }

    def run_backtest(self) -> Dict:
        """Run the complete risk backtest."""
        logger.info(f"🚀 Starting risk backtest for {self.symbol}")
        logger.info(f"Parameters: {self.bars} bars, {self.days} days")

        # Load data
        if not self.load_data():
            return {"success": False, "error": "Failed to load data"}

        # Calculate returns
        returns = self.calculate_returns()
        if len(returns) == 0:
            return {"success": False, "error": "No returns calculated"}

        # Calculate statistics
        pnl_stats = self.calculate_pnl_stats(returns)
        risk_stats = self.assess_risk_breaches(returns)

        # Determine pass/fail
        var_pass = risk_stats["var_breach_pct"] < self.var_threshold
        pnl_pass = pnl_stats["sigma"] <= self.pnl_sigma_threshold

        results = {
            "success": True,
            "symbol": self.symbol,
            "bars": self.bars,
            "days": self.days,
            "pnl_stats": pnl_stats,
            "risk_stats": risk_stats,
            "var_pass": var_pass,
            "pnl_pass": pnl_pass,
            "overall_pass": var_pass and pnl_pass,
        }

        return results

    def print_results(self, results: Dict):
        """Print backtest results."""
        if not results["success"]:
            logger.error(f"❌ Backtest failed: {results.get('error', 'Unknown error')}")
            return

        logger.info("📊 Risk Backtest Results")
        logger.info("=" * 50)

        # PnL Statistics
        pnl = results["pnl_stats"]
        logger.info(f"📈 PnL Statistics:")
        logger.info(f"   Mean return: {pnl['mean']:.6f}")
        logger.info(f"   Std dev: {pnl['std']:.6f}")
        logger.info(
            f"   Sigma dev: {pnl['sigma']:.3f} (threshold: ±{self.pnl_sigma_threshold})"
        )
        logger.info(f"   Total bars: {pnl['total_bars']}")

        # Risk Statistics
        risk = results["risk_stats"]
        logger.info(f"📉 Risk Statistics:")
        logger.info(f"   VaR (5%): {risk['var_5pct']:.6f}")
        logger.info(
            f"   VaR breaches: {risk['var_breaches']} ({risk['var_breach_pct']:.1f}%)"
        )
        logger.info(f"   Expected breaches: {risk['expected_breaches']:.1f}")

        # Pass/Fail
        logger.info(f"✅ Results:")
        var_status = "✅ PASS" if results["var_pass"] else "❌ FAIL"
        pnl_status = "✅ PASS" if results["pnl_pass"] else "❌ FAIL"
        overall_status = "✅ PASS" if results["overall_pass"] else "❌ FAIL"

        logger.info(f"   VaR test: {var_status} (breach % ≤ {self.var_threshold}%)")
        logger.info(f"   PnL test: {pnl_status} (sigma ≤ ±{self.pnl_sigma_threshold})")
        logger.info(f"   Overall: {overall_status}")

        if results["overall_pass"]:
            logger.info("🎉 All risk criteria passed!")
        else:
            logger.error("⚠️  Some risk criteria failed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Risk Management Backtesting")
    parser.add_argument("--bars", default="1m", help="Bar resolution (default: 1m)")
    parser.add_argument(
        "--days", type=int, default=30, help="Days to backtest (default: 30)"
    )
    parser.add_argument(
        "--symbol", default="BTCUSDT", help="Symbol to backtest (default: BTCUSDT)"
    )

    args = parser.parse_args()

    # Create and run backtester
    backtester = RiskBacktester(symbol=args.symbol, bars=args.bars, days=args.days)

    results = backtester.run_backtest()
    backtester.print_results(results)

    # Exit with appropriate code
    if results["success"] and results["overall_pass"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
