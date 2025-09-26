#!/usr/bin/env python3
"""
Smoke Backtest for Trading System

Replays synthetic or historical market data through the trading system
to validate end-to-end functionality and ensure no crashes.
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import time
import json
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.layers.layer0_data_ingestion.schemas import MarketTick, FeatureSnapshot
from src.layers.layer0_data_ingestion.feature_bus import FeatureBus
from src.layers.layer1_alpha_models.order_book_pressure import OrderBookPressure
from src.layers.layer1_alpha_models.ma_momentum import MAMomentum
from src.layers.layer2_ensemble.meta_learner import MetaLearner
from src.layers.layer3_position_sizing.kelly_sizing import KellySizing
from src.layers.layer4_execution.market_order_executor import MarketOrderExecutor
from src.layers.layer5_risk.basic_risk_manager import BasicRiskManager
from src.utils.logger import get_logger
from src.utils.metrics import get_metrics, start_metrics_server


class SmokeBacktest:
    """
    Smoke backtest runner for validating the trading system.

    Generates synthetic market data and runs it through all layers
    to ensure the system works end-to-end without crashes.
    """

    def __init__(self, symbol: str = "BTC-USD", speed: float = 10.0):
        """
        Initialize smoke backtest.

        Args:
            symbol: Trading symbol to test
            speed: Playback speed multiplier (10x = 10 times faster)
        """
        self.symbol = symbol
        self.speed = speed
        self.logger = get_logger("smoke_backtest")

        # Initialize all system components
        self.feature_bus = FeatureBus()
        self.order_book_model = OrderBookPressure()
        self.momentum_model = MAMomentum()
        self.ensemble = MetaLearner()
        self.position_sizer = KellySizing()
        self.executor = MarketOrderExecutor()
        self.risk_manager = BasicRiskManager()

        # Metrics
        self.metrics = get_metrics()

        # State tracking
        self.tick_count = 0
        self.start_time = None
        self.portfolio_values = []
        self.errors = []

        self.logger.info(f"Smoke backtest initialized for {symbol} at {speed}x speed")

    async def run(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """
        Run the smoke backtest.

        Args:
            duration_minutes: How long to run the test

        Returns:
            Results dictionary with performance metrics
        """
        self.logger.info(f"Starting smoke backtest for {duration_minutes} minutes...")
        self.start_time = time.time()

        try:
            # Start metrics server
            start_metrics_server(port=8000)

            # Generate and process synthetic data
            async for tick in self._generate_synthetic_data(duration_minutes):
                await self._process_tick(tick)

                # Sleep to simulate real-time at specified speed
                await asyncio.sleep(1.0 / self.speed)

                self.tick_count += 1

                if self.tick_count % 100 == 0:
                    self.logger.info(f"Processed {self.tick_count} ticks...")

            # Generate final results
            results = self._generate_results()

            self.logger.info(f"Smoke backtest completed successfully!")
            self.logger.info(f"Results: {json.dumps(results, indent=2)}")

            return results

        except Exception as e:
            self.logger.error(f"Smoke backtest failed: {e}")
            self.errors.append(str(e))
            raise

    async def _generate_synthetic_data(self, duration_minutes: int):
        """Generate synthetic market data for testing."""
        start_price = Decimal("50000")  # Starting BTC price
        current_price = start_price

        ticks_per_minute = 60  # 1 tick per second
        total_ticks = duration_minutes * ticks_per_minute

        for i in range(total_ticks):
            # Generate price movement (random walk with slight drift)
            import random

            # Price change: -0.1% to +0.1% per tick
            price_change_pct = (random.random() - 0.5) * 0.002
            current_price = current_price * (1 + Decimal(str(price_change_pct)))

            # Ensure price stays reasonable
            current_price = max(Decimal("10000"), min(Decimal("100000"), current_price))

            # Generate bid/ask spread (2-10 bps)
            spread_bps = random.uniform(2, 10)
            spread_amount = current_price * Decimal(str(spread_bps / 10000))

            bid_price = current_price - spread_amount / 2
            ask_price = current_price + spread_amount / 2

            # Generate volumes
            base_volume = Decimal("100")
            volume_multiplier = random.uniform(0.5, 2.0)
            bid_size = base_volume * Decimal(str(volume_multiplier))
            ask_size = base_volume * Decimal(str(volume_multiplier))

            # Create market tick
            tick = MarketTick(
                symbol=self.symbol,
                exchange="synthetic",
                asset_type="crypto",
                timestamp=datetime.now(timezone.utc),
                bid=bid_price,
                ask=ask_price,
                last=current_price,
                bid_size=bid_size,
                ask_size=ask_size,
                volume=base_volume * Decimal(str(random.uniform(1, 5))),
            )

            yield tick

    async def _process_tick(self, tick: MarketTick):
        """Process a single market tick through all system layers."""
        try:
            # Record metrics
            self.metrics.record_market_tick(
                tick.symbol, tick.exchange, tick.asset_type, 0.001
            )

            # Layer 0: Feature computation
            features = await self.feature_bus.process_tick(tick)
            if not features:
                return

            self.metrics.record_feature_computation(
                tick.symbol, 100.0
            )  # 100μs simulated

            # Layer 1: Alpha models
            alpha_signals = {}

            edge1, conf1 = self.order_book_model.predict(features)
            alpha_signals["order_book_pressure"] = (edge1, conf1)
            self.metrics.record_alpha_prediction(
                "order_book_pressure", tick.symbol, edge1, conf1
            )

            edge2, conf2 = self.momentum_model.predict(features)
            alpha_signals["ma_momentum"] = (edge2, conf2)
            self.metrics.record_alpha_prediction(
                "ma_momentum", tick.symbol, edge2, conf2
            )

            # Layer 2: Ensemble
            ensemble_edge, ensemble_conf = self.ensemble.predict(
                alpha_signals, features
            )
            self.metrics.record_ensemble_prediction(tick.symbol, ensemble_edge)

            # Layer 3: Position sizing
            portfolio_value = self.executor.get_portfolio_value({tick.symbol: tick.mid})
            target_position, reasoning = self.position_sizer.calculate_position_size(
                tick.symbol, ensemble_edge, ensemble_conf, tick.mid, portfolio_value
            )

            # Layer 5: Risk check
            current_positions = {tick.symbol: self.executor.get_position(tick.symbol)}
            current_prices = {tick.symbol: tick.mid}

            risk_metrics = self.risk_manager.check_portfolio_risk(
                current_positions, current_prices, portfolio_value
            )

            trading_allowed, risk_reason = self.risk_manager.is_trading_allowed(
                risk_metrics
            )

            # Update risk metrics
            self.risk_manager.update_portfolio_value(portfolio_value)
            self.metrics.update_risk_metrics(
                risk_metrics.risk_score,
                risk_metrics.current_drawdown,
                risk_metrics.total_exposure,
            )

            # Layer 4: Execution (if trading allowed)
            if trading_allowed and abs(target_position) > Decimal("10"):
                order = await self.executor.execute_order(
                    tick.symbol, target_position, tick.mid
                )

                if order:
                    self.metrics.record_order_submitted(
                        tick.symbol, order.side.value, order.order_type
                    )

                    # Simulate fill
                    if order.status.value == "filled":
                        self.metrics.record_order_filled(
                            tick.symbol,
                            order.side.value,
                            0.1,
                            2.0,  # 0.1s latency, 2bp slippage
                        )

            # Update portfolio metrics
            portfolio_value = self.executor.get_portfolio_value(current_prices)
            cash_balance = self.executor.cash_balance

            self.metrics.update_portfolio_metrics(
                float(portfolio_value), float(cash_balance)
            )
            self.portfolio_values.append(float(portfolio_value))

            # Update position value
            position_value = self.executor.get_position_value(tick.symbol, tick.mid)
            self.metrics.update_position_value(tick.symbol, float(position_value))

        except Exception as e:
            self.logger.error(f"Error processing tick {self.tick_count}: {e}")
            self.errors.append(f"Tick {self.tick_count}: {e}")

    def _generate_results(self) -> Dict[str, Any]:
        """Generate backtest results summary."""
        end_time = time.time()
        duration = end_time - self.start_time

        # Portfolio performance
        start_value = self.portfolio_values[0] if self.portfolio_values else 100000
        end_value = self.portfolio_values[-1] if self.portfolio_values else 100000
        total_return = (end_value - start_value) / start_value

        # Component statistics
        feature_stats = self.feature_bus.get_stats()
        ob_model_stats = self.order_book_model.get_stats()
        ma_model_stats = self.momentum_model.get_stats()
        ensemble_stats = self.ensemble.get_stats()
        position_stats = self.position_sizer.get_stats()
        execution_stats = self.executor.get_stats()
        risk_stats = self.risk_manager.get_stats()

        return {
            "test_summary": {
                "symbol": self.symbol,
                "duration_seconds": round(duration, 2),
                "ticks_processed": self.tick_count,
                "ticks_per_second": round(self.tick_count / duration, 2),
                "errors": len(self.errors),
                "success": len(self.errors) == 0,
            },
            "portfolio_performance": {
                "start_value": start_value,
                "end_value": end_value,
                "total_return_pct": round(total_return * 100, 4),
                "final_cash": float(self.executor.cash_balance),
                "final_positions": {
                    k: float(v) for k, v in self.executor.positions.items()
                },
            },
            "component_stats": {
                "feature_bus": feature_stats,
                "order_book_model": ob_model_stats,
                "momentum_model": ma_model_stats,
                "ensemble": ensemble_stats,
                "position_sizing": position_stats,
                "execution": execution_stats,
                "risk_management": risk_stats,
            },
            "errors": self.errors[:10] if self.errors else [],  # First 10 errors
        }


async def main():
    """Main entry point for smoke backtest."""
    parser = argparse.ArgumentParser(description="Trading System Smoke Backtest")
    parser.add_argument("--symbol", default="BTC-USD", help="Trading symbol")
    parser.add_argument(
        "--speed", type=float, default=10.0, help="Playback speed multiplier"
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Test duration in minutes"
    )
    parser.add_argument(
        "--date", help="Test date (YYYY-MM-DD) - ignored for synthetic data"
    )

    args = parser.parse_args()

    print("🚀 Starting Trading System Smoke Backtest")
    print(f"   Symbol: {args.symbol}")
    print(f"   Speed: {args.speed}x")
    print(f"   Duration: {args.duration} minutes")
    print()

    backtest = SmokeBacktest(args.symbol, args.speed)

    try:
        results = await backtest.run(args.duration)

        print("\n✅ Smoke Backtest PASSED")
        print(f"   Processed {results['test_summary']['ticks_processed']} ticks")
        print(
            f"   Performance: {results['portfolio_performance']['total_return_pct']:.2f}%"
        )
        print(f"   Errors: {results['test_summary']['errors']}")

        if results["test_summary"]["errors"] > 0:
            print("\n⚠️  Errors encountered:")
            for error in results["errors"]:
                print(f"   - {error}")
            sys.exit(1)

        return 0

    except Exception as e:
        print(f"\n❌ Smoke Backtest FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Configure logging
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the backtest
    sys.exit(asyncio.run(main()))
