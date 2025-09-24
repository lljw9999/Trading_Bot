#!/usr/bin/env python3
"""
Real-time Crypto Trading Session

Implements the complete crypto trading pipeline with real-time data feeds.
"""

import asyncio
import logging
import numpy as np
import signal
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.layers.layer0_data_ingestion.crypto_connector import CoinbaseConnector
from src.layers.layer1_alpha_models.ma_momentum import MovingAverageMomentumAlpha
from src.layers.layer2_ensemble.meta_learner import MetaLearner
from src.layers.layer3_position_sizing.enhanced_position_sizing import (
    EnhancedPositionSizing,
    OptimizationMethod,
)
from src.layers.layer5_risk.basic_risk_manager import BasicRiskManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CryptoTradingSession:
    """End-to-end crypto trading session coordinator."""

    def __init__(self, symbols: List[str], duration_minutes: int = 30):
        self.symbols = symbols
        self.duration_minutes = duration_minutes
        self.shutdown_event = asyncio.Event()

        # Initialize pipeline components
        self.coinbase_connector = CoinbaseConnector(symbols=symbols)
        self.alpha_model = MovingAverageMomentumAlpha()
        self.ensemble = MetaLearner()  # Changed from WeightedEnsemble
        self.position_sizer = EnhancedPositionSizing(
            max_total_leverage=3.0,  # Crypto allows higher leverage
            target_volatility=0.15,  # 15% target volatility
            lookback_days=252,
        )
        self.risk_manager = BasicRiskManager()  # Changed from RiskManager

        # Performance tracking
        self.total_signals = 0
        self.portfolio_value = Decimal("100000")  # $100K starting capital
        self.session_start = None

        # Signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()

    async def run(self):
        """Run the complete trading session."""
        self.session_start = datetime.now(timezone.utc)
        logger.info("🚀 Starting crypto trading session...")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Duration: {self.duration_minutes} minutes")
        logger.info(f"Portfolio Value: ${self.portfolio_value:,.2f}")

        try:
            # Start data feed and processing
            data_task = asyncio.create_task(self._run_data_feed())

            # Wait for shutdown signal or duration timeout
            try:
                await asyncio.wait_for(
                    self.shutdown_event.wait(), timeout=self.duration_minutes * 60
                )
                logger.info("Shutdown signal received")
            except asyncio.TimeoutError:
                logger.info(f"Session completed after {self.duration_minutes} minutes")

            # Cancel data feed
            data_task.cancel()

            try:
                await data_task
            except asyncio.CancelledError:
                pass

        except Exception as e:
            logger.error(f"Session error: {e}")
            raise
        finally:
            await self._shutdown()

    async def _run_data_feed(self):
        """Run real-time Coinbase data feed."""
        try:
            logger.info("🔌 Connecting to Coinbase WebSocket...")

            # Start the connector (includes Kafka and WebSocket connection)
            await self.coinbase_connector.start()

            # Stream data using the connector's method
            async for tick_data in self.coinbase_connector.start_data_stream():
                try:
                    # Convert MarketTick to dict format for processing
                    tick_dict = {
                        "symbol": tick_data.symbol,
                        "price": float(tick_data.last) if tick_data.last else 0.0,
                        "timestamp": tick_data.timestamp.isoformat(),
                        "bid": float(tick_data.bid) if tick_data.bid else 0.0,
                        "ask": float(tick_data.ask) if tick_data.ask else 0.0,
                    }

                    # Process tick through the trading pipeline
                    await self._process_tick(tick_dict)

                except Exception as e:
                    logger.error(f"Error processing tick: {e}")

        except asyncio.CancelledError:
            logger.info("📡 Data feed stopped")
        except Exception as e:
            logger.error(f"Data feed error: {e}")
        finally:
            try:
                await self.coinbase_connector.stop()
            except:
                pass

    async def _process_tick(self, tick: Dict):
        """Process a single market tick through all layers."""
        try:
            symbol = tick.get("symbol")
            price = float(tick.get("price", tick.get("last", 0)))
            timestamp = tick.get("timestamp", datetime.now(timezone.utc).isoformat())

            # Add periodic logging to track processing
            self.total_signals += 1
            if self.total_signals % 20 == 1:  # Log every 20th tick
                logger.info(
                    f"🔍 Processing tick #{self.total_signals}: {symbol} @ ${price:.2f}"
                )

            if price <= 0:
                logger.warning(f"Invalid price for {symbol}: {price}")
                return

            # L1: Alpha model
            alpha_signal = self.alpha_model.update_price(symbol, price, timestamp)
            if not alpha_signal:
                # Add debug logging to understand why no signal
                if self.total_signals % 100 == 1:  # Log occasionally
                    logger.info(
                        f"🤔 No alpha signal for {symbol} @ tick #{self.total_signals}"
                    )
                return

            logger.info(
                f"📊 L1 Alpha: {symbol} edge={alpha_signal.edge_bps:.1f}bps "
                f"conf={alpha_signal.confidence:.2f}"
            )

            # L2: Ensemble (using predict_simple for backward compatibility)
            ensemble_edge = self.ensemble.predict_simple([alpha_signal.edge_bps])

            logger.info(f"🎯 L2 Ensemble: {symbol} final_edge={ensemble_edge:.1f}bps")

            # L3: Enhanced Position Sizing
            # First update market data for the enhanced system
            self.position_sizer.update_market_data(
                symbol=symbol,
                price_data={"close": price, "volume": 1000},  # Simplified for real-time
                returns=np.array(
                    [alpha_signal.edge_bps / 10000]
                ),  # Convert bps to return
                features=np.array([alpha_signal.confidence, ensemble_edge, price]),
            )

            # Prepare signals for optimization
            alpha_signals = {symbol: ensemble_edge}
            confidence_scores = {symbol: alpha_signal.confidence}

            # Run portfolio optimization (single symbol for now)
            optimization_result = self.position_sizer.optimize_portfolio(
                symbols=[symbol],
                alpha_signals=alpha_signals,
                confidence_scores=confidence_scores,
                portfolio_value=self.portfolio_value,
                method=OptimizationMethod.COMBINED_OPTIMIZATION,
            )

            # Extract position size from optimization result
            position_result = (
                optimization_result.position_results[0]
                if optimization_result.position_results
                else None
            )
            if position_result:
                position_size = position_result.position_dollars
                sizing_reason = f"Enhanced({optimization_result.optimization_method.value}): {position_result.reasoning}"
            else:
                position_size = Decimal("0")
                sizing_reason = "Enhanced optimization failed"

            logger.info(
                f"💰 L3 Enhanced: {symbol} position=${position_size:.0f} ({sizing_reason})"
            )

            # L5: Risk check (before execution)
            is_allowed, risk_reason, max_allowed = (
                self.risk_manager.check_position_risk(
                    symbol=symbol,
                    proposed_position=position_size,
                    current_price=Decimal(str(price)),
                    portfolio_value=self.portfolio_value,
                )
            )

            if is_allowed:
                logger.info(f"✅ L5 Risk: {symbol} APPROVED ${position_size:.0f}")
                # In a real system, this would go to L4 execution
            else:
                logger.warning(
                    f"❌ L5 Risk: {symbol} BLOCKED - {risk_reason} (max: ${max_allowed:.0f})"
                )

        except Exception as e:
            logger.error(
                f"Error processing tick for {tick.get('symbol', 'unknown')}: {e}"
            )

    async def _shutdown(self):
        """Clean shutdown procedures."""
        logger.info("🛑 Shutting down crypto trading session...")

        # Print session statistics
        session_duration = datetime.now(timezone.utc) - self.session_start
        minutes = session_duration.total_seconds() / 60

        logger.info("📊 SESSION SUMMARY:")
        logger.info(f"   ⏱️  Duration: {minutes:.1f} minutes")
        logger.info(f"   📈 Total signals: {self.total_signals}")
        logger.info(f"   💹 Symbols traded: {', '.join(self.symbols)}")
        logger.info(f"   🏦 Portfolio value: ${self.portfolio_value:,.2f}")

        if self.total_signals > 0:
            signals_per_minute = self.total_signals / minutes
            logger.info(f"   📊 Signal rate: {signals_per_minute:.1f} signals/min")


async def main():
    """Main entry point."""
    # Configuration
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]  # Coinbase format
    duration_minutes = 30

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f'logs/crypto_session_{datetime.now().strftime("%Y%m%d_%H%M")}.log'
            ),
            logging.StreamHandler(),
        ],
    )

    # Enable debug logging for alpha model temporarily
    alpha_logger = logging.getLogger("src.layers.layer1_alpha_models.ma_momentum")
    alpha_logger.setLevel(logging.DEBUG)

    try:
        session = CryptoTradingSession(symbols, duration_minutes)
        await session.run()

    except KeyboardInterrupt:
        logger.info("Session interrupted by user")
    except Exception as e:
        logger.error(f"Session failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
