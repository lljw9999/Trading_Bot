#!/usr/bin/env python3
"""
End-to-End Stocks Trading Session

Runs the complete 6-layer stack:
L0: Alpaca stock connector → raw market data
L1: OBP + MAM alphas → edge signals  
L2: Logistic ensemble → final edge
L3: Kelly sizing → position sizes
L4: Alpaca paper executor → orders
L5: Risk manager → position limits
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional
from datetime import datetime, timezone
from decimal import Decimal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import all layers
from src.layers.layer0_data_ingestion.alpaca_connector import AlpacaConnector
from src.layers.layer1_alpha_models.ob_pressure import OrderBookPressureAlpha
from src.layers.layer1_alpha_models.ma_momentum import MovingAverageMomentumAlpha
from src.layers.layer2_ensemble.meta_learner import LogisticMetaLearner
from src.layers.layer3_position_sizing.kelly_sizing import KellySizing
from src.layers.layer4_execution.alpaca_executor import AlpacaExecutor
from src.layers.layer5_risk.basic_risk_manager import BasicRiskManager
from src.utils.metrics import get_metrics, start_metrics_server

class StocksSession:
    """End-to-end stocks trading session."""
    
    def __init__(self):
        """Initialize all system layers."""
        # Initialize metrics
        self.metrics = get_metrics()
        start_metrics_server(port=8000)
        
        # L0: Data Ingestion
        self.connector = AlpacaConnector()
        
        # L1: Alpha Models
        self.obp_alpha = OrderBookPressureAlpha()
        self.mam_alpha = MovingAverageMomentumAlpha()
        
        # L2: Ensemble
        self.meta_learner = LogisticMetaLearner()
        
        # L3: Position Sizing
        self.kelly_sizer = KellySizing()
        
        # L4: Execution
        self.executor = AlpacaExecutor()
        
        # L5: Risk Management
        self.risk_manager = BasicRiskManager(
            max_position_pct=0.25,  # Max 25% position size
            max_drawdown_pct=0.03,  # 3% max drawdown
            vol_multiplier_limit=4.0,  # 4x vol circuit breaker
            min_trade_size=10.0  # $10 min trade
        )
        
        # Trading symbols
        self.symbols = ['AAPL', 'MSFT', 'NVDA']
        
        # State tracking
        self.positions = {}  # Current positions
        self.portfolio_value = Decimal('0')
        self.fills = []  # Trade fills
        self.pnl = Decimal('0')  # Session P&L
        
        # Initialize metrics for each symbol
        for symbol in self.symbols:
            self.metrics.position_value.labels(symbol=symbol).set(0)
            self.metrics.pnl_unrealized.labels(symbol=symbol).set(0)
        
        logger.info("🚀 Stocks trading session initialized")
        logger.info(f"Trading symbols: {', '.join(self.symbols)}")
    
    async def run_session(self, duration_minutes: int = 60):
        """Run the trading session for specified duration."""
        try:
            logger.info(f"🚀 Starting {duration_minutes}min trading session")
            logger.info(f"📈 Trading {len(self.symbols)} symbols: {', '.join(self.symbols)}")
            
            # Initialize connection
            await self.connector.connect()
            await self.connector.subscribe(self.symbols)
            
            # Get initial portfolio value
            self.portfolio_value = await self.executor.get_portfolio_value()
            logger.info(f"💰 Starting portfolio value: ${float(self.portfolio_value):,.2f}")
            
            # Main trading loop
            end_time = time.time() + (duration_minutes * 60)
            
            while time.time() < end_time:
                try:
                    # Process market data
                    tick = await self.connector.get_next_tick()
                    if tick:
                        await self._process_tick(tick)
                    
                    # Update portfolio value and P&L every minute
                    if int(time.time()) % 60 == 0:
                        self.portfolio_value = await self.executor.get_portfolio_value()
                        
                        # Log progress
                        remaining = int((end_time - time.time()) / 60)
                        logger.info(f"⏳ {remaining}min remaining | "
                                  f"Portfolio: ${float(self.portfolio_value):,.2f} | "
                                  f"P&L: ${float(self.pnl):,.2f} | "
                                  f"Positions: {len(self.positions)}")
                        
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    continue
                    
            # Session complete
            logger.info("🏁 Trading session complete")
            await self._print_session_summary()
            
        except Exception as e:
            logger.error(f"Session error: {e}")
            
        finally:
            # Cleanup
            try:
                await self.connector.disconnect()
                logger.info("👋 Disconnected from market data")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

    async def _print_session_summary(self):
        """Print final session statistics."""
        try:
            # Calculate session duration
            duration = (datetime.now(timezone.utc) - self.connector.session_start).total_seconds() / 60
            
            print("\n" + "="*60)
            print("📊 SESSION SUMMARY")
            print("="*60)
            
            # Trading activity
            print(f"\n🎯 Trading Activity:")
            print(f"  Duration: {duration:.1f}min")
            print(f"  Symbols: {', '.join(self.symbols)}")
            print(f"  Total signals: {self.connector.total_signals}")
            print(f"  Total orders: {len(self.fills)}")
            print(f"  Fill rate: {len(self.fills)/max(1,self.connector.total_signals):.1%}")
            
            # Performance
            print(f"\n💰 Performance:")
            print(f"  Starting value: ${float(self.connector.initial_portfolio):,.2f}")
            print(f"  Ending value: ${float(self.portfolio_value):,.2f}")
            print(f"  Session P&L: ${float(self.pnl):,.2f}")
            print(f"  Return: {(float(self.pnl)/float(self.connector.initial_portfolio)):.2%}")
            
            # Positions
            if self.positions:
                print(f"\n📍 Final Positions:")
                for symbol, pos in self.positions.items():
                    print(f"  {symbol}: ${float(pos):,.2f}")
            
            # Risk metrics
            risk_stats = self.risk_manager.get_stats()
            print(f"\n🛡️ Risk Management:")
            for key, value in risk_stats.items():
                print(f"  {key}: {value}")
            
            print("\n" + "="*60)
            
        except Exception as e:
            logger.error(f"Error printing summary: {e}")

    async def _run_data_layer(self):
        """Run Layer 0: Data ingestion."""
        try:
            async for tick in self.connector.stream_ticks():
                # Forward tick to alpha layer
                await self._process_tick(tick)
                
        except asyncio.CancelledError:
            logger.info("Data layer stopped")
        except Exception as e:
            logger.error(f"Data layer error: {e}")
    
    async def _process_tick(self, tick: Dict):
        """Process a single market tick through all layers."""
        try:
            start_time = time.time()
            symbol = tick.get('symbol')
            price = float(tick.get('price', 0))
            bid_size = float(tick.get('bid_size', 0))
            ask_size = float(tick.get('ask_size', 0))
            timestamp = tick.get('timestamp', datetime.now(timezone.utc).isoformat())
            
            # Record market tick
            self.metrics.record_market_tick(
                symbol=symbol,
                exchange='alpaca',
                asset_type='stock',
                latency=time.time() - start_time
            )
            
            # L1: Alpha Models
            # 1. Order Book Pressure Alpha
            obp_signal = self.obp_alpha.update(
                symbol=symbol,
                bid_size=bid_size,
                ask_size=ask_size,
                timestamp=timestamp
            )
            
            if obp_signal:
                self.metrics.record_alpha_prediction(
                    model_name='ob_pressure',
                    symbol=symbol,
                    edge_bps=obp_signal.edge_bps,
                    confidence=obp_signal.confidence
                )
            
            # 2. Moving Average Momentum Alpha
            mam_signal = self.mam_alpha.update(
                symbol=symbol,
                price=price,
                timestamp=timestamp
            )
            
            if mam_signal:
                self.metrics.record_alpha_prediction(
                    model_name='ma_momentum',
                    symbol=symbol,
                    edge_bps=mam_signal.edge_bps,
                    confidence=mam_signal.confidence
                )
            
            # Skip if no signals generated
            if not obp_signal and not mam_signal:
                return
            
            # L2: Logistic Meta-Learner
            # Combine signals - use 0 if a model didn't generate a signal
            ensemble_edge = self.meta_learner.predict(
                symbol=symbol,
                alpha_signals=[
                    obp_signal.edge_bps if obp_signal else 0,
                    mam_signal.edge_bps if mam_signal else 0
                ],
                confidences=[
                    obp_signal.confidence if obp_signal else 0.5,
                    mam_signal.confidence if mam_signal else 0.5
                ]
            )
            
            if ensemble_edge:
                self.metrics.record_ensemble_prediction(
                    symbol=symbol,
                    edge_bps=ensemble_edge.edge_bps
                )
            else:
                return
            
            # L3: Kelly Position Sizing
            position_size = self.kelly_sizer.calculate_position(
                edge_bps=ensemble_edge.edge_bps,
                confidence=ensemble_edge.confidence,
                price=price,
                portfolio_value=self.portfolio_value
            )
            
            if position_size:
                self.metrics.record_position_sizing(
                    symbol=symbol,
                    kelly_frac=float(position_size) / float(self.portfolio_value)
                )
            else:
                return
                
            # L5: Risk Management
            risk_check, reason, allowed_size = self.risk_manager.check_position_risk(
                symbol=symbol,
                proposed_position=position_size,
                current_price=Decimal(str(price)),
                portfolio_value=self.portfolio_value
            )
            
            if not risk_check:
                logger.warning(f"🛑 Risk check failed: {reason}")
                self.metrics.record_risk_violation(violation_type=reason)
                return
                
            # Adjust position size if needed
            if allowed_size != position_size:
                logger.info(f"⚠️ Position size adjusted from ${position_size:,.2f} to ${allowed_size:,.2f}")
                position_size = allowed_size
            
            # L4: Order Execution
            order_result = await self._execute_order(symbol, position_size, price)
            if order_result:
                logger.info(f"📋 L4 Executor: {symbol} order placed - {order_result}")
            
            # Record final tick processing latency
            self.metrics.market_tick_latency.labels(
                symbol=symbol,
                exchange='alpaca'
            ).observe(time.time() - start_time)
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            self.metrics.component_health.labels(component='tick_processor').set(0)
    
    async def _execute_order(self, symbol: str, position_size: Decimal, price: float) -> Optional[str]:
        """Execute an order through the Alpaca executor."""
        try:
            # Record order start time
            order_start = time.time()
            
            # Determine order side and size
            side = 'buy' if position_size > 0 else 'sell'
            shares = abs(float(position_size) / price)
            
            # Record order submission
            self.metrics.record_order_submitted(
                symbol=symbol,
                side=side,
                order_type='market'
            )
            
            # Place order through executor
            result = await self.executor.execute_order(
                symbol=symbol,
                side=side,
                shares=shares,
                price=price
            )
            
            # Update positions and track fill
            if result and result.get('status') == 'filled':
                # Calculate fill latency and slippage
                fill_latency = time.time() - order_start
                fill_price = float(result.get('fill_price', price))
                slippage_bps = abs(fill_price - price) / price * 10000
                
                # Record fill metrics
                self.metrics.record_order_filled(
                    symbol=symbol,
                    side=side,
                    fill_latency=fill_latency,
                    slippage_bps=slippage_bps
                )
                
                # Update position
                current_pos = self.positions.get(symbol, Decimal('0'))
                new_pos = current_pos + (position_size if side == 'buy' else -position_size)
                self.positions[symbol] = new_pos
                
                # Update position metrics
                self.metrics.update_position_value(symbol, float(new_pos))
                
                # Record fill
                fill = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'side': side,
                    'shares': shares,
                    'price': price,
                    'value': float(position_size),
                    'position_after': float(new_pos)
                }
                self.fills.append(fill)
                
                # Update P&L (simple mark-to-market)
                self.pnl = sum(
                    pos * Decimal(str(self.executor.get_last_price(sym)))
                    for sym, pos in self.positions.items()
                )
                
                # Update P&L metrics
                self.metrics.update_portfolio_metrics(
                    portfolio_value=float(self.portfolio_value),
                    cash_balance=float(self.portfolio_value - sum(abs(p) for p in self.positions.values()))
                )
                
                # Update unrealized P&L
                for sym, pos in self.positions.items():
                    last_price = self.executor.get_last_price(sym)
                    unrealized = float(pos * Decimal(str(last_price)))
                    self.metrics.update_unrealized_pnl(sym, unrealized)
                
                return f"{side.upper()} {shares:.2f} shares @ ${price:.2f}"
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            self.metrics.component_health.labels(component='order_executor').set(0)
            return None
    
    async def _run_pipeline(self):
        """Run the processing pipeline."""
        try:
            while True:
                # Periodic status updates
                await asyncio.sleep(30)  # Every 30 seconds
                
                # Update portfolio value based on positions
                await self._update_portfolio_value()
                
                # Log session stats
                elapsed = (datetime.now(timezone.utc) - self.connector.session_start).total_seconds() / 60
                logger.info(f"📈 Session stats: {elapsed:.1f}min, "
                           f"signals={self.connector.total_signals}, orders={self.connector.total_orders}, "
                           f"positions={len(self.positions)}")
                
        except asyncio.CancelledError:
            logger.info("Pipeline stopped")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
    
    async def _update_portfolio_value(self):
        """Update portfolio value based on current positions."""
        try:
            # For paper trading, we'll simulate P&L
            # In real implementation, this would query actual account value
            total_position_value = sum(abs(pos) for pos in self.positions.values())
            
            # Simple P&L simulation: assume 0.1% random movement
            import random
            pnl_factor = 1 + random.uniform(-0.001, 0.001)  # ±0.1%
            
            self.pnl = Decimal(str(total_position_value * (pnl_factor - 1)))
            
            # Update portfolio metrics
            self.metrics.update_portfolio_metrics(
                portfolio_value=float(self.portfolio_value),
                cash_balance=float(self.portfolio_value - total_position_value)
            )
            
            # Update risk metrics
            drawdown = max(0, 1 - float(self.portfolio_value) / float(self.connector.initial_portfolio))
            exposure = total_position_value / float(self.portfolio_value)
            risk_score = min(100, max(0, drawdown * 100 + exposure * 50))
            
            self.metrics.update_risk_metrics(
                risk_score=risk_score,
                drawdown=drawdown,
                exposure=exposure
            )
            
            # Update system health metrics
            import psutil
            process = psutil.Process()
            self.metrics.update_system_health(
                memory_bytes=process.memory_info().rss,
                cpu_percent=process.cpu_percent()
            )
            
            # Update component health
            self.metrics.set_component_health('portfolio_manager', True)
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
            self.metrics.set_component_health('portfolio_manager', False)
    
    async def _generate_session_report(self):
        """Generate final session report."""
        try:
            elapsed = (datetime.now(timezone.utc) - self.connector.session_start).total_seconds() / 60
            
            # Alpha model stats
            alpha_stats = self.obp_alpha.get_stats() # Assuming OBP is the primary alpha model for stats
            alpha_stats.update(self.mam_alpha.get_stats())
            
            # Risk manager stats
            risk_stats = self.risk_manager.get_stats()
            
            print("\n" + "="*60)
            print("📊 STOCKS TRADING SESSION REPORT")
            print("="*60)
            print(f"⏱️  Duration: {elapsed:.1f} minutes")
            print(f"📈 Symbols: {', '.join(self.symbols)}")
            print(f"💰 Portfolio: ${self.portfolio_value:,.2f}")
            print(f"💸 Session P&L: ${self.pnl:,.2f}")
            print(f"📊 Total signals: {self.connector.total_signals}")
            print(f"📋 Total orders: {self.connector.total_orders}")
            print(f"🎯 Signal-to-order ratio: {self.connector.total_orders/max(1, self.connector.total_signals):.2%}")
            print(f"📍 Active positions: {len(self.positions)}")
            
            if self.positions:
                print(f"📊 Position details:")
                for symbol, position in self.positions.items():
                    print(f"  {symbol}: ${position:,.2f}")
            
            print(f"\n📈 Alpha Model Stats:")
            for key, value in alpha_stats.items():
                print(f"  {key}: {value}")
            
            print(f"\n🛡️  Risk Manager Stats:")
            for key, value in risk_stats.items():
                print(f"  {key}: {value}")
            
            # Performance metrics
            if self.connector.total_signals > 0:
                fill_rate = self.connector.total_orders / self.connector.total_signals
                print(f"\n🎯 Performance Metrics:")
                print(f"  Fill rate: {fill_rate:.2%}")
                print(f"  Avg signals/min: {self.connector.total_signals / elapsed:.1f}")
                print(f"  Avg orders/min: {self.connector.total_orders / elapsed:.1f}")
            
            print("="*60)
            print("✅ Session completed successfully!")
            
        except Exception as e:
            logger.error(f"Error generating session report: {e}")


async def main():
    """Main entry point."""
    logger.info("🚀 Starting Stocks Trading Session...")
    
    # Create and run session
    session = StocksSession()
    
    try:
        # Run for 5 minutes (short test)
        await session.run_session(duration_minutes=5)
        
    except KeyboardInterrupt:
        logger.info("Session interrupted by user")
    except Exception as e:
        logger.error(f"Session failed: {e}")
        raise


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run stocks trading session")
    parser.add_argument("--duration", type=int, default=60,
                      help="Session duration in minutes")
    parser.add_argument("--metrics-port", type=int, default=8000,
                      help="Port for Prometheus metrics server")
    args = parser.parse_args()
    
    # Create and run session
    session = StocksSession()
    
    try:
        # Start metrics server
        start_metrics_server(port=args.metrics_port)
        logger.info(f"📊 Metrics server started on port {args.metrics_port}")
        
        # Initialize component health
        metrics = get_metrics()
        for component in ['data_ingestion', 'alpha_models', 'ensemble', 'position_sizing', 
                         'execution', 'risk_management', 'portfolio_manager']:
            metrics.set_component_health(component, True)
        
        # Run with asyncio
        asyncio.run(session.run_session(duration_minutes=args.duration))
    except KeyboardInterrupt:
        logger.info("\n⚠️ Session interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        # Mark all components as unhealthy
        metrics = get_metrics()
        for component in ['data_ingestion', 'alpha_models', 'ensemble', 'position_sizing', 
                         'execution', 'risk_management', 'portfolio_manager']:
            metrics.set_component_health(component, False)
    finally:
        logger.info("👋 Session ended") 