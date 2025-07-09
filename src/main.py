"""
Main Entry Point for the Multi-Layer Trading System

This is the primary entry point that orchestrates all layers of the trading system.
"""

import argparse
import asyncio
import signal
import sys
from typing import Optional
import logging

from utils.config_manager import config
from utils.logger import get_logger, TradingLogger

# TODO: Import layer modules as they are implemented
# from layers.layer0_data_ingestion import DataIngestionManager
# from layers.layer1_alpha_models import AlphaModelManager
# from layers.layer2_ensemble import EnsembleManager
# from layers.layer3_position_sizing import PositionSizingManager
# from layers.layer4_execution import ExecutionEngine
# from layers.layer5_risk import RiskManager


class TradingSystemOrchestrator:
    """
    Main orchestrator for the multi-layer trading system.
    
    Coordinates all layers and manages the overall system lifecycle.
    """
    
    def __init__(self, strategy: str = "crypto_scalping", mode: str = "paper"):
        """
        Initialize the trading system orchestrator.
        
        Args:
            strategy: Trading strategy to use
            mode: Trading mode (paper, live)
        """
        self.strategy = strategy
        self.mode = mode
        self.logger = get_logger(__name__)
        self.trading_logger = TradingLogger("main")
        
        # System components (to be initialized)
        self.data_manager = None
        self.alpha_manager = None
        self.ensemble_manager = None
        self.position_manager = None
        self.execution_engine = None
        self.risk_manager = None
        
        # System state
        self.is_running = False
        self.shutdown_requested = False
        
        # Load strategy configuration
        self.strategy_config = config.load_strategy_config(strategy)
        
        self.logger.info(f"Trading System initialized - Strategy: {strategy}, Mode: {mode}")
    
    async def initialize(self) -> None:
        """Initialize all system components."""
        try:
            self.logger.info("Initializing trading system components...")
            
            # TODO: Initialize each layer as modules are implemented
            # self.data_manager = DataIngestionManager()
            # self.alpha_manager = AlphaModelManager()
            # self.ensemble_manager = EnsembleManager()
            # self.position_manager = PositionSizingManager()
            # self.execution_engine = ExecutionEngine()
            # self.risk_manager = RiskManager()
            
            # For now, just log that initialization would happen
            self.logger.info("Layer 0 (Data Ingestion) - TODO: Initialize data connectors")
            self.logger.info("Layer 1 (Alpha Models) - TODO: Load alpha signal models")
            self.logger.info("Layer 2 (Ensemble) - TODO: Initialize meta-learner")
            self.logger.info("Layer 3 (Position Sizing) - TODO: Set up position calculator")
            self.logger.info("Layer 4 (Execution) - TODO: Connect to exchanges")
            self.logger.info("Layer 5 (Risk Management) - TODO: Initialize risk monitors")
            
            self.logger.info("Trading system initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading system: {e}")
            raise
    
    async def start(self) -> None:
        """Start the trading system."""
        try:
            self.logger.info("Starting trading system...")
            
            # Initialize all components
            await self.initialize()
            
            # Set running state
            self.is_running = True
            
            # Start main trading loop
            await self.run_trading_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting trading system: {e}")
            raise
    
    async def run_trading_loop(self) -> None:
        """Main trading loop that coordinates all layers."""
        self.logger.info("Starting main trading loop...")
        
        while self.is_running and not self.shutdown_requested:
            try:
                # Main trading cycle
                await self.trading_cycle()
                
                # Sleep briefly before next cycle
                await asyncio.sleep(1.0)  # 1 second cycle for now
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                
                # In production, might want to implement retry logic
                # For now, continue to next cycle
                await asyncio.sleep(5.0)
    
    async def trading_cycle(self) -> None:
        """Execute one complete trading cycle through all layers."""
        
        # TODO: Implement actual trading cycle
        # This is the skeleton that will be filled in as layers are implemented
        
        # Layer 0: Get latest market data and features
        # market_data = await self.data_manager.get_latest_data()
        # features = await self.data_manager.get_latest_features()
        
        # Layer 1: Generate alpha signals from all models
        # signals = await self.alpha_manager.generate_signals(features)
        
        # Layer 2: Combine signals into trading decision
        # trading_decision = await self.ensemble_manager.combine_signals(signals)
        
        # Layer 3: Calculate position sizes
        # target_positions = await self.position_manager.calculate_positions(trading_decision)
        
        # Layer 5: Check risk constraints (before execution)
        # risk_approved = await self.risk_manager.check_risk(target_positions)
        
        # Layer 4: Execute trades if risk approved
        # if risk_approved:
        #     await self.execution_engine.execute_trades(target_positions)
        
        # For now, just log that a cycle would execute
        self.logger.debug("Trading cycle executed (placeholder)")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the trading system."""
        self.logger.info("Initiating trading system shutdown...")
        
        self.shutdown_requested = True
        self.is_running = False
        
        # TODO: Shutdown each component gracefully
        # if self.execution_engine:
        #     await self.execution_engine.cancel_all_orders()
        #     await self.execution_engine.shutdown()
        
        # if self.data_manager:
        #     await self.data_manager.shutdown()
        
        self.logger.info("Trading system shutdown complete")
    
    def signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown())


async def main():
    """Main function to run the trading system."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Layer Trading System")
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="crypto_scalping",
        choices=["crypto_scalping", "crypto_swing", "stocks_intraday", "stocks_swing"],
        help="Trading strategy to use"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="paper",
        choices=["paper", "live"],
        help="Trading mode (paper or live)"
    )
    parser.add_argument(
        "--asset",
        type=str,
        default="BTC-USD",
        help="Primary asset to trade"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )
    
    args = parser.parse_args()
    
    # Initialize logging
    logger = get_logger(__name__)
    logger.info("Starting Multi-Layer Trading System")
    logger.info(f"Strategy: {args.strategy}, Mode: {args.mode}, Asset: {args.asset}")
    
    # Create and run trading system
    trading_system = TradingSystemOrchestrator(
        strategy=args.strategy,
        mode=args.mode
    )
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, trading_system.signal_handler)
    signal.signal(signal.SIGTERM, trading_system.signal_handler)
    
    try:
        # Start the trading system
        await trading_system.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error in trading system: {e}")
        sys.exit(1)
    finally:
        await trading_system.shutdown()


if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main()) 