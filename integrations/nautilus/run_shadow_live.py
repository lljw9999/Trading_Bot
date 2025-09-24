#!/usr/bin/env python3
"""
Nautilus Shadow Live Execution

Shadow execution system using NautilusTrader for live market data
and execution with reduced position sizes for performance measurement.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List
import logging
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from nautilus_trader.live.node import TradingNode
    from nautilus_trader.config import TradingNodeConfig

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False

from integrations.nautilus.strategy_basis_carry_nt import (
    NautilusBasisCarryStrategy,
    BasisCarryConfig,
)
from integrations.nautilus.bridge import NautilusRedisBridge

logger = logging.getLogger("nautilus_shadow_live")


class NautilusShadowLiveRunner:
    """
    Shadow live execution runner using NautilusTrader.

    Runs live trading with reduced position sizes to measure
    real-world performance without significant risk.
    """

    def __init__(self):
        """Initialize shadow live runner."""
        self.node = None
        self.bridge = None
        self.strategy = None
        self.running = False

        if not NAUTILUS_AVAILABLE:
            logger.warning("NautilusTrader not available - using mock mode")

        logger.info("Initialized Nautilus shadow live runner")

    def setup_node(self, venues: List[str], symbols: List[str]) -> bool:
        """
        Setup trading node with live adapters.

        Args:
            venues: List of venues to connect to
            symbols: List of symbols to trade

        Returns:
            True if setup successful
        """
        try:
            if not NAUTILUS_AVAILABLE:
                logger.info(f"Mock setup: venues={venues}, symbols={symbols}")
                return True

            # Create node configuration
            config = TradingNodeConfig(
                trader_id="SHADOW-001",
                log_level="INFO",
            )

            # Add venue configurations (mock - real implementation would have actual configs)
            logger.info(f"Would configure venues: {venues}")
            logger.info(f"Would configure symbols: {symbols}")

            # Create trading node
            # self.node = TradingNode(config=config)

            logger.info("Setup shadow live trading node")
            return True

        except Exception as e:
            logger.error(f"Error setting up trading node: {e}")
            return False

    def setup_bridge(self) -> bool:
        """Setup Redis bridge for integration."""
        try:
            self.bridge = NautilusRedisBridge()
            logger.info("Setup Nautilus-Redis bridge")
            return True

        except Exception as e:
            logger.error(f"Error setting up bridge: {e}")
            return False

    def setup_strategy(self, symbols: List[str], shadow_mode: bool = True) -> bool:
        """
        Setup basis carry strategy for shadow execution.

        Args:
            symbols: Symbols to trade
            shadow_mode: Enable shadow mode (reduced sizes)

        Returns:
            True if setup successful
        """
        try:
            # Create shadow strategy config
            config = BasisCarryConfig()
            config.symbols = symbols

            if shadow_mode:
                # Reduce position sizes for shadow mode
                config.max_notional_usd = min(
                    config.max_notional_usd * 0.01, 1000
                )  # 1% size, max $1k
                config.min_notional_usd = max(
                    config.min_notional_usd * 0.1, 50
                )  # Reduce minimum

            self.strategy = NautilusBasisCarryStrategy(config)

            logger.info(
                f"Setup shadow strategy with max notional: ${config.max_notional_usd}"
            )
            return True

        except Exception as e:
            logger.error(f"Error setting up strategy: {e}")
            return False

    async def start(self) -> bool:
        """Start shadow live execution."""
        try:
            logger.info("🕵️ Starting Nautilus shadow live execution")

            if not NAUTILUS_AVAILABLE:
                logger.info("Running in mock mode")
                return await self._run_mock_shadow()

            # Start bridge
            if self.bridge:
                await self.bridge.start()

            # Start strategy
            if self.strategy:
                self.strategy.on_start()

            # Start trading node
            # if self.node:
            #     await self.node.start()

            self.running = True

            # Main execution loop
            await self._execution_loop()

            return True

        except Exception as e:
            logger.error(f"Error starting shadow execution: {e}")
            return False

    async def stop(self):
        """Stop shadow live execution."""
        try:
            logger.info("Stopping shadow live execution")
            self.running = False

            if self.strategy:
                self.strategy.on_stop()

            if self.bridge:
                await self.bridge.stop()

            # if self.node:
            #     await self.node.stop()

        except Exception as e:
            logger.error(f"Error stopping shadow execution: {e}")

    async def _execution_loop(self):
        """Main execution loop for shadow trading."""
        try:
            while self.running:
                # Monitor positions and performance
                await self._monitor_shadow_performance()

                # Sleep before next iteration
                await asyncio.sleep(10)  # Check every 10 seconds

        except Exception as e:
            logger.error(f"Error in execution loop: {e}")

    async def _monitor_shadow_performance(self):
        """Monitor shadow trading performance."""
        try:
            if not self.strategy:
                return

            # Get strategy metrics
            active_positions = len(self.strategy.active_positions)
            total_pnl = self.strategy.total_pnl
            total_trades = self.strategy.total_trades
            win_rate = self.strategy.metrics.get("basis_win_rate", 0)

            # Log performance periodically
            if total_trades > 0:
                logger.info(
                    f"📊 Shadow performance: {active_positions} active, "
                    f"{total_trades} trades, {win_rate:.1%} win rate, "
                    f"${total_pnl:+.2f} P&L"
                )

        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")

    async def _run_mock_shadow(self) -> bool:
        """Run mock shadow execution for testing."""
        try:
            logger.info("Running mock shadow execution...")

            for i in range(60):  # Run for 1 minute
                # Simulate shadow trading activity
                if i % 20 == 0:
                    logger.info(f"Mock shadow tick {i}: monitoring opportunities...")

                await asyncio.sleep(1)

            logger.info("Mock shadow execution completed")
            return True

        except Exception as e:
            logger.error(f"Error in mock shadow execution: {e}")
            return False

    def get_status(self) -> dict:
        """Get shadow execution status."""
        status = {
            "running": self.running,
            "nautilus_available": NAUTILUS_AVAILABLE,
            "strategy_active": self.strategy is not None,
            "bridge_active": self.bridge is not None,
        }

        if self.strategy:
            status.update(
                {
                    "active_positions": len(self.strategy.active_positions),
                    "total_trades": self.strategy.total_trades,
                    "total_pnl": self.strategy.total_pnl,
                    "win_rate": self.strategy.metrics.get("basis_win_rate", 0),
                }
            )

        return status


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Nautilus Shadow Live Execution")

    parser.add_argument(
        "--venues",
        nargs="+",
        default=["BINANCE", "DERIBIT"],
        help="Venues to connect to",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["BTC", "ETH", "SOL"], help="Symbols to trade"
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        default=True,
        help="Enable shadow mode (reduced sizes)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without actual execution"
    )

    return parser.parse_args()


async def main():
    """Main function to run shadow live execution."""
    args = parse_args()

    logger.info("🌊 Starting Nautilus Shadow Live Execution")
    logger.info(f"  Venues: {args.venues}")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  Shadow mode: {args.shadow}")

    try:
        # Create runner
        runner = NautilusShadowLiveRunner()

        # Setup components
        if not runner.setup_node(args.venues, args.symbols):
            logger.error("Failed to setup trading node")
            return 1

        if not runner.setup_bridge():
            logger.error("Failed to setup bridge")
            return 1

        if not runner.setup_strategy(args.symbols, args.shadow):
            logger.error("Failed to setup strategy")
            return 1

        # Start execution
        try:
            await runner.start()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await runner.stop()

        # Final status
        final_status = runner.get_status()
        logger.info(f"Final status: {json.dumps(final_status, indent=2)}")

        logger.info("✅ Shadow live execution completed")
        return 0

    except Exception as e:
        logger.error(f"Error in shadow live execution: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
