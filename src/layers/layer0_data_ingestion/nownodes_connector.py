#!/usr/bin/env python3
"""
NOWNodes WebSocket Connector for Real-time Crypto Data

Connects to NOWNodes WebSocket API to fetch live crypto data
and publishes metrics to the FastAPI metrics endpoint.
"""

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
import websockets
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Demo mode control
DEMO = bool(os.getenv("DEMO_MODE", "false").lower() == "true")

logger = logging.getLogger(__name__)

class NOWNodesConnector:
    """NOWNodes WebSocket connector for real-time crypto data."""
    
    def __init__(self, api_key: str, endpoint: str, metrics_url: str = "http://localhost:8001"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.metrics_url = metrics_url
        self.websockets = {}
        self.tick_counts = {}
        self.running = False
        
        # WebSocket endpoint mapping
        self.endpoint_map = {
            "BTC": os.getenv("NOWNODES_BTC_WS", "wss://btc.blockbook.ws.nownodes.io/?api_key=" + api_key),
            "ETH": os.getenv("NOWNODES_ETH_WS", "wss://eth.blockbook.ws.nownodes.io/?api_key=" + api_key), 
            "SOL": os.getenv("NOWNODES_SOL_WS", "wss://solana.blockbook.ws.nownodes.io/?api_key=" + api_key)
        }
        
    async def connect_symbol(self, symbol: str) -> None:
        """Connect to WebSocket for a specific symbol."""
        if DEMO:
            logger.warning(f"Running in DEMO fake-price mode for {symbol} – NO real WS traffic")
            await self._simulate_data_stream(symbol)
            return
            
        try:
            url = self.endpoint_map.get(symbol)
            if not url:
                logger.error(f"No WebSocket endpoint configured for {symbol}")
                await self._simulate_data_stream(symbol)
                return
                
            logger.info(f"Connecting to real NOWNodes WebSocket for {symbol}: {url}")
            
            async with websockets.connect(url, ping_interval=20) as ws:
                logger.info(f"✅ WebSocket connected for {symbol}")
                
                # Optional: Send subscription message if needed
                # await ws.send('{"event":"subscribe","data":"prices"}')
                
                async for raw_message in ws:
                    try:
                        msg = json.loads(raw_message)
                        
                        # Parse NOWNodes message format
                        # Adjust based on actual NOWNodes schema
                        if "timestamp" in msg and "price" in msg:
                            ts = datetime.fromtimestamp(msg["timestamp"], timezone.utc)
                            price = float(msg.get("price", msg.get("bestBid", msg.get("last", 0))))
                        else:
                            # Fallback for different message formats
                            ts = datetime.now(timezone.utc)
                            price = float(msg.get("bestBid", msg.get("price", msg.get("last", 0))))
                        
                        if price > 0:
                            tick_data = {
                                "symbol": f"{symbol}USDT",
                                "price": price,
                                "timestamp": ts.isoformat(),
                                "volume": msg.get("volume", 1000),
                                "source": "nownodes_real"
                            }
                            
                            # Process tick
                            await self._process_tick(tick_data)
                            self._update_metrics(symbol, tick_data)
                            
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse message for {symbol}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing message for {symbol}: {e}")
                        
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed for {symbol}: {e}")
            # For SOL, fall back to simulation if connection fails
            if symbol == "SOL":
                logger.info(f"Falling back to simulation for {symbol}")
                await self._simulate_data_stream(symbol)
            else:
                raise
                
        except Exception as e:
            logger.error(f"Failed to connect to {symbol} WebSocket: {e}")
            # Fall back to simulation
            logger.info(f"Falling back to simulation for {symbol}")
            await self._simulate_data_stream(symbol)
    
    async def _simulate_data_stream(self, symbol: str) -> None:
        """Simulate real-time data stream for testing."""
        import random
        
        base_prices = {
            "BTC": 45000,
            "ETH": 2500,
            "SOL": 100
        }
        
        base_price = base_prices.get(symbol, 1000)
        current_price = base_price
        
        source = "nownodes_sim" if DEMO else "nownodes_fallback"
        logger.info(f"Starting simulated data stream for {symbol} (source: {source})")
        
        while self.running:
            try:
                # Simulate price movement with realistic precision
                change = random.uniform(-0.005, 0.005)  # ±0.5% change
                current_price *= (1 + change)
                
                # Use realistic precision (2 decimal places for realistic prices)
                current_price = round(current_price, 2)
                
                # Create tick data
                tick_data = {
                    "symbol": f"{symbol}USDT",
                    "price": current_price,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "volume": random.uniform(1000, 10000),
                    "source": source
                }
                
                # Process tick
                await self._process_tick(tick_data)
                
                # Update metrics
                self._update_metrics(symbol, tick_data)
                
                # Wait before next tick (1 second for demo)
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in data stream for {symbol}: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_tick(self, tick_data: Dict) -> None:
        """Process a single tick of data."""
        symbol = tick_data["symbol"]
        
        # Update tick count
        if symbol not in self.tick_counts:
            self.tick_counts[symbol] = 0
        self.tick_counts[symbol] += 1
        
        # Log every 10th tick with full precision
        if self.tick_counts[symbol] % 10 == 0:
            price_str = f"${tick_data['price']:.2f}" if tick_data['price'] < 1000 else f"${tick_data['price']:,.2f}"
            logger.info(f"Processed {self.tick_counts[symbol]} ticks for {symbol} - Price: {price_str} (source: {tick_data['source']})")
    
    def _update_metrics(self, symbol: str, tick_data: Dict) -> None:
        """Update metrics via HTTP API."""
        try:
            # Update counter
            counter_url = f"{self.metrics_url}/metrics/counter/crypto_ticks_total"
            counter_data = {
                "labels": {
                    "symbol": tick_data["symbol"],
                    "source": tick_data["source"]
                }
            }
            
            # Update gauge for price (full precision)
            gauge_url = f"{self.metrics_url}/metrics/gauge/crypto_price_usd"
            gauge_data = {
                "value": tick_data["price"],
                "labels": {
                    "symbol": tick_data["symbol"]
                }
            }
            
            # Send metrics (fire and forget)
            requests.post(counter_url, json=counter_data, timeout=1)
            requests.post(gauge_url, json=gauge_data, timeout=1)
            
        except Exception as e:
            # Don't let metrics errors stop data processing
            logger.debug(f"Metrics update failed: {e}")
    
    async def start(self, symbols: List[str], duration_minutes: int = 30) -> None:
        """Start the connector for specified symbols."""
        self.running = True
        
        mode_str = "DEMO simulation" if DEMO else "LIVE NOWNodes WebSocket"
        logger.info(f"Starting NOWNodes connector in {mode_str} mode")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Duration: {duration_minutes} minutes")
        
        # Start WebSocket connections for each symbol
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self.connect_symbol(symbol))
            tasks.append(task)
        
        # Run for specified duration
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        try:
            while time.time() < end_time and self.running:
                await asyncio.sleep(1)
                
                # Print status every 60 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 60 == 0 and int(elapsed) > 0:
                    total_ticks = sum(self.tick_counts.values())
                    logger.info(f"Status: {elapsed/60:.1f}min elapsed, {total_ticks} total ticks")
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        
        finally:
            logger.info("Shutting down NOWNodes connector")
            self.running = False
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Print final statistics
            total_ticks = sum(self.tick_counts.values())
            elapsed = time.time() - start_time
            logger.info(f"Session complete – exiting after {elapsed/60:.1f} minutes")
            logger.info(f"Final statistics: {total_ticks} total ticks")
            for symbol, count in self.tick_counts.items():
                logger.info(f"  {symbol}: {count} ticks")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NOWNodes WebSocket Connector")
    parser.add_argument("--symbols", default="BTC,ETH,SOL", help="Comma-separated symbols")
    parser.add_argument("--duration-min", type=int, default=30, help="Duration in minutes")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Configure logging
    log_config = {
        "level": getattr(logging, args.log_level.upper()),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S"
    }
    
    if args.log_file:
        log_config["filename"] = args.log_file
    
    logging.basicConfig(**log_config)
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Get credentials from environment
    api_key = os.getenv("NOWNODES_WS_APIKEY", "demo_key")
    endpoint = os.getenv("NOWNODES_WS_ENDPOINT", "wss://demo.endpoint")
    
    if api_key == "demo_key":
        logger.warning("Using demo API key - falling back to simulation mode")
        os.environ["DEMO_MODE"] = "true"
        global DEMO
        DEMO = True
    
    # Create and start connector
    connector = NOWNodesConnector(api_key, endpoint)
    
    try:
        asyncio.run(connector.start(symbols, args.duration_min))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Connector failed: {e}")
        raise

if __name__ == "__main__":
    main() 