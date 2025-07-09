#!/usr/bin/env python3
"""
Stock Replay Demo for Trading System (L0-L5)

Reads NVDA CSV data and processes it through the complete 6-layer pipeline:
- L0: Data ingestion (Redis)
- L1: Alpha models (OBP + MAM)
- L2: Ensemble blending
- L3: Kelly position sizing
- L4: Paper trading execution
- L5: Risk management
"""

import pandas as pd
import time
import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import requests
import redis
import math

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from layers.layer1_alpha_models.ob_pressure import OrderBookPressureAlpha
from layers.layer1_alpha_models.ma_momentum import MovingAverageMomentumAlpha

class StockReplayDemo:
    def __init__(self, symbol: str, speed_multiplier: float = 30.0, 
                 metrics_url: str = "http://localhost:8001",
                 redis_host: str = "localhost", redis_port: int = 6379):
        self.symbol = symbol
        self.speed_multiplier = speed_multiplier
        self.metrics_url = metrics_url
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
            logging.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            logging.warning(f"Redis connection failed: {e}. Continuing without Redis.")
            self.redis_client = None
        
        # Initialize alpha models
        self.obp = OrderBookPressureAlpha()
        self.mam = MovingAverageMomentumAlpha()
        
        # Trading state
        self.position_usd = 0.0  # Current position in USD
        self.cash_usd = 100000.0  # Starting cash
        self.pnl_cumulative_usd = 0.0
        self.max_position_pct = 0.25  # 25% max position size (Kelly cap)
        self.risk_events = 0
        
        # Statistics
        self.tick_count = 0
        self.signal_count = 0
        self.start_time = None
        self.trades = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load CSV data from file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Clean volume data (remove underscores)
        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype(str).str.replace('_', '').astype(float)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def simulate_order_book(self, price: float, volume: float) -> tuple:
        """Simulate order book data from OHLCV."""
        # Tighter spread for stocks (0.05% vs 0.1% for crypto)
        spread_pct = 0.0005  
        spread = price * spread_pct
        
        bid_price = price - spread / 2
        ask_price = price + spread / 2
        
        # Volume distribution (roughly 55/45 split for stocks)
        bid_size = volume * 0.55
        ask_size = volume * 0.45
        
        return bid_price, ask_price, bid_size, ask_size
    
    def publish_to_redis(self, data: dict) -> None:
        """Publish tick data to Redis (L0)."""
        if not self.redis_client:
            return
            
        try:
            # Publish to stock ticks channel
            channel = f"stock.ticks.{self.symbol}"
            self.redis_client.publish(channel, json.dumps(data))
            
            # Store in Redis with expiration
            key = f"stock:tick:{self.symbol}:latest"
            self.redis_client.setex(key, 300, json.dumps(data))  # 5 min expiration
            
        except Exception as e:
            logging.debug(f"Redis publish failed: {e}")
    
    def update_metrics(self, tick_data: dict, signals: dict, position_usd: float, pnl: float) -> None:
        """Update metrics via HTTP API."""
        try:
            # L0: Tick counter
            counter_url = f"{self.metrics_url}/metrics/counter/stock_ticks_total"
            counter_data = {"labels": {"symbol": self.symbol}}
            requests.post(counter_url, json=counter_data, timeout=1)
            
            # L0: Price gauge
            price_url = f"{self.metrics_url}/metrics/gauge/stock_price_usd"
            price_data = {"value": tick_data['close'], "labels": {"symbol": self.symbol}}
            requests.post(price_url, json=price_data, timeout=1)
            
            # L1: Alpha signal metrics
            for signal_type, signal_data in signals.items():
                if signal_type == 'ensemble':
                    continue
                    
                # Edge metric
                edge_url = f"{self.metrics_url}/metrics/gauge/alpha_signal_edge_bps"
                edge_data = {
                    "value": signal_data['edge_bps'],
                    "labels": {"symbol": self.symbol, "model": signal_type}
                }
                requests.post(edge_url, json=edge_data, timeout=1)
                
                # Confidence metric
                conf_url = f"{self.metrics_url}/metrics/gauge/alpha_signal_confidence"
                conf_data = {
                    "value": signal_data['confidence'],
                    "labels": {"symbol": self.symbol, "model": signal_type}
                }
                requests.post(conf_url, json=conf_data, timeout=1)
            
            # L2: Ensemble metrics
            if 'ensemble' in signals:
                ensemble_edge_url = f"{self.metrics_url}/metrics/gauge/alpha_signal_edge_bps"
                ensemble_edge_data = {
                    "value": signals['ensemble']['edge_bps'],
                    "labels": {"symbol": self.symbol, "model": "ensemble"}
                }
                requests.post(ensemble_edge_url, json=ensemble_edge_data, timeout=1)
            
            # L3: Position size
            position_url = f"{self.metrics_url}/metrics/gauge/position_usd"
            position_data = {"value": position_usd, "labels": {"symbol": self.symbol}}
            requests.post(position_url, json=position_data, timeout=1)
            
            # L4: Cumulative P&L
            pnl_url = f"{self.metrics_url}/metrics/gauge/pnl_cumulative_usd"
            pnl_data = {"value": pnl, "labels": {"symbol": self.symbol}}
            requests.post(pnl_url, json=pnl_data, timeout=1)
            
            # L5: Risk events
            risk_url = f"{self.metrics_url}/metrics/gauge/risk_events_total"
            risk_data = {"value": self.risk_events, "labels": {"symbol": self.symbol}}
            requests.post(risk_url, json=risk_data, timeout=1)
                
        except Exception as e:
            # Don't let metrics errors stop the demo
            logging.debug(f"Metrics update failed: {e}")
    
    def kelly_sizing(self, edge_bps: float, confidence: float, price: float) -> float:
        """L3: Kelly position sizing with 25% cap."""
        if abs(edge_bps) < 5.0 or confidence < 0.6:  # Minimum thresholds
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # Where bp = win probability, q = loss probability, b = win/loss ratio
        # Simplified: assume win/loss ratio = 1, so f = 2*bp - 1
        
        win_prob = confidence
        kelly_fraction = 2 * win_prob - 1
        
        # Apply edge scaling
        kelly_fraction *= abs(edge_bps) / 100.0
        
        # Cap at 25% as specified
        kelly_fraction = min(kelly_fraction, self.max_position_pct)
        kelly_fraction = max(kelly_fraction, -self.max_position_pct)
        
        # Convert to USD position size
        portfolio_value = self.cash_usd + self.position_usd
        target_position_usd = kelly_fraction * portfolio_value
        
        # Apply direction based on edge sign
        if edge_bps < 0:
            target_position_usd = -abs(target_position_usd)
        
        return target_position_usd
    
    def execute_trade(self, target_position_usd: float, price: float) -> dict:
        """L4: Paper trading execution."""
        position_change = target_position_usd - self.position_usd
        
        if abs(position_change) < 100:  # Minimum $100 trade size
            return {"executed": False, "reason": "Below minimum trade size"}
        
        # Calculate shares to trade
        shares_to_trade = position_change / price
        
        # Execute the trade (paper)
        self.cash_usd -= position_change
        self.position_usd = target_position_usd
        
        # Calculate P&L from the trade
        if len(self.trades) > 0:
            # Update P&L based on price movement
            last_price = self.trades[-1]['price']
            price_change = price - last_price
            pnl_change = (self.position_usd / last_price) * price_change
            self.pnl_cumulative_usd += pnl_change
        
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "price": price,
            "shares": shares_to_trade,
            "position_change_usd": position_change,
            "new_position_usd": self.position_usd,
            "cash_remaining": self.cash_usd,
            "executed": True
        }
        
        self.trades.append(trade)
        return trade
    
    def risk_management(self, position_usd: float, price: float) -> dict:
        """L5: Risk management guard-rails."""
        risk_event = None
        
        portfolio_value = self.cash_usd + position_usd
        position_pct = abs(position_usd) / portfolio_value if portfolio_value > 0 else 0
        
        # Check position size limit
        if position_pct > self.max_position_pct:
            risk_event = {
                "type": "position_limit_breach",
                "current_pct": position_pct,
                "limit_pct": self.max_position_pct,
                "action": "reduce_position"
            }
            self.risk_events += 1
        
        # Check portfolio drawdown
        if self.pnl_cumulative_usd < -10000:  # $10k max drawdown
            risk_event = {
                "type": "drawdown_limit",
                "current_pnl": self.pnl_cumulative_usd,
                "limit": -10000,
                "action": "stop_trading"
            }
            self.risk_events += 1
        
        return risk_event
    
    def process_tick(self, row: pd.Series) -> dict:
        """Process a single tick through all 6 layers."""
        # Extract OHLCV data
        tick_data = {
            'timestamp': row['timestamp'].isoformat(),
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume'],
            'symbol': self.symbol
        }
        
        price = row['close']
        volume = row['volume']
        
        # L0: Data ingestion
        self.publish_to_redis(tick_data)
        
        # Simulate order book
        bid_price, ask_price, bid_size, ask_size = self.simulate_order_book(price, volume)
        
        signals = {}
        
        # L1: Alpha models
        # OBP (Order Book Pressure)
        obp_signal = self.obp.generate_signal(
            symbol=self.symbol,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp=tick_data['timestamp']
        )
        
        if obp_signal:
            signals['obp'] = {
                'edge_bps': obp_signal.edge_bps,
                'confidence': obp_signal.confidence,
                'reasoning': obp_signal.reasoning
            }
        
        # MAM (Moving Average Momentum)
        mam_signal = self.mam.update_price(
            symbol=self.symbol,
            price=price,
            timestamp=tick_data['timestamp']
        )
        
        if mam_signal:
            signals['mam'] = {
                'edge_bps': mam_signal.edge_bps,
                'confidence': mam_signal.confidence,
                'reasoning': mam_signal.reasoning
            }
        
        # L2: Ensemble (Logistic blending)
        ensemble_edge = 0.0
        ensemble_conf = 0.0
        
        if 'obp' in signals and 'mam' in signals:
            # Logistic blending
            obp_scaled = signals['obp']['edge_bps'] / 100.0
            mam_scaled = signals['mam']['edge_bps'] / 100.0
            
            logit = 1.0 * obp_scaled + 1.0 * mam_scaled
            try:
                prob = 1.0 / (1.0 + math.exp(-logit))
            except OverflowError:
                prob = 0.5
            
            ensemble_edge = (prob - 0.5) * 100
            ensemble_conf = (signals['obp']['confidence'] + signals['mam']['confidence']) / 2.0
            
            signals['ensemble'] = {
                'edge_bps': ensemble_edge,
                'confidence': ensemble_conf
            }
            self.signal_count += 1
        
        # L3: Kelly position sizing
        target_position = 0.0
        if 'ensemble' in signals:
            target_position = self.kelly_sizing(
                signals['ensemble']['edge_bps'],
                signals['ensemble']['confidence'],
                price
            )
        
        # L4: Execute trade
        trade_result = None
        if abs(target_position - self.position_usd) > 100:  # Minimum trade threshold
            trade_result = self.execute_trade(target_position, price)
        
        # L5: Risk management
        risk_event = self.risk_management(self.position_usd, price)
        
        # Update metrics
        self.update_metrics(tick_data, signals, self.position_usd, self.pnl_cumulative_usd)
        
        return {
            'tick_data': tick_data,
            'signals': signals,
            'target_position': target_position,
            'current_position': self.position_usd,
            'trade_result': trade_result,
            'risk_event': risk_event,
            'pnl_cumulative': self.pnl_cumulative_usd
        }
    
    def run_replay(self, file_path: str):
        """Run the complete replay through all 6 layers."""
        logging.info(f"🚀 Starting NVDA stock replay demo")
        logging.info(f"📊 Symbol: {self.symbol}")
        logging.info(f"⚡ Speed: {self.speed_multiplier}x")
        logging.info(f"📁 Data file: {file_path}")
        logging.info(f"📈 Metrics: {self.metrics_url}")
        logging.info(f"💰 Starting cash: ${self.cash_usd:,.2f}")
        
        # Load data
        try:
            df = self.load_data(file_path)
            logging.info(f"✅ Loaded {len(df)} ticks for {self.symbol}")
        except FileNotFoundError as e:
            logging.error(f"❌ {e}")
            return
        
        self.start_time = time.time()
        
        # Process each tick
        for i, (_, row) in enumerate(df.iterrows()):
            result = self.process_tick(row)
            
            # Print progress for significant events
            if (i % 10 == 0 or result['trade_result'] or result['risk_event'] or 
                'ensemble' in result['signals']):
                self.print_tick_info(result, i, len(df))
            
            self.tick_count += 1
            
            # Sleep to simulate real-time (adjusted by speed multiplier)
            if self.speed_multiplier > 0:
                time.sleep(60.0 / self.speed_multiplier)  # 1 minute per tick
        
        self.print_summary()
    
    def print_tick_info(self, result: dict, tick_num: int, total_ticks: int):
        """Print information about a processed tick."""
        progress = (tick_num + 1) / total_ticks * 100
        tick_data = result['tick_data']
        
        info_parts = [f"[{progress:5.1f}%] {self.symbol} ${tick_data['close']:.2f}"]
        
        # Add signals
        signals_info = []
        for signal_type, signal_data in result['signals'].items():
            edge = signal_data['edge_bps']
            conf = signal_data['confidence']
            signals_info.append(f"{signal_type.upper()}={edge:+.1f}bp({conf:.2f})")
        
        if signals_info:
            info_parts.append(" | ".join(signals_info))
        
        # Add position info
        if abs(result['current_position']) > 0:
            info_parts.append(f"POS=${result['current_position']:+.0f}")
        
        # Add P&L
        if abs(result['pnl_cumulative']) > 0:
            info_parts.append(f"P&L=${result['pnl_cumulative']:+.2f}")
        
        # Add trade info
        if result['trade_result'] and result['trade_result']['executed']:
            trade = result['trade_result']
            info_parts.append(f"TRADE={trade['shares']:+.0f}@${trade['price']:.2f}")
        
        # Add risk events
        if result['risk_event']:
            info_parts.append(f"RISK={result['risk_event']['type']}")
        
        logging.info(" | ".join(info_parts))
    
    def print_summary(self):
        """Print replay summary."""
        elapsed = time.time() - self.start_time
        ticks_per_sec = self.tick_count / elapsed if elapsed > 0 else 0
        
        logging.info("\n🎉 NVDA Replay completed!")
        logging.info(f"📊 Processed {self.tick_count:,} ticks in {elapsed:.1f}s ({ticks_per_sec:.1f} ticks/s)")
        logging.info(f"🎯 Generated {self.signal_count} ensemble signals")
        logging.info(f"💰 Final P&L: ${self.pnl_cumulative_usd:+.2f}")
        logging.info(f"📈 Final position: ${self.position_usd:+.2f}")
        logging.info(f"💵 Final cash: ${self.cash_usd:,.2f}")
        logging.info(f"⚠️  Risk events: {self.risk_events}")
        logging.info(f"🔄 Total trades: {len(self.trades)}")
        
        if self.trades:
            logging.info("\n📋 Trade Summary:")
            for i, trade in enumerate(self.trades[-3:]):  # Show last 3 trades
                logging.info(f"  {i+1}. {trade['shares']:+.0f} shares @ ${trade['price']:.2f} → ${trade['new_position_usd']:+.0f}")
        
        logging.info("\n💡 Next steps:")
        logging.info("   • Open Grafana at http://localhost:3000")
        logging.info("   • Check Prometheus at http://localhost:9090")
        logging.info("   • View NVDA trading metrics in dashboards")

def main():
    parser = argparse.ArgumentParser(description="NVDA Stock Replay Demo (L0-L5)")
    parser.add_argument("--file", required=True, help="Path to NVDA CSV file")
    parser.add_argument("--symbol", default="NVDA", help="Stock symbol")
    parser.add_argument("--speed", type=float, default=30.0, help="Replay speed multiplier")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--metrics-url", default="http://localhost:8001", help="Metrics server URL")
    
    args = parser.parse_args()
    
    # Configure logging
    log_config = {
        "level": logging.INFO,
        "format": "%(asctime)s - %(levelname)s - %(message)s",
        "datefmt": "%H:%M:%S"
    }
    
    if args.log_file:
        log_config["filename"] = args.log_file
    
    logging.basicConfig(**log_config)
    
    # Create and run replay
    replay = StockReplayDemo(args.symbol, args.speed, args.metrics_url)
    replay.run_replay(args.file)

if __name__ == "__main__":
    main() 