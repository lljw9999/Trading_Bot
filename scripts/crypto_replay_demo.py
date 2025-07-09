#!/usr/bin/env python3
"""
Simple crypto replay demo for the trading system.
Reads CSV data and processes it through the L0-L2 pipeline.
"""

import pandas as pd
import time
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import requests

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from layers.layer1_alpha_models.ob_pressure import OrderBookPressureAlpha
from layers.layer1_alpha_models.ma_momentum import MovingAverageMomentumAlpha

class CryptoReplayDemo:
    def __init__(self, data_dir: str, speed_multiplier: float = 5.0, metrics_url: str = "http://localhost:8001"):
        self.data_dir = Path(data_dir)
        self.speed_multiplier = speed_multiplier
        self.metrics_url = metrics_url
        
        # Initialize alpha models
        self.obp = OrderBookPressureAlpha()
        self.mam = MovingAverageMomentumAlpha()
        
        # Statistics
        self.tick_count = 0
        self.signal_count = 0
        self.start_time = None
        self.pnl = 0.0
        self.positions = {}
        
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load CSV data for a symbol."""
        csv_path = self.data_dir / f"{symbol}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def simulate_order_book(self, price: float, volume: float) -> tuple:
        """Simulate order book data from OHLCV."""
        # Simple simulation: bid/ask spread around mid price
        spread_pct = 0.001  # 0.1% spread
        spread = price * spread_pct
        
        bid_price = price - spread / 2
        ask_price = price + spread / 2
        
        # Volume distribution (roughly 60/40 split)
        bid_size = volume * 0.6
        ask_size = volume * 0.4
        
        return bid_price, ask_price, bid_size, ask_size
    
    def update_metrics(self, symbol: str, signals: dict, price: float) -> None:
        """Update metrics via HTTP API."""
        try:
            # Update tick counter
            counter_url = f"{self.metrics_url}/metrics/counter/crypto_ticks_total"
            counter_data = {
                "labels": {
                    "symbol": symbol,
                    "source": "replay"
                }
            }
            requests.post(counter_url, json=counter_data, timeout=1)
            
            # Update price gauge
            gauge_url = f"{self.metrics_url}/metrics/gauge/crypto_price_usd"
            gauge_data = {
                "value": price,
                "labels": {
                    "symbol": symbol
                }
            }
            requests.post(gauge_url, json=gauge_data, timeout=1)
            
            # Update alpha signal metrics
            for signal_type, signal_data in signals.items():
                if signal_type == 'ensemble':
                    continue
                    
                # Edge metric
                edge_url = f"{self.metrics_url}/metrics/gauge/alpha_signal_edge_bps"
                edge_data = {
                    "value": signal_data['edge_bps'],
                    "labels": {
                        "symbol": symbol,
                        "model": signal_type
                    }
                }
                requests.post(edge_url, json=edge_data, timeout=1)
                
                # Confidence metric
                conf_url = f"{self.metrics_url}/metrics/gauge/alpha_signal_confidence"
                conf_data = {
                    "value": signal_data['confidence'],
                    "labels": {
                        "symbol": symbol,
                        "model": signal_type
                    }
                }
                requests.post(conf_url, json=conf_data, timeout=1)
            
            # Update P&L if we have positions
            if symbol in self.positions:
                pnl_url = f"{self.metrics_url}/metrics/gauge/portfolio_pnl_usd"
                pnl_data = {
                    "value": self.pnl,
                    "labels": {
                        "symbol": symbol
                    }
                }
                requests.post(pnl_url, json=pnl_data, timeout=1)
                
        except Exception as e:
            # Don't let metrics errors stop the demo
            pass
    
    def simulate_trading(self, symbol: str, signals: dict, price: float) -> None:
        """Simulate simple trading based on ensemble signals."""
        if 'ensemble' not in signals:
            return
            
        edge = signals['ensemble']['edge_bps']
        confidence = signals['ensemble']['confidence']
        
        # Simple trading logic: trade if edge > 10bp and confidence > 0.6
        if abs(edge) > 10 and confidence > 0.6:
            position_size = 1000  # $1000 position
            
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'size': 0,
                    'entry_price': 0,
                    'unrealized_pnl': 0
                }
            
            pos = self.positions[symbol]
            
            # Close existing position if signal flips
            if (pos['size'] > 0 and edge < 0) or (pos['size'] < 0 and edge > 0):
                # Close position
                pnl = pos['size'] * (price - pos['entry_price'])
                self.pnl += pnl
                pos['size'] = 0
                pos['entry_price'] = 0
                pos['unrealized_pnl'] = 0
            
            # Open new position
            if pos['size'] == 0:
                pos['size'] = position_size if edge > 0 else -position_size
                pos['entry_price'] = price
            
            # Update unrealized P&L
            if pos['size'] != 0:
                pos['unrealized_pnl'] = pos['size'] * (price - pos['entry_price'])
    
    def process_tick(self, symbol: str, row: pd.Series) -> dict:
        """Process a single tick through the alpha models."""
        price = row['close']
        volume = row['volume']
        timestamp = row['timestamp'].isoformat()
        
        # Simulate order book
        bid_price, ask_price, bid_size, ask_size = self.simulate_order_book(price, volume)
        
        signals = {}
        
        # Generate OBP signal
        obp_signal = self.obp.generate_signal(
            symbol=symbol,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp=timestamp
        )
        
        if obp_signal:
            signals['obp'] = {
                'edge_bps': obp_signal.edge_bps,
                'confidence': obp_signal.confidence,
                'reasoning': obp_signal.reasoning
            }
        
        # Generate MAM signal
        mam_signal = self.mam.update_price(
            symbol=symbol,
            price=price,
            timestamp=timestamp
        )
        
        if mam_signal:
            signals['mam'] = {
                'edge_bps': mam_signal.edge_bps,
                'confidence': mam_signal.confidence,
                'reasoning': mam_signal.reasoning
            }
        
        # Simple ensemble if we have both signals
        ensemble_edge = 0.0
        ensemble_conf = 0.0
        
        if 'obp' in signals and 'mam' in signals:
            # Logistic blending
            import math
            obp_scaled = signals['obp']['edge_bps'] / 100.0
            mam_scaled = signals['mam']['edge_bps'] / 100.0
            
            logit = 1.0 * obp_scaled + 1.0 * mam_scaled
            try:
                prob = 1.0 / (1.0 + math.exp(-logit))
            except OverflowError:
                prob = 0.5
            
            ensemble_edge = (prob - 0.5) * 100
            ensemble_conf = prob * (signals['obp']['confidence'] + signals['mam']['confidence']) / 2.0
            
            signals['ensemble'] = {
                'edge_bps': ensemble_edge,
                'confidence': ensemble_conf
            }
            self.signal_count += 1
        
        # Simulate trading
        self.simulate_trading(symbol, signals, price)
        
        # Update metrics
        self.update_metrics(symbol, signals, price)
        
        return {
            'symbol': symbol,
            'timestamp': timestamp,
            'price': price,
            'volume': volume,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'signals': signals
        }
    
    def run_replay(self, symbols: list, max_ticks: int = None):
        """Run the replay for specified symbols."""
        print(f"🚀 Starting crypto replay demo")
        print(f"📊 Symbols: {', '.join(symbols)}")
        print(f"⚡ Speed: {self.speed_multiplier}x")
        print(f"📁 Data dir: {self.data_dir}")
        print(f"📈 Metrics: {self.metrics_url}")
        print()
        
        # Load data for all symbols
        data = {}
        for symbol in symbols:
            try:
                df = self.load_data(symbol)
                data[symbol] = df
                print(f"✅ Loaded {len(df)} ticks for {symbol}")
            except FileNotFoundError as e:
                print(f"❌ {e}")
                return
        
        print()
        
        # Find common time range
        min_length = min(len(df) for df in data.values())
        if max_ticks:
            min_length = min(min_length, max_ticks)
        
        print(f"📈 Processing {min_length} ticks per symbol...")
        print()
        
        self.start_time = time.time()
        
        # Process ticks
        for i in range(min_length):
            for symbol in symbols:
                row = data[symbol].iloc[i]
                result = self.process_tick(symbol, row)
                
                # Print progress
                if i % 50 == 0 or 'ensemble' in result['signals']:
                    self.print_tick_info(result, i, min_length)
                
                self.tick_count += 1
            
            # Sleep to simulate real-time (adjusted by speed multiplier)
            if self.speed_multiplier > 0:
                time.sleep(60.0 / self.speed_multiplier)  # 1 minute per tick
        
        self.print_summary()
    
    def print_tick_info(self, result: dict, tick_num: int, total_ticks: int):
        """Print information about a processed tick."""
        progress = (tick_num + 1) / total_ticks * 100
        
        signals_info = []
        for signal_type, signal_data in result['signals'].items():
            edge = signal_data['edge_bps']
            conf = signal_data['confidence']
            signals_info.append(f"{signal_type.upper()}={edge:+.1f}bp({conf:.2f})")
        
        # Add P&L info if we have positions
        symbol = result['symbol']
        if symbol in self.positions and self.positions[symbol]['size'] != 0:
            pnl = self.positions[symbol]['unrealized_pnl']
            signals_info.append(f"P&L=${pnl:+.2f}")
        
        if signals_info:
            signals_str = " | ".join(signals_info)
            print(f"[{progress:5.1f}%] {result['symbol']} ${result['price']:.2f} | {signals_str}")
    
    def print_summary(self):
        """Print replay summary."""
        elapsed = time.time() - self.start_time
        ticks_per_sec = self.tick_count / elapsed if elapsed > 0 else 0
        
        print()
        print("🎉 Replay completed!")
        print(f"📊 Processed {self.tick_count:,} ticks in {elapsed:.1f}s ({ticks_per_sec:.1f} ticks/s)")
        print(f"🎯 Generated {self.signal_count} ensemble signals")
        print(f"💰 Total P&L: ${self.pnl:+.2f}")
        
        # Print position summary
        if self.positions:
            print("\n📈 Position Summary:")
            for symbol, pos in self.positions.items():
                if pos['size'] != 0:
                    print(f"  {symbol}: Size=${pos['size']:+.0f}, Entry=${pos['entry_price']:.2f}, P&L=${pos['unrealized_pnl']:+.2f}")
        
        print()
        print("💡 Next steps:")
        print("   • Open Grafana at http://localhost:3000")
        print("   • Check Prometheus at http://localhost:9090")
        print("   • View trading system metrics")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto replay demo")
    parser.add_argument("data_dir", nargs="?", default="data/crypto/2025-07-02",
                       help="Data directory path")
    parser.add_argument("--speed", type=float, default=5.0,
                       help="Replay speed multiplier (default: 5x)")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                       help="Symbols to replay")
    parser.add_argument("--max-ticks", type=int, help="Maximum ticks to process")
    parser.add_argument("--metrics-url", default="http://localhost:8001",
                       help="Metrics server URL")
    
    args = parser.parse_args()
    
    replay = CryptoReplayDemo(args.data_dir, args.speed, args.metrics_url)
    replay.run_replay(args.symbols, args.max_ticks)

if __name__ == "__main__":
    main() 