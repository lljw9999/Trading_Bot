#!/usr/bin/env python3
"""
Whale Alert API Integration for SAC-DiF Trading Bot
Tracks large cryptocurrency transfers that may signal market movements
"""

import requests
import redis
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class WhaleAlertMonitor:
    """Monitor large cryptocurrency transfers via Whale Alert API"""

    def __init__(self, api_key: str, redis_client=None):
        self.api_key = api_key
        self.base_url = "https://api.whale-alert.io/v1"
        self.redis_client = redis_client or redis.Redis(
            host="localhost", port=6379, decode_responses=True
        )

        # Whale thresholds (USD values)
        self.thresholds = {
            "bitcoin": 1_000_000,  # $1M+ BTC moves
            "ethereum": 500_000,  # $500k+ ETH moves
            "critical": 10_000_000,  # $10M+ critical alerts
        }

        # Exchange addresses for flow analysis
        self.known_exchanges = [
            "binance",
            "coinbase",
            "kraken",
            "bitfinex",
            "huobi",
            "okex",
            "bitstamp",
            "gemini",
        ]

    def fetch_transactions(self, start_time: Optional[int] = None) -> List[Dict]:
        """Fetch recent whale transactions"""
        if not start_time:
            # Default: last 10 minutes
            start_time = int((datetime.now() - timedelta(minutes=10)).timestamp())

        try:
            url = f"{self.base_url}/transactions"
            params = {"api_key": self.api_key, "start": start_time, "limit": 100}

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("transactions", [])
            else:
                print(f"❌ Whale Alert API error: {response.status_code}")
                return []

        except Exception as e:
            print(f"❌ Whale Alert fetch error: {e}")
            return []

    def classify_transfer(self, transaction: Dict) -> Dict:
        """Classify whale transfer and generate trading signals"""
        symbol = transaction.get("symbol", "").lower()
        amount_usd = transaction.get("amount_usd", 0)
        from_owner = transaction.get("from", {}).get("owner", "")
        to_owner = transaction.get("to", {}).get("owner", "")

        classification = {
            "symbol": symbol,
            "amount_usd": amount_usd,
            "timestamp": transaction.get("timestamp"),
            "hash": transaction.get("hash"),
            "signal": "NEUTRAL",
            "signal_strength": 0,
            "flow_type": "unknown",
            "exchange_involved": False,
        }

        # Detect exchange involvement
        from_exchange = any(ex in from_owner.lower() for ex in self.known_exchanges)
        to_exchange = any(ex in to_owner.lower() for ex in self.known_exchanges)

        classification["exchange_involved"] = from_exchange or to_exchange

        # Classify flow patterns
        if from_exchange and not to_exchange:
            # Exchange outflow (potentially bullish - accumulation)
            classification["flow_type"] = "exchange_outflow"
            classification["signal"] = "BULLISH"
            classification["signal_strength"] = min(
                5, int(amount_usd / 2_000_000)
            )  # 1-5 scale

        elif not from_exchange and to_exchange:
            # Exchange inflow (potentially bearish - selling pressure)
            classification["flow_type"] = "exchange_inflow"
            classification["signal"] = "BEARISH"
            classification["signal_strength"] = min(5, int(amount_usd / 2_000_000))

        elif from_exchange and to_exchange:
            # Exchange-to-exchange (neutral but worth tracking)
            classification["flow_type"] = "exchange_transfer"

        else:
            # Wallet-to-wallet (whale redistribution)
            classification["flow_type"] = "whale_transfer"
            # Large moves between unknown wallets can signal accumulation/distribution
            if amount_usd > 5_000_000:
                classification["signal"] = "WHALE_ACTIVITY"
                classification["signal_strength"] = 3

        return classification

    def generate_trading_signals(self, transactions: List[Dict]) -> Dict:
        """Analyze whale transactions and generate trading signals"""
        if not transactions:
            return {"status": "no_data"}

        btc_signals = []
        eth_signals = []

        for tx in transactions:
            if tx.get("amount_usd", 0) < 500_000:  # Skip small transfers
                continue

            classification = self.classify_transfer(tx)

            if classification["symbol"] == "bitcoin":
                btc_signals.append(classification)
            elif classification["symbol"] == "ethereum":
                eth_signals.append(classification)

        # Aggregate signals
        def aggregate_symbol_signals(signals):
            if not signals:
                return {"signal": "NEUTRAL", "strength": 0, "count": 0}

            bullish_strength = sum(
                s["signal_strength"] for s in signals if s["signal"] == "BULLISH"
            )
            bearish_strength = sum(
                s["signal_strength"] for s in signals if s["signal"] == "BEARISH"
            )

            net_strength = bullish_strength - bearish_strength

            if net_strength > 3:
                return {
                    "signal": "STRONG_BULLISH",
                    "strength": min(10, net_strength),
                    "count": len(signals),
                }
            elif net_strength > 0:
                return {
                    "signal": "BULLISH",
                    "strength": net_strength,
                    "count": len(signals),
                }
            elif net_strength < -3:
                return {
                    "signal": "STRONG_BEARISH",
                    "strength": abs(net_strength),
                    "count": len(signals),
                }
            elif net_strength < 0:
                return {
                    "signal": "BEARISH",
                    "strength": abs(net_strength),
                    "count": len(signals),
                }
            else:
                return {"signal": "NEUTRAL", "strength": 0, "count": len(signals)}

        return {
            "timestamp": datetime.now().isoformat(),
            "btc": aggregate_symbol_signals(btc_signals),
            "eth": aggregate_symbol_signals(eth_signals),
            "total_volume_usd": sum(tx.get("amount_usd", 0) for tx in transactions),
            "transaction_count": len(transactions),
            "raw_transactions": transactions[:10],  # Store sample for debugging
        }

    def store_whale_signals(self, signals: Dict):
        """Store whale signals in Redis for dashboard consumption"""
        try:
            # Store current signals
            self.redis_client.set("whale:current_signals", json.dumps(signals), ex=3600)

            # Add to time series
            timestamp = int(time.time() * 1000)

            for symbol in ["btc", "eth"]:
                if symbol in signals:
                    signal_data = signals[symbol]

                    whale_entry = {
                        "timestamp": timestamp,
                        "signal": signal_data["signal"],
                        "strength": signal_data["strength"],
                        "count": signal_data["count"],
                        "volume_usd": signals["total_volume_usd"],
                    }

                    self.redis_client.xadd(f"whale:{symbol}", whale_entry, maxlen=1000)

            print(
                f"✅ Stored whale signals: BTC={signals['btc']['signal']}, ETH={signals['eth']['signal']}"
            )

        except Exception as e:
            print(f"❌ Error storing whale signals: {e}")

    def run_monitoring_loop(self, interval_seconds: int = 300):
        """Run continuous whale monitoring"""
        print(f"🐋 Starting Whale Alert monitoring (every {interval_seconds}s)")

        last_check = int((datetime.now() - timedelta(minutes=10)).timestamp())

        while True:
            try:
                # Fetch recent transactions
                transactions = self.fetch_transactions(last_check)

                if transactions:
                    print(f"🔍 Found {len(transactions)} whale transactions")

                    # Generate signals
                    signals = self.generate_trading_signals(transactions)

                    # Store for dashboard
                    self.store_whale_signals(signals)

                    # Update last check time
                    last_check = max(
                        tx.get("timestamp", last_check) for tx in transactions
                    )

                    # Log significant signals
                    for symbol in ["btc", "eth"]:
                        if symbol in signals and signals[symbol]["strength"] > 2:
                            signal_info = signals[symbol]
                            print(
                                f"🚨 {symbol.upper()} Whale Signal: {signal_info['signal']} "
                                f"(strength: {signal_info['strength']}, count: {signal_info['count']})"
                            )
                else:
                    print("📊 No significant whale activity detected")

                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                print("\n🛑 Whale monitoring stopped")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(60)  # Wait before retry


def main():
    """Demo whale alert monitoring"""
    # Note: You need a Whale Alert API key from https://whale-alert.io/
    api_key = "DEMO_KEY_GET_FROM_WHALE_ALERT"

    if api_key == "DEMO_KEY_GET_FROM_WHALE_ALERT":
        print("⚠️  Please get a real API key from https://whale-alert.io/")
        print("💡 For demo, simulating whale alerts...")

        # Simulate whale signals for demo
        redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        demo_signals = {
            "timestamp": datetime.now().isoformat(),
            "btc": {"signal": "BULLISH", "strength": 4, "count": 3},
            "eth": {"signal": "BEARISH", "strength": 2, "count": 1},
            "total_volume_usd": 15_000_000,
            "transaction_count": 4,
            "raw_transactions": [],
        }

        monitor = WhaleAlertMonitor(api_key, redis_client)
        monitor.store_whale_signals(demo_signals)
        print("✅ Demo whale signals stored in Redis")

    else:
        monitor = WhaleAlertMonitor(api_key)
        monitor.run_monitoring_loop(interval_seconds=300)  # Check every 5 minutes


if __name__ == "__main__":
    main()
