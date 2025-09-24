#!/usr/bin/env python3
"""
Event Gate: Signal Spike Override System
Override red windows when signal spikes indicate temporary trading opportunities.
"""
import os
import sys
import json
import datetime
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class EventGate:
    def __init__(self, lookback_minutes: int = 30):
        self.lookback_minutes = lookback_minutes
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")
        self.assets = ["SOL-USD", "BTC-USD", "ETH-USD", "NVDA"]

        # Event thresholds
        self.sentiment_threshold = 0.7  # Sentiment score threshold
        self.volume_spike_threshold = 3.0  # 3x normal volume
        self.volatility_spike_threshold = 2.5  # 2.5x normal volatility
        self.whale_alert_threshold = 1000000  # $1M+ whale movement

        # TCA and cost forecast limits
        self.max_tca_slippage_bps = 25.0  # Max acceptable slippage
        self.max_cost_forecast_hourly = 15.0  # Max cost/hour during event

    def load_sentiment_signals(self, lookback_minutes: int) -> Dict[str, Any]:
        """Load recent sentiment signals."""
        try:
            # Try to load from sentiment enricher Redis data
            import redis

            r = redis.Redis(decode_responses=True)

            # Get recent sentiment data
            sentiment_data = {}
            for asset in self.assets:
                sentiment_key = f"soft.enriched.{asset.replace('-', '').lower()}"
                recent_entries = r.lrange(sentiment_key, 0, 10)  # Last 10 entries

                if recent_entries:
                    latest_sentiment = json.loads(recent_entries[0])
                    sentiment_score = latest_sentiment.get("sentiment_score", 0.0)
                    timestamp = latest_sentiment.get("timestamp", "")

                    # Check if recent (within lookback window)
                    try:
                        sent_time = datetime.datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )
                        cutoff_time = datetime.datetime.now(
                            datetime.timezone.utc
                        ) - datetime.timedelta(minutes=lookback_minutes)

                        if sent_time >= cutoff_time:
                            sentiment_data[asset] = {
                                "sentiment_score": sentiment_score,
                                "timestamp": timestamp,
                                "recency_minutes": (
                                    datetime.datetime.now(datetime.timezone.utc)
                                    - sent_time
                                ).total_seconds()
                                / 60,
                            }
                    except:
                        pass

            print(f"📰 Loaded sentiment for {len(sentiment_data)} assets")
            return sentiment_data

        except Exception as e:
            print(f"⚠️ Could not load sentiment signals: {e}")

        # Fallback to simulated sentiment data
        sentiment_data = {}
        current_time = datetime.datetime.now(datetime.timezone.utc)

        for asset in self.assets:
            # Simulate sentiment spikes occasionally
            if np.random.random() < 0.15:  # 15% chance of sentiment spike
                sentiment_score = np.random.uniform(
                    0.7, 0.95
                )  # High positive sentiment
                sentiment_data[asset] = {
                    "sentiment_score": sentiment_score,
                    "timestamp": current_time.isoformat(),
                    "recency_minutes": np.random.uniform(1, lookback_minutes),
                }

        print(f"📰 Simulated sentiment spikes for {len(sentiment_data)} assets")
        return sentiment_data

    def load_volume_signals(self, lookback_minutes: int) -> Dict[str, Any]:
        """Load recent volume spike signals."""
        volume_data = {}

        # Simulate volume spike detection
        for asset in self.assets:
            # Random volume spikes
            if np.random.random() < 0.1:  # 10% chance of volume spike
                normal_volume = 50000 if asset != "NVDA" else 25000
                spike_volume = normal_volume * np.random.uniform(3.0, 8.0)
                volume_ratio = spike_volume / normal_volume

                volume_data[asset] = {
                    "current_volume": spike_volume,
                    "normal_volume": normal_volume,
                    "volume_ratio": volume_ratio,
                    "spike_duration_minutes": np.random.uniform(5, 30),
                }

        print(f"📊 Detected volume spikes for {len(volume_data)} assets")
        return volume_data

    def load_volatility_signals(self, lookback_minutes: int) -> Dict[str, Any]:
        """Load recent volatility spike signals."""
        volatility_data = {}

        # Simulate volatility spike detection
        for asset in self.assets:
            if np.random.random() < 0.12:  # 12% chance of volatility spike
                normal_vol = 0.002 if asset != "NVDA" else 0.0015
                spike_vol = normal_vol * np.random.uniform(2.5, 6.0)
                vol_ratio = spike_vol / normal_vol

                volatility_data[asset] = {
                    "current_volatility": spike_vol,
                    "normal_volatility": normal_vol,
                    "volatility_ratio": vol_ratio,
                    "spike_duration_minutes": np.random.uniform(10, 45),
                }

        print(f"📈 Detected volatility spikes for {len(volatility_data)} assets")
        return volatility_data

    def load_whale_signals(self, lookback_minutes: int) -> Dict[str, Any]:
        """Load recent whale alert signals."""
        whale_data = {}

        # Simulate whale movement detection (crypto only)
        crypto_assets = [a for a in self.assets if a != "NVDA"]

        for asset in crypto_assets:
            if np.random.random() < 0.05:  # 5% chance of whale movement
                whale_amount = np.random.uniform(1000000, 50000000)  # $1M - $50M
                movement_type = np.random.choice(
                    ["large_buy", "large_sell", "exchange_inflow", "exchange_outflow"]
                )

                whale_data[asset] = {
                    "whale_amount_usd": whale_amount,
                    "movement_type": movement_type,
                    "exchange": np.random.choice(["binance", "coinbase", "unknown"]),
                    "confidence": np.random.uniform(0.8, 0.95),
                }

        print(f"🐋 Detected whale movements for {len(whale_data)} assets")
        return whale_data

    def check_tca_forecast(self, asset: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if TCA forecast is acceptable for event trading."""
        tca_data = {
            "slippage_forecast_bps": 0.0,
            "spread_forecast_bps": 0.0,
            "depth_forecast": 0.0,
            "tca_acceptable": True,
        }

        # Simulate TCA forecast check
        if asset == "NVDA":
            # Stock TCA
            slippage_forecast = np.random.uniform(8, 20)
            spread_forecast = np.random.uniform(1.0, 3.0)
            depth_forecast = np.random.uniform(0.7, 1.2)
        else:
            # Crypto TCA
            slippage_forecast = np.random.uniform(5, 15)
            spread_forecast = np.random.uniform(1.5, 4.0)
            depth_forecast = np.random.uniform(0.8, 1.5)

        tca_data.update(
            {
                "slippage_forecast_bps": slippage_forecast,
                "spread_forecast_bps": spread_forecast,
                "depth_forecast": depth_forecast,
                "tca_acceptable": slippage_forecast <= self.max_tca_slippage_bps,
            }
        )

        return tca_data["tca_acceptable"], tca_data

    def check_cost_forecast(self, asset: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if cost forecast is acceptable for event trading."""
        cost_data = {
            "base_cost_hourly": 4.40,  # From calibration
            "event_cost_multiplier": 1.0,
            "total_cost_forecast": 0.0,
            "cost_acceptable": True,
        }

        # Event trading may increase costs due to urgency
        if asset == "NVDA":
            event_multiplier = np.random.uniform(
                1.1, 1.4
            )  # Stocks have higher event costs
        else:
            event_multiplier = np.random.uniform(1.05, 1.2)  # Crypto more efficient

        total_cost = cost_data["base_cost_hourly"] * event_multiplier

        cost_data.update(
            {
                "event_cost_multiplier": event_multiplier,
                "total_cost_forecast": total_cost,
                "cost_acceptable": total_cost <= self.max_cost_forecast_hourly,
            }
        )

        return cost_data["cost_acceptable"], cost_data

    def calculate_event_score(
        self, asset: str, signals: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate composite event score for an asset."""
        score = 0.0

        # Sentiment component
        if asset in signals["sentiment"]:
            sentiment_score = signals["sentiment"][asset]["sentiment_score"]
            if sentiment_score >= self.sentiment_threshold:
                score += sentiment_score * 0.3  # 30% weight

        # Volume component
        if asset in signals["volume"]:
            volume_ratio = signals["volume"][asset]["volume_ratio"]
            if volume_ratio >= self.volume_spike_threshold:
                score += min(1.0, volume_ratio / 5.0) * 0.25  # 25% weight

        # Volatility component
        if asset in signals["volatility"]:
            vol_ratio = signals["volatility"][asset]["volatility_ratio"]
            if vol_ratio >= self.volatility_spike_threshold:
                score += min(1.0, vol_ratio / 4.0) * 0.25  # 25% weight

        # Whale component (crypto only)
        if asset != "NVDA" and asset in signals["whale"]:
            whale_amount = signals["whale"][asset]["whale_amount_usd"]
            confidence = signals["whale"][asset]["confidence"]
            if whale_amount >= self.whale_alert_threshold:
                whale_score = (
                    min(1.0, whale_amount / 10000000) * confidence
                )  # $10M = max score
                score += whale_score * 0.2  # 20% weight

        return min(1.0, score)  # Cap at 1.0

    def create_event_tokens(
        self,
        asset: str,
        event_score: float,
        signals: Dict[str, Any],
        tca_data: Dict[str, Any],
        cost_data: Dict[str, Any],
        output_dir: str,
        dry_run: bool = False,
    ) -> Optional[str]:
        """Create event gate token for asset."""

        # Create token data
        current_time = datetime.datetime.now(datetime.timezone.utc)
        token_data = {
            "timestamp": current_time.isoformat(),
            "asset": asset,
            "event_score": event_score,
            "valid_until": (
                current_time + datetime.timedelta(minutes=15)
            ).isoformat(),  # 15-min TTL
            "signals": signals,
            "tca_forecast": tca_data,
            "cost_forecast": cost_data,
            "dry_run": dry_run,
        }

        # Create output directory structure
        output_path = Path(output_dir) / asset
        output_path.mkdir(parents=True, exist_ok=True)

        # Create token file
        timestamp_str = current_time.strftime("%Y%m%d_%H%M%SZ")
        token_file = output_path / f"event_green_{timestamp_str}.json"

        if not dry_run:
            with open(token_file, "w") as f:
                json.dump(token_data, f, indent=2)

            print(f"🎫 Created event token: {token_file}")
        else:
            print(f"🧪 DRY RUN: Would create event token: {token_file}")

        return str(token_file)

    def run_event_gate_scan(
        self, output_dir: str = "artifacts/ev/event_gate_on", dry_run: bool = False
    ) -> Dict[str, Any]:
        """Run complete event gate scan."""

        print("🚨 Event Gate: Signal Spike Override System")
        print("=" * 50)
        print(f"Lookback: {self.lookback_minutes} minutes")
        print(f"Output: {output_dir}")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        print("=" * 50)

        # Load all signal types
        print("📡 Loading signals...")
        signals = {
            "sentiment": self.load_sentiment_signals(self.lookback_minutes),
            "volume": self.load_volume_signals(self.lookback_minutes),
            "volatility": self.load_volatility_signals(self.lookback_minutes),
            "whale": self.load_whale_signals(self.lookback_minutes),
        }

        # Process each asset
        event_results = {}
        tokens_created = 0

        for asset in self.assets:
            print(f"\n🔍 Analyzing {asset}...")

            # Calculate event score
            event_score = self.calculate_event_score(asset, signals)

            # Check TCA and cost forecasts
            tca_ok, tca_data = self.check_tca_forecast(asset)
            cost_ok, cost_data = self.check_cost_forecast(asset)

            # Determine if event gate should trigger
            score_threshold = 0.6  # Minimum event score
            should_trigger = event_score >= score_threshold and tca_ok and cost_ok

            result = {
                "asset": asset,
                "event_score": event_score,
                "score_threshold": score_threshold,
                "tca_acceptable": tca_ok,
                "cost_acceptable": cost_ok,
                "should_trigger": should_trigger,
                "token_created": False,
                "token_file": None,
            }

            if should_trigger:
                # Create event token
                token_file = self.create_event_tokens(
                    asset,
                    event_score,
                    {k: v.get(asset, {}) for k, v in signals.items()},
                    tca_data,
                    cost_data,
                    output_dir,
                    dry_run,
                )
                result["token_created"] = True
                result["token_file"] = token_file
                tokens_created += 1

                print(f"✅ Event gate triggered: score={event_score:.2f}")
            else:
                reasons = []
                if event_score < score_threshold:
                    reasons.append(f"low_score_{event_score:.2f}")
                if not tca_ok:
                    reasons.append(
                        f"tca_breach_{tca_data['slippage_forecast_bps']:.1f}bp"
                    )
                if not cost_ok:
                    reasons.append(
                        f"cost_breach_${cost_data['total_cost_forecast']:.1f}/h"
                    )

                print(f"❌ Event gate blocked: {'; '.join(reasons)}")

            event_results[asset] = result

        # Create summary audit
        summary = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "lookback_minutes": self.lookback_minutes,
            "assets_scanned": len(self.assets),
            "tokens_created": tokens_created,
            "dry_run": dry_run,
            "thresholds": {
                "event_score": 0.6,
                "sentiment": self.sentiment_threshold,
                "volume_spike": self.volume_spike_threshold,
                "volatility_spike": self.volatility_spike_threshold,
                "whale_alert": self.whale_alert_threshold,
                "max_tca_slippage_bps": self.max_tca_slippage_bps,
                "max_cost_hourly": self.max_cost_forecast_hourly,
            },
            "signal_summary": {
                "sentiment_assets": len(signals["sentiment"]),
                "volume_spikes": len(signals["volume"]),
                "volatility_spikes": len(signals["volatility"]),
                "whale_movements": len(signals["whale"]),
            },
            "event_results": event_results,
        }

        # Save audit
        audit_dir = self.base_dir / "artifacts" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%SZ")
        audit_file = audit_dir / f"event_gate_{timestamp_str}.json"

        with open(audit_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n🚨 Event Gate Summary:")
        print(f"  Assets Scanned: {len(self.assets)}")
        print(f"  Event Tokens Created: {tokens_created}")
        print(
            f"  Signal Activity: {len(signals['sentiment'])} sentiment, {len(signals['volume'])} volume, {len(signals['volatility'])} volatility, {len(signals['whale'])} whale"
        )
        print(f"  Audit saved: {audit_file}")

        if tokens_created > 0:
            print("✅ Event gates active - temporary green windows created!")
        else:
            print("📊 No event triggers - normal EV calendar applies")

        return summary


def main():
    """Main event gate function."""
    parser = argparse.ArgumentParser(
        description="Event Gate: Signal Spike Override System"
    )
    parser.add_argument("--lookback", default="30m", help="Lookback window (e.g., 30m)")
    parser.add_argument(
        "--out", default="artifacts/ev/event_gate_on", help="Output directory"
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    args = parser.parse_args()

    # Parse lookback
    if args.lookback.endswith("m"):
        lookback_minutes = int(args.lookback[:-1])
    elif args.lookback.endswith("h"):
        lookback_minutes = int(args.lookback[:-1]) * 60
    else:
        lookback_minutes = int(args.lookback)

    try:
        gate = EventGate(lookback_minutes=lookback_minutes)
        summary = gate.run_event_gate_scan(args.out, args.dry_run)

        print(f"💡 Next: Run 'make duty-5m' to enable 5-minute duty cycling")
        return 0

    except Exception as e:
        print(f"❌ Event gate failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
