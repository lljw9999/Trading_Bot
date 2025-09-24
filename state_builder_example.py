#!/usr/bin/env python3
"""
State Builder Example - Integration with Whale Alert
Shows how to add whale impact features to RL state
"""

import redis
import numpy as np
from typing import Dict, Any


def build_rl_state_with_whale_features() -> Dict[str, Any]:
    """
    Build RL state with whale alert features integrated
    This would be integrated into your existing state builder
    """

    # Connect to Redis
    R = redis.Redis(host="localhost", port=6379, decode_responses=True)

    # Initialize state dictionary (your existing state features would go here)
    state = {
        # Example existing features
        "price": 50000.0,
        "volume": 1000.0,
        "rsi": 55.0,
        "macd": 0.1,
        "sentiment": 0.2,
        "entropy": 1.2,
        "q_spread": 45.0,
        # ... other features
    }

    # Add whale alert features as specified in the task brief
    try:
        whale_event = R.xrevrange("event:whale", count=1)
        state["whale_high_impact"] = int(
            bool(whale_event and whale_event[0][1].get("high_impact") == "1")
        )

        # Additional whale features for enhanced alpha
        if whale_event:
            latest_event = whale_event[0][1]
            state["whale_amount_usd"] = (
                float(latest_event.get("amount_usd", 0)) / 1e9
            )  # Normalize to billions
            state["whale_to_exchange"] = int(latest_event.get("to", "") == "exchange")
            state["whale_from_exchange"] = int(
                latest_event.get("from", "") == "exchange"
            )
        else:
            state["whale_amount_usd"] = 0.0
            state["whale_to_exchange"] = 0
            state["whale_from_exchange"] = 0

    except Exception as e:
        print(f"⚠️ Error adding whale features: {e}")
        # Default values if whale data unavailable
        state["whale_high_impact"] = 0
        state["whale_amount_usd"] = 0.0
        state["whale_to_exchange"] = 0
        state["whale_from_exchange"] = 0

    # Add IV surface features as specified in the task brief
    try:
        iv = R.hgetall("iv_surface")
        sym = "BTC"  # Would be dynamic based on current symbol being traded
        state["atm_iv"] = float(iv.get(f"{sym}:atm".encode(), b"0"))
        state["iv_slope"] = float(iv.get(f"{sym}:slope".encode(), b"0"))
    except Exception as e:
        print(f"⚠️ Error adding IV surface features: {e}")
        state["atm_iv"] = 0.0
        state["iv_slope"] = 0.0

    # Add options flow features as specified in task brief
    try:
        state["call_sweep_usd"] = float(R.hget("optsweep", "call_usd") or 0)
        state["put_sweep_usd"] = float(R.hget("optsweep", "put_usd") or 0)

        # Additional derived features for enhanced alpha
        call_put_ratio = float(R.hget("optsweep", "call_put_ratio") or 1.0)
        net_flow = float(R.hget("optsweep", "net_flow") or 0)

        state["options_call_put_ratio"] = call_put_ratio
        state["options_net_flow"] = net_flow / 1e6  # Normalize to millions
        state["options_bullish_signal"] = min(
            1.0, max(-1.0, net_flow / 5e6)
        )  # Clamp to [-1, 1]

    except Exception as e:
        print(f"⚠️ Error adding options flow features: {e}")
        state["call_sweep_usd"] = 0.0
        state["put_sweep_usd"] = 0.0
        state["options_call_put_ratio"] = 1.0
        state["options_net_flow"] = 0.0
        state["options_bullish_signal"] = 0.0

    return state


def get_whale_trading_signal() -> Dict[str, float]:
    """
    Generate trading signals based on whale activity
    Returns signal strength for BTC and ETH (-1 to +1)
    """
    R = redis.Redis(host="localhost", port=6379, decode_responses=True)

    try:
        # Get recent whale events (last hour)
        import time

        cutoff_time = int((time.time() - 3600) * 1000)  # 1 hour ago in milliseconds

        recent_events = R.xrevrange("event:whale", min=cutoff_time, count=50)

        btc_signal = 0.0
        eth_signal = 0.0

        for event_id, fields in recent_events:
            symbol = fields.get("sym", "").upper()
            amount_usd = float(fields.get("amount_usd", 0))
            from_type = fields.get("from", "")
            to_type = fields.get("to", "")

            # Calculate signal strength based on amount (logarithmic scale)
            signal_strength = min(1.0, np.log10(amount_usd / 1e7) / 2)  # Max at $1B

            # Determine bullish/bearish direction
            if to_type == "exchange":
                # Selling pressure (bearish)
                signal_strength *= -1
            elif from_type == "exchange":
                # Accumulation (bullish)
                pass  # Keep positive
            else:
                # Wallet-to-wallet (neutral)
                signal_strength *= 0.5

            # Apply to relevant symbol
            if symbol == "BITCOIN":
                btc_signal += signal_strength
            elif symbol == "ETHEREUM":
                eth_signal += signal_strength

        # Normalize signals to [-1, 1] range
        btc_signal = np.tanh(btc_signal)
        eth_signal = np.tanh(eth_signal)

        return {"btc_whale_signal": btc_signal, "eth_whale_signal": eth_signal}

    except Exception as e:
        print(f"⚠️ Error generating whale signals: {e}")
        return {"btc_whale_signal": 0.0, "eth_whale_signal": 0.0}


if __name__ == "__main__":
    print("🔄 Testing Whale Alert State Integration")

    # Build state with whale features
    state = build_rl_state_with_whale_features()
    print(f"📊 State with whale features:")
    whale_features = {k: v for k, v in state.items() if "whale" in k}
    for feature, value in whale_features.items():
        print(f"   {feature}: {value}")

    # Get trading signals
    signals = get_whale_trading_signal()
    print(f"\n🎯 Whale trading signals:")
    print(f"   BTC signal: {signals['btc_whale_signal']:.3f}")
    print(f"   ETH signal: {signals['eth_whale_signal']:.3f}")
