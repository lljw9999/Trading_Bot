import os, requests, redis, time, hashlib, json

R = redis.Redis(host="localhost", port=6379, decode_responses=True)
API_KEY = os.getenv("WHALE_ALERT_KEY", "DEMO_KEY")
BASE = "https://api.whale-alert.io/v1/transactions"


def fetch_whale_tx(min_value_usd=10000000):
    """Fetch whale transactions from Whale Alert API"""
    params = dict(
        api_key=API_KEY, min_value=min_value_usd, currency="btc,eth", limit=50
    )

    if API_KEY == "DEMO_KEY":
        # Return demo data when no real API key
        return generate_demo_whale_transactions()

    resp = requests.get(BASE, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()["transactions"]


def generate_demo_whale_transactions():
    """Generate demo whale transactions for testing"""
    import random

    demo_transactions = []

    for i in range(random.randint(0, 5)):
        tx = {
            "hash": f"demo_tx_{int(time.time())}_{i}",
            "symbol": random.choice(["bitcoin", "ethereum"]),
            "amount_usd": random.randint(10000000, 100000000),  # $10M - $100M
            "from": {"owner_type": random.choice(["exchange", "wallet", "unknown"])},
            "to": {"owner_type": random.choice(["exchange", "wallet", "unknown"])},
            "timestamp": int(time.time()) - random.randint(0, 3600),  # Last hour
        }
        demo_transactions.append(tx)

    return demo_transactions


def push_events():
    """Fetch and push whale events to Redis streams"""
    try:
        transactions = fetch_whale_tx()
        events_added = 0

        for tx in transactions:
            h = hashlib.sha1(tx["hash"].encode()).hexdigest()

            # Skip if we've already seen this transaction
            if R.sismember("whale:seen", h):
                continue

            # Mark as seen
            R.sadd("whale:seen", h)

            # Create event payload
            payload = {
                "ts": int(time.time()),
                "sym": tx["symbol"].upper(),
                "amount_usd": tx["amount_usd"],
                "from": tx["from"]["owner_type"],
                "to": tx["to"]["owner_type"],
                "high_impact": str(
                    int(tx["amount_usd"] > 1e7 and tx["to"]["owner_type"] == "exchange")
                ),
            }

            # Add to whale events stream
            R.xadd("event:whale", payload, maxlen=10000)
            events_added += 1

            # Log significant events
            if payload["high_impact"] == "1":
                print(
                    f"🚨 HIGH IMPACT WHALE: ${tx['amount_usd']:,.0f} {tx['symbol'].upper()} → {tx['to']['owner_type']}"
                )

        if events_added > 0:
            print(f"✅ Added {events_added} new whale events to stream")

        return events_added

    except Exception as e:
        print(f"❌ Error fetching whale transactions: {e}")
        return 0


def get_recent_whale_signals(hours_back: int = 1):
    """Get recent whale signals for trading decision making"""
    try:
        cutoff_timestamp = int((time.time() - hours_back * 3600) * 1000)

        # Get recent whale events
        events = R.xrevrange("event:whale", min=cutoff_timestamp, count=100)

        whale_signals = {
            "btc_bullish": 0,
            "btc_bearish": 0,
            "eth_bullish": 0,
            "eth_bearish": 0,
            "total_volume_usd": 0,
            "high_impact_count": 0,
        }

        for event_id, fields in events:
            symbol = fields.get("sym", "").lower()
            amount_usd = float(fields.get("amount_usd", 0))
            from_type = fields.get("from", "")
            to_type = fields.get("to", "")
            high_impact = fields.get("high_impact", "0") == "1"

            whale_signals["total_volume_usd"] += amount_usd

            if high_impact:
                whale_signals["high_impact_count"] += 1

            # Classify as bullish or bearish
            if to_type == "exchange":
                # Moving to exchange = bearish (selling pressure)
                if symbol == "bitcoin":
                    whale_signals["btc_bearish"] += 1
                elif symbol == "ethereum":
                    whale_signals["eth_bearish"] += 1

            elif from_type == "exchange":
                # Moving from exchange = bullish (accumulation)
                if symbol == "bitcoin":
                    whale_signals["btc_bullish"] += 1
                elif symbol == "ethereum":
                    whale_signals["eth_bullish"] += 1

        return whale_signals

    except Exception as e:
        print(f"❌ Error getting whale signals: {e}")
        return {
            "btc_bullish": 0,
            "btc_bearish": 0,
            "eth_bullish": 0,
            "eth_bearish": 0,
            "total_volume_usd": 0,
            "high_impact_count": 0,
        }


if __name__ == "__main__":
    print("🐋 Whale Alert Client Demo")

    # Test the client
    events_added = push_events()
    signals = get_recent_whale_signals(hours_back=24)

    print("\n📊 Recent Whale Signals (24h):")
    print(f"   BTC: {signals['btc_bullish']} bullish, {signals['btc_bearish']} bearish")
    print(f"   ETH: {signals['eth_bullish']} bullish, {signals['eth_bearish']} bearish")
    print(f"   Total volume: ${signals['total_volume_usd']:,.0f}")
    print(f"   High impact events: {signals['high_impact_count']}")
