#!/usr/bin/env python
"""
nownodes_smoke_test.py  –  quick connectivity test for NOWNodes.

Usage examples
--------------
# run with defaults (BTC for 10 s, 5 ticks)
python nownodes_smoke_test.py

# specify symbol & API-key explicitly
python nownodes_smoke_test.py --symbol ETH \
  --apikey YOUR-UUID-HERE \
  --timeout 15 --ticks 10
"""
import argparse, asyncio, os, ssl, sys, json, time
from datetime import datetime

import aiohttp
import websockets


################################################################################
# CLI args
################################################################################
def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--symbol",
        default="BTC",
        choices=["BTC", "ETH", "SOL"],
        help="asset (upper-case)",
    )
    p.add_argument(
        "--apikey",
        default=os.getenv("NOWNODES_APIKEY", ""),
        help="NOWNodes API key (env:NOWNODES_APIKEY)",
    )
    p.add_argument(
        "--ticks", type=int, default=5, help="number of ticks to print before exit"
    )
    p.add_argument(
        "--timeout", type=int, default=10, help="max seconds to wait for first WSS msg"
    )
    return p.parse_args()


################################################################################
# URL helpers
################################################################################
def ws_url(symbol, apikey):
    return f"wss://{symbol.lower()}.blockbook.ws.nownodes.io/?api_key={apikey}"


def rest_url(symbol, apikey):
    return f"https://{symbol.lower()}.blockbook.ws.nownodes.io/api/v2/ticker?api_key={apikey}"


def parse_rest(json_data):
    # Blockbook REST format → price (float)
    return float(json_data["ticker"]["price"])


################################################################################
# Main logic
################################################################################
async def run_wss(ws_uri, want_ticks, first_timeout):
    """Try WSS stream; yield (price, ts) until want_ticks reached."""
    ssl_ctx = ssl.create_default_context()
    async with websockets.connect(ws_uri, ssl=ssl_ctx, max_queue=0) as ws:
        # Blockbook requires a SUBSCRIBE msg after connect
        await ws.send(json.dumps({"event": "subscribe", "data": "ticker"}))
        for _ in range(want_ticks):
            msg = await asyncio.wait_for(ws.recv(), timeout=first_timeout)
            data = json.loads(msg)
            price = float(data["price"])
            yield price, data.get("time", time.time())


async def run_rest(rest_uri, want_ticks, poll=1.0):
    async with aiohttp.ClientSession() as sess:
        for _ in range(want_ticks):
            async with sess.get(rest_uri, timeout=10) as r:
                r.raise_for_status()
                price = parse_rest(await r.json())
            yield price, time.time()
            await asyncio.sleep(poll)


async def main():
    args = get_args()
    if not args.apikey:
        sys.exit("❌  Provide --apikey or set NOWNODES_APIKEY")

    ws_uri = ws_url(args.symbol, args.apikey)
    rest_uri = rest_url(args.symbol, args.apikey)

    print(f"🔌  Opening WSS → {ws_uri}")
    try:
        async for px, ts in run_wss(ws_uri, args.ticks, args.timeout):
            print(f"🟢 WSS {args.symbol} {datetime.fromtimestamp(ts)}  ${px:,.2f}")
        print("✅  WSS test passed")
        sys.exit(0)

    except (
        asyncio.TimeoutError,
        websockets.InvalidURI,
        websockets.InvalidStatusCode,
        ssl.SSLError,
        ConnectionRefusedError,
    ) as e:
        print(
            f"⚠️   WSS failed ({type(e).__name__}: {e}) – falling back to REST polling…"
        )

    # ----------- REST fallback -----------
    try:
        async for px, ts in run_rest(rest_uri, args.ticks):
            print(f"🔵 REST {args.symbol} {datetime.fromtimestamp(ts)}  ${px:,.2f}")
        print("✅  REST fallback passed")
        sys.exit(0)

    except Exception as e:
        print(f"❌  REST fallback failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
