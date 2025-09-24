import os, time, math, hmac, hashlib, argparse, sys, json
from decimal import Decimal, ROUND_DOWN, getcontext
from binance.spot import Spot as Client

getcontext().prec = 28

API_KEY = os.getenv("BINANCE_TRADING_API_KEY", "")
API_SECRET = os.getenv("BINANCE_TRADING_SECRET_KEY", "")

BASE_URL = "https://api.binance.com"
RECV_WINDOW = 5000

CANARY_FIRST = True  # can be overridden by --canary
CANARY_USD = Decimal("5")  # will auto-bump to minNotional if needed
ALLOCATION = {"BTCUSDT": Decimal("0.5"), "ETHUSDT": Decimal("0.5")}
TIMEOUT_SEC = 180


def die(msg):
    print(f"[FATAL] {msg}")
    sys.exit(1)


def decimal_step_round_down(value: Decimal, step: Decimal) -> Decimal:
    # quantize to step size, rounding down
    if step == 0:
        return value
    q = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return q * step


def step_to_decimals(step_str: str) -> int:
    if "." in step_str:
        return len(step_str.rstrip("0").split(".")[1])
    return 0


def get_filters(client, symbol):
    ex = client.exchange_info(symbol=symbol)
    s = ex["symbols"][0]
    f = {fl["filterType"]: fl for fl in s["filters"]}
    lot = f.get("LOT_SIZE") or f.get("MARKET_LOT_SIZE")
    pricef = f["PRICE_FILTER"]
    notional = f.get("NOTIONAL") or f.get(
        "MIN_NOTIONAL"
    )  # (Binance has both varieties)
    return {
        "minQty": Decimal(lot["minQty"]),
        "stepSize": Decimal(lot["stepSize"]),
        "tickSize": Decimal(pricef["tickSize"]),
        "minNotional": Decimal(notional.get("minNotional", "0")),
    }


def round_price(p: Decimal, tick: Decimal) -> (str, Decimal):
    rp = decimal_step_round_down(p, tick)
    dp = step_to_decimals(str(tick))
    return (f"{rp:.{dp}f}", rp)


def round_qty(q: Decimal, step: Decimal) -> (str, Decimal):
    rq = decimal_step_round_down(q, step)
    dq = step_to_decimals(str(step))
    return (f"{rq:.{dq}f}", rq)


def check_api_restrictions(client):
    # signed check of API perms
    import time, hmac, hashlib

    ts = str(int(time.time() * 1000))
    qs = f"timestamp={ts}"
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    import requests

    r = requests.get(
        f"{BASE_URL}/sapi/v1/account/apiRestrictions",
        params={"timestamp": ts, "signature": sig},
        headers={"X-MBX-APIKEY": API_KEY},
        timeout=10,
    )
    data = r.json()
    ok = (
        data.get("enableReading")
        and data.get("enableSpotAndMarginTrading")
        and data.get("ipRestrict")
    )
    print("[INFO] API restrictions:", json.dumps(data, indent=2))
    if not ok:
        die(
            "API key not ready for trading (need enableSpotAndMarginTrading + ipRestrict + enableReading)."
        )
    if data.get("enableWithdrawals", False):
        print(
            "[WARN] Withdrawals are enabled on this key. Strongly recommended to disable."
        )


def ensure_symbol_whitelist(client, symbols):
    # optional: try a harmless call to catch 'symbol not in whitelist' early
    try:
        info = client.exchange_info()
        allowed = {s["symbol"] for s in info["symbols"]}
        for sym in symbols:
            if sym not in allowed:
                die(f"Symbol {sym} not available on this account/cluster.")
    except Exception as e:
        print("[WARN] Could not verify symbol whitelist early:", e)


def get_usdt_free(client) -> Decimal:
    acct = client.account(recvWindow=RECV_WINDOW)
    for bal in acct["balances"]:
        if bal["asset"] == "USDT":
            return Decimal(bal["free"])
    return Decimal("0")


def maker_buy_order_plan(client, symbol, usd_amount, filters):
    # discover price via book ticker
    bt = client.book_ticker(symbol)
    bid = Decimal(bt["bidPrice"])
    # price for maker BUY: below or equal to best bid (use bid - 1 tick for safety)
    price_str, pr = round_price(bid - filters["tickSize"], filters["tickSize"])
    if pr <= 0:
        price_str, pr = round_price(bid, filters["tickSize"])

    # Calculate initial quantity
    qty = Decimal(usd_amount) / pr
    qty_str, qrd = round_qty(qty, filters["stepSize"])

    # enforce minQty
    if qrd < filters["minQty"]:
        qrd = filters["minQty"]

    # Calculate notional and ensure it meets minimum
    notional = qrd * pr
    if notional < filters["minNotional"]:
        # Calculate exact quantity needed for minimum notional + buffer
        min_qty_for_notional = (filters["minNotional"] * Decimal("1.01")) / pr
        qrd = decimal_step_round_down(min_qty_for_notional, filters["stepSize"])
        # If rounding down still too small, round up
        if qrd * pr < filters["minNotional"]:
            qrd = qrd + filters["stepSize"]
        notional = qrd * pr

    # Format final quantity
    qty_str = f"{qrd:.{step_to_decimals(str(filters['stepSize']))}f}"

    return {
        "symbol": symbol,
        "side": "BUY",
        "type": "LIMIT_MAKER",
        "price": price_str,
        "quantity": qty_str,
        "price_decimal": str(pr),
        "qty_decimal": str(qrd),
        "notional_decimal": str(notional),
    }


def place_and_wait(client, order_args, timeout_sec=120):
    # Clean order args - only send Binance API parameters
    clean_args = {
        "symbol": order_args["symbol"],
        "side": order_args["side"],
        "type": order_args["type"],
        "price": order_args["price"],
        "quantity": order_args["quantity"],
        "newClientOrderId": f"live_{order_args['symbol']}_{int(time.time()*1000)}",
        "recvWindow": RECV_WINDOW,
    }

    oid = client.new_order(**clean_args)
    order_id = oid["orderId"]
    t0 = time.time()
    status = "NEW"
    while time.time() - t0 < timeout_sec:
        st = client.get_order(symbol=order_args["symbol"], orderId=order_id)
        status = st["status"]
        if status in ("FILLED", "PARTIALLY_FILLED", "CANCELED", "REJECTED", "EXPIRED"):
            print(
                f"[ORDER] {order_args['symbol']} status: {status} | {st.get('executedQty')} filled"
            )
            if status == "FILLED":
                return True, st
            else:
                return False, st
        time.sleep(5)
    # timeout → cancel to avoid stale maker
    client.cancel_order(symbol=order_args["symbol"], orderId=order_id)
    return False, {"status": "TIMEOUT_CANCELED"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--alloc", default="0.5,0.5")
    parser.add_argument("--canary", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if not API_KEY or not API_SECRET:
        die(
            "Missing BINANCE_TRADING_API_KEY or BINANCE_TRADING_SECRET_KEY in environment."
        )

    syms = [s.strip().upper() for s in args.symbols.split(",")]
    allocs = [Decimal(x.strip()) for x in args.alloc.split(",")]
    if len(syms) != len(allocs) or abs(sum(allocs) - Decimal("1")) > Decimal("0.0001"):
        die("Allocation must match symbols and sum to 1.0 (e.g., --alloc 0.5,0.5)")

    client = Client(api_key=API_KEY, api_secret=API_SECRET, base_url=BASE_URL)

    # 0) Basic connectivity and restrictions
    client.ping()
    server_time = client.time()
    print("[INFO] Binance server time:", server_time)
    check_api_restrictions(client)
    ensure_symbol_whitelist(client, syms)

    # 1) Fetch filters
    filters = {s: get_filters(client, s) for s in syms}
    print(
        "[INFO] Filters:",
        json.dumps(
            {k: {kk: str(v[kk]) for kk in v} for k, v in filters.items()}, indent=2
        ),
    )

    # 2) Get balance
    usdt = get_usdt_free(client)
    print(f"[INFO] USDT free balance: {usdt}")
    if usdt <= 0:
        die("No USDT available.")

    # 3) Canary (min notional) buys to verify live trading works
    global CANARY_FIRST
    CANARY_FIRST = bool(args.canary)
    if CANARY_FIRST:
        print("[STEP] Running canary trades...")
        for sym in syms:
            f = filters[sym]
            # choose max(minNotional, CANARY_USD) for canary size
            canary_usd = max(f["minNotional"], CANARY_USD)
            plan = maker_buy_order_plan(client, sym, canary_usd, f)
            print("[PLAN][CANARY]", sym, plan)
            ok, detail = place_and_wait(client, plan, timeout_sec=args.timeout)
            if not ok:
                die(f"Canary for {sym} did not FILLED cleanly: {detail}")
        print("[OK] Canary trades filled. Proceeding to full allocation.")

    # 4) Full allocation (use 100% of free USDT, split by allocs)
    print("[STEP] Placing full-allocation orders (LIMIT_MAKER, post-only).")
    # refresh balance after canaries
    usdt = get_usdt_free(client)
    if usdt <= 0:
        die("No USDT left after canaries.")

    results = []
    for sym, a in zip(syms, allocs):
        usd_chunk = usdt * a
        plan = maker_buy_order_plan(client, sym, usd_chunk, filters[sym])
        print("[PLAN][FULL]", sym, plan)
        ok, detail = place_and_wait(client, plan, timeout_sec=args.timeout)
        results.append((sym, ok, detail))

    print("[SUMMARY]")
    for sym, ok, det in results:
        print(sym, "FILLED" if ok else f"NOT FILLED ({det.get('status')})")

    # 5) Print final balances
    acct = client.account(recvWindow=RECV_WINDOW)
    bals = {
        b["asset"]: Decimal(b["free"])
        for b in acct["balances"]
        if Decimal(b["free"]) > 0
    }
    print("[BALANCES]", json.dumps({k: str(v) for k, v in bals.items()}, indent=2))


if __name__ == "__main__":
    main()
