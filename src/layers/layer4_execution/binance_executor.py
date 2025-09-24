import os, time, math, ccxt
from typing import Dict, Any

LOT_CACHE = {}


def _round_step(value, step):
    return math.floor(value / step) * step


class BinanceExecutor:
    def __init__(self, sandbox=False):
        params = {
            "apiKey": os.getenv("BINANCE_TRADING_API_KEY"),
            "secret": os.getenv("BINANCE_TRADING_SECRET_KEY"),
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        self.client = ccxt.binance(params)
        if sandbox:
            self.client.set_sandbox_mode(True)

    def _ensure_market(self, symbol: str):
        global LOT_CACHE
        if symbol not in LOT_CACHE:
            mkts = self.client.load_markets()
            LOT_CACHE[symbol] = mkts[symbol]
        return LOT_CACHE[symbol]

    def place_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        qty: float,
        price: float = None,
        ttl_ms: int = 3000,
    ) -> Dict[str, Any]:
        m = self._ensure_market(symbol)
        # Enforce minNotional/stepSize/precision
        step_q = m["limits"]["amount"]["min"] or m["precision"].get("amount", None)
        step_p = m["precision"].get("price", None)
        min_notional = m["limits"]["cost"]["min"] or 10.0

        ticker = self.client.fetch_ticker(symbol)
        px = price or ticker["last"]
        if px is None:
            raise ValueError("No price available for symbol")

        notional = qty * px
        if notional < min_notional:
            raise ValueError(
                f"Order notional {notional:.2f} < minNotional {min_notional:.2f}"
            )

        if step_q:
            if isinstance(step_q, float):
                qty = _round_step(qty, step_q)
        if step_p and price is not None:
            price = round(price, step_p)

        params = {"timeInForce": "GTC"}
        if type_.upper() == "IOC":
            params["timeInForce"] = "IOC"

        order = self.client.create_order(
            symbol, type_.upper(), side.upper(), qty, price, params
        )
        order["submitted_ts"] = time.time()
        order["ttl_ms"] = ttl_ms
        return order

    def cancel(self, symbol: str, order_id: str):
        return self.client.cancel_order(order_id, symbol)

    def get_balances(self):
        return self.client.fetch_balance()
