import numpy as np, redis, json, math

R = redis.Redis()


def kelly_size(edge, vol, risk_cap=0.02):
    """edge = expected return per trade, vol = annualised σ (in %)."""
    k = edge / (vol**2 + 1e-9)
    return float(np.clip(k * risk_cap, -risk_cap, risk_cap))


def compute_size(symbol, edge):
    iv = float(R.hget("iv_surface", f"{symbol}:atm") or 0.7)  # as %
    vol_ann = iv / 100 * math.sqrt(365 / 1)  # 1-day horizon
    size_frac = kelly_size(edge, vol_ann)
    R.hset("risk:size_frac", symbol, size_frac)
    return size_frac
