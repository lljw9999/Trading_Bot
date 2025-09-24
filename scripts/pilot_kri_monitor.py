#!/usr/bin/env python3
import os, time, sys, json, requests, datetime, pathlib
from src.rl.influence_controller import InfluenceController

LIMITS = {
    "max_drawdown_pct_day": float(os.getenv("PILOT_MAX_DD", "2.0")),
    "max_slippage_bps": float(os.getenv("PILOT_MAX_SLIP", "35")),
    "entropy_floor": float(os.getenv("PILOT_ENTROPY_FLOOR", "0.9")),
    "qspread_max_ratio_24h": float(os.getenv("PILOT_QSPREAD_RATIO", "2.0")),
}


def metric(name, default=0.0):
    # TODO: fetch from Prometheus/Exporter; stub returns healthy
    return {
        "drawdown_pct_day": 0.5,
        "slippage_bps": 12,
        "entropy": 1.2,
        "q_spread_ratio": 1.1,
    }.get(name, default)


def audit(kind, payload):
    pathlib.Path("artifacts/audit").mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat().replace(":", "_")
    open(f"artifacts/audit/{ts}_{kind}.json", "w").write(json.dumps(payload, indent=2))


def main():
    dd = metric("drawdown_pct_day")
    sl = metric("slippage_bps")
    ent = metric("entropy")
    qsr = metric("q_spread_ratio")
    breach = (
        (dd > LIMITS["max_drawdown_pct_day"])
        or (sl > LIMITS["max_slippage_bps"])
        or (ent < LIMITS["entropy_floor"])
        or (qsr > LIMITS["qspread_max_ratio_24h"])
    )
    if breach:
        ic = InfluenceController(ttl_sec=3600)
        ic.set_weight(0)
        audit(
            "pilot_auto_rollback",
            {
                "dd": dd,
                "slip": sl,
                "entropy": ent,
                "qspread_ratio": qsr,
                "action": "influence->0",
            },
        )
        print("AUTO_ROLLBACK: influence -> 0% due to KRI breach")
    else:
        print("KRI OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
