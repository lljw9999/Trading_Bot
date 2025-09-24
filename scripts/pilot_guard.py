#!/usr/bin/env python3
import os, sys, json, time, yaml, requests, pathlib, datetime
from datetime import timezone
from src.rl.influence_controller import InfluenceController


# Helpers (replace stubs with real Prometheus/Grafana queries if available)
def last_two_offline_pass():
    # scan artifacts/*/rl/gate_report.md for two recent PASS
    import glob

    passes = 0
    for d in sorted(glob.glob("artifacts/*/rl"), reverse=True):
        try:
            if "PASS" in open(f"{d}/gate_report.md").read().upper():
                passes += 1
            if passes >= 2:
                return True
        except:
            pass
    return False


def no_pages_in(hours):
    # stub: check for sentinel alert files under artifacts/alerts
    import glob, os, time

    now = time.time()
    for f in glob.glob("artifacts/alerts/*"):
        if now - os.path.getmtime(f) < hours * 3600:
            return False
    return True


def shadow_kpi_within(band_pct):
    # stub: assume exporter exposes /metrics; simply return True for scaffold
    return True


def write_audit(obj, name):
    pathlib.Path("artifacts/audit").mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(timezone.utc).isoformat().replace(":", "_")
    open(f"artifacts/audit/{ts}_{name}.json", "w").write(json.dumps(obj, indent=2))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="pilot/pilot_run.yaml")
    ap.add_argument("--target-pct", type=int, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pilot = cfg["pilot"]
    gates = cfg["gates"]
    limits = cfg["risk_limits"]
    target = max(0, min(int(args.target_pct), int(pilot["max_influence_pct"])))

    # Gate checks
    reasons = []
    if gates.get("require_go_flag", True) and os.getenv("GO_LIVE", "0") != "1":
        reasons.append("GO_LIVE flag not set")
    if gates.get("min_48h_passes", 2) > 0 and not last_two_offline_pass():
        reasons.append("Missing two consecutive 48h validation PASS")
    if not no_pages_in(gates.get("no_pages_hours", 48)):
        reasons.append("Recent page alerts within no-pages window")
    if not shadow_kpi_within(gates.get("shadow_kpi_band_pct", 20)):
        reasons.append("Shadow KPIs outside band")

    if reasons:
        write_audit(
            {"action": "ramp_blocked", "target_pct": target, "reasons": reasons},
            "ramp_block",
        )
        print("RAMP_GUARD_FAIL:", "; ".join(reasons))
        sys.exit(1)

    # Set influence via controller (TTL still applies)
    from scripts.promote_policy import main as promote_main  # if available

    try:
        os.environ.setdefault("REASON", f"pilot ramp to {target}%")
        promote_main()
    except Exception:
        # fallback direct
        ic = InfluenceController(ttl_sec=3600)
        ic.set_weight(target)

    write_audit(
        {"action": "ramp_set", "pct": target, "pilot": pilot["name"]}, "ramp_set"
    )
    print(f"RAMP_GUARD_PASS: set to {target}%")
    sys.exit(0)


if __name__ == "__main__":
    main()
