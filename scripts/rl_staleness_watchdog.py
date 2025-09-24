#!/usr/bin/env python3
"""
RL Staleness Watchdog
Auto-restarts RL policy service if heartbeat is stale beyond threshold
"""
import os, time, sys, json, subprocess, argparse
import redis
from datetime import datetime, timezone


def read_redis_ts(r, key):
    """Read timestamp from Redis, return None if missing/invalid."""
    val = r.get(key)
    if not val:
        return None
    try:
        return float(val)
    except Exception:
        return None


def restart_service(svc):
    """Restart systemd service, return (returncode, stdout, stderr)."""
    try:
        out = subprocess.run(
            ["systemctl", "restart", svc], capture_output=True, text=True, timeout=30
        )
        return out.returncode, out.stdout, out.stderr
    except Exception as e:
        return 1, "", str(e)


def main():
    parser = argparse.ArgumentParser(description="RL Staleness Watchdog")
    parser.add_argument(
        "--threshold-sec",
        type=int,
        default=24 * 3600,
        help="stale if last update exceeds this",
    )
    parser.add_argument(
        "--service", default="rl-policy", help="systemd service to restart"
    )
    parser.add_argument(
        "--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="don't actually restart service"
    )
    parser.add_argument("--out", default=None, help="JSON output file")
    args = parser.parse_args()

    try:
        r = redis.Redis.from_url(args.redis_url, decode_responses=True)
        r.ping()  # Test connection
    except Exception as e:
        print(f"❌ Redis connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    now = time.time()
    last = read_redis_ts(r, "policy:last_update_ts")

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "now": now,
        "last_update_ts": last,
        "age_sec": None if last is None else now - last,
        "threshold_sec": args.threshold_sec,
        "action": "none",
        "service": args.service,
        "dry_run": args.dry_run,
        "ok": True,
    }

    # Check staleness conditions
    if last is None:
        result.update({"ok": False, "action": "restart:missing_heartbeat"})
        print(f"⚠️  Missing heartbeat - policy:last_update_ts not found")

        if not args.dry_run:
            print(f"🔄 Restarting service: {args.service}")
            rc, out, err = restart_service(args.service)
            result.update({"restart_rc": rc, "stdout": out, "stderr": err})
            if rc == 0:
                print(f"✅ Service restart successful")
            else:
                print(f"❌ Service restart failed: {err}")
    else:
        age = now - last
        result["age_sec"] = age
        age_hours = age / 3600

        if age > args.threshold_sec:
            result.update({"ok": False, "action": "restart:stale"})
            print(
                f"⚠️  Stale heartbeat - {age_hours:.1f}h old (threshold: {args.threshold_sec/3600:.1f}h)"
            )

            if not args.dry_run:
                print(f"🔄 Restarting service: {args.service}")
                rc, out, err = restart_service(args.service)
                result.update({"restart_rc": rc, "stdout": out, "stderr": err})
                if rc == 0:
                    print(f"✅ Service restart successful")
                else:
                    print(f"❌ Service restart failed: {err}")
        else:
            print(
                f"✅ Heartbeat healthy - {age_hours:.1f}h old (threshold: {args.threshold_sec/3600:.1f}h)"
            )

    # Save detailed results if requested
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"📝 Detailed results saved to: {args.out}")

    # Console status output
    if result["ok"]:
        status = "OK"
    else:
        age_sec = result.get("age_sec")
        age_display = int(age_sec) if age_sec is not None else -1
        status = f"STALE ({age_display}s) → {result['action']}"

    print(f"🏁 Status: {status}")

    # Exit 0 even on stale - alerting handles the notification, this is just remediation
    sys.exit(0)


if __name__ == "__main__":
    main()
