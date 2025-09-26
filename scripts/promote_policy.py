#!/usr/bin/env python3
"""
Policy Promotion Script
Sets allowed influence percentage in Redis with WORM audit trail
"""

import argparse
import redis
import os
import sys
import json
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description="Promote RL Policy Influence")
    parser.add_argument(
        "--pct", type=int, default=0, help="Influence percentage (0-100)"
    )
    parser.add_argument(
        "--reason", default="Manual promotion", help="Reason for change"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )
    args = parser.parse_args()

    # Validate percentage
    pct = max(0, min(100, args.pct))
    if pct != args.pct:
        print(f"⚠️  Clamped percentage from {args.pct} to {pct}")

    # Connect to Redis
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = redis.Redis.from_url(redis_url, decode_responses=True)
        r.ping()  # Test connection
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        sys.exit(1)

    # Get current value for audit trail
    try:
        current_pct = r.get("policy:allowed_influence_pct") or "0"
        current_pct = int(current_pct)
    except (ValueError, TypeError):
        current_pct = 0

    # Create audit entry
    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "action": "policy_promotion",
        "previous_pct": current_pct,
        "new_pct": pct,
        "reason": args.reason,
        "user": os.getenv("USER", "unknown"),
        "host": os.getenv("HOSTNAME", "unknown"),
    }

    if args.dry_run:
        print("🧪 DRY RUN - Would perform:")
        print(f"   Current influence: {current_pct}%")
        print(f"   New influence: {pct}%")
        print(f"   Reason: {args.reason}")
        print(f"   Audit entry: {json.dumps(audit_entry, indent=2)}")
        sys.exit(0)

    try:
        # Set the policy influence percentage
        r.set("policy:allowed_influence_pct", pct)

        # Store audit trail (WORM-style - append only)
        audit_key = f"audit:policy_promotion:{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        r.hset(audit_key, mapping=audit_entry)
        r.expire(audit_key, 365 * 24 * 3600)  # 1 year retention

        # Also add to the promotion history list
        r.lpush("audit:policy_promotions", json.dumps(audit_entry))
        r.ltrim("audit:policy_promotions", 0, 99)  # Keep last 100 changes

        # WORM-style audit trail (as specified in Future_instruction.txt)
        from datetime import datetime, timezone
        from pathlib import Path

        aud = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "actor": os.getenv("USER", "ops"),
            "action": "set_influence_pct",
            "pct": pct,
            "previous_pct": current_pct,
            "reason": args.reason,
            "source": "promote_policy.py",
            "host": os.getenv("HOSTNAME", "unknown"),
            "ttl_hours": 1.0,  # 1 hour TTL from InfluenceController
        }

        Path("artifacts/audit").mkdir(parents=True, exist_ok=True)
        audit_filename = f"artifacts/audit/{aud['ts'].replace(':', '_')}_influence.json"
        with open(audit_filename, "w") as f:
            json.dump(aud, f, indent=2)

        print(f"✅ Policy influence updated:")
        print(f"   Previous: {current_pct}%")
        print(f"   Current:  {pct}%")
        print(f"   Reason:   {args.reason}")
        print(f"   Audit:    {audit_key}")
        print(f"   WORM Trail: {audit_filename}")

        # Warning for non-zero percentages
        if pct > 0:
            print(
                f"⚠️  WARNING: Policy influence is now {pct}% - shadow trading impact enabled!"
            )
            print(
                f"   Monitor dashboards: curl localhost:9108/metrics | grep influence"
            )
            print(f"   Kill-switch: python scripts/kill_switch.py")
            print(f"   Manual revert: python scripts/promote_policy.py --pct 0")
            print(f"   ⏱️  Auto-expires in 1 hour if not refreshed")
        else:
            print(f"🛡️  Policy influence at 0% - shadow mode only, no trading impact")

    except Exception as e:
        print(f"❌ Failed to update policy influence: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
