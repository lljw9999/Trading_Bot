#!/usr/bin/env python3
"""
RL Policy Kill-Switch - Emergency 0% Influence 
Immediately sets policy influence to 0% for emergency situations
"""
import sys
import time
from datetime import datetime
from pathlib import Path
from src.rl.influence_controller import InfluenceController
import logging
import json


def main():
    """Execute emergency kill-switch."""
    print("🚨 RL POLICY KILL-SWITCH ACTIVATED")
    print(f"⏰ Timestamp: {datetime.utcnow().isoformat()}Z")
    print()

    try:
        # Initialize influence controller
        ic = InfluenceController()

        # Get current status before kill
        print("📊 Current status:")
        status_before = ic.get_status()
        current_pct = status_before.get("percentage", 0)
        print(f"   Influence before: {current_pct}%")

        if current_pct == 0:
            print("ℹ️  Policy influence already at 0%")
        else:
            print(f"🔄 Setting influence from {current_pct}% to 0%...")

        # Execute emergency stop
        success = ic.emergency_stop()

        if success:
            # Verify the change
            status_after = ic.get_status()
            final_pct = status_after.get("percentage", -1)

            if final_pct == 0:
                print("✅ KILL-SWITCH SUCCESS: Influence set to 0%")
                print("🛡️  Policy outputs now have NO trading impact")
                print("📝 All future actions will use baseline strategies only")
            else:
                print(f"⚠️  WARNING: Influence shows {final_pct}% after kill-switch")

        else:
            print("❌ KILL-SWITCH FAILED: Could not set influence to 0%")
            print("🚨 MANUAL INTERVENTION REQUIRED")

        # Create audit trail
        audit_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": "kill_switch_executed",
            "actor": "kill_switch.py",
            "influence_before": current_pct,
            "influence_after": status_after.get("percentage", -1) if success else -1,
            "success": success,
            "status_before": status_before,
            "status_after": (
                status_after if success else {"error": "kill switch failed"}
            ),
        }

        # Write audit record
        audit_path = Path("artifacts/audit")
        audit_path.mkdir(parents=True, exist_ok=True)

        with open(audit_path / f"kill_switch_{int(time.time())}.json", "w") as f:
            json.dump(audit_data, f, indent=2)

        print(f"📋 Audit record: artifacts/audit/kill_switch_{int(time.time())}.json")

        # Exit codes
        if success and status_after.get("percentage", -1) == 0:
            print("\n🏁 Kill-switch completed successfully")
            sys.exit(0)
        else:
            print("\n💥 Kill-switch failed or incomplete")
            sys.exit(1)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Ensure InfluenceController is properly installed")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Kill-switch error: {e}")
        print("🚨 MANUAL REDIS INTERVENTION MAY BE REQUIRED:")
        print("   redis-cli SET policy:allowed_influence_pct 0")
        sys.exit(1)


if __name__ == "__main__":
    main()
