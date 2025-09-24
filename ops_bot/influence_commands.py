#!/usr/bin/env python3
"""
Ops Bot Influence Commands - Safe Slack Integration
Provides secure influence control with guard rails and audit trails
"""
import os
import sys
import json
import redis
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class InfluenceBotCommands:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.max_influence_default = 10  # Default max 10% without GO_LIVE flag
        self.go_live_key = "ops:go_live"

    def get_redis_client(self):
        """Get Redis client with error handling."""
        try:
            return redis.Redis.from_url(self.redis_url, decode_responses=True)
        except Exception as e:
            raise Exception(f"Redis connection failed: {e}")

    def is_go_live_enabled(self):
        """Check if GO_LIVE flag is set in Redis."""
        try:
            r = self.get_redis_client()
            go_live = r.get(self.go_live_key)
            return go_live == "1"
        except:
            return False

    def create_audit_record(self, action, details):
        """Create WORM-style audit record."""
        audit_data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": f"ops_bot_{action}",
            "details": details,
            "source": "ops_bot/influence_commands.py",
            "actor": os.getenv("SLACK_USER", "ops_bot"),
            "go_live_enabled": self.is_go_live_enabled(),
        }

        # Write audit artifact
        os.makedirs("artifacts/audit", exist_ok=True)
        audit_filename = (
            f"artifacts/audit/{audit_data['ts'].replace(':', '_')}_bot_{action}.json"
        )
        with open(audit_filename, "w") as f:
            json.dump(audit_data, f, indent=2)

        return audit_filename

    def run_ramp_guard(self):
        """Run ramp guard safety checks."""
        try:
            result = subprocess.run(
                ["python", "scripts/ramp_guard.py"],
                capture_output=True,
                text=True,
                cwd=".",
            )

            if result.returncode == 0:
                return True, "PASS"
            else:
                return False, result.stderr or result.stdout
        except Exception as e:
            return False, str(e)

    def command_status(self):
        """Get current influence status."""
        try:
            from src.rl.influence_controller import InfluenceController

            ic = InfluenceController()
            status = ic.get_status()

            response = {
                "status": "success",
                "current_influence": status.get("percentage", 0),
                "current_weight": status.get("weight", 0.0),
                "key_exists": status.get("key_exists", False),
                "ttl_seconds": status.get("ttl_seconds", 0),
                "go_live_enabled": self.is_go_live_enabled(),
                "max_allowed": (
                    100 if self.is_go_live_enabled() else self.max_influence_default
                ),
            }

            # Create audit record
            self.create_audit_record("status", response)

            return response

        except Exception as e:
            error_response = {"status": "error", "message": str(e)}
            self.create_audit_record("status_error", error_response)
            return error_response

    def command_set(self, percentage, reason):
        """Set influence percentage with guard rails."""
        try:
            # Validate percentage
            if not isinstance(percentage, (int, float)):
                raise ValueError("Percentage must be a number")

            percentage = int(percentage)
            if percentage < 0 or percentage > 100:
                raise ValueError("Percentage must be between 0-100")

            # Check GO_LIVE flag for high percentages
            go_live = self.is_go_live_enabled()
            max_allowed = 100 if go_live else self.max_influence_default

            if percentage > max_allowed:
                error_msg = f"Percentage {percentage}% exceeds max allowed {max_allowed}% (GO_LIVE={'enabled' if go_live else 'disabled'})"
                response = {
                    "status": "error",
                    "message": error_msg,
                    "current_influence": 0,
                    "requested": percentage,
                    "max_allowed": max_allowed,
                }
                self.create_audit_record("set_blocked", response)
                return response

            # Run ramp guard for non-zero percentages
            if percentage > 0:
                guard_passed, guard_details = self.run_ramp_guard()
                if not guard_passed:
                    response = {
                        "status": "error",
                        "message": f"Ramp guard failed: {guard_details}",
                        "current_influence": 0,
                        "requested": percentage,
                    }
                    self.create_audit_record("set_guard_fail", response)
                    return response

            # Execute the influence change
            result = subprocess.run(
                [
                    "python",
                    "scripts/promote_policy.py",
                    "--pct",
                    str(percentage),
                    "--reason",
                    f"Ops Bot: {reason}",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            if result.returncode == 0:
                response = {
                    "status": "success",
                    "message": f"Set influence to {percentage}%",
                    "previous_influence": 0,  # Would need to get from promote_policy output
                    "new_influence": percentage,
                    "reason": reason,
                    "guard_passed": percentage == 0 or guard_passed,
                }
                self.create_audit_record("set_success", response)
                return response
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                response = {
                    "status": "error",
                    "message": f"Failed to set influence: {error_msg}",
                    "requested": percentage,
                }
                self.create_audit_record("set_fail", response)
                return response

        except Exception as e:
            error_response = {
                "status": "error",
                "message": str(e),
                "requested": percentage if "percentage" in locals() else None,
            }
            self.create_audit_record("set_error", error_response)
            return error_response

    def command_kill(self):
        """Execute emergency kill-switch."""
        try:
            result = subprocess.run(
                ["python", "scripts/kill_switch.py"],
                capture_output=True,
                text=True,
                cwd=".",
            )

            if result.returncode == 0:
                response = {
                    "status": "success",
                    "message": "Emergency kill-switch executed - influence set to 0%",
                    "new_influence": 0,
                }
                self.create_audit_record("kill_success", response)
                return response
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                response = {
                    "status": "error",
                    "message": f"Kill-switch failed: {error_msg}",
                }
                self.create_audit_record("kill_fail", response)
                return response

        except Exception as e:
            error_response = {"status": "error", "message": str(e)}
            self.create_audit_record("kill_error", error_response)
            return error_response

    def command_go_live(self, enable=True):
        """Enable/disable GO_LIVE flag (admin only)."""
        try:
            r = self.get_redis_client()

            if enable:
                r.set(self.go_live_key, "1")
                message = (
                    "GO_LIVE flag ENABLED - high influence percentages now allowed"
                )
            else:
                r.delete(self.go_live_key)
                message = "GO_LIVE flag DISABLED - influence capped at 10%"

            response = {
                "status": "success",
                "message": message,
                "go_live_enabled": enable,
            }

            self.create_audit_record("go_live_change", response)
            return response

        except Exception as e:
            error_response = {"status": "error", "message": str(e)}
            self.create_audit_record("go_live_error", error_response)
            return error_response


# CLI interface for testing/debugging
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ops Bot Influence Commands")
    parser.add_argument(
        "command",
        choices=["status", "set", "kill", "go-live"],
        help="Command to execute",
    )
    parser.add_argument("--pct", type=int, help="Percentage for set command")
    parser.add_argument("--reason", default="CLI test", help="Reason for set command")
    parser.add_argument("--enable", action="store_true", help="Enable GO_LIVE flag")
    parser.add_argument("--disable", action="store_true", help="Disable GO_LIVE flag")
    args = parser.parse_args()

    bot = InfluenceBotCommands()

    if args.command == "status":
        result = bot.command_status()
    elif args.command == "set":
        if args.pct is None:
            print("❌ --pct required for set command")
            sys.exit(1)
        result = bot.command_set(args.pct, args.reason)
    elif args.command == "kill":
        result = bot.command_kill()
    elif args.command == "go-live":
        if args.enable:
            result = bot.command_go_live(True)
        elif args.disable:
            result = bot.command_go_live(False)
        else:
            print("❌ --enable or --disable required for go-live command")
            sys.exit(1)

    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    main()
