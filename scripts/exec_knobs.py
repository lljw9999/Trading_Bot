#!/usr/bin/env python3
"""
Execution Knobs: Live Parameter Tuning System
Adjust execution parameters in real-time without code deployment.
"""
import os
import sys
import json
import datetime
import argparse
import redis
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path


class ExecutionKnobs:
    """Live-editable execution parameter management."""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.base_dir = Path("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

        # Redis connection
        try:
            self.redis_client = redis.Redis(
                host=redis_host, port=redis_port, decode_responses=True
            )
            self.redis_client.ping()
        except Exception as e:
            print(f"⚠️ Redis not available, using file-based fallback: {e}")
            self.redis_client = None

        # Load policy configuration
        self.policy_config = self.load_policy_config()

        # WORM audit log
        self.audit_log = []

    def load_policy_config(self) -> Dict[str, Any]:
        """Load execution policy configuration."""
        config_file = self.base_dir / "config" / "exec_policy.yaml"

        try:
            if config_file.exists():
                with open(config_file, "r") as f:
                    return yaml.safe_load(f)
            else:
                print(f"⚠️ Policy config not found: {config_file}")
                return {}
        except Exception as e:
            print(f"⚠️ Error loading policy config: {e}")
            return {}

    def get_knob_value(self, knob_name: str, default_value: Any = None) -> Any:
        """Get current knob value from Redis or config."""

        # Try Redis first
        if self.redis_client:
            try:
                redis_key = f"exec:{knob_name}"
                redis_value = self.redis_client.get(redis_key)

                if redis_value is not None:
                    # Try to parse as JSON first, then as string
                    try:
                        return json.loads(redis_value)
                    except:
                        # Return as string/number
                        try:
                            return float(redis_value)
                        except:
                            return redis_value

            except Exception as e:
                print(f"⚠️ Redis get error for {knob_name}: {e}")

        # Fallback to config file
        return self.get_config_value(knob_name, default_value)

    def get_config_value(self, knob_name: str, default_value: Any = None) -> Any:
        """Get value from policy configuration."""

        # Parse nested key path (e.g., "sizer_v2.slice_pct_max")
        keys = knob_name.split(".")
        config = self.policy_config

        try:
            for key in keys:
                config = config[key]
            return config
        except (KeyError, TypeError):
            return default_value

    def set_knob_value(
        self, knob_name: str, value: Any, reason: str = "manual"
    ) -> bool:
        """Set knob value with audit logging."""

        # Validate the parameter
        if not self.validate_knob(knob_name, value):
            print(f"❌ Invalid value {value} for knob {knob_name}")
            return False

        # Get previous value for audit
        old_value = self.get_knob_value(knob_name)

        # Set in Redis
        success = False
        if self.redis_client:
            try:
                redis_key = f"exec:{knob_name}"

                # Store as JSON if complex type
                if isinstance(value, (dict, list)):
                    redis_value = json.dumps(value)
                else:
                    redis_value = str(value)

                self.redis_client.set(redis_key, redis_value)

                # Set expiry to prevent stale values
                self.redis_client.expire(redis_key, 86400)  # 24 hours

                success = True
                print(f"✅ Set {knob_name} = {value}")

            except Exception as e:
                print(f"❌ Redis set error for {knob_name}: {e}")

        # Audit logging (WORM - Write Once Read Many)
        audit_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "knob_name": knob_name,
            "old_value": old_value,
            "new_value": value,
            "reason": reason,
            "success": success,
            "user": os.getenv("USER", "unknown"),
        }

        self.audit_log.append(audit_entry)
        self.write_audit_log(audit_entry)

        return success

    def validate_knob(self, knob_name: str, value: Any) -> bool:
        """Validate knob value against policy constraints."""

        # Get validation rules from config
        validation = self.policy_config.get("validation", {})

        if knob_name in validation:
            rules = validation[knob_name]

            # Range validation for numeric values
            if isinstance(rules, list) and len(rules) == 2:
                min_val, max_val = rules
                if isinstance(value, (int, float)):
                    return min_val <= value <= max_val

        # Known parameter types
        knob_validations = {
            "slice_pct_max": lambda v: isinstance(v, (int, float)) and 0.1 <= v <= 10.0,
            "post_only_base": lambda v: isinstance(v, (int, float))
            and 0.3 <= v <= 0.95,
            "sla_ms_crypto": lambda v: isinstance(v, (int, float)) and 50 <= v <= 500,
            "sla_ms_equities": lambda v: isinstance(v, (int, float))
            and 100 <= v <= 1000,
            "adverse_tau": lambda v: isinstance(v, (int, float)) and 0.1 <= v <= 1.0,
            "pov_cap": lambda v: isinstance(v, (int, float)) and 0.01 <= v <= 0.5,
            "eta_conf_hi": lambda v: isinstance(v, (int, float)) and 0.5 <= v <= 0.95,
        }

        # Extract base parameter name
        base_knob = knob_name.split(".")[-1]

        if base_knob in knob_validations:
            return knob_validations[base_knob](value)

        # Default: allow if reasonable type
        return isinstance(value, (str, int, float, bool, dict, list))

    def write_audit_log(self, audit_entry: Dict[str, Any]) -> None:
        """Write audit entry to WORM log."""

        audit_dir = self.base_dir / "artifacts" / "exec" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)

        # Daily audit log file
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        audit_file = audit_dir / f"exec_knobs_audit_{date_str}.jsonl"

        try:
            # Append to daily log (JSONL format)
            with open(audit_file, "a") as f:
                f.write(json.dumps(audit_entry) + "\n")

        except Exception as e:
            print(f"⚠️ Audit log write error: {e}")

    def list_knobs(self, pattern: str = None) -> Dict[str, Any]:
        """List current knob values."""

        knobs = {}

        # Get from Redis
        if self.redis_client:
            try:
                redis_pattern = f"exec:{pattern}*" if pattern else "exec:*"
                for key in self.redis_client.keys(redis_pattern):
                    knob_name = key.replace("exec:", "")
                    knobs[knob_name] = self.get_knob_value(knob_name)
            except Exception as e:
                print(f"⚠️ Redis list error: {e}")

        # Add config defaults for missing knobs
        if pattern is None:
            self._add_config_defaults(knobs)

        return knobs

    def _add_config_defaults(self, knobs: Dict[str, Any]) -> None:
        """Add config defaults for knobs not in Redis."""

        # Flatten config structure
        config_knobs = {}
        self._flatten_config(self.policy_config, "", config_knobs)

        for knob_name, default_value in config_knobs.items():
            if knob_name not in knobs:
                knobs[knob_name] = default_value

    def _flatten_config(
        self, config: Dict[str, Any], prefix: str, result: Dict[str, Any]
    ) -> None:
        """Flatten nested config structure."""

        for key, value in config.items():
            if key == "validation":  # Skip validation section
                continue

            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._flatten_config(value, full_key, result)
            else:
                result[full_key] = value

    def reset_knob(self, knob_name: str, reason: str = "reset") -> bool:
        """Reset knob to config default."""

        default_value = self.get_config_value(knob_name)

        if default_value is not None:
            # Delete from Redis to use config default
            if self.redis_client:
                try:
                    redis_key = f"exec:{knob_name}"
                    self.redis_client.delete(redis_key)
                    print(f"🔄 Reset {knob_name} to config default: {default_value}")

                    # Audit the reset
                    audit_entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "knob_name": knob_name,
                        "old_value": self.get_knob_value(knob_name),
                        "new_value": default_value,
                        "reason": f"reset: {reason}",
                        "success": True,
                        "user": os.getenv("USER", "unknown"),
                    }
                    self.write_audit_log(audit_entry)

                    return True

                except Exception as e:
                    print(f"❌ Reset error for {knob_name}: {e}")
        else:
            print(f"❌ No default value found for {knob_name}")

        return False

    def export_current_config(self, output_file: str = None) -> Dict[str, Any]:
        """Export current knob configuration."""

        current_config = self.list_knobs()

        export_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "exec_knobs_export",
            "knobs": current_config,
            "audit_entries": len(self.audit_log),
        }

        if output_file:
            output_path = Path(output_file)
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)
            print(f"📄 Config exported to: {output_path}")

        return export_data

    def bulk_update(self, updates: Dict[str, Any], reason: str = "bulk_update") -> int:
        """Update multiple knobs in one operation."""

        success_count = 0

        for knob_name, value in updates.items():
            if self.set_knob_value(knob_name, value, reason):
                success_count += 1

        print(f"✅ Successfully updated {success_count}/{len(updates)} knobs")
        return success_count


def main():
    """Main exec knobs CLI."""
    parser = argparse.ArgumentParser(
        description="Execution Knobs: Live Parameter Tuning"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Set command
    set_parser = subparsers.add_parser("set", help="Set knob value")
    set_parser.add_argument("knob_name", help="Knob name (e.g., exec:post_only_base)")
    set_parser.add_argument("value", help="New value")
    set_parser.add_argument("--reason", default="manual", help="Reason for change")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get knob value")
    get_parser.add_argument("knob_name", help="Knob name")

    # List command
    list_parser = subparsers.add_parser("list", help="List knobs")
    list_parser.add_argument("--pattern", help="Filter pattern")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset knob to default")
    reset_parser.add_argument("knob_name", help="Knob name")
    reset_parser.add_argument("--reason", default="reset", help="Reason for reset")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export current config")
    export_parser.add_argument("--out", help="Output file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        knobs = ExecutionKnobs()

        if args.command == "set":
            # Parse value
            try:
                # Try numeric first
                if "." in args.value:
                    value = float(args.value)
                else:
                    value = int(args.value)
            except ValueError:
                # Try boolean
                if args.value.lower() in ("true", "false"):
                    value = args.value.lower() == "true"
                else:
                    # Keep as string
                    value = args.value

            knob_name = args.knob_name.replace("exec:", "")  # Remove prefix if present
            success = knobs.set_knob_value(knob_name, value, args.reason)
            return 0 if success else 1

        elif args.command == "get":
            knob_name = args.knob_name.replace("exec:", "")
            value = knobs.get_knob_value(knob_name)
            print(f"{args.knob_name}: {value}")
            return 0

        elif args.command == "list":
            knob_list = knobs.list_knobs(args.pattern)
            print(f"📋 Execution Knobs ({len(knob_list)} total):")

            for knob_name, value in sorted(knob_list.items()):
                print(f"  {knob_name}: {value}")
            return 0

        elif args.command == "reset":
            knob_name = args.knob_name.replace("exec:", "")
            success = knobs.reset_knob(knob_name, args.reason)
            return 0 if success else 1

        elif args.command == "export":
            config = knobs.export_current_config(args.out)
            print(f"📊 Exported {len(config['knobs'])} knobs")
            return 0

    except Exception as e:
        print(f"❌ Exec knobs error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
