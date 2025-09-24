#!/usr/bin/env python3
"""
RL Policy Influence Controller
Safe, guarded influence weighting with Redis persistence and TTL safety
"""
import os
import time
import redis
import logging
from typing import Optional


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
KEY = "policy:allowed_influence_pct"
ASSET_KEY_FMT = "policy:allowed_influence_pct:{asset}"
FALLBACK = 0  # hard default to 0% for safety


class InfluenceController:
    """
    Controls RL policy influence weight with safety guardrails.

    Safety features:
    - Always defaults to 0% influence if key missing/invalid
    - TTL expiration prevents stale non-zero weights
    - Clamping prevents invalid percentage values
    - Read-only weight getter for execution paths
    """

    def __init__(self, ttl_sec: int = 3600):
        """
        Initialize influence controller.

        Args:
            ttl_sec: Time-to-live for influence keys (prevents stale non-zero)
                    Set to 0 to disable TTL. Default 1 hour.
        """
        self.r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        self.ttl_sec = ttl_sec
        self.logger = logging.getLogger("influence_controller")

    def get_weight(self) -> float:
        """
        Return influence weight w in [0,1].

        Safety: Falls back to 0.0 if key missing, invalid, or any error occurs.
        This ensures that policy influence is never accidentally enabled.

        Returns:
            float: Influence weight between 0.0 (baseline only) and 1.0 (policy only)
        """
        try:
            v = self.r.get(KEY)
            if v is None:
                return 0.0

            # Convert and clamp to valid percentage
            pct = max(0, min(100, int(float(v))))
            weight = pct / 100.0

            self.logger.debug(f"Retrieved influence weight: {weight:.2%}")
            return weight

        except redis.ConnectionError:
            self.logger.error("Redis connection failed - defaulting to 0% influence")
            return 0.0
        except Exception as e:
            self.logger.error(
                f"Error retrieving influence weight: {e} - defaulting to 0%"
            )
            return 0.0

    def set_weight(self, pct: int, reason: str = "ops_manual") -> int:
        """
        Setter for ops scripts; clamps percentage and sets TTL to avoid stale non-zero.

        Args:
            pct: Influence percentage (0-100)
            reason: Reason for change (for logging)

        Returns:
            int: Actual percentage set after clamping

        Raises:
            redis.ConnectionError: If Redis is unavailable
            Exception: For other Redis errors
        """
        # Safety clamp to valid range
        original_pct = pct
        pct = max(0, min(100, int(pct)))

        if pct != original_pct:
            self.logger.warning(f"Clamped influence from {original_pct}% to {pct}%")

        try:
            pipe = self.r.pipeline()
            pipe.set(KEY, pct)

            # Set TTL to prevent stale non-zero influence
            if self.ttl_sec > 0:
                pipe.expire(KEY, self.ttl_sec)

            pipe.execute()

            self.logger.info(f"Set influence to {pct}% (reason: {reason})")
            if self.ttl_sec > 0:
                self.logger.info(f"TTL set to {self.ttl_sec}s - will auto-expire to 0%")

            return pct

        except Exception as e:
            self.logger.error(f"Failed to set influence: {e}")
            raise

    def get_status(self) -> dict:
        """
        Get detailed status information for monitoring/debugging.

        Returns:
            dict: Status information including weight, TTL, key existence
        """
        try:
            raw_value = self.r.get(KEY)
            ttl = self.r.ttl(KEY) if raw_value else -1
            weight = self.get_weight()

            return {
                "weight": weight,
                "percentage": int(weight * 100),
                "raw_value": raw_value,
                "key_exists": raw_value is not None,
                "ttl_seconds": ttl,
                "ttl_expired": ttl == -2,
                "redis_key": KEY,
                "redis_url": REDIS_URL,
            }
        except Exception as e:
            return {"weight": 0.0, "percentage": 0, "error": str(e), "redis_key": KEY}

    def get_weight_asset(self, asset: str) -> float:
        """
        Return per-asset influence weight w in [0,1].

        Safety: Falls back to 0.0 if key missing, invalid, or any error occurs.

        Args:
            asset: Asset symbol (e.g., 'SOL-USD', 'BTC-USD')

        Returns:
            float: Influence weight between 0.0 and 1.0
        """
        try:
            v = self.r.get(ASSET_KEY_FMT.format(asset=asset))
            if v is None:
                return 0.0

            pct = max(0, min(100, int(float(v))))
            return pct / 100.0

        except Exception as e:
            self.logger.warning(f"Error getting asset weight for {asset}: {e}")
            return 0.0

    def set_weight_asset(self, asset: str, pct: int, reason: str = "n/a") -> int:
        """
        Set per-asset influence percentage with TTL and audit.

        Args:
            asset: Asset symbol (e.g., 'SOL-USD', 'BTC-USD')
            pct: Percentage influence (0-100)
            reason: Reason for change (for audit trail)

        Returns:
            int: Clamped percentage value that was set
        """
        try:
            pct = max(0, min(100, int(pct)))

            pipe = self.r.pipeline()
            asset_key = ASSET_KEY_FMT.format(asset=asset)
            pipe.set(asset_key, pct)

            if self.ttl_sec > 0:
                pipe.expire(asset_key, self.ttl_sec)

            pipe.execute()

            # WORM audit
            from datetime import datetime, timezone
            import json
            import pathlib

            audit_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "asset": asset,
                "pct": pct,
                "reason": reason,
                "action": "set_influence_pct_asset",
                "operator": os.getenv("USER", "system"),
                "ttl_seconds": self.ttl_sec,
            }

            pathlib.Path("artifacts/audit").mkdir(parents=True, exist_ok=True)
            audit_file = f"artifacts/audit/{audit_record['timestamp'].replace(':', '_')}_asset_influence.json"

            with open(audit_file, "w") as f:
                json.dump(audit_record, f, indent=2)

            self.logger.info(f"Set {asset} influence to {pct}% (reason: {reason})")
            if self.ttl_sec > 0:
                self.logger.info(f"TTL set to {self.ttl_sec}s - will auto-expire to 0%")

            return pct

        except Exception as e:
            self.logger.error(f"Failed to set asset influence for {asset}: {e}")
            raise

    def get_all_asset_weights(self) -> dict:
        """
        Get influence weights for all assets.

        Returns:
            dict: Asset symbol -> influence weight (0.0-1.0)
        """
        try:
            pattern = ASSET_KEY_FMT.format(asset="*")
            keys = self.r.keys(pattern)

            weights = {}
            for key in keys:
                # Extract asset from key pattern
                asset = key.split(":")[-1]
                weights[asset] = self.get_weight_asset(asset)

            return weights

        except Exception as e:
            self.logger.error(f"Failed to get all asset weights: {e}")
            return {}

    def emergency_stop(self) -> bool:
        """
        Emergency stop: immediately set influence to 0% with no TTL.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.r.set(KEY, 0)
            self.logger.critical("EMERGENCY STOP: Influence set to 0%")
            return True
        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
            return False


# Convenience functions for scripts
def get_current_influence() -> float:
    """Get current influence weight [0,1]."""
    return InfluenceController().get_weight()


def set_influence(pct: int, reason: str = "script_call") -> int:
    """Set influence percentage with reason."""
    return InfluenceController().set_weight(pct, reason)


def emergency_stop() -> bool:
    """Emergency stop - set influence to 0%."""
    return InfluenceController().emergency_stop()


if __name__ == "__main__":
    # CLI usage
    import sys

    if len(sys.argv) == 1:
        # Show current status
        ic = InfluenceController()
        status = ic.get_status()
        print(f"Current influence: {status['percentage']}%")
        if status.get("ttl_seconds", -1) > 0:
            print(f"TTL: {status['ttl_seconds']}s")
        sys.exit(0)
    elif sys.argv[1] == "stop":
        # Emergency stop
        success = emergency_stop()
        sys.exit(0 if success else 1)
    elif sys.argv[1].isdigit():
        # Set percentage
        pct = int(sys.argv[1])
        reason = sys.argv[2] if len(sys.argv) > 2 else "cli_manual"
        result = set_influence(pct, reason)
        print(f"Set influence to {result}%")
        sys.exit(0)
    else:
        print("Usage: python influence_controller.py [percentage] [reason] | stop")
        sys.exit(1)
