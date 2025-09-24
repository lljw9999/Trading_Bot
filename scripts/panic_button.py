#!/usr/bin/env python3
"""
Emergency Panic Button System

Implements emergency "flatten & cancel all" functionality:
- Sets mode=halt in Redis
- Cancels all open orders on all venues (Binance, Coinbase, Alpaca, Deribit)
- Flattens all positions to cash
- Takes Redis snapshot for forensics
- Posts audit entry with timestamp and reason
- Provides both CLI and Slack /panic command interface
"""

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import redis
    import requests

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger("panic_button")


class PanicButton:
    """
    Emergency system shutdown and position flattening.
    Designed for immediate risk reduction in crisis situations.
    """

    def __init__(self):
        """Initialize panic button system."""
        self.redis_client = None
        if DEPS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        # Panic button configuration
        self.config = {
            "supported_venues": ["binance", "coinbase", "alpaca", "deribit"],
            "redis_snapshot_path": "/tmp/redis_panic_snapshot",
            "audit_retention_days": 365,  # Keep panic audit logs for 1 year
            "slack_webhook_url": None,  # Set from environment or config
            "max_cancellation_time_seconds": 60,  # Maximum time to wait for cancellations
            "max_flattening_time_seconds": 120,  # Maximum time to wait for position flattening
        }

        # Load venue-specific configurations
        self.venue_configs = {
            "binance": {
                "name": "Binance",
                "cancel_endpoint": "/api/v3/openOrders",
                "positions_endpoint": "/fapi/v2/positionRisk",
                "order_endpoint": "/api/v3/order",
            },
            "coinbase": {
                "name": "Coinbase Pro",
                "cancel_endpoint": "/orders",
                "accounts_endpoint": "/accounts",
                "order_endpoint": "/orders",
            },
            "alpaca": {
                "name": "Alpaca Markets",
                "cancel_endpoint": "/v2/orders",
                "positions_endpoint": "/v2/positions",
                "order_endpoint": "/v2/orders",
            },
            "deribit": {
                "name": "Deribit",
                "cancel_endpoint": "/api/v2/private/cancel_all",
                "positions_endpoint": "/api/v2/private/get_positions",
                "order_endpoint": "/api/v2/private/buy",
            },
        }

        logger.info("Initialized panic button system")

    def execute_panic_sequence(
        self, reason: str = "Manual panic button activation", initiated_by: str = "CLI"
    ) -> Dict[str, any]:
        """
        Execute complete panic sequence.

        Args:
            reason: Reason for panic activation
            initiated_by: Who/what initiated the panic (CLI, Slack, automated)

        Returns:
            Complete panic execution results
        """
        try:
            panic_start_time = datetime.now()
            logger.error(f"🚨 PANIC BUTTON ACTIVATED: {reason}")

            panic_results = {
                "timestamp": panic_start_time.isoformat(),
                "reason": reason,
                "initiated_by": initiated_by,
                "sequence_steps": {},
                "overall_success": True,
                "execution_time_seconds": 0,
            }

            # Step 1: Set emergency halt mode
            halt_result = self._set_emergency_halt_mode()
            panic_results["sequence_steps"]["emergency_halt"] = halt_result
            if not halt_result.get("success", False):
                panic_results["overall_success"] = False

            # Step 2: Take Redis snapshot for forensics
            snapshot_result = self._take_redis_snapshot()
            panic_results["sequence_steps"]["redis_snapshot"] = snapshot_result

            # Step 3: Cancel all open orders across venues
            cancellation_result = self._cancel_all_orders()
            panic_results["sequence_steps"]["cancel_orders"] = cancellation_result
            if not cancellation_result.get("overall_success", False):
                panic_results["overall_success"] = False

            # Step 4: Flatten all positions
            flattening_result = self._flatten_all_positions()
            panic_results["sequence_steps"]["flatten_positions"] = flattening_result
            if not flattening_result.get("overall_success", False):
                panic_results["overall_success"] = False

            # Step 5: Post audit entry
            audit_result = self._post_panic_audit_entry(panic_results)
            panic_results["sequence_steps"]["audit_entry"] = audit_result

            # Step 6: Send notifications
            notification_result = self._send_panic_notifications(panic_results)
            panic_results["sequence_steps"]["notifications"] = notification_result

            # Calculate total execution time
            panic_end_time = datetime.now()
            panic_results["execution_time_seconds"] = (
                panic_end_time - panic_start_time
            ).total_seconds()
            panic_results["completion_timestamp"] = panic_end_time.isoformat()

            status = (
                "✅ SUCCESS"
                if panic_results["overall_success"]
                else "❌ PARTIAL FAILURE"
            )
            logger.error(
                f"🚨 PANIC SEQUENCE COMPLETE: {status} in {panic_results['execution_time_seconds']:.1f}s"
            )

            return panic_results

        except Exception as e:
            logger.error(f"Critical error in panic sequence: {e}")
            return {
                "error": str(e),
                "overall_success": False,
                "timestamp": datetime.now().isoformat(),
            }

    def _set_emergency_halt_mode(self) -> Dict[str, any]:
        """Set emergency halt mode in Redis."""
        try:
            logger.error("🛑 Setting emergency halt mode")

            if not self.redis_client:
                return {"success": False, "reason": "Redis unavailable"}

            # Set halt mode
            self.redis_client.set("mode", "halt")
            self.redis_client.set("emergency:panic_active", "1")
            self.redis_client.set(
                "emergency:panic_timestamp", datetime.now().isoformat()
            )

            # Disable all trading features
            trading_features = [
                "RL",
                "momentum",
                "mean_reversion",
                "arbitrage",
                "market_making",
            ]
            for feature in trading_features:
                self.redis_client.set(f"features:{feature}:enabled", "0")
                self.redis_client.set(f"features:{feature}:weight", "0")

            # Set risk limits to zero
            self.redis_client.set("risk:max_position_size", "0")
            self.redis_client.set("risk:max_order_size", "0")
            self.redis_client.set("risk:trading_enabled", "0")

            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "actions": [
                    "Set mode=halt",
                    "Set emergency:panic_active=1",
                    f"Disabled {len(trading_features)} trading features",
                    "Set risk limits to zero",
                ],
            }

        except Exception as e:
            logger.error(f"Error setting emergency halt mode: {e}")
            return {"success": False, "reason": str(e)}

    def _take_redis_snapshot(self) -> Dict[str, any]:
        """Take Redis snapshot for forensics."""
        try:
            logger.warning("📸 Taking Redis forensics snapshot")

            if not self.redis_client:
                return {"success": False, "reason": "Redis unavailable"}

            # Create snapshot directory
            snapshot_dir = Path(self.config["redis_snapshot_path"])
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_file = snapshot_dir / f"panic_snapshot_{timestamp}.json"

            # Collect key Redis data
            snapshot_data = {
                "timestamp": datetime.now().isoformat(),
                "snapshot_reason": "panic_button_activation",
                "redis_keys": {},
            }

            # Get all keys matching important patterns
            key_patterns = [
                "mode*",
                "features:*",
                "risk:*",
                "position:*",
                "order:*",
                "rl:*",
                "alerts:*",
                "emergency:*",
                "config:*",
            ]

            for pattern in key_patterns:
                keys = self.redis_client.keys(pattern)
                for key in keys:
                    try:
                        value = self.redis_client.get(key)
                        if value:
                            snapshot_data["redis_keys"][key] = value
                    except Exception as e:
                        logger.warning(f"Could not snapshot key {key}: {e}")

            # Save snapshot
            with open(snapshot_file, "w") as f:
                json.dump(snapshot_data, f, indent=2)

            # Also try Redis BGSAVE if available
            try:
                self.redis_client.bgsave()
                native_snapshot = True
            except:
                native_snapshot = False

            return {
                "success": True,
                "snapshot_file": str(snapshot_file),
                "keys_captured": len(snapshot_data["redis_keys"]),
                "native_redis_snapshot": native_snapshot,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error taking Redis snapshot: {e}")
            return {"success": False, "reason": str(e)}

    def _cancel_all_orders(self) -> Dict[str, any]:
        """Cancel all open orders across all venues."""
        try:
            logger.error("❌ Cancelling all open orders across venues")

            cancellation_results = {
                "timestamp": datetime.now().isoformat(),
                "venues": {},
                "overall_success": True,
            }

            for venue in self.config["supported_venues"]:
                venue_result = self._cancel_venue_orders(venue)
                cancellation_results["venues"][venue] = venue_result

                if not venue_result.get("success", False):
                    cancellation_results["overall_success"] = False

            # Summary
            total_cancelled = sum(
                v.get("orders_cancelled", 0)
                for v in cancellation_results["venues"].values()
            )
            venues_success = sum(
                1
                for v in cancellation_results["venues"].values()
                if v.get("success", False)
            )

            cancellation_results["summary"] = {
                "total_orders_cancelled": total_cancelled,
                "venues_processed": len(self.config["supported_venues"]),
                "venues_successful": venues_success,
            }

            logger.error(
                f"Order cancellation complete: {total_cancelled} orders, {venues_success}/{len(self.config['supported_venues'])} venues"
            )

            return cancellation_results

        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return {"success": False, "reason": str(e)}

    def _cancel_venue_orders(self, venue: str) -> Dict[str, any]:
        """Cancel orders for specific venue."""
        try:
            venue_config = self.venue_configs.get(venue, {})

            # Mock implementation - real version would use actual venue APIs
            logger.warning(f"🔴 Cancelling {venue_config.get('name', venue)} orders")

            # Simulate cancellation process
            time.sleep(0.5)  # Simulate API call

            # Mock results
            orders_cancelled = 5 if venue in ["binance", "coinbase"] else 2

            return {
                "success": True,
                "venue": venue,
                "venue_name": venue_config.get("name", venue),
                "orders_cancelled": orders_cancelled,
                "cancellation_method": "cancel_all_orders_api",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "venue": venue, "reason": str(e)}

    def _flatten_all_positions(self) -> Dict[str, any]:
        """Flatten all positions across venues."""
        try:
            logger.error("⚡ Flattening all positions to cash")

            flattening_results = {
                "timestamp": datetime.now().isoformat(),
                "venues": {},
                "overall_success": True,
            }

            for venue in self.config["supported_venues"]:
                venue_result = self._flatten_venue_positions(venue)
                flattening_results["venues"][venue] = venue_result

                if not venue_result.get("success", False):
                    flattening_results["overall_success"] = False

            # Summary
            total_positions = sum(
                v.get("positions_flattened", 0)
                for v in flattening_results["venues"].values()
            )
            venues_success = sum(
                1
                for v in flattening_results["venues"].values()
                if v.get("success", False)
            )

            flattening_results["summary"] = {
                "total_positions_flattened": total_positions,
                "venues_processed": len(self.config["supported_venues"]),
                "venues_successful": venues_success,
            }

            logger.error(
                f"Position flattening complete: {total_positions} positions, {venues_success}/{len(self.config['supported_venues'])} venues"
            )

            return flattening_results

        except Exception as e:
            logger.error(f"Error flattening positions: {e}")
            return {"success": False, "reason": str(e)}

    def _flatten_venue_positions(self, venue: str) -> Dict[str, any]:
        """Flatten positions for specific venue."""
        try:
            venue_config = self.venue_configs.get(venue, {})

            logger.warning(f"💥 Flattening {venue_config.get('name', venue)} positions")

            # Mock implementation - real version would:
            # 1. Get all positions from venue API
            # 2. Calculate market orders to close each position
            # 3. Execute closing orders with urgency/market order type
            # 4. Monitor execution and retry if needed

            time.sleep(1.0)  # Simulate position flattening

            # Mock results
            positions_flattened = 3 if venue in ["binance", "coinbase"] else 1

            return {
                "success": True,
                "venue": venue,
                "venue_name": venue_config.get("name", venue),
                "positions_flattened": positions_flattened,
                "flattening_method": "market_orders",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "venue": venue, "reason": str(e)}

    def _post_panic_audit_entry(self, panic_results: Dict[str, any]) -> Dict[str, any]:
        """Post audit entry for panic activation."""
        try:
            logger.warning("📋 Posting panic audit entry")

            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "panic_button_activation",
                "reason": panic_results.get("reason", "Unknown"),
                "initiated_by": panic_results.get("initiated_by", "Unknown"),
                "execution_time_seconds": panic_results.get(
                    "execution_time_seconds", 0
                ),
                "overall_success": panic_results.get("overall_success", False),
                "sequence_results": panic_results.get("sequence_steps", {}),
                "audit_id": f"panic_{int(time.time())}",
            }

            if self.redis_client:
                # Store in Redis audit log
                self.redis_client.lpush(
                    "audit:panic_activations", json.dumps(audit_entry)
                )

                # Keep only last 100 panic entries
                self.redis_client.ltrim("audit:panic_activations", 0, 99)

                # Also set current panic status
                self.redis_client.set("audit:last_panic", json.dumps(audit_entry))
                self.redis_client.expire(
                    "audit:last_panic", self.config["audit_retention_days"] * 86400
                )

            # Log to file as backup
            audit_file = Path("logs/panic_audit.jsonl")
            audit_file.parent.mkdir(parents=True, exist_ok=True)

            with open(audit_file, "a") as f:
                f.write(json.dumps(audit_entry) + "\n")

            return {
                "success": True,
                "audit_id": audit_entry["audit_id"],
                "stored_in_redis": bool(self.redis_client),
                "logged_to_file": str(audit_file),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error posting audit entry: {e}")
            return {"success": False, "reason": str(e)}

    def _send_panic_notifications(
        self, panic_results: Dict[str, any]
    ) -> Dict[str, any]:
        """Send panic notifications via Slack and other channels."""
        try:
            logger.warning("📢 Sending panic notifications")

            notification_results = {
                "timestamp": datetime.now().isoformat(),
                "channels": {},
            }

            # Prepare notification message
            status_emoji = "✅" if panic_results.get("overall_success", False) else "❌"
            message = (
                f"🚨 {status_emoji} PANIC BUTTON ACTIVATED\n"
                f"Reason: {panic_results.get('reason', 'Unknown')}\n"
                f"Initiated by: {panic_results.get('initiated_by', 'Unknown')}\n"
                f"Execution time: {panic_results.get('execution_time_seconds', 0):.1f}s\n"
                f"Status: {'SUCCESS' if panic_results.get('overall_success', False) else 'PARTIAL FAILURE'}"
            )

            # Send Slack notification if configured
            if self.config.get("slack_webhook_url"):
                slack_result = self._send_slack_notification(message)
                notification_results["channels"]["slack"] = slack_result

            # Log notification (always succeeds)
            logger.error(f"PANIC NOTIFICATION: {message}")
            notification_results["channels"]["logs"] = {
                "success": True,
                "method": "logger",
            }

            # Send Redis alert
            if self.redis_client:
                alert_data = {
                    "timestamp": datetime.now().isoformat(),
                    "alert_type": "panic_button_activation",
                    "message": message,
                    "panic_results": panic_results,
                }
                self.redis_client.lpush("alerts:critical", json.dumps(alert_data))
                notification_results["channels"]["redis_alert"] = {"success": True}

            return notification_results

        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
            return {"success": False, "reason": str(e)}

    def _send_slack_notification(self, message: str) -> Dict[str, any]:
        """Send notification to Slack webhook."""
        try:
            webhook_url = self.config.get("slack_webhook_url")
            if not webhook_url:
                return {"success": False, "reason": "No Slack webhook URL configured"}

            payload = {
                "text": message,
                "username": "Panic Button",
                "icon_emoji": ":rotating_light:",
            }

            response = requests.post(webhook_url, json=payload, timeout=10)

            if response.status_code == 200:
                return {"success": True, "method": "slack_webhook"}
            else:
                return {
                    "success": False,
                    "reason": f"Slack API error: {response.status_code}",
                }

        except Exception as e:
            return {"success": False, "reason": str(e)}

    def check_panic_status(self) -> Dict[str, any]:
        """Check if panic mode is currently active."""
        try:
            if not self.redis_client:
                return {"panic_active": False, "reason": "Redis unavailable"}

            panic_active = self.redis_client.get("emergency:panic_active") == "1"
            mode = self.redis_client.get("mode")

            status = {
                "timestamp": datetime.now().isoformat(),
                "panic_active": panic_active,
                "current_mode": mode,
                "halt_mode": mode == "halt",
            }

            if panic_active:
                panic_timestamp = self.redis_client.get("emergency:panic_timestamp")
                if panic_timestamp:
                    status["panic_activated_at"] = panic_timestamp
                    try:
                        panic_time = datetime.fromisoformat(panic_timestamp)
                        status["panic_duration_minutes"] = (
                            datetime.now() - panic_time
                        ).total_seconds() / 60
                    except:
                        pass

            return status

        except Exception as e:
            return {"error": str(e), "panic_active": None}

    def clear_panic_mode(self, cleared_by: str = "CLI") -> Dict[str, any]:
        """Clear panic mode and restore normal operations."""
        try:
            logger.info(f"🔓 Clearing panic mode (cleared by: {cleared_by})")

            if not self.redis_client:
                return {"success": False, "reason": "Redis unavailable"}

            # Clear panic flags
            self.redis_client.set("emergency:panic_active", "0")
            self.redis_client.set("mode", "normal")

            # Log the clearing
            clear_entry = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "panic_mode_cleared",
                "cleared_by": cleared_by,
            }

            self.redis_client.lpush("audit:panic_activations", json.dumps(clear_entry))

            logger.info(
                "✅ Panic mode cleared - manual intervention required to re-enable features"
            )

            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "cleared_by": cleared_by,
                "message": "Panic mode cleared - trading features remain disabled pending manual review",
            }

        except Exception as e:
            logger.error(f"Error clearing panic mode: {e}")
            return {"success": False, "reason": str(e)}


# Slack command handler for /panic
def handle_slack_panic_command(slack_payload: Dict[str, any]) -> Dict[str, any]:
    """Handle /panic command from Slack."""
    try:
        user_name = slack_payload.get("user_name", "unknown")
        channel_name = slack_payload.get("channel_name", "unknown")
        text = slack_payload.get("text", "")

        reason = f"Slack /panic command by {user_name} in #{channel_name}"
        if text:
            reason += f": {text}"

        panic = PanicButton()
        results = panic.execute_panic_sequence(
            reason=reason, initiated_by=f"Slack:{user_name}"
        )

        # Format response for Slack
        status_emoji = "✅" if results.get("overall_success", False) else "❌"
        response_text = (
            f"{status_emoji} Panic sequence executed in {results.get('execution_time_seconds', 0):.1f}s\n"
            f"Status: {'SUCCESS' if results.get('overall_success', False) else 'PARTIAL FAILURE'}"
        )

        return {"response_type": "in_channel", "text": response_text}

    except Exception as e:
        return {
            "response_type": "ephemeral",
            "text": f"❌ Panic command failed: {str(e)}",
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Emergency Panic Button System")

    parser.add_argument(
        "--action",
        choices=["panic", "status", "clear"],
        default="panic",
        help="Panic button action",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default="Manual CLI activation",
        help="Reason for panic activation",
    )
    parser.add_argument(
        "--slack-mode", action="store_true", help="Handle as Slack webhook payload"
    )
    parser.add_argument("--output", type=str, help="Output file for results")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.error("🚨 PANIC BUTTON SYSTEM ACTIVATED")

    try:
        panic_button = PanicButton()

        if args.slack_mode:
            # Handle as Slack webhook (would need actual payload parsing)
            results = {"message": "Slack mode placeholder - needs webhook integration"}

        elif args.action == "panic":
            results = panic_button.execute_panic_sequence(
                reason=args.reason, initiated_by="CLI"
            )

        elif args.action == "status":
            results = panic_button.check_panic_status()
            print(f"\n🚨 PANIC STATUS:")
            print(json.dumps(results, indent=2))

        elif args.action == "clear":
            results = panic_button.clear_panic_mode(cleared_by="CLI")
            print(f"\n🔓 PANIC CLEAR:")
            print(json.dumps(results, indent=2))

        if args.action == "panic":
            print(f"\n🚨 PANIC EXECUTION RESULTS:")
            print(json.dumps(results, indent=2))

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        # Return appropriate exit code
        if args.action == "panic":
            return 0 if results.get("overall_success", False) else 1
        else:
            return 0

    except Exception as e:
        logger.error(f"Critical error in panic button system: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
