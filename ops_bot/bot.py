#!/usr/bin/env python3
"""
Ops Chatbot - Slack integration for trading bot operations
Provides real-time control and monitoring via Slack commands
"""

import os
import redis
import subprocess
import json
import time
import logging
from typing import Dict, Any
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("ops_bot")

# Initialize Slack app
app = App(
    token=os.environ.get("SLACK_BOT_TOKEN", "xoxb-test-token"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET", "test-secret"),
)

# Initialize Redis
r = redis.Redis(host="localhost", port=6379, decode_responses=True)


class OpsBot:
    """Operations bot with Slack integration."""

    def __init__(self):
        self.redis = r
        logger.info("🤖 Ops Bot initialized")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Get basic Redis stats
            risk_stats = self.redis.hgetall("risk:stats") or {}

            # Get deployment info
            active_color = self.redis.get("mode:active_color") or "unknown"
            mode = self.redis.get("mode") or "auto"

            # Get recent metrics
            gpu_mem_frac = self.redis.get("gpu:mem_frac") or "0.8"
            model_hash = self.redis.get("model:hash") or "unknown"

            # Calculate uptime from Redis
            try:
                redis_info = self.redis.info()
                uptime_seconds = redis_info.get("uptime_in_seconds", 0)
                uptime_hours = uptime_seconds / 3600
            except:
                uptime_hours = 0

            status = {
                "deployment": {
                    "active_color": active_color,
                    "mode": mode,
                    "uptime_hours": f"{uptime_hours:.1f}h",
                },
                "resources": {
                    "gpu_mem_fraction": gpu_mem_frac,
                    "model_hash": (
                        model_hash[:8] if model_hash != "unknown" else "unknown"
                    ),
                },
                "risk": risk_stats,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }

            return status

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}

    def format_status_message(self, status: Dict[str, Any]) -> str:
        """Format status for Slack display."""
        if "error" in status:
            return f"❌ Error getting status: {status['error']}"

        deployment = status.get("deployment", {})
        resources = status.get("resources", {})
        risk = status.get("risk", {})

        # Color emoji mapping
        color_emoji = {"blue": "🔵", "green": "🟢", "unknown": "⚪"}
        active_emoji = color_emoji.get(deployment.get("active_color", "unknown"), "⚪")

        message_parts = [
            "🚀 *Trading Bot Status*",
            f"{active_emoji} *Deployment:* {deployment.get('active_color', 'unknown').title()}",
            f"⚙️ *Mode:* {deployment.get('mode', 'unknown')}",
            f"⏰ *Uptime:* {deployment.get('uptime_hours', '0h')}",
            "",
            "📊 *Resources:*",
            f"  • GPU Memory: {resources.get('gpu_mem_fraction', 'unknown')}",
            f"  • Model Hash: `{resources.get('model_hash', 'unknown')}`",
            "",
        ]

        # Add risk metrics if available
        if risk:
            message_parts.extend(
                [
                    "⚡ *Risk Metrics:*",
                    f"  • Position Size: {risk.get('position_size', 'N/A')}",
                    f"  • Volatility: {risk.get('volatility', 'N/A')}",
                    f"  • Drawdown: {risk.get('drawdown', 'N/A')}",
                    "",
                ]
            )

        message_parts.append(f"🕒 *Last Updated:* {status.get('timestamp', 'unknown')}")

        return "\n".join(message_parts)

    def set_mode(self, mode: str) -> Dict[str, Any]:
        """Set trading mode."""
        valid_modes = ["auto", "manual", "paper", "halt"]

        if mode not in valid_modes:
            return {
                "success": False,
                "message": f"Invalid mode: {mode}. Valid modes: {', '.join(valid_modes)}",
            }

        try:
            # Set mode in Redis
            self.redis.set("mode", mode)

            # Log mode change
            mode_change = {
                "timestamp": int(time.time()),
                "mode": mode,
                "source": "slack_bot",
            }

            self.redis.lpush("mode:history", json.dumps(mode_change))
            self.redis.ltrim("mode:history", 0, 99)

            logger.info(f"Mode changed to {mode} via Slack")

            return {
                "success": True,
                "message": f"✅ Mode set to `{mode}`",
                "previous_mode": self.redis.get("mode:previous") or "unknown",
            }

        except Exception as e:
            logger.error(f"Error setting mode: {e}")
            return {"success": False, "message": f"❌ Error setting mode: {str(e)}"}

    def trigger_canary_switch(self, target_color: str) -> Dict[str, Any]:
        """Trigger canary deployment switch."""
        if target_color not in ["blue", "green"]:
            return {
                "success": False,
                "message": f"Invalid color: {target_color}. Must be 'blue' or 'green'",
            }

        try:
            # Import canary switch functionality
            import sys

            sys.path.append("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")
            from scripts.canary_switch import CanarySwitch

            switch = CanarySwitch()
            success = switch.switch(target_color)

            if success:
                return {
                    "success": True,
                    "message": f"🔄 Canary switch to `{target_color}` completed successfully",
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Canary switch to `{target_color}` failed",
                }

        except Exception as e:
            logger.error(f"Error in canary switch: {e}")
            return {"success": False, "message": f"❌ Canary switch error: {str(e)}"}

    def capture_incident_snapshot(self) -> Dict[str, Any]:
        """Capture incident snapshot."""
        try:
            # Import state capture
            import sys

            sys.path.append("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")
            from scripts.capture_state import StateCapture

            capture = StateCapture()
            result = capture.capture_snapshot()

            if result.get("status") == "completed":
                return {
                    "success": True,
                    "message": f"📸 Snapshot captured successfully",
                    "details": {
                        "size_mb": result.get("size_bytes", 0) / (1024 * 1024),
                        "s3_url": result.get("s3_url"),
                        "ipfs_cid": result.get("ipfs_cid"),
                        "components": len(result.get("components", [])),
                    },
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Snapshot capture failed: {result.get('error', 'unknown')}",
                }

        except Exception as e:
            logger.error(f"Error capturing snapshot: {e}")
            return {"success": False, "message": f"❌ Snapshot capture error: {str(e)}"}


# Initialize bot instance
ops_bot = OpsBot()


# Slack command handlers
@app.command("/status")
def handle_status(ack, respond):
    """Handle /status command."""
    ack()  # Acknowledge command immediately

    logger.info("Received /status command")
    status = ops_bot.get_system_status()
    message = ops_bot.format_status_message(status)
    respond(message)


@app.command("/mode")
def handle_mode(ack, respond, command):
    """Handle /mode command."""
    ack()  # Acknowledge command immediately

    mode_text = command["text"].strip() if command["text"] else ""
    logger.info(f"Received /mode command: {mode_text}")

    if not mode_text:
        current_mode = r.get("mode") or "auto"
        respond(
            f"📋 Current mode: `{current_mode}`\nValid modes: `auto`, `manual`, `paper`, `halt`"
        )
        return

    result = ops_bot.set_mode(mode_text)
    respond(result["message"])


@app.command("/canary")
def handle_canary(ack, respond, command):
    """Handle /canary command."""
    ack()  # Acknowledge command immediately

    color_text = command["text"].strip() if command["text"] else ""
    logger.info(f"Received /canary command: {color_text}")

    if not color_text:
        current_color = r.get("mode:active_color") or "unknown"
        respond(
            f"🔄 Current deployment: `{current_color}`\nUsage: `/canary blue` or `/canary green`"
        )
        return

    result = ops_bot.trigger_canary_switch(color_text)
    respond(result["message"])


@app.command("/capital")
def handle_capital(ack, respond, command):
    """Handle /capital command for staging capital increases."""
    ack()  # Acknowledge command immediately

    command_text = command["text"].strip() if command["text"] else ""
    logger.info(f"Received /capital command: {command_text}")

    if not command_text:
        # Show current capital status
        try:
            current_effective = float(r.get("risk:capital_effective") or 0.4)
            current_cap = float(r.get("risk:capital_cap") or 1.0)
            staged_request = r.get("risk:capital_stage_request")

            status_lines = [
                "💰 *Capital Status:*",
                f"  • Current Effective: {current_effective:.0%}",
                f"  • Current Cap: {current_cap:.0%}",
                f"  • Staged Request: {staged_request or 'None'}",
                "",
                "Usage: `/capital stage <10|20|30|50|70|100>`",
            ]

            respond("\n".join(status_lines))

        except Exception as e:
            respond(f"❌ Error getting capital status: {str(e)}")
        return

    # Parse stage command
    parts = command_text.split()
    if len(parts) != 2 or parts[0] != "stage":
        respond("❌ Invalid format. Use: `/capital stage <percentage>`")
        return

    try:
        percentage = int(parts[1])
        if percentage not in [10, 20, 30, 50, 70, 100]:
            respond("❌ Invalid percentage. Use: 10, 20, 30, 50, 70, or 100")
            return

        # Store staging request
        stage_fraction = percentage / 100.0
        r.set("risk:capital_stage_request", stage_fraction)
        r.set("risk:capital_stage_timestamp", int(time.time()))
        r.set("risk:capital_stage_user", command.get("user_id", "unknown"))

        # Create approval buttons
        approval_blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"💰 *Capital Staging Request*\n\nRequesting to stage capital allocation to *{percentage}%*\n\nThis will take effect next week after approval.",
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Approve"},
                        "style": "primary",
                        "action_id": "capital_approve",
                        "value": str(percentage),
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌ Reject"},
                        "style": "danger",
                        "action_id": "capital_reject",
                        "value": str(percentage),
                    },
                ],
            },
        ]

        respond(
            text=f"📋 Staging request recorded: {percentage}% (awaiting approval)",
            blocks=approval_blocks,
        )

        logger.info(
            f"Capital staging request: {percentage}% by user {command.get('user_id', 'unknown')}"
        )

    except ValueError:
        respond("❌ Invalid percentage value. Must be a number.")
    except Exception as e:
        logger.error(f"Error in capital staging: {e}")
        respond(f"❌ Error processing request: {str(e)}")


@app.action("capital_approve")
def handle_capital_approve(ack, body, respond):
    """Handle capital staging approval."""
    ack()

    try:
        # Get staged request details
        staged_fraction = float(r.get("risk:capital_stage_request") or 0)
        staged_percentage = int(staged_fraction * 100)
        user_id = body["user"]["id"]

        # Check SLO Gate and recon breaches
        slo_tier = r.get("slo:tier") or "A"
        recon_breaches = int(r.get("recon:breaches_24h") or 0)

        # Validate approval conditions
        if slo_tier == "C":
            respond("❌ Approval denied: SLO Gate shows Tier C performance")
            return

        if recon_breaches > 0:
            respond(
                f"❌ Approval denied: {recon_breaches} reconciliation breaches in last 24h"
            )
            return

        # Set next week's capital cap
        r.set("risk:capital_cap_next_week", staged_fraction)

        # Log approval to audit database
        import sqlite3
        from pathlib import Path

        db_path = Path(
            "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D/ab_history.db"
        )
        db_path.parent.mkdir(exist_ok=True)

        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS capital_approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                requested_percentage INTEGER,
                approved_fraction REAL,
                approved_by TEXT,
                slo_tier TEXT,
                recon_breaches INTEGER
            )
        """
        )

        conn.execute(
            """
            INSERT INTO capital_approvals 
            (timestamp, requested_percentage, approved_fraction, approved_by, slo_tier, recon_breaches)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                int(time.time()),
                staged_percentage,
                staged_fraction,
                user_id,
                slo_tier,
                recon_breaches,
            ),
        )
        conn.commit()
        conn.close()

        # Clear staging request
        r.delete("risk:capital_stage_request")
        r.delete("risk:capital_stage_timestamp")
        r.delete("risk:capital_stage_user")

        # Update response
        respond(
            f"✅ Capital staging approved: {staged_percentage}% will take effect next week\n\nApproved by: <@{user_id}>"
        )

        logger.info(f"Capital staging approved: {staged_percentage}% by {user_id}")

    except Exception as e:
        logger.error(f"Error approving capital staging: {e}")
        respond(f"❌ Error processing approval: {str(e)}")


@app.action("capital_reject")
def handle_capital_reject(ack, body, respond):
    """Handle capital staging rejection."""
    ack()

    try:
        staged_fraction = float(r.get("risk:capital_stage_request") or 0)
        staged_percentage = int(staged_fraction * 100)
        user_id = body["user"]["id"]

        # Clear staging request
        r.delete("risk:capital_stage_request")
        r.delete("risk:capital_stage_timestamp")
        r.delete("risk:capital_stage_user")

        respond(
            f"❌ Capital staging rejected: {staged_percentage}% request cancelled\n\nRejected by: <@{user_id}>"
        )

        logger.info(f"Capital staging rejected: {staged_percentage}% by {user_id}")

    except Exception as e:
        logger.error(f"Error rejecting capital staging: {e}")
        respond(f"❌ Error processing rejection: {str(e)}")


@app.command("/health")
def handle_health(ack, respond):
    """Handle /health command for quick system check."""
    ack()

    logger.info("Received /health command")

    try:
        # Quick health checks
        redis_ok = r.ping()
        mode = r.get("mode") or "auto"
        active_color = r.get("mode:active_color") or "unknown"

        # Add capital staging info
        current_effective = float(r.get("risk:capital_effective") or 0.4)
        staged_request = r.get("risk:capital_stage_request")

        health_items = [
            f"🟢 Redis: {'OK' if redis_ok else 'FAIL'}",
            f"⚙️ Mode: {mode}",
            f"🔄 Deployment: {active_color}",
            f"💰 Capital: {current_effective:.0%}{' (staged: ' + str(int(float(staged_request) * 100)) + '%)' if staged_request else ''}",
            f"🕒 Timestamp: {time.strftime('%H:%M:%S UTC', time.gmtime())}",
        ]

        health_message = "🏥 *Quick Health Check*\n" + "\n".join(health_items)
        respond(health_message)

    except Exception as e:
        logger.error(f"Health check error: {e}")
        respond(f"❌ Health check failed: {str(e)}")


@app.command("/snapshot")
def handle_snapshot(ack, respond, command):
    """Handle /snapshot command for incident snapshots."""
    ack()  # Acknowledge command immediately

    command_text = command["text"].strip() if command["text"] else ""
    logger.info(f"Received /snapshot command: {command_text}")

    if command_text != "now":
        respond(
            "📸 *Incident Snapshot*\n\nUsage: `/snapshot now` - Capture full system state"
        )
        return

    # Show immediate response
    respond("📸 Starting incident snapshot capture... this may take a moment.")

    try:
        # Capture snapshot
        result = ops_bot.capture_incident_snapshot()

        if result["success"]:
            details = result.get("details", {})
            s3_info = f"\n📤 S3: `{details['s3_url']}`" if details.get("s3_url") else ""
            ipfs_info = (
                f"\n📤 IPFS: `ipfs://{details['ipfs_cid']}`"
                if details.get("ipfs_cid")
                else ""
            )

            success_message = f"""✅ **Incident snapshot captured successfully**

📦 Size: {details.get('size_mb', 0):.1f} MB
🧩 Components: {details.get('components', 0)}{s3_info}{ipfs_info}

The snapshot contains:
• Feature flags state
• Current positions
• Active orders
• Model/Git hashes  
• Service status
• Key metrics
• Recent logs"""

            respond(success_message)
        else:
            respond(result["message"])

    except Exception as e:
        logger.error(f"Error in snapshot command: {e}")
        respond(f"❌ Error capturing snapshot: {str(e)}")


@app.event("app_mention")
def handle_app_mention(event, say):
    """Handle app mentions."""
    user = event.get("user", "unknown")
    text = event.get("text", "").lower()

    logger.info(f"App mentioned by {user}: {text}")

    if "status" in text:
        status = ops_bot.get_system_status()
        message = ops_bot.format_status_message(status)
        say(f"<@{user}> {message}")
    elif "help" in text:
        help_text = """🤖 *Ops Bot Commands:*
        
• `/status` - Get full system status
• `/mode [auto|manual|paper|halt]` - Set trading mode  
• `/canary [blue|green]` - Switch deployment
• `/capital [stage <10|20|30|50|70|100>]` - Capital staging with approval
• `/snapshot now` - Capture incident snapshot for audit
• `/health` - Quick health check

You can also mention me with `status` or `help`!"""
        say(f"<@{user}> {help_text}")
    else:
        say(
            f"<@{user}> Hi! Try `/status` for system info or mention me with `help` for commands."
        )


def main():
    """Main entry point."""
    logger.info("🚀 Starting Ops Bot")

    # Check environment variables
    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    app_token = os.environ.get("SLACK_APP_TOKEN")

    if not bot_token or bot_token == "xoxb-test-token":
        logger.warning("⚠️ SLACK_BOT_TOKEN not set - using test mode")

    if not app_token:
        logger.warning("⚠️ SLACK_APP_TOKEN not set - using HTTP mode")

    try:
        # Try socket mode first (requires app-level token)
        if app_token and app_token != "test-token":
            handler = SocketModeHandler(app, app_token)
            logger.info("🔌 Starting in Socket Mode")
            handler.start()
        else:
            # Fallback to HTTP mode
            logger.info("🌐 Starting in HTTP Mode on port 3001")
            app.start(port=int(os.environ.get("PORT", 3001)))

    except KeyboardInterrupt:
        logger.info("🛑 Ops Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Ops Bot error: {e}")
        raise


if __name__ == "__main__":
    main()
