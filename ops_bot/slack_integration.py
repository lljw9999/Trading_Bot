#!/usr/bin/env python3
"""
Slack Bot Integration for Influence Commands
Example integration showing how to wire Slack commands to influence controls
"""
import os
import re
import json
from datetime import datetime
from ops_bot.influence_commands import InfluenceBotCommands


class SlackInfluenceBot:
    def __init__(self):
        self.commands = InfluenceBotCommands()
        auth_env = os.getenv("AUTHORIZED_USERS", "")
        self.authorized_users = {
            user.strip() for user in auth_env.split(",") if user.strip()
        }

    def is_user_authorized(self, user_id):
        """Check if user is authorized for influence commands."""
        if not self.authorized_users:
            return True  # If no restriction set, allow all
        return user_id in self.authorized_users

    def parse_influence_command(self, text):
        """Parse Slack influence command text."""
        text = text.strip().lower()

        # /influence status
        if text == "status":
            return "status", {}

        # /influence kill
        if text == "kill":
            return "kill", {}

        # /influence set 25 because reason text
        set_match = re.match(r"set\s+(\d+)\s+because\s+(.+)", text)
        if set_match:
            percentage = int(set_match.group(1))
            reason = set_match.group(2).strip()
            return "set", {"percentage": percentage, "reason": reason}

        # /influence set 25 reason text (without "because")
        set_match2 = re.match(r"set\s+(\d+)\s+(.+)", text)
        if set_match2:
            percentage = int(set_match2.group(1))
            reason = set_match2.group(2).strip()
            return "set", {"percentage": percentage, "reason": reason}

        return None, {}

    def format_status_response(self, status_data):
        """Format status response for Slack."""
        if status_data.get("status") == "error":
            return f"❌ Error: {status_data.get('message', 'Unknown error')}"

        influence = status_data.get("current_influence", 0)
        weight = status_data.get("current_weight", 0.0)
        ttl = status_data.get("ttl_seconds", 0)
        go_live = status_data.get("go_live_enabled", False)
        max_allowed = status_data.get("max_allowed", 10)

        status_icon = "🛡️" if influence == 0 else "⚠️" if influence <= 10 else "🚨"

        response = f"{status_icon} *RL Policy Influence Status*\n"
        response += f"• Current Influence: *{influence}%* (weight: {weight:.2f})\n"
        response += f"• Max Allowed: *{max_allowed}%*\n"
        response += f"• GO_LIVE Mode: {'🟢 ENABLED' if go_live else '🔴 DISABLED'}\n"

        if ttl > 0:
            hours = ttl // 3600
            minutes = (ttl % 3600) // 60
            response += f"• TTL Remaining: {hours}h {minutes}m\n"
        elif influence > 0:
            response += f"• TTL: No expiration\n"

        if influence == 0:
            response += f"\n🛡️ *Shadow Mode* - No trading impact"
        elif influence <= 10:
            response += f"\n⚠️ *Limited Live* - {influence}% trading impact"
        else:
            response += f"\n🚨 *High Impact* - {influence}% trading impact"

        return response

    def format_set_response(self, set_data):
        """Format set command response for Slack."""
        if set_data.get("status") == "error":
            return f"❌ *Failed to set influence*\n{set_data.get('message', 'Unknown error')}"

        new_influence = set_data.get("new_influence", 0)
        reason = set_data.get("reason", "")

        if new_influence == 0:
            icon = "🛡️"
            impact = "Shadow mode - no trading impact"
        elif new_influence <= 10:
            icon = "⚠️"
            impact = f"{new_influence}% trading impact"
        else:
            icon = "🚨"
            impact = f"{new_influence}% HIGH IMPACT"

        response = f"{icon} *Influence Updated*\n"
        response += f"• New Influence: *{new_influence}%*\n"
        response += f"• Impact: {impact}\n"
        response += f"• Reason: {reason}\n"
        response += f"• Auto-expires in 1 hour if not refreshed"

        return response

    def format_kill_response(self, kill_data):
        """Format kill command response for Slack."""
        if kill_data.get("status") == "error":
            return (
                f"❌ *Kill-switch failed*\n{kill_data.get('message', 'Unknown error')}"
            )

        return f"🚨 *EMERGENCY KILL-SWITCH EXECUTED*\n• Influence set to 0%\n• All trading impact disabled\n• System in shadow mode"

    def handle_slack_command(self, user_id, channel_id, text):
        """Handle incoming Slack command."""
        # Check authorization
        if not self.is_user_authorized(user_id):
            return {
                "response_type": "ephemeral",
                "text": "❌ You are not authorized to use influence commands",
            }

        # Parse command
        command, params = self.parse_influence_command(text)

        if command is None:
            help_text = """*RL Policy Influence Commands*
            
`/influence status` - Show current influence status
`/influence set <pct> because <reason>` - Set influence percentage
`/influence kill` - Emergency kill-switch (set to 0%)

*Examples:*
• `/influence status`
• `/influence set 10 because canary test`
• `/influence kill`

*Safety Notes:*
• Default max: 10% (unless GO_LIVE enabled)
• Ramp guard checks run automatically
• All changes are audited
• Auto-expires in 1 hour"""

            return {"response_type": "ephemeral", "text": help_text}

        # Execute command
        try:
            if command == "status":
                result = self.commands.command_status()
                response_text = self.format_status_response(result)

            elif command == "set":
                result = self.commands.command_set(
                    params["percentage"], params["reason"]
                )
                response_text = self.format_set_response(result)

            elif command == "kill":
                result = self.commands.command_kill()
                response_text = self.format_kill_response(result)

            # Determine response type
            if result.get("status") == "success":
                response_type = "in_channel"  # Public success
            else:
                response_type = "ephemeral"  # Private error

            return {"response_type": response_type, "text": response_text}

        except Exception as e:
            return {
                "response_type": "ephemeral",
                "text": f"❌ Command failed: {str(e)}",
            }


# Example Flask integration
def create_slack_app():
    """Example Flask app for Slack integration."""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask not available - install with: pip install flask")
        return None

    app = Flask(__name__)
    bot = SlackInfluenceBot()

    @app.route("/slack/influence", methods=["POST"])
    def slack_influence_command():
        """Handle /influence Slack command."""
        data = request.form
        user_id = data.get("user_id")
        channel_id = data.get("channel_id")
        text = data.get("text", "")

        response = bot.handle_slack_command(user_id, channel_id, text)
        return jsonify(response)

    return app


if __name__ == "__main__":
    # CLI test interface
    import sys

    if len(sys.argv) < 2:
        print("Usage: python slack_integration.py '<command text>'")
        print("Example: python slack_integration.py 'status'")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    bot = SlackInfluenceBot()

    # Test command parsing and execution
    response = bot.handle_slack_command("test_user", "test_channel", text)
    print(json.dumps(response, indent=2))
