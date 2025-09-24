#!/usr/bin/env python3
"""
Slack Panic Button Webhook Handler

Provides Flask webhook endpoint for Slack /panic slash command integration.
Handles authentication and executes emergency panic sequence.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from flask import Flask, request, jsonify
    from scripts.panic_button import PanicButton, handle_slack_panic_command

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger("slack_panic_webhook")

app = Flask(__name__)

# Configuration
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")  # For request verification
AUTHORIZED_SLACK_USERS = os.getenv("AUTHORIZED_PANIC_USERS", "").split(
    ","
)  # Comma-separated list


@app.route("/slack/panic", methods=["POST"])
def slack_panic_webhook():
    """Handle Slack /panic slash command."""
    try:
        # Verify request is from Slack (simplified - real implementation would verify signature)
        if not request.form:
            return jsonify({"error": "Invalid request format"}), 400

        # Extract Slack payload
        slack_payload = {
            "user_name": request.form.get("user_name", "unknown"),
            "user_id": request.form.get("user_id", "unknown"),
            "channel_name": request.form.get("channel_name", "unknown"),
            "channel_id": request.form.get("channel_id", "unknown"),
            "text": request.form.get("text", ""),
            "command": request.form.get("command", ""),
            "team_domain": request.form.get("team_domain", ""),
        }

        # Log panic activation attempt
        logger.error(
            f"🚨 Slack panic command attempted by {slack_payload['user_name']} ({slack_payload['user_id']}) in #{slack_payload['channel_name']}"
        )

        # Check authorization if configured
        if (
            AUTHORIZED_SLACK_USERS
            and slack_payload["user_name"] not in AUTHORIZED_SLACK_USERS
        ):
            logger.warning(
                f"❌ Unauthorized panic attempt by {slack_payload['user_name']}"
            )
            return (
                jsonify(
                    {
                        "response_type": "ephemeral",
                        "text": f"❌ Unauthorized: {slack_payload['user_name']} is not authorized to use panic button",
                    }
                ),
                403,
            )

        # Execute panic sequence
        response = handle_slack_panic_command(slack_payload)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error handling Slack panic webhook: {e}")
        return (
            jsonify(
                {
                    "response_type": "ephemeral",
                    "text": f"❌ Panic command failed: {str(e)}",
                }
            ),
            500,
        )


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "service": "slack_panic_webhook",
            "timestamp": "2025-01-15T12:00:00Z",
        }
    )


if __name__ == "__main__":
    if not DEPS_AVAILABLE:
        print("❌ Missing dependencies: flask")
        sys.exit(1)

    port = int(os.getenv("PANIC_WEBHOOK_PORT", 8080))
    logger.info(f"🚨 Starting Slack panic webhook on port {port}")

    app.run(host="0.0.0.0", port=port, debug=False)
