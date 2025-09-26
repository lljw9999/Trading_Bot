#!/usr/bin/env python3
"""
Unit tests for Influence Slack Integration
Tests Slack command parsing, authorization, and response formatting
"""
import pytest
import os
import sys
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ops_bot.slack_integration import SlackInfluenceBot
from ops_bot.influence_commands import InfluenceBotCommands


class TestSlackInfluenceBot:
    def setup_method(self):
        """Set up test fixtures."""
        self.bot = SlackInfluenceBot()

    def test_parse_influence_command_status(self):
        """Test parsing status command."""
        command, params = self.bot.parse_influence_command("status")
        assert command == "status"
        assert params == {}

    def test_parse_influence_command_kill(self):
        """Test parsing kill command."""
        command, params = self.bot.parse_influence_command("kill")
        assert command == "kill"
        assert params == {}

    def test_parse_influence_command_set_with_because(self):
        """Test parsing set command with 'because' keyword."""
        command, params = self.bot.parse_influence_command(
            "set 25 because testing canary deployment"
        )
        assert command == "set"
        assert params["percentage"] == 25
        assert params["reason"] == "testing canary deployment"

    def test_parse_influence_command_set_without_because(self):
        """Test parsing set command without 'because' keyword."""
        command, params = self.bot.parse_influence_command("set 10 emergency response")
        assert command == "set"
        assert params["percentage"] == 10
        assert params["reason"] == "emergency response"

    def test_parse_influence_command_invalid(self):
        """Test parsing invalid command."""
        command, params = self.bot.parse_influence_command("invalid command")
        assert command is None
        assert params == {}

    def test_is_user_authorized_no_restriction(self):
        """Test user authorization when no restriction is set."""
        # Default behavior - no authorized users set
        assert self.bot.is_user_authorized("any_user_id") is True

    def test_is_user_authorized_with_restriction(self):
        """Test user authorization with restrictions."""
        self.bot.authorized_users = {"user1", "user2", "admin"}

        assert self.bot.is_user_authorized("user1") is True
        assert self.bot.is_user_authorized("user2") is True
        assert self.bot.is_user_authorized("unauthorized") is False

    def test_format_status_response_shadow_mode(self):
        """Test formatting status response for shadow mode."""
        status_data = {
            "status": "success",
            "current_influence": 0,
            "current_weight": 0.0,
            "ttl_seconds": 0,
            "go_live_enabled": False,
            "max_allowed": 10,
        }

        response = self.bot.format_status_response(status_data)

        assert "🛡️" in response  # Shadow mode icon
        assert "0%" in response
        assert "Shadow Mode" in response
        assert "No trading impact" in response

    def test_format_status_response_limited_live(self):
        """Test formatting status response for limited live mode."""
        status_data = {
            "status": "success",
            "current_influence": 10,
            "current_weight": 0.1,
            "ttl_seconds": 1800,  # 30 minutes
            "go_live_enabled": False,
            "max_allowed": 10,
        }

        response = self.bot.format_status_response(status_data)

        assert "⚠️" in response  # Limited live icon
        assert "10%" in response
        assert "Limited Live" in response
        assert "0h 30m" in response  # TTL formatting

    def test_format_status_response_high_impact(self):
        """Test formatting status response for high impact mode."""
        status_data = {
            "status": "success",
            "current_influence": 50,
            "current_weight": 0.5,
            "ttl_seconds": 3600,  # 1 hour
            "go_live_enabled": True,
            "max_allowed": 100,
        }

        response = self.bot.format_status_response(status_data)

        assert "🚨" in response  # High impact icon
        assert "50%" in response
        assert "High Impact" in response
        assert "1h 0m" in response
        assert "🟢 ENABLED" in response  # GO_LIVE enabled

    def test_format_status_response_error(self):
        """Test formatting status response for error case."""
        status_data = {"status": "error", "message": "Redis connection failed"}

        response = self.bot.format_status_response(status_data)

        assert "❌ Error:" in response
        assert "Redis connection failed" in response

    def test_format_set_response_success(self):
        """Test formatting set response for successful change."""
        set_data = {
            "status": "success",
            "new_influence": 25,
            "reason": "Testing deployment",
        }

        response = self.bot.format_set_response(set_data)

        assert "🚨" in response  # High impact icon for 25%
        assert "25%" in response
        assert "Testing deployment" in response
        assert "Auto-expires in 1 hour" in response

    def test_format_set_response_error(self):
        """Test formatting set response for error case."""
        set_data = {
            "status": "error",
            "message": "Ramp guard failed: insufficient validation history",
        }

        response = self.bot.format_set_response(set_data)

        assert "❌" in response
        assert "Failed to set influence" in response
        assert "Ramp guard failed" in response

    def test_format_kill_response_success(self):
        """Test formatting kill response for successful execution."""
        kill_data = {
            "status": "success",
            "message": "Emergency kill-switch executed - influence set to 0%",
        }

        response = self.bot.format_kill_response(kill_data)

        assert "🚨" in response
        assert "EMERGENCY KILL-SWITCH EXECUTED" in response
        assert "0%" in response
        assert "shadow mode" in response

    def test_format_kill_response_error(self):
        """Test formatting kill response for error case."""
        kill_data = {"status": "error", "message": "Kill-switch script not found"}

        response = self.bot.format_kill_response(kill_data)

        assert "❌" in response
        assert "Kill-switch failed" in response

    def test_handle_slack_command_unauthorized(self):
        """Test handling command from unauthorized user."""
        self.bot.authorized_users = {"admin"}

        response = self.bot.handle_slack_command(
            "unauthorized_user", "channel", "status"
        )

        assert response["response_type"] == "ephemeral"
        assert "not authorized" in response["text"]

    def test_handle_slack_command_invalid_command(self):
        """Test handling invalid command."""
        response = self.bot.handle_slack_command(
            "user", "channel", "invalid command syntax"
        )

        assert response["response_type"] == "ephemeral"
        assert "RL Policy Influence Commands" in response["text"]
        assert "Examples:" in response["text"]

    def test_handle_slack_command_status_success(self):
        """Test handling successful status command."""
        with patch.object(self.bot.commands, "command_status") as mock_status:
            mock_status.return_value = {
                "status": "success",
                "current_influence": 0,
                "current_weight": 0.0,
                "ttl_seconds": 0,
                "go_live_enabled": False,
                "max_allowed": 10,
            }

            response = self.bot.handle_slack_command("user", "channel", "status")

            assert response["response_type"] == "in_channel"  # Public success
            assert "🛡️" in response["text"]  # Shadow mode
            assert "Shadow Mode" in response["text"]

    def test_handle_slack_command_set_success(self):
        """Test handling successful set command."""
        with patch.object(self.bot.commands, "command_set") as mock_set:
            mock_set.return_value = {
                "status": "success",
                "new_influence": 15,
                "reason": "testing deployment",
            }

            response = self.bot.handle_slack_command(
                "user", "channel", "set 15 because testing deployment"
            )

            assert response["response_type"] == "in_channel"  # Public success
            assert "15%" in response["text"]
            assert "testing deployment" in response["text"]

    def test_handle_slack_command_set_failure(self):
        """Test handling failed set command."""
        with patch.object(self.bot.commands, "command_set") as mock_set:
            mock_set.return_value = {"status": "error", "message": "Ramp guard failed"}

            response = self.bot.handle_slack_command(
                "user", "channel", "set 25 because testing"
            )

            assert response["response_type"] == "ephemeral"  # Private error
            assert "Failed to set influence" in response["text"]

    def test_handle_slack_command_kill_success(self):
        """Test handling successful kill command."""
        with patch.object(self.bot.commands, "command_kill") as mock_kill:
            mock_kill.return_value = {
                "status": "success",
                "message": "Kill-switch executed",
            }

            response = self.bot.handle_slack_command("user", "channel", "kill")

            assert response["response_type"] == "in_channel"  # Public success
            assert "EMERGENCY KILL-SWITCH EXECUTED" in response["text"]

    def test_handle_slack_command_exception(self):
        """Test handling command when exception occurs."""
        with patch.object(
            self.bot.commands, "command_status", side_effect=Exception("Test error")
        ):
            response = self.bot.handle_slack_command("user", "channel", "status")

            assert response["response_type"] == "ephemeral"
            assert "Command failed:" in response["text"]
            assert "Test error" in response["text"]


class TestInfluenceBotCommands:
    def setup_method(self):
        """Set up test fixtures."""
        self.commands = InfluenceBotCommands()

    def test_is_go_live_enabled_true(self):
        """Test GO_LIVE flag detection when enabled."""
        with patch("redis.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.from_url.return_value = mock_client
            mock_client.get.return_value = "1"

            assert self.commands.is_go_live_enabled() is True

    def test_is_go_live_enabled_false(self):
        """Test GO_LIVE flag detection when disabled."""
        with patch("redis.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.from_url.return_value = mock_client
            mock_client.get.return_value = None

            assert self.commands.is_go_live_enabled() is False

    def test_command_status_success(self):
        """Test successful status command."""
        with patch("src.rl.influence_controller.InfluenceController") as mock_ic:
            mock_instance = MagicMock()
            mock_instance.get_status.return_value = {
                "percentage": 10,
                "weight": 0.1,
                "key_exists": True,
                "ttl_seconds": 1800,
            }
            mock_ic.return_value = mock_instance

            with patch.object(self.commands, "is_go_live_enabled", return_value=False):
                with patch.object(self.commands, "create_audit_record"):
                    result = self.commands.command_status()

                    assert result["status"] == "success"
                    assert result["current_influence"] == 10
                    assert result["max_allowed"] == 10  # Default max without GO_LIVE

    def test_command_set_success_within_limit(self):
        """Test successful set command within default limit."""
        with patch.object(self.commands, "is_go_live_enabled", return_value=False):
            with patch.object(
                self.commands, "run_ramp_guard", return_value=(True, "PASS")
            ):
                with patch("subprocess.run") as mock_run:
                    mock_result = MagicMock()
                    mock_result.returncode = 0
                    mock_run.return_value = mock_result

                    with patch.object(self.commands, "create_audit_record"):
                        result = self.commands.command_set(5, "testing")

                        assert result["status"] == "success"
                        assert result["new_influence"] == 5

    def test_command_set_blocked_by_limit(self):
        """Test set command blocked by influence limit."""
        with patch.object(self.commands, "is_go_live_enabled", return_value=False):
            with patch.object(self.commands, "create_audit_record"):
                result = self.commands.command_set(
                    25, "testing"
                )  # Above 10% default limit

                assert result["status"] == "error"
                assert "exceeds max allowed" in result["message"]
                assert result["max_allowed"] == 10

    def test_command_set_blocked_by_ramp_guard(self):
        """Test set command blocked by ramp guard failure."""
        with patch.object(
            self.commands, "is_go_live_enabled", return_value=True
        ):  # Allow high percentage
            with patch.object(
                self.commands,
                "run_ramp_guard",
                return_value=(False, "Insufficient validations"),
            ):
                with patch.object(self.commands, "create_audit_record"):
                    result = self.commands.command_set(25, "testing")

                    assert result["status"] == "error"
                    assert "Ramp guard failed" in result["message"]

    def test_command_set_zero_percentage(self):
        """Test set command with zero percentage (no ramp guard needed)."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            with patch.object(self.commands, "create_audit_record"):
                result = self.commands.command_set(0, "emergency stop")

                assert result["status"] == "success"
                assert result["new_influence"] == 0

    def test_command_kill_success(self):
        """Test successful kill command."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            with patch.object(self.commands, "create_audit_record"):
                result = self.commands.command_kill()

                assert result["status"] == "success"
                assert "kill-switch executed" in result["message"]
                assert result["new_influence"] == 0

    def test_command_kill_failure(self):
        """Test failed kill command."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "Kill script not found"
            mock_run.return_value = mock_result

            with patch.object(self.commands, "create_audit_record"):
                result = self.commands.command_kill()

                assert result["status"] == "error"
                assert "Kill-switch failed" in result["message"]

    def test_command_go_live_enable(self):
        """Test enabling GO_LIVE flag."""
        with patch("redis.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.from_url.return_value = mock_client

            with patch.object(self.commands, "create_audit_record"):
                result = self.commands.command_go_live(True)

                assert result["status"] == "success"
                assert "ENABLED" in result["message"]
                assert result["go_live_enabled"] is True
                mock_client.set.assert_called_with("ops:go_live", "1")

    def test_command_go_live_disable(self):
        """Test disabling GO_LIVE flag."""
        with patch("redis.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.from_url.return_value = mock_client

            with patch.object(self.commands, "create_audit_record"):
                result = self.commands.command_go_live(False)

                assert result["status"] == "success"
                assert "DISABLED" in result["message"]
                assert result["go_live_enabled"] is False
                mock_client.delete.assert_called_with("ops:go_live")

    def test_run_ramp_guard_pass(self):
        """Test ramp guard execution - pass case."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            success, details = self.commands.run_ramp_guard()

            assert success is True
            assert details == "PASS"

    def test_run_ramp_guard_fail(self):
        """Test ramp guard execution - fail case."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "Validation history insufficient"
            mock_run.return_value = mock_result

            success, details = self.commands.run_ramp_guard()

            assert success is False
            assert "Validation history insufficient" in details

    def test_create_audit_record(self):
        """Test audit record creation."""
        with patch("os.makedirs"):
            written_files = {}

            def mock_open(filename, mode):
                if mode == "w":
                    return MockFile(written_files, filename)

            with patch("builtins.open", side_effect=mock_open):
                audit_file = self.commands.create_audit_record(
                    "test_action", {"key": "value"}
                )

                assert len(written_files) == 1
                audit_content = list(written_files.values())[0]
                audit_data = json.loads(audit_content)

                assert "ts" in audit_data
                assert audit_data["action"] == "ops_bot_test_action"
                assert audit_data["details"]["key"] == "value"


class MockFile:
    """Mock file object for testing file writes."""

    def __init__(self, storage, filename):
        self.storage = storage
        self.filename = filename
        self.content = ""

    def write(self, data):
        self.content += data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.storage[self.filename] = self.content


def test_module_imports():
    """Test that Slack integration modules can be imported."""
    from ops_bot.slack_integration import SlackInfluenceBot
    from ops_bot.influence_commands import InfluenceBotCommands

    bot = SlackInfluenceBot()
    commands = InfluenceBotCommands()

    assert bot is not None
    assert commands is not None
    assert hasattr(bot, "handle_slack_command")
    assert hasattr(commands, "command_status")


if __name__ == "__main__":
    # Run tests directly
    import subprocess

    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v"], cwd=Path(__file__).parent.parent
    )
    # Note: sys.exit removed to avoid pytest return-value warnings
    # Test framework will handle success/failure appropriately
