#!/usr/bin/env python3
"""
Param Server v1 - Hot-reloading parameter store

Provides hot-reloading functionality for model router rules with sub-100ms latency.
Uses watchdog for file system events and Redis pub/sub for distributed notifications.
"""

import os
import time
import logging
import signal
import threading
import yaml
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import redis
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .schemas import ModelRouterRules, ModelRoute, ParamServerConfig

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ReloadEvent:
    """Represents a configuration reload event."""

    timestamp: float
    config_hash: str
    rules_count: int
    latency_ms: float
    success: bool
    error: Optional[str] = None


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration changes."""

    def __init__(self, param_server: "ParamServer"):
        self.param_server = param_server
        self.last_reload = 0
        self.debounce_seconds = 0.1  # Debounce rapid file changes

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        # Check if this is our config file
        if not self._is_config_file(event.src_path):
            return

        # Debounce rapid changes
        current_time = time.time()
        if current_time - self.last_reload < self.debounce_seconds:
            return

        self.last_reload = current_time
        logger.info(f"Config file change detected: {event.src_path}")

        # Trigger reload asynchronously
        threading.Thread(
            target=self.param_server._handle_file_change,
            args=(event.src_path,),
            daemon=True,
        ).start()

    def _is_config_file(self, path: str) -> bool:
        """Check if the modified file is our configuration file."""
        return Path(path).resolve() == Path(self.param_server.config_path).resolve()


class ParamServer:
    """
    Hot-reloading parameter server for model router rules.

    Features:
    - File-watch based hot reloading with <100ms latency
    - Redis pub/sub for distributed notifications
    - Pydantic validation with fallback to previous config on error
    - Zero-allocation fastpath for rule access
    - SIGHUP signal support for manual reloading
    """

    def __init__(self, config_path: str, redis_url: Optional[str] = None):
        """Initialize parameter server with configuration file path."""
        self.config_path = Path(config_path).resolve()
        self.redis_url = redis_url
        self.redis_client = None

        # Thread-safe rule storage
        self._rules_lock = threading.RLock()
        self._current_rules: Optional[ModelRouterRules] = None
        self._backup_rules: Optional[ModelRouterRules] = None

        # File watching
        self._observer: Optional[Observer] = None
        self._file_handler: Optional[ConfigFileHandler] = None

        # Performance tracking
        self._reload_events: List[ReloadEvent] = []
        self._load_count = 0
        self._total_load_time_ms = 0.0

        # Signal handlers
        self._original_sighup_handler = None

        # Initialize Redis if URL provided
        if self.redis_url:
            try:
                self.redis_client = redis.Redis.from_url(self.redis_url)
                self.redis_client.ping()  # Test connection
                logger.info(f"Connected to Redis at {self.redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.redis_client = None

        # Load initial configuration
        self._load_config()

        # Setup file watching
        self._setup_file_watching()

        # Setup signal handler
        self._setup_signal_handler()

        logger.info(
            f"ParamServer initialized with {len(self._current_rules.rules) if self._current_rules else 0} rules"
        )

    def get_rules(self) -> List[ModelRoute]:
        """
        Get current routing rules.

        Zero-allocation fastpath for high-frequency access.
        Thread-safe with minimal locking overhead.
        """
        with self._rules_lock:
            if self._current_rules is None:
                return []
            return self._current_rules.get_sorted_rules()

    def get_config(self) -> Optional[ModelRouterRules]:
        """Get the complete current configuration."""
        with self._rules_lock:
            return self._current_rules

    def watch(self):
        """
        Start file watching and hot reload mechanism.

        Enables automatic reloading on file changes and SIGHUP signals.
        """
        if self._observer and not self._observer.is_alive():
            self._observer.start()
            logger.info("File watching started")

    def stop_watching(self):
        """Stop file watching."""
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join(timeout=5.0)
            logger.info("File watching stopped")

    def reload_config(self) -> bool:
        """
        Manually trigger configuration reload.

        Returns:
            True if reload was successful, False otherwise.
        """
        logger.info("Manual config reload triggered")
        return self._load_config()

    def _load_config(self) -> bool:
        """Load configuration from file with validation and error handling."""
        start_time = time.perf_counter()

        try:
            # Read YAML file
            with open(self.config_path, "r") as f:
                raw_config = yaml.safe_load(f)

            # Handle nested model_router structure
            if "model_router" in raw_config:
                config_data = raw_config["model_router"]
            else:
                config_data = raw_config

            # Validate with Pydantic
            new_rules = ModelRouterRules.model_validate(config_data)

            # Calculate config hash for change detection
            config_str = json.dumps(config_data, sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

            # Atomic update with backup
            with self._rules_lock:
                self._backup_rules = self._current_rules
                self._current_rules = new_rules

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Update performance metrics
            self._load_count += 1
            self._total_load_time_ms += latency_ms

            # Record reload event
            reload_event = ReloadEvent(
                timestamp=time.time(),
                config_hash=config_hash,
                rules_count=len(new_rules.rules),
                latency_ms=latency_ms,
                success=True,
            )
            self._reload_events.append(reload_event)
            if len(self._reload_events) > 100:
                self._reload_events = self._reload_events[-100:]

            # Publish reload notification
            self._publish_reload_notification(reload_event)

            logger.info(
                f"Config loaded successfully: {len(new_rules.rules)} rules, "
                f"latency={latency_ms:.1f}ms, hash={config_hash}"
            )

            return True

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record failed reload event
            reload_event = ReloadEvent(
                timestamp=time.time(),
                config_hash="error",
                rules_count=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )
            self._reload_events.append(reload_event)
            if len(self._reload_events) > 100:
                self._reload_events = self._reload_events[-100:]

            logger.error(f"Config load failed: {e} (latency={latency_ms:.1f}ms)")

            # Keep previous configuration on error
            if self._current_rules is None:
                # No previous config, create minimal fallback
                self._create_fallback_config()

            return False

    def _create_fallback_config(self):
        """Create minimal fallback configuration when initial load fails."""
        from .schemas import create_default_router_rules

        try:
            fallback_rules = create_default_router_rules()
            with self._rules_lock:
                self._current_rules = fallback_rules
            logger.warning("Using fallback configuration due to load failure")
        except Exception as e:
            logger.error(f"Failed to create fallback config: {e}")

    def _setup_file_watching(self):
        """Setup file system watching for hot reload."""
        try:
            self._observer = Observer()
            self._file_handler = ConfigFileHandler(self)

            # Watch the directory containing the config file
            watch_dir = self.config_path.parent
            self._observer.schedule(self._file_handler, str(watch_dir), recursive=False)

            logger.info(f"File watching setup for {watch_dir}")

        except Exception as e:
            logger.error(f"Failed to setup file watching: {e}")

    def _setup_signal_handler(self):
        """Setup SIGHUP signal handler for manual reload."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, triggering config reload")
            self.reload_config()

        try:
            self._original_sighup_handler = signal.signal(signal.SIGHUP, signal_handler)
            logger.info("SIGHUP signal handler registered")
        except Exception as e:
            logger.warning(f"Failed to setup signal handler: {e}")

    def _handle_file_change(self, file_path: str):
        """Handle file system change event."""
        logger.debug(f"Processing file change: {file_path}")
        self._load_config()

    def _publish_reload_notification(self, event: ReloadEvent):
        """Publish reload notification to Redis pub/sub."""
        if not self.redis_client:
            return

        try:
            message = {
                "component": "router",
                "timestamp": event.timestamp,
                "config_hash": event.config_hash,
                "rules_count": event.rules_count,
                "latency_ms": event.latency_ms,
                "success": event.success,
            }

            if event.error:
                message["error"] = event.error

            channel = "param.reload"
            self.redis_client.publish(channel, json.dumps(message))

            # Also store in Redis hash for persistence
            redis_key = "param:model_router"
            if event.success and self._current_rules:
                config_dict = self._current_rules.model_dump()
                for key, value in config_dict.items():
                    self.redis_client.hset(redis_key, key, json.dumps(value))
                self.redis_client.expire(redis_key, 3600)  # 1 hour TTL

            logger.debug(f"Published reload notification to {channel}")

        except Exception as e:
            logger.warning(f"Failed to publish reload notification: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        with self._rules_lock:
            current_rules_count = (
                len(self._current_rules.rules) if self._current_rules else 0
            )

        avg_load_time = self._total_load_time_ms / max(1, self._load_count)

        recent_events = self._reload_events[-10:]  # Last 10 events
        successful_reloads = sum(1 for e in recent_events if e.success)

        return {
            "load_count": self._load_count,
            "avg_load_time_ms": avg_load_time,
            "current_rules_count": current_rules_count,
            "recent_reload_success_rate": successful_reloads
            / max(1, len(recent_events)),
            "recent_events": [
                {
                    "timestamp": e.timestamp,
                    "success": e.success,
                    "latency_ms": e.latency_ms,
                    "rules_count": e.rules_count,
                }
                for e in recent_events
            ],
        }

    def get_reload_events(self, limit: int = 50) -> List[ReloadEvent]:
        """Get recent reload events for debugging."""
        return self._reload_events[-limit:]

    def validate_config_file(
        self, config_path: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Validate configuration file without loading it."""
        path = Path(config_path) if config_path else self.config_path

        try:
            with open(path, "r") as f:
                raw_config = yaml.safe_load(f)

            if "model_router" in raw_config:
                config_data = raw_config["model_router"]
            else:
                config_data = raw_config

            # Validate with Pydantic
            ModelRouterRules.model_validate(config_data)
            return True, None

        except Exception as e:
            return False, str(e)

    def __enter__(self):
        """Context manager entry."""
        self.watch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_watching()

        # Restore signal handler
        if self._original_sighup_handler is not None:
            signal.signal(signal.SIGHUP, self._original_sighup_handler)


# Factory function for easy instantiation
def create_param_server(
    config_path: str, redis_url: Optional[str] = None
) -> ParamServer:
    """Factory function to create ParamServer instance."""
    return ParamServer(config_path=config_path, redis_url=redis_url)


# Context manager for temporary parameter server
class TempParamServer:
    """Temporary parameter server for testing."""

    def __init__(self, rules: ModelRouterRules):
        self.rules = rules

    def get_rules(self) -> List[ModelRoute]:
        return self.rules.get_sorted_rules()

    def get_config(self) -> ModelRouterRules:
        return self.rules
