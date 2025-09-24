#!/usr/bin/env python3
"""
Feature Flags Service
Centralized service for checking and managing A/B testing feature flags
"""

import redis
import logging
from typing import Dict, Optional

# Import with fallback for testing
try:
    from .param_server.schemas import FeatureFlags
except ImportError:
    try:
        from src.core.param_server.schemas import FeatureFlags
    except ImportError:
        # Fallback schema for testing
        class FeatureFlags:
            model_fields = {
                "EXEC_RL_LIVE": {"default": False},
                "BANDIT_WEIGHTS": {"default": False},
                "LLM_SENTIMENT": {"default": False},
                "TAIL_RISK_HEDGE": {"default": False},
                "DL_FINE_TUNE": {"default": True},
            }


logger = logging.getLogger(__name__)


class FeatureFlagService:
    """Service for managing feature flags with Redis backend."""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        """Initialize feature flag service."""
        self.redis = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )
        self.flags_key = "features:flags"

        # Cache for frequently accessed flags
        self._cache = {}
        self._cache_ttl = 60  # seconds

        logger.info("🚩 FeatureFlagService initialized")

    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature flag is enabled."""
        try:
            # Try cache first
            if flag_name in self._cache:
                return self._cache[flag_name]

            # Get from Redis
            value = self.redis.hget(self.flags_key, flag_name)
            enabled = bool(int(value)) if value is not None else False

            # Cache result
            self._cache[flag_name] = enabled

            return enabled

        except Exception as e:
            logger.warning(f"Error checking flag {flag_name}: {e}")
            return False  # Fail safe - disable feature

    def set_flag(self, flag_name: str, enabled: bool) -> bool:
        """Set a feature flag."""
        try:
            self.redis.hset(self.flags_key, flag_name, int(enabled))

            # Update cache
            self._cache[flag_name] = enabled

            logger.info(f"🚩 Set {flag_name} = {enabled}")
            return True

        except Exception as e:
            logger.error(f"Error setting flag {flag_name}: {e}")
            return False

    def get_all_flags(self) -> Dict[str, bool]:
        """Get all feature flags."""
        try:
            raw_flags = self.redis.hgetall(self.flags_key)
            return {k: bool(int(v)) for k, v in raw_flags.items()}

        except Exception as e:
            logger.error(f"Error getting all flags: {e}")
            return {}

    def initialize_defaults(self) -> None:
        """Initialize default feature flags if not present."""
        try:
            defaults = FeatureFlags()
            current_flags = self.get_all_flags()

            # Set defaults for missing flags
            for field_name, field_info in defaults.model_fields.items():
                if field_name not in current_flags:
                    default_value = field_info.default
                    self.set_flag(field_name, default_value)

            logger.info("✅ Initialized default feature flags")

        except Exception as e:
            logger.error(f"Error initializing defaults: {e}")

    def clear_cache(self) -> None:
        """Clear the local cache."""
        self._cache.clear()


# Global feature flag service instance
_feature_service: Optional[FeatureFlagService] = None


def get_feature_service() -> FeatureFlagService:
    """Get global feature flag service instance."""
    global _feature_service
    if _feature_service is None:
        _feature_service = FeatureFlagService()
        _feature_service.initialize_defaults()
    return _feature_service


def is_enabled(flag_name: str) -> bool:
    """Convenient function to check if a feature flag is enabled."""
    return get_feature_service().is_enabled(flag_name)
