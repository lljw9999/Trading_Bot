#!/usr/bin/env python3
"""
Pydantic schemas for Param Server v1

Defines the structure and validation for model router rules YAML configuration.
Supports glob patterns for asset_class and range expressions for horizon_ms.
"""

import re
import fnmatch
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class MatchCriteria(BaseModel):
    """Matching criteria for routing rules."""

    asset_class: str = Field(..., description="Asset class pattern (supports glob)")
    horizon_ms: str = Field(..., description="Horizon pattern with operators")

    @field_validator("asset_class")
    @classmethod
    def validate_asset_class(cls, v):
        if not isinstance(v, str):
            raise ValueError("Asset class must be a string")
        return v

    @field_validator("horizon_ms")
    @classmethod
    def validate_horizon_ms(cls, v):
        if not isinstance(v, str):
            raise ValueError("Horizon must be a string")
        return v

    def matches_asset_class(self, asset_class: str) -> bool:
        """Check if asset_class matches the pattern."""
        if self.asset_class == "*":
            return True
        return fnmatch.fnmatch(asset_class.lower(), self.asset_class.lower())

    def matches_horizon(self, horizon_ms: int) -> bool:
        """Check if horizon_ms matches the pattern."""
        if self.horizon_ms == "*":
            return True

        # Parse comparison expressions
        horizon_str = self.horizon_ms.strip()

        # Handle range expressions with &
        if "&" in horizon_str:
            parts = [part.strip() for part in horizon_str.split("&")]
            return all(
                self._evaluate_single_condition(part, horizon_ms) for part in parts
            )

        return self._evaluate_single_condition(horizon_str, horizon_ms)

    def _evaluate_single_condition(self, condition: str, value: int) -> bool:
        """Evaluate a single condition like '<60000' or '>=14400000'."""
        condition = condition.strip()

        # Extract operator and threshold
        if condition.startswith(">="):
            threshold = int(condition[2:])
            return value >= threshold
        elif condition.startswith("<="):
            threshold = int(condition[2:])
            return value <= threshold
        elif condition.startswith(">"):
            threshold = int(condition[1:])
            return value > threshold
        elif condition.startswith("<"):
            threshold = int(condition[1:])
            return value < threshold
        elif condition.startswith("="):
            threshold = int(condition[1:])
            return value == threshold
        else:
            # Try direct comparison (equal)
            try:
                threshold = int(condition)
                return value == threshold
            except ValueError:
                return False


class ModelRoute(BaseModel):
    """Individual routing rule with match criteria and target model."""

    match: MatchCriteria = Field(..., description="Matching criteria")
    model: str = Field(..., description="Target model ID")
    priority: int = Field(100, description="Rule priority (lower = higher priority)")
    description: Optional[str] = Field(None, description="Human-readable description")

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v):
        if v < 0:
            raise ValueError("Priority must be non-negative")
        return v

    def matches(self, asset_class: str, horizon_ms: int) -> bool:
        """Check if this rule matches the given criteria."""
        return self.match.matches_asset_class(
            asset_class
        ) and self.match.matches_horizon(horizon_ms)


class RouterConfig(BaseModel):
    """Router configuration settings."""

    default_model: str = Field(
        "tlob_tiny", description="Default model when no rules match"
    )
    cache_ttl_seconds: int = Field(300, description="Cache TTL in seconds")
    redis_url: str = Field(
        "redis://localhost:6379/0", description="Redis connection URL"
    )
    performance_logging: bool = Field(True, description="Enable performance logging")
    max_latency_us: int = Field(50, description="Target max latency in microseconds")

    @field_validator("cache_ttl_seconds")
    @classmethod
    def validate_cache_ttl(cls, v):
        if v <= 0:
            raise ValueError("Cache TTL must be positive")
        return v

    @field_validator("max_latency_us")
    @classmethod
    def validate_max_latency(cls, v):
        if v <= 0:
            raise ValueError("Max latency must be positive")
        return v


class ModelThreshold(BaseModel):
    """Performance thresholds for a model."""

    max_latency_ms: float = Field(..., description="Maximum allowed latency in ms")
    min_accuracy: float = Field(..., description="Minimum required accuracy")

    @field_validator("max_latency_ms")
    @classmethod
    def validate_latency(cls, v):
        if v <= 0:
            raise ValueError("Max latency must be positive")
        return v

    @field_validator("min_accuracy")
    @classmethod
    def validate_accuracy(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Accuracy must be between 0 and 1")
        return v


class FeatureFlags(BaseModel):
    """Feature flags for A/B testing and safe rollouts."""

    EXEC_RL_LIVE: bool = Field(
        False, description="Enable live RL execution (vs shadow)"
    )
    BANDIT_WEIGHTS: bool = Field(
        False, description="Use contextual bandit ensemble weights"
    )
    LLM_SENTIMENT: bool = Field(False, description="Include LLM sentiment in state")
    TAIL_RISK_HEDGE: bool = Field(False, description="Enable tail risk hedging overlay")
    DL_FINE_TUNE: bool = Field(True, description="Enable nightly DL model fine-tuning")

    def to_redis_mapping(self) -> dict:
        """Convert to Redis-compatible mapping (int values)."""
        return {k: int(v) for k, v in self.model_dump().items()}

    @classmethod
    def from_redis_mapping(cls, redis_data: dict) -> "FeatureFlags":
        """Create from Redis hash data."""
        flags = {}
        for field in cls.model_fields:
            value = redis_data.get(field, "0")
            flags[field] = bool(int(value))
        return cls(**flags)


class ReloadConfig(BaseModel):
    """Hot-reload configuration."""

    enabled: bool = Field(True, description="Enable hot reloading")
    signal: str = Field("SIGHUP", description="Unix signal for reload trigger")
    validation: bool = Field(True, description="Validate config before applying")
    backup_on_reload: bool = Field(True, description="Backup old config on reload")


class ModelRouterRules(BaseModel):
    """Complete model router rules configuration."""

    rules: List[ModelRoute] = Field(..., description="List of routing rules")
    config: RouterConfig = Field(
        default_factory=RouterConfig, description="Router configuration"
    )
    model_thresholds: Dict[str, ModelThreshold] = Field(
        default_factory=dict, description="Performance thresholds per model"
    )
    feature_flags: FeatureFlags = Field(
        default_factory=FeatureFlags, description="A/B testing feature flags"
    )
    reload: ReloadConfig = Field(
        default_factory=ReloadConfig, description="Hot-reload settings"
    )

    @model_validator(mode="after")
    def validate_rules(self):
        """Validate that rules are properly structured."""
        rules = self.rules
        if not rules:
            raise ValueError("At least one routing rule must be defined")

        # Check for priority conflicts
        priorities = [rule.priority for rule in rules]
        if len(priorities) != len(set(priorities)):
            raise ValueError("Duplicate priorities found in rules")

        # Verify default model is referenced
        default_model_id = self.config.default_model
        model_ids = {rule.model for rule in rules}

        if default_model_id not in model_ids:
            # Add a catch-all rule for the default model
            fallback_rule = ModelRoute(
                match=MatchCriteria(asset_class="*", horizon_ms="*"),
                model=default_model_id,
                priority=9999,
                description="Auto-generated fallback rule",
            )
            self.rules = rules + [fallback_rule]

        return self

    def get_sorted_rules(self) -> List[ModelRoute]:
        """Get rules sorted by priority (ascending)."""
        return sorted(self.rules, key=lambda r: r.priority)

    def find_matching_rule(
        self, asset_class: str, horizon_ms: int
    ) -> Optional[ModelRoute]:
        """Find the highest priority rule that matches the criteria."""
        for rule in self.get_sorted_rules():
            if rule.matches(asset_class, horizon_ms):
                return rule
        return None


class ParamServerConfig(BaseModel):
    """Configuration for ParamServer instance."""

    config_path: str = Field(..., description="Path to YAML configuration file")
    redis_url: Optional[str] = Field(None, description="Redis URL for pub/sub")
    watch_enabled: bool = Field(True, description="Enable file watching")
    redis_key_prefix: str = Field("param", description="Redis key prefix")

    @field_validator("config_path")
    @classmethod
    def validate_config_path(cls, v):
        if not v:
            raise ValueError("Config path cannot be empty")
        return v


# Factory function for creating default configuration
def create_default_router_rules() -> ModelRouterRules:
    """Create default router rules configuration."""
    rules = [
        # Crypto high-frequency
        ModelRoute(
            match=MatchCriteria(asset_class="crypto", horizon_ms="<60000"),
            model="tlob_tiny",
            priority=10,
            description="TLOB-Tiny for crypto microstructure analysis",
        ),
        # Crypto medium-frequency
        ModelRoute(
            match=MatchCriteria(asset_class="crypto", horizon_ms=">=60000 & <7200000"),
            model="patchtst_small",
            priority=20,
            description="PatchTST for crypto medium-term patterns",
        ),
        # Crypto long-term
        ModelRoute(
            match=MatchCriteria(asset_class="crypto", horizon_ms=">=7200000"),
            model="mamba_ts_small",
            priority=30,
            description="MambaTS for crypto long-term regime analysis",
        ),
        # US Stocks intraday
        ModelRoute(
            match=MatchCriteria(asset_class="us_stocks", horizon_ms="<14400000"),
            model="timesnet_base",
            priority=40,
            description="TimesNet for US equity intraday patterns",
        ),
        # US Stocks overnight
        ModelRoute(
            match=MatchCriteria(asset_class="us_stocks", horizon_ms=">=14400000"),
            model="mamba_ts_small",
            priority=50,
            description="MambaTS for US equity swing/overnight",
        ),
        # A-shares intraday
        ModelRoute(
            match=MatchCriteria(asset_class="a_shares", horizon_ms="<14400000"),
            model="timesnet_base",
            priority=60,
            description="TimesNet for A-share intraday",
        ),
        # A-shares overnight
        ModelRoute(
            match=MatchCriteria(asset_class="a_shares", horizon_ms=">=14400000"),
            model="chronos_bolt_base",
            priority=70,
            description="Chronos Bolt for A-share overnight with macro factors",
        ),
    ]

    config = RouterConfig(
        default_model="tlob_tiny",
        cache_ttl_seconds=300,
        redis_url="redis://localhost:6379/0",
        performance_logging=True,
        max_latency_us=50,
    )

    model_thresholds = {
        "tlob_tiny": ModelThreshold(max_latency_ms=3.0, min_accuracy=0.52),
        "patchtst_small": ModelThreshold(max_latency_ms=10.0, min_accuracy=0.54),
        "timesnet_base": ModelThreshold(max_latency_ms=15.0, min_accuracy=0.53),
        "mamba_ts_small": ModelThreshold(max_latency_ms=20.0, min_accuracy=0.55),
        "chronos_bolt_base": ModelThreshold(max_latency_ms=25.0, min_accuracy=0.56),
    }

    reload_config = ReloadConfig(
        enabled=True, signal="SIGHUP", validation=True, backup_on_reload=True
    )

    feature_flags = FeatureFlags()

    return ModelRouterRules(
        rules=rules,
        config=config,
        model_thresholds=model_thresholds,
        feature_flags=feature_flags,
        reload=reload_config,
    )
