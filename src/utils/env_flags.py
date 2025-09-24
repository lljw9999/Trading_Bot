"""Simple environment flag utilities used for compatibility toggles."""

from __future__ import annotations

import os


def env_flag(name: str, default: bool = False) -> bool:
    """Return True when environment variable is truthy.

    Truthy values: ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    """

    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
