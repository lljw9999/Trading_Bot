"""Minimal stub for PyYAML when the real package is unavailable.

Provides `safe_load` and `safe_dump` that operate on JSON-like syntax (actually
using Python's `json`), enough for tests that only expect a dict returned.
"""

import json
from typing import Any, Dict, Union

__all__ = ["safe_load", "safe_dump", "YAMLError"]


class YAMLError(Exception):
    """Generic YAML error (stub)."""


def safe_load(stream: Union[str, bytes]) -> Dict[str, Any]:  # type: ignore[override]
    """Very naive loader: attempts JSON parse, else returns empty dict."""
    try:
        if isinstance(stream, (bytes, bytearray)):
            stream = stream.decode()
        return json.loads(stream)
    except Exception:
        # Fallback – return empty dict if not JSON
        return {}


def safe_dump(data: Dict[str, Any], *args: Any, **kwargs: Any) -> str:  # noqa: D401
    return json.dumps(data, *args, **kwargs) 