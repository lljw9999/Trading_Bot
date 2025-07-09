"""websockets stub module for offline/offline testing environment.

Provides minimal interfaces (connect, exceptions.ConnectionClosed) needed by our
connectors so that the code can import `websockets` even if the real package is
not available. The stub does **not** establish real network connections.
"""

from __future__ import annotations

import asyncio
import types
from typing import Any, AsyncIterator


class _DummyWebSocketClient:
    """A dummy async iterator that yields no messages and supports basic methods."""

    async def send(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        await asyncio.sleep(0)

    async def recv(self) -> str:  # noqa: D401
        await asyncio.sleep(0)
        raise StopAsyncIteration

    # Allow `async for message in websocket` syntax
    def __aiter__(self) -> "_DummyWebSocketClient":  # noqa: D401
        return self

    async def __anext__(self) -> str:  # noqa: D401
        raise StopAsyncIteration

    async def close(self) -> None:  # noqa: D401
        await asyncio.sleep(0)


async def connect(*args: Any, **kwargs: Any) -> _DummyWebSocketClient:  # noqa: D401
    """Return a dummy websocket client instance."""
    await asyncio.sleep(0)
    return _DummyWebSocketClient()


# Minimal exceptions namespace expected by code
exceptions = types.SimpleNamespace(ConnectionClosed=Exception) 