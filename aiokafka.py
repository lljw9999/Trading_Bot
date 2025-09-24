"""aiokafka stub module for offline testing.

Provides minimal implementations of AIOKafkaProducer and AIOKafkaConsumer so that
code importing `aiokafka` can run without the real library or a Kafka broker.
This is **not** suitable for production use – it only fulfils method signatures
used in the unit-tests and simulation modes.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any


class AIOKafkaProducer:  # noqa: N801 – keep original class name
    """Minimal async stub of the real AIOKafkaProducer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        self._started = False

    async def start(self) -> None:  # noqa: D401
        self._started = True

    async def stop(self) -> None:  # noqa: D401
        self._started = False

    async def send_and_wait(
        self, topic: str, value: Any, *args: Any, **kwargs: Any
    ) -> None:  # noqa: D401
        # In stub mode we just swallow the message.
        if not self._started:
            raise RuntimeError("Producer not started – stub")
        await asyncio.sleep(0)  # Yield control to the event loop


class AIOKafkaConsumer:  # noqa: N801 – keep original name
    """Minimal async stub of the real AIOKafkaConsumer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        self._started = False

    async def start(self) -> None:  # noqa: D401
        self._started = True

    async def stop(self) -> None:  # noqa: D401
        self._started = False

    def __aiter__(self):  # noqa: D401
        return self

    async def __anext__(self):  # noqa: D401
        # Immediately stop iteration – no real Kafka messages.
        raise StopAsyncIteration


# Provide errors namespace compatible with aiokafka.errors
errors = SimpleNamespace(KafkaError=Exception)
