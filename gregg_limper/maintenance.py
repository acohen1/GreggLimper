"""
Common utilities for background maintenance tasks.

This module exposes helpers for scheduling periodic maintenance jobs and
cancelling them when the application shuts down.  Individual maintenance
modules register their task functions here rather than duplicating scheduling
logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


async def startup(task_fn: Callable[[], Awaitable[None]], interval: float) -> asyncio.Task:
    """
    Schedule ``task_fn`` to run periodically every ``interval`` seconds.

    The task function is invoked in an endless loop until cancelled.  Any
    exceptions raised by the task function are logged but do not stop the
    periodic execution.

    Returns the created :class:`asyncio.Task` handle.
    """

    async def _periodic() -> None:
        await asyncio.sleep(interval)   # delay initial loop
        while True:
            try:
                await task_fn()
            except Exception as exc:  # pragma: no cover - best effort logging
                logger.error("Maintenance cycle failed: %s", exc)
            await asyncio.sleep(interval)

    return asyncio.create_task(_periodic())


async def shutdown(task: asyncio.Task | None) -> None:
    """
    Cancel a maintenance task started with :func:`startup`.

    The function is tolerant of ``None`` and awaits task cancellation to finish
    silently.
    """

    if not task:
        return

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:  # pragma: no cover - normal cancellation
        pass