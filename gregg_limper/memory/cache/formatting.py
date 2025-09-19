"""Formatting helpers used by the cache."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, Iterable, List

from discord import Message

from gregg_limper.formatter import format_message as _format_message

logger = logging.getLogger(__name__)


async def format_for_cache(message: Message) -> dict:
    """Format ``message`` into the memo payload stored by the cache."""

    return await _format_message(message)


async def format_missing_messages(
    messages: Iterable[Message],
    has_memo: Callable[[int], bool],
    concurrency: int,
) -> Dict[int, dict]:
    """Format messages lacking memos with bounded concurrency."""

    semaphore = asyncio.Semaphore(concurrency)
    tasks: List[asyncio.Task[tuple[int, dict | None]]] = []

    async def _format_one(msg: Message) -> tuple[int, dict | None]:
        async with semaphore:
            try:
                payload = await format_for_cache(msg)
            except Exception:
                logger.exception("Failed to format message %s", msg.id)
                payload = None
        return msg.id, payload

    for message in messages:
        if not has_memo(message.id):
            tasks.append(asyncio.create_task(_format_one(message)))

    results: Dict[int, dict] = {}
    for coro in asyncio.as_completed(tasks):
        mid, payload = await coro
        if payload is not None:
            results[mid] = payload
    return results
