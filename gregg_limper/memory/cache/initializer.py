"""Discord history hydration for the cache manager."""

from __future__ import annotations

import asyncio
import logging
from typing import List, TYPE_CHECKING

from discord import Client, Message, TextChannel

from gregg_limper.config import cache

from .formatting import format_missing_messages
from .ingestion import evaluate_ingestion, ingest_message
from .memo_store import MemoStore

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from .manager import GLCache

logger = logging.getLogger(__name__)


class CacheInitializer:
    """Populate the cache from Discord history and persistent memo files."""

    def __init__(self, cache_manager: "GLCache", memo_store: MemoStore) -> None:
        self._cache = cache_manager
        self._memo_store = memo_store

    async def hydrate(self, client: Client, channel_ids: List[int]) -> None:
        """Hydrate ``channel_ids`` from Discord and persisted memos."""

        for channel_id in channel_ids:
            loaded_ids = self._memo_store.load_channel(channel_id)
            channel = client.get_channel(channel_id)
            if not isinstance(channel, TextChannel):
                logger.warning(
                    "Channel %s is not a text channel or not found. Skipping.",
                    channel_id,
                )
                continue

            logger.info("Fetching history for channel %s...", channel_id)
            history = [
                message
                async for message in channel.history(limit=cache.CACHE_LENGTH)
            ]
            messages: List[Message] = list(reversed(history))

            formatted_missing = await format_missing_messages(
                messages, self._memo_store.has, cache.INIT_CONCURRENCY
            )

            ingest_sem = asyncio.Semaphore(cache.INGEST_CONCURRENCY)
            ingest_tasks: list[asyncio.Task[None]] = []

            for message in messages:
                payload = formatted_missing.get(message.id)
                try:
                    await self._cache.add_message(
                        channel_id, message, ingest=False, cache_msg=payload
                    )
                except Exception:
                    logger.exception("Failed to add message %s during init", message.id)
                    continue

                should_ingest, resources = await evaluate_ingestion(
                    message, ingest_requested=True, memo_present=True
                )
                if should_ingest and not resources.sqlite:
                    ingest_tasks.append(
                        asyncio.create_task(
                            self._bounded_ingest(channel_id, message, ingest_sem)
                        )
                    )

            if ingest_tasks:
                await asyncio.gather(*ingest_tasks)

            state = self._cache._get_state(channel_id)
            self._memo_store.reconcile_channel(
                channel_id, state.message_ids(), loaded_ids
            )

        logger.info("Initialized caches for %s channels", len(channel_ids))

    async def _bounded_ingest(
        self, channel_id: int, message: Message, semaphore: asyncio.Semaphore
    ) -> None:
        async with semaphore:
            cache_message = self._memo_store.get(message.id)
            await ingest_message(channel_id, message, cache_message)
