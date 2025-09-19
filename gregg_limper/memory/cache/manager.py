"""Cache manager coordinating channel state, memos, and ingestion."""

from __future__ import annotations

import logging
from typing import List

from discord import Client, Message

from gregg_limper.config import cache

from . import formatting, ingestion
from .channel_state import ChannelCacheState
from .initializer import CacheInitializer
from .memo_store import MemoStore
from .serialization import Mode, copy_memo_entry, serialize
from .utils import _frags_preview

logger = logging.getLogger(__name__)


class GLCache:
    """Singleton, channel-aware cache with memoized message formatting."""

    _instance: "GLCache" | None = None

    def __new__(cls) -> "GLCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._states: dict[int, ChannelCacheState] = {}
            cls._instance._memo_store = MemoStore()
        return cls._instance

    # ------------------------------------------------------------------ #
    # WRITE helpers
    # ------------------------------------------------------------------ #

    async def add_message(
        self,
        channel_id: int,
        message_obj: Message,
        ingest: bool = True,
        cache_msg: dict | None = None,
    ) -> None:
        """Append a raw Discord message and memoize its formatted payload."""

        state = self._get_state(channel_id)
        msg_id = message_obj.id

        memo_present = self._memo_store.has(msg_id)
        evicted_id = state.append(message_obj)
        if evicted_id is not None:
            self._memo_store.delete(evicted_id)

        should_ingest, resources = await ingestion.evaluate_ingestion(
            message_obj, ingest_requested=ingest, memo_present=memo_present
        )

        if cache_msg is not None:
            record = cache_msg
            self._memo_store.set(msg_id, cache_msg)
        elif memo_present:
            record = self._memo_store.get(msg_id)
        else:
            record = await formatting.format_for_cache(message_obj)
            self._memo_store.set(msg_id, record)

        if should_ingest and not resources.sqlite:
            await ingestion.ingest_message(channel_id, message_obj, record)

        if not memo_present or evicted_id is not None or cache_msg is not None:
            self._memo_store.save_channel_snapshot(channel_id, state.message_ids())

        if logger.isEnabledFor(logging.INFO):
            preview = _frags_preview(
                record.get("fragments", []), width_each=20, max_total_chars=200
            )
            logger.info(
                "Cached msg %s in channel %s (%s) by %s | Frags: %s",
                msg_id,
                channel_id,
                "new" if not memo_present else "reuse",
                record.get("author"),
                preview,
            )

    # ------------------------------------------------------------------ #
    # READ helpers
    # ------------------------------------------------------------------ #

    def _get_state(self, channel_id: int) -> ChannelCacheState:
        try:
            return self._states[channel_id]
        except KeyError as exc:
            raise KeyError(
                f"Channel ID {channel_id} is not configured for caching."
            ) from exc

    def _get_memo_entry(self, channel_id: int, message_id: int) -> dict:
        state = self._get_state(channel_id)
        if not state.contains(message_id):
            raise KeyError(
                f"Message ID {message_id} is not cached for channel {channel_id}."
            )
        return self._memo_store.get(message_id)

    def list_raw_messages(self, channel_id: int) -> list[Message]:
        return self._get_state(channel_id).iter_messages()

    def list_formatted_messages(
        self, channel_id: int, mode: Mode, n: int | None = None
    ) -> list[dict]:
        state = self._get_state(channel_id)
        messages = state.iter_messages(n)
        return [serialize(self._memo_store.get(msg.id), mode) for msg in messages]

    def list_memo_records(
        self, channel_id: int, n: int | None = None
    ) -> list[dict]:
        state = self._get_state(channel_id)
        messages = state.iter_messages(n)
        return [copy_memo_entry(self._memo_store.get(msg.id)) for msg in messages]

    def get_formatted_message(
        self, channel_id: int, message_id: int, mode: Mode
    ) -> dict:
        cache_msg = self._get_memo_entry(channel_id, message_id)
        return serialize(cache_msg, mode)

    def get_memo_record(self, channel_id: int, message_id: int) -> dict:
        cache_msg = self._get_memo_entry(channel_id, message_id)
        return copy_memo_entry(cache_msg)

    # ------------------------------------------------------------------ #
    # INITIALIZATION
    # ------------------------------------------------------------------ #

    async def initialize(self, client: Client, channel_ids: List[int]) -> None:
        if self._states and set(self._states.keys()) == set(channel_ids):
            logger.info("Cache already initialized with same channel IDs. Skipping.")
            return

        self._states = {
            cid: ChannelCacheState(cid, cache.CACHE_LENGTH) for cid in channel_ids
        }
        self._memo_store.reset()

        initializer = CacheInitializer(self, self._memo_store)
        await initializer.hydrate(client, channel_ids)

    # ------------------------------------------------------------------ #
    # MAINTENANCE
    # ------------------------------------------------------------------ #

    def clear_cache(self, channel_id: int) -> None:
        state = self._get_state(channel_id)
        self._memo_store.remove_many(state.message_ids())
        state.clear()
