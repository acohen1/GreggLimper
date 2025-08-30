"""
Per-channel caches backed by a deque (raw Discord message objects)
    - Leftmost element  = oldest message
    - Rightmost element = newest message

Memo table (dict) keyed by Discord message id (formatted payload)
    - msg_id -> {"author", "fragments"}
    - ``fragments`` holds :class:`Fragment` objects which are serialized on
      demand for downstream consumers.
"""

from typing import Any, Iterable, List, Literal, Tuple
from collections import deque
from discord import (
    Attachment,
    Client,
    File,
    Member,
    Message,
    Role,
    TextChannel,
    User,
)

from gregg_limper.config import cache
from gregg_limper.formatter import format_message
from .. import rag
from ..rag import consent
from . import memo
from .utils import _frags_preview

import logging
import datetime

logger = logging.getLogger(__name__)

# ---------- GLCache ------------------------

Mode = Literal["llm", "full"]

class GLCache:
    """Singleton, channel-aware cache with memoized message formatting."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._caches = {}
            cls._instance._memo = {}
        return cls._instance

    # ------------------------------------------------------------------ #
    # WRITE helpers
    # ------------------------------------------------------------------ #

    async def add_message(self, channel_id: int, message_obj: Message, ingest: bool = True) -> None:
        """Append *raw* Discord message to that channel's deque and memoize its
        formatted representation. Raises KeyError if channel_id is unknown.

        :param channel_id: The ID of the channel the message belongs to.
        :param message_obj: The raw Discord message object to cache.
        :param ingest: Whether to ingest the message into the RAG database.
        :returns: ``None``.
        """
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")

        cache = self._caches[channel_id]

        # Preemptively check if we're about to evict a message from the cache
        evicted = None
        if len(cache) == cache.maxlen:
            evicted = cache[0].id  # Leftmost = oldest = next to be evicted

        cache.append(message_obj)

        # Memo cleanup evicted message
        if evicted is not None:
            self._memo.pop(evicted, None)

        # Enumerate resource availability to cleanly route message ingestion
        msg_id = message_obj.id
        resources = {
            "memo": msg_id in self._memo,
            "sqlite": False,
            "vector": False,
        }

        if ingest:
            try:
                if not await consent.is_opted_in(message_obj.author.id):
                    ingest = False
                else:
                    # NOTE: We call message_exists() as a safeguard to prevent duplicate ingestion. When rehydrating
                    # the cache without a local memo, Discord messages may be re-processed through the LLM pipeline,
                    # which can yield slightly different fragments. Tracking this helps us avoid inconsistent duplicates
                    # and maintain cleaner storage across both SQLite and the vector index. If the I/O overhead of this
                    # check proves too high, it can be reconsidered, since downstream upserts are already designed to
                    # deduplicate. Removing message_exists should only be an issue if the local memo is lost or deleted.
                    resources["sqlite"] = await rag.message_exists(msg_id)
                    # Vector index hydration is coupled with SQLite writes so we mirror its availability here.
                    resources["vector"] = resources["sqlite"]
            except Exception:
                resources["sqlite"] = resources["vector"] = False
                ingest = False

        # Memoize if missing
        if not resources["memo"]:
            self._memo[msg_id] = await format_message(message_obj)

        # Ingest into RAG if backing stores are missing
        if ingest and not resources["sqlite"]:
            try:
                created_at = message_obj.created_at
                # Discord test fixtures may have naive timestamps; normalize to UTC
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=datetime.timezone.utc)
                await rag.ingest_cache_message(
                    server_id=message_obj.guild.id if message_obj.guild else 0,
                    channel_id=channel_id,
                    message_id=msg_id,
                    author_id=message_obj.author.id,
                    ts=created_at.timestamp(),
                    cache_message=self._memo[msg_id],
                )  # crosses into SQLite + vector stores
            except Exception:
                logger.exception("RAG ingestion failed for message %s", msg_id)

        # Persist memo if new or eviction occurred
        if not resources["memo"] or evicted is not None:
            memo_dict = {m.id: self._memo[m.id] for m in cache}
            memo_dict = memo.prune(channel_id, memo_dict)
            memo.save(channel_id, memo_dict)

        # Log message preview (only build if enabled)
        if logger.isEnabledFor(logging.INFO):
            preview = _frags_preview(
                self._memo[msg_id]["fragments"], width_each=20, max_total_chars=200
            )
            logger.info(
                "Cached msg %s in channel %s (%s) by %s | Frags: %s",
                msg_id,
                channel_id,
                "new" if not resources["memo"] else "reuse",
                self._memo[msg_id]["author"],
                preview,
            )

    # ------------------------------------------------------------------ #
    # READ helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _serialize(cache_msg: dict, mode: Mode) -> dict:
        """Serialize a cached message for a specific consumer.

        :param cache_msg: Memoized message record.
        :param mode: ``"llm"`` for prompt form or ``"full"`` for persistence.
        :returns: Dict with ``author`` and ``fragments`` fields.
        """
        frags = cache_msg.get("fragments", [])
        if mode == "llm":
            return {
                "author": cache_msg.get("author"),
                "fragments": [f.to_llm() for f in frags],
            }
        return {
            "author": cache_msg.get("author"),
            "fragments": [f.to_dict() for f in frags],
        }

    def _iter_msgs(self, channel_id: int, n: int | None = None) -> list[Message]:
        """Return up to ``n`` most recent messages from a channel.

        :param channel_id: Discord channel id.
        :param n: Optional limit. ``None`` or ≥ cache length returns all.
        :returns: Messages ordered oldest -> newest.
        """
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")
        dq = self._caches[channel_id]
        if n is None:
            return list(dq)  # oldest -> newest
        if n < 0:
            raise ValueError("n must be >= 0 or None")
        if n >= len(dq):
            return list(dq)
        return list(dq)[len(dq) - n :]  # tail, still oldest -> newest

    def get_raw_messages(self, channel_id: int) -> List[Message]:
        """Return raw :class:`discord.Message` objects.

        :param channel_id: Discord channel id.
        :returns: Messages ordered oldest -> newest.
        """
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")
        return list(self._caches[channel_id])

    # TODO: If we need even more views, we could collapse the get_messages_llm and get_messages_full
    # into a single method with a "mode" parameter (e.g., "llm", "full", "debug").

    def get_messages_llm(self, channel_id: int, n: int | None = None) -> list[dict]:
        """Return recent messages formatted for LLM prompts.

        :param channel_id: Discord channel id.
        :param n: Optional max number of messages to return.
        :returns: List of serialized message dicts (oldest -> newest).
        """
        return [
            self._serialize(self._memo[m.id], "llm")
            for m in self._iter_msgs(channel_id, n)
        ]

    def get_messages_full(self, channel_id: int, n: int | None = None) -> list[dict]:
        """Return recent messages with full fragment details.

        :param channel_id: Discord channel id.
        :param n: Optional max number of messages to return.
        :returns: List of serialized message dicts (oldest -> newest).
        """
        return [
            self._serialize(self._memo[m.id], "full")
            for m in self._iter_msgs(channel_id, n)
        ]

    # ------------------------------------------------------------------ #
    # INITIALIZATION
    # ------------------------------------------------------------------ #

    async def initialize(self, client: Client, channel_ids: list[int]) -> None:
        """Hydrate caches from Discord history.

        :param client: Discord client for API calls.
        :param channel_ids: Channels to populate.
        :returns: ``None``.
        """
        if self._caches and set(self._caches.keys()) == set(channel_ids):
            logger.info("Cache already initialized with same channel IDs. Skipping.")
            return

        # Initialize empty data structures
        self._caches = {cid: deque(maxlen=cache.CACHE_LENGTH) for cid in channel_ids}
        self._memo = {}  # Clear to be safe -- this is a fresh start either way

        # Populate caches with recent messages (UP TO cache.CACHE_LENGTH)
        for cid in channel_ids:
            # First seed memo table from on-disk store before fetching
            loaded = memo.load(cid) if memo.exists(cid) else {}
            self._memo.update(loaded)

            # Fetch the diff
            channel = client.get_channel(cid)
            if not isinstance(channel, TextChannel):
                logger.warning(f"Channel {cid} is not a text channel or not found. Skipping.")
                continue

            logger.info(f"Fetching history for channel {cid}...")
            history = [
                msg
                async for msg in channel.history(limit=cache.CACHE_LENGTH)
            ]
            # Returned newest -> oldest; reverse to store oldest -> newest
            for msg in reversed(history):
                await self.add_message(cid, msg)

            # Prune memo to match deque and persist
            memo_dict = {m.id: self._memo[m.id] for m in self._caches[cid]}
            for mid in set(loaded) - set(memo_dict):
                self._memo.pop(mid, None)
            memo_dict = memo.prune(cid, memo_dict)
            memo.save(cid, memo_dict)

        logger.info(f"Initialized caches for {len(channel_ids)} channels")

    # ------------------------------------------------------------------ #
    # MAINTENANCE
    # ------------------------------------------------------------------ #

    def clear_cache(self, channel_id: int) -> None:
        """Remove all cached messages for a channel.

        :param channel_id: Discord channel id.
        :returns: ``None``.
        """
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")

        for msg in self._caches[channel_id]:
            self._memo.pop(msg.id, None)  # safe even if already removed
        self._caches[channel_id].clear()
