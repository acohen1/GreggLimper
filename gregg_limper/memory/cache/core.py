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
import asyncio

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

    async def add_message(
        self,
        channel_id: int,
        message_obj: Message,
        ingest: bool = True,
        cache_msg: dict | None = None,
    ) -> None:
        """
        Append *raw* Discord message to that channel's deque and memoize its
        formatted representation. Raises KeyError if channel_id is unknown.

        :param channel_id: The ID of the channel the message belongs to.
        :param message_obj: The raw Discord message object to cache.
        :param ingest: Whether to ingest the message into the RAG database.
        :param cache_msg: Optional pre-formatted representation of ``message_obj``.
            If provided, ``format_message`` will not be called.
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
            if cache_msg is None:
                self._memo[msg_id] = await format_message(message_obj)
            else:
                self._memo[msg_id] = cache_msg

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
        """
        Serialize a cached message for a specific consumer.

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
        """
        Return up to ``n`` most recent messages from a channel.

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
        """
        Return raw :class:`discord.Message` objects.

        :param channel_id: Discord channel id.
        :returns: Messages ordered oldest -> newest.
        """
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")
        return list(self._caches[channel_id])

    def get_serialized_messages(self, channel_id: int, mode: Mode, n: int | None = None) -> list[dict]:
        """
        Retrieve up to ``n`` most recent cached messages serialized for the requested mode.

        Notes
        - ``mode`` controls the shape: ``"llm"`` yields a prompt-ready subset; ``"full"`` returns all memo fields.
        - Order is chronological (oldest -> newest) among the selected messages.

        :param channel_id: Discord channel id.
        :param mode: Serialization mode to use (e.g., ``"llm"`` or ``"full"``).
        :param n: Optional limit of the most recent messages to include. ``None`` returns all cached messages.
        :returns: List of serialized message dicts for up to ``n`` messages ordered oldest -> newest.
        """
        return [
            self._serialize(self._memo[m.id], mode)
            for m in self._iter_msgs(channel_id, n)
        ]

    def get_unserialized_messages(self, channel_id: int, n: int | None = None) -> list[dict]:
        """
        Retrieve up to ``n`` most recent cached messages in raw memo form.

        Notes
        - Provides a thin bridge to the memo table for consumers that need raw fragments.
        - Each entry is a shallow copy of the memo record containing ``author`` and ``fragments``.
        - ``fragments`` holds :class:`gregg_limper.formatter.model.Fragment` instances; treat them as read-only unless deep-copied.
        - Order is chronological (oldest -> newest) among the selected messages.

        :param channel_id: Discord channel id.
        :param n: Optional limit of the most recent messages to include. ``None`` returns all cached messages.
        :returns: List of memo dicts for up to ``n`` messages ordered oldest -> newest.
        """
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")

        out: list[dict] = []
        for m in self._iter_msgs(channel_id, n):
            rec = self._memo.get(m.id, {"author": None, "fragments": []})
            out.append({
                "author": rec.get("author"),
                "fragments": list(rec.get("fragments", [])),  # shallow copy list
            })
        return out

    def get_serialized_message_by_id(self, channel_id: int, message_id: int, mode: Mode) -> dict:
        """
        Retrieve a cached message by id serialized for the requested mode.

        Notes
        - Raises ``KeyError`` if the channel is unknown or the message is not cached.
        - Each result is produced via ``_serialize`` for the requested ``mode``.

        :param channel_id: Discord channel id.
        :param message_id: Discord message id.
        :param mode: Serialization mode to use (e.g., ``"llm"`` or ``"full"``).
        :returns: Serialized representation of the cached message.
        """
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")

        if not any(m.id == message_id for m in self._caches[channel_id]):
            raise KeyError(
                f"Message ID {message_id} is not cached for channel {channel_id}."
            )

        cache_msg = self._memo.get(message_id)
        if cache_msg is None:
            raise KeyError(
                f"Message ID {message_id} does not have a memoized record."
            )

        return self._serialize(cache_msg, mode)

    def get_unserialized_message_by_id(self, channel_id: int, message_id: int) -> dict:
        """
        Retrieve a cached message by id in raw memo form.

        Notes
        - Raises ``KeyError`` if the channel is unknown or the message is not cached.
        - Provides a shallow copy of the memo entry with fragments preserved by reference.
        - ``fragments`` holds :class:`gregg_limper.formatter.model.Fragment` instances; treat them as read-only unless deep-copied.

        :param channel_id: Discord channel id.
        :param message_id: Discord message id.
        :returns: Memo dict for the cached message.
        """
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")

        if not any(m.id == message_id for m in self._caches[channel_id]):
            raise KeyError(
                f"Message ID {message_id} is not cached for channel {channel_id}."
            )

        rec = self._memo.get(message_id)
        if rec is None:
            raise KeyError(
                f"Message ID {message_id} does not have a memoized record."
            )

        return {
            "author": rec.get("author"),
            "fragments": list(rec.get("fragments", [])),
        }

    # ------------------------------------------------------------------ #
    # INITIALIZATION
    # ------------------------------------------------------------------ #

    async def initialize(self, client: Client, channel_ids: list[int]) -> None:
        """
        Hydrate caches from Discord history.

        Messages are added to the cache with ``ingest=False`` and a separate
        ingestion step runs afterward. This keeps cache insertion ordered while
        allowing RAG upserts to execute with bounded concurrency. Using
        ``add_message(..., ingest=True)`` directly here would couple these two
        operations and serialize initialization.

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
            # Seed the memo table from on-disk store. This ensures we don’t waste
            # time re-formatting messages we already have cached payloads for.
            loaded = memo.load(cid) if memo.exists(cid) else {}
            self._memo.update(loaded)
            
            # Grab the history from Discord
            channel = client.get_channel(cid)
            if not isinstance(channel, TextChannel):
                logger.warning(f"Channel {cid} is not a text channel or not found. Skipping.")
                continue
            logger.info(f"Fetching history for channel {cid}...")
            history = [
                msg
                async for msg in channel.history(limit=cache.CACHE_LENGTH)
            ]
            messages = list(reversed(history))  # oldest -> newest

            # Begin formatting parallelization with bounded concurrency
            sem = asyncio.Semaphore(cache.INIT_CONCURRENCY)
            tasks: list[asyncio.Task[tuple[int, Message, dict | None]]] = []

            async def _format_bounded(idx: int, msg: Message) -> tuple[int, Message, dict | None]:
                """Format one message while respecting concurrency limits."""
                async with sem:
                    try:
                        cache_msg = await format_message(msg)
                    except Exception:
                        logger.exception("Failed to format message %s during init", msg.id)
                        cache_msg = None
                return idx, msg, cache_msg

            # Ingest tasks collected after cache population (bounded later)
            # We avoid calling ``add_message(..., ingest=True)`` so that cache
            # insertion remains ordered while RAG upserts happen in parallel.
            ingest_sem = asyncio.Semaphore(cache.INGEST_CONCURRENCY)
            ingest_tasks: list[asyncio.Future] = []

            async def _ingest_bounded(msg: Message, cache_msg: dict) -> None:
                """Ingest one message while respecting concurrency limits."""
                async with ingest_sem:
                    try:
                        # Skip if author is not opted in
                        if not await consent.is_opted_in(msg.author.id):
                            return
                        # Skip if this message has already been ingested
                        if await rag.message_exists(msg.id):
                            return
                        created_at = msg.created_at
                        if created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=datetime.timezone.utc)
                        await rag.ingest_cache_message(
                            server_id=msg.guild.id if msg.guild else 0,
                            channel_id=cid,
                            message_id=msg.id,
                            author_id=msg.author.id,
                            ts=created_at.timestamp(),
                            cache_message=cache_msg,
                        )
                    except Exception:
                        logger.exception(
                            "RAG ingestion failed for message %s during init", msg.id
                        )

            # Build a results list aligned to message order. Each result slot corresponds to a chronological index.
            # - If the message is already memoized, mark it as ready immediately.  
            # - Otherwise, schedule a format task to fill its slot later.
            results: list[tuple[Message, dict | None] | None] = [None] * len(messages)
            for idx, msg in enumerate(messages):
                if msg.id in self._memo:
                    results[idx] = (msg, None)
                else:
                    tasks.append(asyncio.create_task(_format_bounded(idx, msg)))

            # Flush contiguous ready items at the start (likely memo hits).
            # This ensures we start populating the cache immediately.
            next_idx = 0
            while next_idx < len(results) and results[next_idx] is not None:
                m, _ = results[next_idx]
                try:
                    await self.add_message(cid, m, ingest=False)
                except Exception:
                    logger.exception(
                        "Failed to add message %s during init", m.id
                    )
                else:
                    if not await rag.message_exists(m.id):
                        ingest_tasks.append(_ingest_bounded(m, self._memo[m.id]))
                next_idx += 1

            # As formatting tasks finish (possibly out of order), fill their slots.
            # Every time a gap at `next_idx` becomes ready, flush forward until the next gap.
            for coro in asyncio.as_completed(tasks):
                idx, msg, cache_msg = await coro
                results[idx] = (msg, cache_msg)
                while next_idx < len(results) and results[next_idx] is not None:
                    m, cm = results[next_idx]
                    if cm is None:
                        try:
                            await self.add_message(cid, m, ingest=False)
                        except Exception:
                            logger.exception(
                                "Failed to add message %s during init", m.id
                            )
                        else:
                            if not await rag.message_exists(m.id):
                                ingest_tasks.append(
                                    _ingest_bounded(m, self._memo[m.id])
                                )
                    else:
                        await self.add_message(
                            cid, m, ingest=False, cache_msg=cm
                        )
                        if not await rag.message_exists(m.id):
                            ingest_tasks.append(_ingest_bounded(m, self._memo[m.id]))
                    next_idx += 1

            # Final flush: if anything remained unprocessed (e.g. the tail),
            # walk forward and push those into the cache in order.
            while next_idx < len(results):
                m, cm = results[next_idx]
                if cm is None:
                    try:
                        await self.add_message(cid, m, ingest=False)
                    except Exception:
                        logger.exception(
                            "Failed to add message %s during init", m.id
                        )
                    else:
                        if not await rag.message_exists(m.id):
                            ingest_tasks.append(
                                _ingest_bounded(m, self._memo[m.id])
                            )
                else:
                    await self.add_message(cid, m, ingest=False, cache_msg=cm)
                    if not await rag.message_exists(m.id):
                        ingest_tasks.append(_ingest_bounded(m, self._memo[m.id]))
                next_idx += 1

            # Perform all RAG ingests with bounded concurrency
            if ingest_tasks:
                await asyncio.gather(*ingest_tasks)

            # Reconcile on-disk memo with final deque contents
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
        """
        Remove all cached messages for a channel.

        :param channel_id: Discord channel id.
        :returns: ``None``.
        """
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")

        for msg in self._caches[channel_id]:
            self._memo.pop(msg.id, None)  # safe even if already removed
        self._caches[channel_id].clear()



