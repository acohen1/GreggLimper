"""
Per-channel caches backed by a deque (raw Discord message objects)
    - Leftmost element  = oldest message
    - Rightmost element = newest message

Memo table (dict) keyed by Discord message id (formatted payload)
    - msg_id -> {"author", "fragments"}
    - ``fragments`` holds :class:`Fragment` objects which are serialized on
      demand for downstream consumers.
"""

from typing import Any, List, Tuple, Literal, Iterable
from discord import Message, TextChannel, User, Member, Role, Attachment, File, Client
from collections import deque
import textwrap

from gregg_limper.config import Config
from gregg_limper.formatter import format_message

import logging

logger = logging.getLogger(__name__)

# ---------- local logging helpers ----------

def _frag_summary(frag, *, width: int = 20) -> str:
    """Return a compact one-line summary for logs: e.g., text:'Hello…'."""
    d = frag.to_llm()  # lean dict
    t = d.get("type", "?")
    val = d.get("description") or d.get("caption") or d.get("title") or ""
    if not val:
        return t
    return f"{t}:'{textwrap.shorten(str(val), width=width, placeholder='…')}'"


def _frags_preview(frags, *, width_each: int = 20, max_total_chars: int = 200) -> str:
    """Join multiple summaries and cap total length to avoid noisy logs."""
    parts = []
    total = 0
    for f in frags:
        s = _frag_summary(f, width=width_each)
        if total + len(s) + (2 if parts else 0) > max_total_chars:
            parts.append("…")
            break
        parts.append(s)
        total += len(s) + (2 if parts else 0)
    return ", ".join(parts)

# ---------- GLCache ------------------------

Mode = Literal["llm", "full"]

class GLCache:
    """
    Singleton, channel-aware cache with memoized message formatting.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._caches = {}
            cls._instance._memo = {}
        return cls._instance

    # ------------------------------------------------------------------ #
    # WRITE helpers
    # Note: these methods are async to allow for message formatting
    # ------------------------------------------------------------------ #

    async def add_message(self, channel_id: int, message_obj: Message) -> None:
        """
        Append *raw* Discord message to that channel's deque and memoize its
        formatted representation. Raises KeyError if channel_id is unknown.
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

        # Memoize if needed
        msg_id = message_obj.id
        needs_memo = msg_id not in self._memo
        if needs_memo:
            self._memo[msg_id] = await format_message(message_obj)

        # Log message preview (only build if enabled)
        if logger.isEnabledFor(logging.INFO):
            preview = _frags_preview(
                self._memo[msg_id]["fragments"], width_each=20, max_total_chars=200
            )
            logger.info(
                "Cached msg %s in channel %s (%s) by %s | Frags: %s",
                msg_id,
                channel_id,
                "new" if needs_memo else "reuse",
                self._memo[msg_id]["author"],
                preview,
            )

    # ------------------------------------------------------------------ #
    # READ helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _serialize(cache_msg: dict, mode: Mode) -> dict:
        """
        Internal helper to serialize a cached message.

        - mode="llm": minimal form for prompt construction (omits IDs/URLs).
        - mode="full": complete form for persistence or downstream consumers.

        Returns a dict with "author" and "fragments".
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
        Return up to ``n`` most recent messages from a channel, oldest -> newest.

        If ``n`` is None or >= cache length, all messages are returned.
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
        Return raw Discord.Message objects [oldest -> newest].
        """
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")
        return list(self._caches[channel_id])

    # TODO: If we need even more views, we could collapse the get_messages_llm and get_messages_full
    # into a single method with a "mode" parameter (e.g., "llm", "full", "debug").

    def get_messages_llm(self, channel_id: int, n: int | None = None) -> list[dict]:
        """
        Return the ``n`` most-recent messages formatted for LLM consumption.

        Messages are returned oldest -> newest. Each fragment dict is produced
        on demand via ``Fragment.to_llm`` and thus omits fields like ``id`` or
        ``url`` that are unnecessary for prompting.
        """
        return [self._serialize(self._memo[m.id], "llm") for m in self._iter_msgs(channel_id, n)]

    def get_messages_full(self, channel_id: int, n: int | None = None) -> list[dict]:
        """
        Return the ``n`` most-recent messages with all fragment details.

        Messages are returned oldest -> newest. Each fragment dict is produced
        on demand via ``Fragment.to_dict`` and therefore retains all fields
        for downstream consumers requiring full fidelity.
        """
        return [self._serialize(self._memo[m.id], "full") for m in self._iter_msgs(channel_id, n)]

    # ------------------------------------------------------------------ #
    # INITIALIZATION
    # ------------------------------------------------------------------ #

    async def initialize(self, client: Client, channel_ids: list[int]) -> None:
        """
        Fetch and cache recent messages from Discord channels.
        """
        if self._caches and set(self._caches.keys()) == set(channel_ids):
            logger.info("Cache already initialized with same channel IDs. Skipping.")
            return

        # Initialize empty data structures
        self._caches = {cid: deque(maxlen=Config.CACHE_LENGTH) for cid in channel_ids}
        self._memo = {}  # Clear to be safe -- this is a fresh start either way

        # Populate caches with recent messages (UP TO Config.CACHE_LENGTH)
        for cid in channel_ids:
            channel = client.get_channel(cid)
            if not isinstance(channel, TextChannel):
                logger.warning(
                    f"Channel {cid} is not a text channel or not found. Skipping."
                )
                continue

            logger.info(f"Fetching history for channel {cid}...")
            async for msg in channel.history(
                limit=Config.CACHE_LENGTH, oldest_first=True
            ):
                await self.add_message(cid, msg)

        logger.info(f"Initialized caches for {len(channel_ids)} channels")

    # ------------------------------------------------------------------ #
    # MAINTENANCE
    # ------------------------------------------------------------------ #

    def clear_cache(self, channel_id: int) -> None:
        """
        Remove all cached messages and their memoized fragments for a channel.
        """
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")

        for msg in self._caches[channel_id]:
            self._memo.pop(msg.id, None)  # safe even if already removed
        self._caches[channel_id].clear()
