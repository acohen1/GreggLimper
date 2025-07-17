"""
Per-channel caches backed by a deque (raw Discord message objects)
    - Leftmost element  = oldest message
    - Rightmost element = newest message

Memo table (dict) keyed by Discord message id (Formatted payload)
    - msg_id -> FORMATTED payload (via MessageFormatter)
    - Prevents re-formatting the same Discord message multiple times
"""

from typing import Any, List, Tuple
from discord import Message, TextChannel, User, Member, Role, Attachment, File, Client
from collections import deque

from gregg_limper.config import Config
from gregg_limper.formatter import format_message

import logging
logger = logging.getLogger(__name__)

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
    # Note: these methods are async to allow for message formatting
    # ------------------------------------------------------------------ #

    async def add_message(self, channel_id: str, message_obj: Message) -> None:
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

        logger.info(
            f"Added message {msg_id} to channel {channel_id} cache "
            f"(memo {'created' if needs_memo else 'reused'}) - "
            f"Message: {self._memo[msg_id]}... "
        )

    # ------------------------------------------------------------------ #
    # READ helpers
    # ------------------------------------------------------------------ #

    def get_all_messages(self, channel_id: int) -> List[Message]:
        """Return raw Discord.Message objects [oldest -> newest]."""
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")
        return list(self._caches[channel_id])

    def get_all_formatted(self, channel_id: int) -> List[Tuple[int, str]]:
        """Return formatted (author_id, message) tuples [oldest -> newest] for a channel."""
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")
        return [(msg.author.id, self._memo[msg.id]) for msg in self._caches[channel_id]]

    def get_recent_formatted(self, channel_id: int, n: int) -> List[Tuple[int, str]]:
        """Return the n most-recent formatted (author_id, message) tuples [oldest -> newest]."""
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")
        msgs = self._caches[channel_id]
        slice_start = max(len(msgs) - n, 0)
        return [(m.author.id, self._memo[m.id]) for m in list(msgs)[slice_start:]]

    # ------------------------------------------------------------------ #
    # INITIALIZATION
    # ------------------------------------------------------------------ #

    async def initialize(self, client: Client, channel_ids: list[str]) -> None:
        """Fetch and cache recent messages from Discord channels."""
        if self._caches and set(self._caches.keys()) == set(channel_ids):
            logger.info("Cache already initialized with same channel IDs. Skipping.")
            return

        # Initialize empty data structures
        self._caches = {
            cid: deque(maxlen=Config.CACHE_LENGTH)
            for cid in channel_ids
        }
        self._memo = {}  # Clear to be safe -- this is a fresh start either way

        # Populate caches with recent messages (UP TO Config.CACHE_LENGTH)
        for cid in channel_ids:
            channel = client.get_channel(int(cid))
            if not isinstance(channel, TextChannel):
                logger.warning(f"Channel {cid} is not a text channel or not found. Skipping.")
                continue

            logger.info(f"Fetching history for channel {cid}...")
            async for msg in channel.history(limit=Config.CACHE_LENGTH, oldest_first=True):
                await self.add_message(cid, msg)

        
        logger.info(f"Initialized caches for {len(channel_ids)} channels")

    # ------------------------------------------------------------------ #
    # MAINTENANCE
    # ------------------------------------------------------------------ #

    def clear_cache(self, channel_id: int) -> None:
        """Wipe the raw/formatted cache for the given channel."""
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")

        for msg in self._caches[channel_id]:
            self._memo.pop(msg.id, None)  # safe even if already removed
        self._caches[channel_id].clear()
