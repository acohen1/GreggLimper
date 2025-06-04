"""
Per-channel caches backed by a deque
    - Leftmost element  = oldest message
    - Rightmost element = newest message

Memo table (dict) keyed by Discord message id
    - msg_id -> formatted payload (via MessageFormatter)
    - Prevents re-formatting the same Discord message multiple times
"""

from typing import Any, List
from discord import Message, TextChannel, User, Member, Role, Attachment, File
from collections import deque

from config import Config
from formatter import format_message

class GLCache:
    """Singleton, channel-aware cache with memoized message formatting."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            # Deque cache for each configured channel
            cls._instance._caches = {
                cid: deque(maxlen=Config.MAX_CACHE_SIZE)
                for cid in Config.CHANNEL_IDS
            }

            # Global memo table: {message_id: formatted_payload}
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

        self._caches[channel_id].append(message_obj)

        # Memoize formatted payload (only once per message_id)
        msg_id = message_obj.id
        if msg_id not in self._memo:
            self._memo[msg_id] = await format_message(message_obj)

    # ------------------------------------------------------------------ #
    # READ helpers
    # ------------------------------------------------------------------ #

    def get_all_messages(self, channel_id: int) -> List[Any]:
        """Return raw messages [oldest → newest]."""
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")
        return list(self._caches[channel_id])

    def get_all_formatted(self, channel_id: int) -> List[Any]:
        """Return formatted payloads [oldest → newest] for a channel."""
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")
        return [self._memo[msg.id] for msg in self._caches[channel_id]]

    def get_recent_formatted(self, channel_id: int, n: int) -> List[Any]:
        """Return the n most-recent formatted payloads [oldest → newest]."""
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")
        msgs = self._caches[channel_id]
        slice_start = max(len(msgs) - n, 0)
        return [self._memo[m.id] for m in list(msgs)[slice_start:]]

    # ------------------------------------------------------------------ #
    # MAINTENANCE
    # ------------------------------------------------------------------ #

    def clear_cache(self, channel_id: int) -> None:
        """Wipe the raw/formatted cache for that channel (keeps memo for other channels)."""
        if channel_id not in self._caches:
            raise KeyError(f"Channel ID {channel_id} is not configured for caching.")

        for msg in self._caches[channel_id]:
            self._memo.pop(msg.id, None)  # safe even if already removed
        self._caches[channel_id].clear()
