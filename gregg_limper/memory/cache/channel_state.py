"""
Channel-local cache state helpers.

The :class:`ChannelCacheState` dataclass wraps the deque of Discord messages and
an accompanying membership index for a single channel. The cache manager uses
these helpers to append messages with eviction awareness, iterate in chronological
order, and keep the memo store synchronized with the currently buffered IDs.
Callers should only interact with this module via :class:`ChannelCacheState`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from discord import Message


@dataclass
class ChannelCacheState:
    """In-memory state for a cached Discord channel."""

    channel_id: int
    maxlen: int
    _messages: deque[Message] = field(init=False, repr=False)
    _index: set[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._messages = deque(maxlen=self.maxlen)
        self._index = set()

    def append(self, message: Message) -> int | None:
        """Append ``message`` to the channel, returning any evicted id."""

        evicted_id: int | None = None
        # Cache eviction happens on append; capture the outgoing id so we can retire it below.
        if len(self._messages) == self._messages.maxlen:
            evicted = self._messages[0]
            evicted_id = evicted.id
        self._messages.append(message)
        self._index.add(message.id)
        # The deque mutates implicitly, so purge the mirrored membership set when needed.
        if evicted_id is not None:
            self._index.discard(evicted_id)
        return evicted_id

    def clear(self) -> None:
        """Clear cached messages and membership index."""

        self._messages.clear()
        self._index.clear()

    def contains(self, message_id: int) -> bool:
        """Return ``True`` if ``message_id`` is cached for this channel."""

        return message_id in self._index

    def iter_messages(self, limit: int | None = None) -> List[Message]:
        """Return up to ``limit`` messages ordered oldest -> newest."""

        if limit is None:
            return list(self._messages)
        if limit < 0:
            raise ValueError("limit must be >= 0 or None")
        if limit >= len(self._messages):
            return list(self._messages)
        start = len(self._messages) - limit
        return list(self._messages)[start:]

    def message_ids(self) -> List[int]:
        """Return cached message ids ordered oldest -> newest."""

        return [m.id for m in self._messages]

    def sync_from_messages(self, messages: Sequence[Message]) -> None:
        """Replace state with ``messages`` while rebuilding membership index."""

        # Used during hydration: rebuild the deque and index as a single atomic snapshot.
        self._messages = deque(messages, maxlen=self.maxlen)
        self._index = {m.id for m in self._messages}

    def remove_many(self, message_ids: Iterable[int]) -> None:
        """Remove ``message_ids`` from the membership index."""

        for mid in message_ids:
            self._index.discard(mid)

    @property
    def messages(self) -> deque[Message]:
        """Expose underlying deque (read-only operations only)."""

        return self._messages
