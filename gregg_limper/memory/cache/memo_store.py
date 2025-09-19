"""
In-memory memo registry with disk persistence helpers.

``MemoStore`` mirrors the memoized payloads that back each cached message.
It offers dictionary-like accessors plus utilities for loading channel memos
from disk, pruning them to the configured cache length, and saving snapshots
after mutations. The cache manager instantiates a single store and shares it
across channel operations.
"""

from __future__ import annotations

from typing import Iterable

from . import memo


class MemoStore:
    """Centralized memo cache that mirrors disk persistence."""

    def __init__(self) -> None:
        self._records: dict[int, dict] = {}

    def reset(self) -> None:
        """Drop all in-memory memo records."""

        self._records.clear()

    def has(self, message_id: int) -> bool:
        """Return ``True`` if ``message_id`` is memoized."""

        return message_id in self._records

    def get(self, message_id: int) -> dict:
        """Return the memoized payload for ``message_id``."""

        try:
            return self._records[message_id]
        except KeyError as exc:
            raise KeyError(f"Memo record missing for message {message_id}") from exc

    def set(self, message_id: int, payload: dict) -> None:
        """Memoize ``payload`` for ``message_id``."""

        self._records[message_id] = payload

    def delete(self, message_id: int) -> None:
        """Remove ``message_id`` from the memo store if present."""

        self._records.pop(message_id, None)

    def remove_many(self, message_ids: Iterable[int]) -> None:
        """Remove each id in ``message_ids`` from the memo store."""

        for mid in message_ids:
            self._records.pop(mid, None)

    def load_channel(self, channel_id: int) -> set[int]:
        """Load memo records from disk for ``channel_id``."""

        loaded = memo.load(channel_id) if memo.exists(channel_id) else {}
        self._records.update(loaded)
        return set(loaded.keys())

    def save_channel_snapshot(self, channel_id: int, message_ids: Iterable[int]) -> None:
        """Persist memo snapshot for ``channel_id`` covering ``message_ids``."""

        ordered_ids = list(message_ids)
        memo_dict = {mid: self._records[mid] for mid in ordered_ids if mid in self._records}
        memo_dict = memo.prune(channel_id, memo_dict)
        memo.save(channel_id, memo_dict)

    def reconcile_channel(
        self,
        channel_id: int,
        message_ids: Iterable[int],
        previously_loaded: Iterable[int],
    ) -> None:
        """Sync ``channel_id``'s memo file against the current cache."""

        keep_ids = list(message_ids)
        stale_ids = set(previously_loaded) - set(keep_ids)
        for mid in stale_ids:
            self._records.pop(mid, None)
        self.save_channel_snapshot(channel_id, keep_ids)

