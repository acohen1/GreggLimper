"""
Core helpers shared across cache ingestion paths.

This module defines the memo access surface used by :func:`process_message_for_rag`.
``_MemoLike`` is a small :class:`typing.Protocol` describing the trio of
``has``/``get``/``set`` methods required to participate in memo reuse so both
``dict``-like mappings and dedicated stores (for example the on-disk
``MemoStore``) satisfy the same static contract.

The ``_memo_has``/``_memo_get``/``_memo_set`` helpers normalize interactions
with those memo containers. Each helper handles ``None`` up front, delegates to
the object's bespoke method when present, and otherwise falls back to standard
mapping operations. This keeps the ingestion pipeline unaware of the concrete
memo implementation while still enabling consistent fragment reuse.
"""

from __future__ import annotations

from typing import Mapping, MutableMapping, Protocol, Tuple

from discord import Message
from discord.abc import User

from . import formatting, ingestion


class _MemoLike(Protocol):
    def has(self, message_id: int) -> bool: ...

    def get(self, message_id: int) -> dict: ...

    def set(self, message_id: int, payload: dict) -> None: ...


def _memo_has(
    memo: Mapping[int, dict] | MutableMapping[int, dict] | _MemoLike | None,
    message_id: int,
) -> bool:
    if memo is None:
        return False
    if hasattr(memo, "has"):
        return getattr(memo, "has")(message_id)
    return message_id in memo  # type: ignore[arg-type]


def _memo_get(
    memo: Mapping[int, dict] | MutableMapping[int, dict] | _MemoLike | None,
    message_id: int,
) -> dict:
    if memo is None:
        raise KeyError(message_id)
    if hasattr(memo, "get"):
        getter = getattr(memo, "get")
        return getter(message_id)
    return memo[message_id]  # type: ignore[index]


def _memo_set(
    memo: Mapping[int, dict] | MutableMapping[int, dict] | _MemoLike | None,
    message_id: int,
    payload: dict,
) -> None:
    if memo is None:
        return
    if hasattr(memo, "set"):
        setter = getattr(memo, "set")
        setter(message_id, payload)
        return
    memo[message_id] = payload  # type: ignore[index]


async def process_message_for_rag(
    message_obj: Message,
    channel_id: int,
    *,
    ingest: bool = True,
    cache_msg: dict | None = None,
    memo: Mapping[int, dict] | MutableMapping[int, dict] | _MemoLike | None = None,
    bot_user: User | None = None,
) -> Tuple[dict, bool]:
    """Format ``message_obj`` and optionally ingest it into downstream RAG stores."""

    msg_id = message_obj.id
    memo_present = _memo_has(memo, msg_id)

    should_ingest, resources = await ingestion.evaluate_ingestion(
        message_obj,
        ingest_requested=ingest,
        memo_present=memo_present,
        bot_user=bot_user,
    )

    if cache_msg is not None:
        record = cache_msg
        _memo_set(memo, msg_id, cache_msg)
    elif memo_present:
        record = _memo_get(memo, msg_id)
    else:
        record = await formatting.format_for_cache(message_obj)
        _memo_set(memo, msg_id, record)

    did_ingest = False
    if should_ingest and not resources.sqlite:
        await ingestion.ingest_message(channel_id, message_obj, record)
        did_ingest = True

    return record, did_ingest


__all__ = ["process_message_for_rag"]
