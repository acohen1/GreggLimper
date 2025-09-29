from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import List, Set

from gregg_limper.memory.cache import GLCache
from gregg_limper.clients import disc
from gregg_limper.memory.rag import consent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HistoryContext:
    """Container for cached chat history and the participants it references."""

    messages: List[dict]
    participant_ids: Set[int]


async def build_history(channel_id: int, limit: int) -> HistoryContext:
    """Return cached history and participant IDs for prompting.

    Notes:
        - ``messages`` matches the format expected by chat completion APIs.
        - ``participant_ids`` includes unique user identifiers observed in the returned history.
    """
    if limit < 1:
        logger.error("Message limit < 1; cannot include latest message.")
        raise ValueError("Message limit must be >= 1")

    cache = GLCache()
    formatted_messages = cache.list_formatted_messages(
        channel_id, mode="llm", n=limit
    )

    if not formatted_messages:
        return HistoryContext(messages=[], participant_ids=set())

    context: List[dict] = []
    for formatted in formatted_messages:
        role = (
            "assistant"
            if formatted.get("author") == disc.client.user.display_name
            else "user"
        )
        content_str = json.dumps(formatted, ensure_ascii=False, separators=(",", ":"))
        context.append({"role": role, "content": content_str})

    raw_messages = cache.list_raw_messages(channel_id, n=limit)
    participants: set[int] = set()

    for raw in raw_messages:
        author_id = getattr(raw.author, "id", None)
        if author_id is not None and author_id != disc.client.user.id:
            participants.add(author_id)

        for mentioned in getattr(raw, "mentions", []) or []:
            mentioned_id = getattr(mentioned, "id", None)
            if mentioned_id is not None and mentioned_id != disc.client.user.id:
                participants.add(mentioned_id)

    if not participants:
        return HistoryContext(messages=context, participant_ids=set())

    participant_list = sorted(participants)
    consent_checks = await asyncio.gather(
        *(consent.is_opted_in(uid) for uid in participant_list)
    )
    consented_ids = {
        uid for uid, opted_in in zip(participant_list, consent_checks) if opted_in
    }

    return HistoryContext(messages=context, participant_ids=consented_ids)


__all__ = ["HistoryContext", "build_history"]
