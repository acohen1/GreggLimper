from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Tuple

from gregg_limper.memory.cache import GLCache
from gregg_limper.clients import disc

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HistoryContext:
    """Container for cached chat history and the participants it references."""

    messages: List[dict]
    participant_ids: Tuple[int, ...]


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
        return HistoryContext(messages=[], participant_ids=())

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

    return HistoryContext(
        messages=context, participant_ids=tuple(sorted(participants))
    )


__all__ = ["HistoryContext", "build_history"]
