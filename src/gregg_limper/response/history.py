"""Helpers for assembling cached conversation history."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

from gregg_limper.clients import disc
from gregg_limper.memory.cache import GLCache

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HistoryContext:
    """Cached history messages and referenced participant identifiers."""

    messages: List[dict[str, str]]
    participant_ids: Set[int]


async def build_history(channel_id: int, limit: int) -> HistoryContext:
    """Return cached history formatted for the chat completion API."""

    if limit < 1:
        logger.error("Message limit < 1; cannot include latest message.")
        raise ValueError("Message limit must be >= 1")

    cache = GLCache()
    formatted_messages = cache.list_formatted_messages(
        channel_id, mode="llm", n=limit
    )

    if not formatted_messages:
        return HistoryContext(messages=[], participant_ids=set())

    history_messages = _convert_history(formatted_messages)
    raw_messages = cache.list_raw_messages(channel_id, n=limit)
    participants = _extract_participants(raw_messages)

    return HistoryContext(messages=history_messages, participant_ids=participants)


def _convert_history(formatted_messages: Sequence[dict]) -> List[dict[str, str]]:
    """Translate cached structured messages to readable chat messages."""

    converted: list[dict[str, str]] = []
    for formatted in formatted_messages:
        author_name = formatted.get("author") or "Unknown speaker"
        role = _resolve_role(author_name)
        fragments = formatted.get("fragments", [])
        content = _render_fragments(fragments)
        if content:
            body = f"{author_name} said:\n{content}"
        else:
            body = f"{author_name} shared non-text content."
        converted.append({"role": role, "content": body})
    return converted


def _resolve_role(author_name: str) -> str:
    bot_user = getattr(getattr(disc, "bot", None), "user", None)
    bot_display_name = getattr(bot_user, "display_name", None)
    if bot_display_name and author_name == bot_display_name:
        return "assistant"
    return "user"


def _render_fragments(fragments: Iterable[dict]) -> str:
    lines: list[str] = []
    for fragment in fragments:
        fragment_type = fragment.get("type") or "text"
        description = fragment.get("description") or fragment.get("content")
        caption = fragment.get("caption")
        title = fragment.get("title")
        url = fragment.get("url") or fragment.get("href")

        if fragment_type == "text" and description:
            lines.append(description.strip())
            continue

        pieces: list[str] = [fragment_type.upper()]
        if title:
            pieces.append(title.strip())
        if description:
            pieces.append(description.strip())
        if caption and caption not in pieces:
            pieces.append(caption.strip())
        if url:
            pieces.append(f"({url})")

        rendered = " - ".join(piece for piece in pieces if piece)
        if rendered:
            lines.append(rendered)

    return "\n".join(lines).strip()


def _extract_participants(raw_messages: Iterable) -> Set[int]:
    """Collect participant IDs from raw Discord messages."""

    participants: set[int] = set()
    bot_user = getattr(getattr(disc, "bot", None), "user", None)
    bot_user_id = getattr(bot_user, "id", None)

    for raw in raw_messages:
        author_id = getattr(raw.author, "id", None)
        if author_id is not None and author_id != bot_user_id:
            participants.add(author_id)

        for mentioned in getattr(raw, "mentions", []) or []:
            mentioned_id = getattr(mentioned, "id", None)
            if mentioned_id is not None and mentioned_id != bot_user_id:
                participants.add(mentioned_id)

    return participants


__all__ = ["HistoryContext", "build_history"]
