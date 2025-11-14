from __future__ import annotations

import logging
from typing import Dict, List

from discord import Message

from gregg_limper.memory.cache.formatting import format_for_cache
from gregg_limper.memory.cache.serialization import serialize as serialize_cache

from .types import SegmentedConversation

logger = logging.getLogger(__name__)


async def relabel_segment(
    segment: SegmentedConversation,
    *,
    message_lookup: Dict[int, Message],
) -> List[dict]:
    """
    Translate a refined segment into alternating user/assistant role dictionaries.

    Output mirrors the history fragments produced by gregg_limper.response.history:
        - assistant turns omit the "<name> said:" prefix
        - user turns include "<display name> said:\n<content>"
    """

    relabeled: list[dict] = []
    for message_id in segment.message_ids:
        message = message_lookup.get(message_id)
        if message is None:
            logger.debug("Segment message %s missing from lookup", message_id)
            continue

        cache_entry = await format_for_cache(message)
        serialized = serialize_cache(cache_entry, mode="llm")
        fragments = serialized.get("fragments", [])
        author_name = serialized.get("author") or getattr(
            message.author, "display_name", None
        )
        author_name = author_name or getattr(message.author, "name", "Unknown speaker")

        content = _render_fragments(fragments)
        role = (
            "assistant"
            if message.author.id == segment.assigned_assistant_id
            else "user"
        )

        if role == "assistant":
            body = content or "Assistant shared non-text content."
        else:
            body = (
                f"{author_name} said:\n{content}"
                if content
                else f"{author_name} shared non-text content."
            )

        relabeled.append(
            {
                "role": role,
                "content": body,
                "author_id": message.author.id,
                "message_id": message_id,
            }
        )

    return relabeled


def _render_fragments(fragments: List[dict]) -> str:
    """Minimal clone of gregg_limper.response.history._render_fragments."""

    lines: list[str] = []
    for fragment in fragments:
        fragment_type = fragment.get("type") or "text"
        description = fragment.get("description")
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


__all__ = ["relabel_segment"]
