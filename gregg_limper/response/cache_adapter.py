from __future__ import annotations
import json
import logging
from typing import List

from gregg_limper.memory.cache import GLCache
from gregg_limper.clients import disc

logger = logging.getLogger(__name__)


async def build_history(channel_id: int, limit: int) -> List[dict]:
    """
    Return the last `limit` cached messages as [{role, content}] in oldest -> newest order.
    Serializes the cached message dictionaries.
    """
    if limit < 1:
        logger.error("Message limit < 1; cannot include latest message.")
        raise ValueError("Message limit must be >= 1")

    cache = GLCache()
    # Fetch the last `limit` messages in oldest -> newest order
    formatted = cache.list_formatted_messages(channel_id, "llm", n=limit)

    if not formatted:
        return []

    context: List[dict] = []
    for msg in formatted:
        role = (
            "assistant"
            if msg.get("author") == disc.client.user.display_name
            else "user"
        )
        content_str = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
        context.append({"role": role, "content": content_str})

    return context
