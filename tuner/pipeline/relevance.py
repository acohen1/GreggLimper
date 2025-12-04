from __future__ import annotations

import logging
from typing import List

from gregg_limper.clients import oai

logger = logging.getLogger(__name__)


async def is_relevant(messages: List[dict], *, model: str) -> bool:
    """
    Use a lightweight LLM to judge whether the final assistant reply
    is a coherent, relevant response to the immediately preceding user turn.
    """
    if not messages or not model:
        return True

    last_assistant_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant" and messages[idx].get("content"):
            last_assistant_idx = idx
            break
    if last_assistant_idx is None:
        return False

    last_user_idx = None
    for idx in range(last_assistant_idx - 1, -1, -1):
        if messages[idx].get("role") == "user" and messages[idx].get("content"):
            last_user_idx = idx
            break
    if last_user_idx is None:
        return False

    user_content = messages[last_user_idx]["content"]
    assistant_content = messages[last_assistant_idx]["content"]

    prompt = [
        {
            "role": "system",
            "content": (
                "Determine if the assistant reply is a relevant, coherent response "
                "to the preceding user message. Reply only with 'yes' or 'no'."
            ),
        },
        {"role": "user", "content": f"User: {user_content}\nAssistant: {assistant_content}"},
    ]

    try:
        resp = await oai.chat(prompt, model=model)
    except Exception:
        logger.warning("Relevance check failed; defaulting to keep sample.", exc_info=True)
        return True

    return resp.strip().lower().startswith("y")


__all__ = ["is_relevant"]
