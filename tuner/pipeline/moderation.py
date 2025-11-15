from __future__ import annotations

import logging
from typing import Sequence

from gregg_limper.clients import oai

logger = logging.getLogger(__name__)


async def moderate_messages(messages: Sequence[dict], *, model: str) -> bool:
    """
    Run OpenAI's moderation API on a fully rendered conversation.

    Returns True when the sample is safe to keep, False when it should be dropped.
    """

    if not model:
        return True

    try:
        # Flatten the conversation into a single text blob for moderation.
        content = "\n\n".join(
            f"{entry.get('role', 'unknown')}: {entry.get('content', '')}"
            for entry in messages
            if entry.get("content")
        )
        if not content:
            return True

        result = await oai.moderate(model=model, input=content)
    except Exception:
        logger.warning("Moderation request failed; defaulting to keep sample.", exc_info=True)
        return True

    flagged = False
    for entry in result.get("results", []):
        if entry.get("flagged"):
            flagged = True
            break
    return not flagged


__all__ = ["moderate_messages"]
