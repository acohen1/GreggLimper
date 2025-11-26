from __future__ import annotations

import logging
from typing import List

from .types import SegmentedConversation, TrainingSample

logger = logging.getLogger(__name__)

ALLOWED_KEYS = {
    "role",
    "content",
}


async def build_prompt_shaped_sample(
    *,
    segment: SegmentedConversation,
    relabeled_history: List[dict],
) -> TrainingSample | None:
    """
    Build a training sample containing only user/assistant dialogue turns.
    """

    target = _find_terminal_assistant(relabeled_history)
    if target is None:
        logger.debug(
            "Skipping segment %s because no terminal assistant reply was found",
            segment.message_ids,
        )
        return None

    sanitized_history = _sanitize_history(relabeled_history)
    if (
        not sanitized_history
        or sanitized_history[-1].get("role") != "assistant"
        or not sanitized_history[-1].get("content")
    ):
        logger.debug(
            "Skipping segment %s because last turn is not an assistant reply",
            segment.message_ids,
        )
        return None

    prompt_messages: list[dict] = sanitized_history

    metadata = {
        "channel_id": segment.channel_id,
        "message_ids": segment.message_ids,
        "assistant_user_id": segment.assigned_assistant_id,
        "target_message_id": target.get("message_id"),
    }
    return TrainingSample(
        messages=prompt_messages,
        metadata=metadata,
        parallel_tool_calls=False,
    )


def _sanitize_history(history: List[dict]) -> List[dict]:
    sanitized: list[dict] = []
    for entry in history:
        role = entry.get("role")
        if role not in ("user", "assistant"):
            continue
        if entry.get("tool_calls"):
            continue
        clean = {k: v for k, v in entry.items() if k in ALLOWED_KEYS}
        clean.setdefault("role", role)
        clean.setdefault("content", entry.get("content", ""))
        clean.pop("name", None)
        sanitized.append(clean)
    return sanitized


def _find_terminal_assistant(history: List[dict]) -> dict | None:
    for entry in reversed(history):
        if entry.get("role") != "assistant":
            continue
        if entry.get("tool_calls"):
            continue
        if not entry.get("content"):
            continue
        return entry
    return None


__all__ = ["build_prompt_shaped_sample"]
