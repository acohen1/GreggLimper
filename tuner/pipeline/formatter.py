from __future__ import annotations

import logging
from typing import List

from gregg_limper.response.system_prompt import get_system_prompt
from gregg_limper.tools import build_tool_prompt, get_registered_tool_specs

from .types import SegmentedConversation, TrainingSample

logger = logging.getLogger(__name__)

ALLOWED_KEYS = {"role", "content", "tool_calls", "tool_call_id", "name"}


async def build_prompt_shaped_sample(
    *,
    segment: SegmentedConversation,
    relabeled_history: List[dict],
    synthetic_tool_uses: int,
    context_messages: List[dict] | None = None,
) -> TrainingSample:
    """
    Reconstruct the production prompt stack for a given assistant reply.

    The function:
        - emits the system prompt, tool announcement, and context block
        - appends relabeled history (including synthetic tool turns)
        - returns TrainingSample ready for JSONL serialization
    """

    messages = []
    messages.extend(_build_prompt_header())
    if context_messages:
        messages.extend(context_messages)
    messages.extend(_sanitize_history(relabeled_history))

    metadata = {
        "channel_id": segment.channel_id,
        "message_ids": segment.message_ids,
        "assistant_user_id": segment.assigned_assistant_id,
        "synthetic_tool_calls": synthetic_tool_uses,
    }
    return TrainingSample(messages=messages, metadata=metadata)


def _build_prompt_header() -> List[dict]:
    entries = [{"role": "system", "content": get_system_prompt()}]

    specs = get_registered_tool_specs()
    if specs:
        entries.append({"role": "assistant", "content": build_tool_prompt(specs)})
    return entries


def _sanitize_history(history: List[dict]) -> List[dict]:
    sanitized: list[dict] = []
    for entry in history:
        clean = {k: v for k, v in entry.items() if k in ALLOWED_KEYS}
        # Always retain role/content even if absent in ALLOWED_KEYS filtering.
        clean.setdefault("role", entry.get("role"))
        clean.setdefault("content", entry.get("content", ""))
        sanitized.append(clean)
    return sanitized


__all__ = ["build_prompt_shaped_sample"]
