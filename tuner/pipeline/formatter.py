from __future__ import annotations

import logging
from typing import List, Tuple

from gregg_limper.response.system_prompt import get_system_prompt
from gregg_limper.tools import build_tool_prompt, get_registered_tool_specs

from .types import SegmentedConversation, TrainingSample

logger = logging.getLogger(__name__)

ALLOWED_KEYS = {
    "role",
    "content",
    "tool_calls",
    "tool_call_id",
    "name",
}


async def build_prompt_shaped_sample(
    *,
    segment: SegmentedConversation,
    relabeled_history: List[dict],
    synthetic_tool_uses: int,
    context_messages: List[dict] | None = None,
) -> TrainingSample | None:
    """
    Reconstruct the production prompt stack for a given assistant reply.

    The function:
        - emits the system prompt, tool announcement, and context block
        - appends relabeled history (including synthetic tool turns)
        - isolates the terminal assistant reply as the supervised target
    """

    target = _find_terminal_assistant(relabeled_history)
    if target is None:
        logger.debug(
            "Skipping segment %s because no terminal assistant reply was found",
            segment.message_ids,
        )
        return None

    sanitized_history = _sanitize_history(relabeled_history)

    prompt_messages, tools = _build_prompt_header()
    if context_messages:
        prompt_messages.extend(context_messages)
    prompt_messages.extend(sanitized_history)

    metadata = {
        "channel_id": segment.channel_id,
        "message_ids": segment.message_ids,
        "assistant_user_id": segment.assigned_assistant_id,
        "synthetic_tool_calls": synthetic_tool_uses,
        "target_message_id": target.get("message_id"),
    }
    return TrainingSample(
        messages=prompt_messages,
        metadata=metadata,
        tools=tools,
        parallel_tool_calls=False,
    )


def _build_prompt_header() -> Tuple[List[dict], List[dict]]:
    entries = [{"role": "system", "content": get_system_prompt()}]
    specs = get_registered_tool_specs()
    if specs:
        entries.append({"role": "assistant", "content": build_tool_prompt(specs)})
    tools = [spec.to_openai() for spec in specs]
    return entries, tools


def _sanitize_history(history: List[dict]) -> List[dict]:
    sanitized: list[dict] = []
    for entry in history:
        clean = {k: v for k, v in entry.items() if k in ALLOWED_KEYS}
        clean.setdefault("role", entry.get("role"))
        clean.setdefault("content", entry.get("content", ""))
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
