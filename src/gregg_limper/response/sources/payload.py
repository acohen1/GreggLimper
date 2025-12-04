"""High-level orchestration for assembling the prompt payload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from discord import Message

from gregg_limper.config import core
from gregg_limper.tools import get_registered_tool_specs, build_tool_description

from .context import ConversationContext, gather_context
from .context_messages import build_context_messages
from .history import HistoryContext, build_history
from .system_prompt import get_system_prompt

__all__ = ["PromptPayload", "build_prompt_payload"]


@dataclass(slots=True)
class PromptPayload:
    """All artifacts required to call the chat completion API."""

    messages: List[dict[str, str]]
    history: HistoryContext
    context: ConversationContext


async def build_prompt_payload(message: Message) -> PromptPayload:
    """Gather history, context, and format messages for a completion call."""

    history = await build_history(message.channel.id, core.CONTEXT_LENGTH)
    context = await gather_context(
        message, participant_ids=history.participant_ids
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": get_system_prompt()}
    ]

    tool_specs = get_registered_tool_specs()
    if tool_specs:
        messages.append({"role": "assistant", "content": build_tool_description(tool_specs)})
    messages.extend(build_context_messages(context))
    messages.extend(history.messages)

    return PromptPayload(messages=messages, history=history, context=context)
