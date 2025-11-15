from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Sequence

from discord import Message


@dataclass(slots=True)
class RawConversation:
    """Raw Discord history scoped to a channel after filtering."""

    channel_id: int
    guild_id: int | None
    messages: List[Message]


@dataclass(slots=True)
class SegmentCandidate:
    """
    Preliminary slice of a conversation before LLM refinement.

    Attributes:
        channel_id: Channel containing the messages.
        message_ids: Ordered Discord message IDs describing the span.
    """

    channel_id: int
    message_ids: List[int]


@dataclass(slots=True)
class SegmentedConversation:
    """
    LLM-approved conversation slice aligned to alternating user/assistant turns.

    Attributes:
        channel_id: Channel containing the conversation.
        message_ids: Ordered Discord message IDs.
        assigned_assistant_id: Discord user id chosen to play Gregg.
    """

    channel_id: int
    message_ids: List[int]
    assigned_assistant_id: int


@dataclass(slots=True)
class TrainingSample:
    """
    Final, prompt-shaped dataset record ready for JSONL export.

    Attributes:
        messages: ChatML-style conversation matching OpenAI finetune schema.
        metadata: Trace data (segment id, participants, synthetic flags).
        tools: Tool definitions announced to the model.
        parallel_tool_calls: Whether parallel tool calling is permitted.
    """

    messages: List[dict]
    metadata: dict[str, object] = field(default_factory=dict)
    tools: List[dict] = field(default_factory=list)
    parallel_tool_calls: bool = False


__all__ = [
    "RawConversation",
    "SegmentCandidate",
    "SegmentedConversation",
    "TrainingSample",
]
