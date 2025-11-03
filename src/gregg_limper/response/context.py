"""Utilities for collecting dynamic context for LLM prompts."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Iterable

from discord import Message

from gregg_limper.memory.rag import consent
from gregg_limper.memory.rag import (
    channel_summary as fetch_channel_summary,
    user_profile as fetch_user_profile,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConversationContext:
    """Dynamic data retrieved to aid a model response."""

    channel_summary: str | None
    user_profiles: list[dict[str, Any]]


async def gather_context(
    message: Message, *, participant_ids: Iterable[int]
) -> ConversationContext:
    """
    Collect all dynamic context required for a reply.
    
    NOTE: Tool-based retrieval now surfaces richer semantic context on demand.
    This function remains for surfacing channel summaries and user profiles.
    """

    channel_summary_task = asyncio.create_task(
        fetch_channel_summary(message.channel.id)
    )
    user_profiles_task = asyncio.create_task(
        _gather_user_profiles(participant_ids)
    )
    channel_summary, user_profiles = await asyncio.gather(
        channel_summary_task,
        user_profiles_task,
    )

    return ConversationContext(
        channel_summary=channel_summary,
        user_profiles=user_profiles,
    )


async def _gather_user_profiles(
    participant_ids: Iterable[int],
) -> list[dict[str, Any]]:
    """Fetch user profiles for opted-in participants."""

    participants = [pid for pid in participant_ids if pid is not None]
    if not participants:
        return []

    consent_checks = await asyncio.gather(
        *(consent.is_opted_in(pid) for pid in participants)
    )

    consenting = [
        pid for pid, opted_in in zip(participants, consent_checks) if opted_in
    ]
    if not consenting:
        return []

    profiles = await asyncio.gather(*(fetch_user_profile(pid) for pid in consenting))
    return [profile for profile in profiles if profile]


__all__ = ["ConversationContext", "gather_context"]
