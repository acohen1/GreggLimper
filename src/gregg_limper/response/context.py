"""Utilities for collecting dynamic context for LLM prompts."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from discord import Message

from gregg_limper.clients import disc
from gregg_limper.config import core, prompt
from gregg_limper.memory.cache import GLCache
from gregg_limper.memory.rag import consent
from gregg_limper.memory.rag import (
    channel_summary as fetch_channel_summary,
    user_profile as fetch_user_profile,
    vector_search,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConversationContext:
    """Dynamic data retrieved to aid a model response."""

    channel_summary: str | None
    user_profiles: list[dict[str, Any]]
    semantic_memory: list[dict[str, Any]]


async def gather_context(
    message: Message, *, participant_ids: Iterable[int]
) -> ConversationContext:
    """Collect all dynamic context required for a reply."""

    channel_summary_task = asyncio.create_task(
        fetch_channel_summary(message.channel.id)
    )
    user_profiles_task = asyncio.create_task(
        _gather_user_profiles(participant_ids)
    )
    semantic_memory_task = asyncio.create_task(_gather_semantic_memory(message))

    channel_summary, user_profiles, semantic_memory = await asyncio.gather(
        channel_summary_task,
        user_profiles_task,
        semantic_memory_task,
    )

    return ConversationContext(
        channel_summary=channel_summary,
        user_profiles=user_profiles,
        semantic_memory=semantic_memory,
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


async def _gather_semantic_memory(message: Message) -> list[dict[str, Any]]:
    """Retrieve semantic memories relevant to the provided message."""

    fragments = _collect_fragment_text(message)
    if not fragments:
        return []

    per_fragment_candidates = await asyncio.gather(
        *(_vector_search_across_channels(message, fragment) for fragment in fragments)
    )

    semantic_candidates = _merge_semantic_candidates(
        per_fragment_candidates, prompt.VECTOR_SEARCH_K
    )
    if not semantic_candidates:
        return []

    author_lookup = await _map_author_display_names(semantic_candidates)
    formatted: list[dict[str, Any]] = []
    for candidate in semantic_candidates:
        author_id = candidate.get("author_id")
        formatted.append(
            {
                "author": author_lookup.get(author_id, _fallback_author_name(author_id)),
                "title": candidate.get("title"),
                "content": candidate.get("content"),
                "type": candidate.get("type"),
                "url": candidate.get("url"),
            }
        )

    return formatted


def _collect_fragment_text(message: Message) -> list[str]:
    """Return textual fragments cached for the given message."""

    cache = GLCache()
    memo_record = cache.get_memo_record(message.channel.id, message.id)
    fragments = memo_record.get("fragments", [])

    contents: list[str] = []
    for fragment in fragments:
        content = fragment.content_text()
        if content and content.strip():
            contents.append(content.strip())

    logger.info("Fragment contents for vector search: %s", contents)
    return contents


async def _vector_search_across_channels(
    message: Message, content: str
) -> list[dict[str, Any]]:
    """Run a vector search for the content across configured channels."""

    fragment_candidates: list[dict[str, Any]] = []
    for channel_id in core.CHANNEL_IDS:
        results = await vector_search(
            message.guild.id,
            channel_id,
            content,
            k=prompt.VECTOR_SEARCH_K + 1,
        )
        filtered = [
            candidate
            for candidate in results
            if candidate.get("message_id") != message.id
        ]
        fragment_candidates.extend(filtered)

    return fragment_candidates


def _merge_semantic_candidates(
    per_fragment_candidates: Sequence[Sequence[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    """Merge semantic search results ensuring diversity and uniqueness."""

    if not per_fragment_candidates or limit <= 0:
        return []

    merged: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    indices = [0] * len(per_fragment_candidates)

    while len(merged) < limit:
        progress_made = False
        for frag_idx, candidates in enumerate(per_fragment_candidates):
            if len(merged) >= limit:
                break

            while indices[frag_idx] < len(candidates):
                candidate = candidates[indices[frag_idx]]
                indices[frag_idx] += 1

                candidate_id = candidate.get("id")
                if candidate_id is not None:
                    if candidate_id in seen_ids:
                        continue
                    seen_ids.add(candidate_id)

                merged.append(candidate)
                progress_made = True
                break

        if not progress_made:
            break

    return merged


async def _map_author_display_names(
    candidates: Iterable[dict[str, Any]]
) -> dict[int, str]:
    """Fetch Discord display names for candidate authors."""

    author_ids = {
        candidate.get("author_id")
        for candidate in candidates
        if candidate.get("author_id") is not None
    }

    author_names: dict[int, str] = {}
    for author_id in author_ids:
        try:
            user = await disc.bot.fetch_user(author_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to fetch author %s display name: %s", author_id, exc
            )
            continue
        author_names[author_id] = user.display_name

    return author_names


def _fallback_author_name(author_id: Any) -> str:
    if author_id is None:
        return "Unknown"
    return str(author_id)


__all__ = ["ConversationContext", "gather_context"]
