from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from discord import Message

from gregg_limper.clients import disc
from gregg_limper.config import core, prompt
from gregg_limper.memory.cache import GLCache
from gregg_limper.memory.rag import consent
from gregg_limper.memory.rag import (
    channel_summary as get_channel_summary,
    user_profile,
    vector_search,
)

from .history_builder import HistoryContext
from .prompt_template import render_sys_prompt


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PromptContext:
    """Aggregated context that feeds into the system prompt template."""

    channel_summary: Any
    user_profiles: List[Dict[str, Any]]
    semantic_memory: List[Dict[str, Any]]


async def build_sys_prompt(
    message: Message,
    history: HistoryContext | None = None,
) -> str:
    """Build the Markdown system prompt for the assistant."""

    context = await _build_prompt_context(message, history)

    return render_sys_prompt(
        channel_summary=context.channel_summary,
        user_profiles=context.user_profiles,
        semantic_memory=context.semantic_memory,
    )


async def _build_prompt_context(
    message: Message, history: HistoryContext | None
) -> PromptContext:
    """Gather all data required to render the system prompt."""

    channel_summary_task = asyncio.create_task(
        get_channel_summary(message.channel.id)
    )
    user_profiles_task = asyncio.create_task(
        _gather_user_profiles(history.participant_ids if history else set())
    )
    semantic_memory_task = asyncio.create_task(_gather_semantic_memory(message))

    channel_summary, user_profiles, semantic_memory = await asyncio.gather(
        channel_summary_task,
        user_profiles_task,
        semantic_memory_task,
    )

    return PromptContext(
        channel_summary=channel_summary,
        user_profiles=user_profiles,
        semantic_memory=semantic_memory,
    )


async def _gather_user_profiles(participant_ids: Iterable[int]) -> List[Dict[str, Any]]:
    """Fetch consented user profiles for the participants in the conversation."""

    participant_list = list(participant_ids)
    if not participant_list:
        return []

    consent_checks = await asyncio.gather(
        *(consent.is_opted_in(uid) for uid in participant_list)
    )

    consenting_ids = [
        uid for uid, opted_in in zip(participant_list, consent_checks) if opted_in
    ]
    if not consenting_ids:
        return []

    profiles = await asyncio.gather(*(user_profile(uid) for uid in consenting_ids))
    return list(profiles)


async def _gather_semantic_memory(message: Message) -> List[Dict[str, Any]]:
    """Retrieve semantically relevant memories for the provided message."""

    fragments = _collect_fragment_text(message)
    if not fragments:
        return []

    per_fragment_candidates = await asyncio.gather(
        *(_vector_search_across_channels(message, text) for text in fragments)
    )

    semantic_candidates = _merge_semantic_candidates(
        per_fragment_candidates, prompt.VECTOR_SEARCH_K
    )
    if not semantic_candidates:
        return []

    author_lookup = await _map_author_display_names(semantic_candidates)
    return [
        {
            "author": author_lookup.get(
                candidate.get("author_id"),
                "Unknown"
                if candidate.get("author_id") is None
                else str(candidate.get("author_id")),
            ),
            "title": candidate.get("title"),
            "content": candidate.get("content"),
            "type": candidate.get("type"),
            "url": candidate.get("url"),
        }
        for candidate in semantic_candidates
    ]


def _collect_fragment_text(message: Message) -> List[str]:
    """Extract searchable fragment text for a message from the memo cache."""

    cache = GLCache()
    memo_record = cache.get_memo_record(message.channel.id, message.id)
    fragments = memo_record.get("fragments", [])

    contents: List[str] = []
    for fragment in fragments:
        content = fragment.content_text()
        if content and content.strip():
            contents.append(content)

    logger.info("Fragment contents for vector search: %s", contents)
    return contents


async def _vector_search_across_channels(
    message: Message, content: str
) -> List[Dict[str, Any]]:
    """Run a vector search for a fragment across all configured channels."""

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
    per_fragment_candidates: Sequence[Sequence[Dict[str, Any]]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Combine semantic candidates ensuring diversity and uniqueness."""

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
    candidates: Iterable[Dict[str, Any]]
) -> Dict[int, str]:
    """Fetch Discord display names for candidate authors."""

    author_ids = {
        candidate.get("author_id")
        for candidate in candidates
        if candidate.get("author_id") is not None
    }

    author_names: dict[int, str] = {}
    for author_id in author_ids:
        try:
            user = await disc.client.fetch_user(author_id)
        except Exception as exc:
            logger.warning(
                "Failed to fetch author %s display name: %s", author_id, exc
            )
            continue
        author_names[author_id] = user.display_name

    return author_names


__all__ = ["build_sys_prompt"]
