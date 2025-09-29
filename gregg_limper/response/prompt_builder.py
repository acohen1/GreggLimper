from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Set, List

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


async def build_sys_prompt(
    message: Message,
    history: HistoryContext | None = None,
) -> str:
    """Build the Markdown system prompt for the assistant."""

    # ------- Channel summary -------
    chan_summary = await get_channel_summary(message.channel.id)

    # ------- User profiles (conversation participants + mentioned members) -------
    uids: List[int] = history.participant_ids if history else []

    # Only include consenting users
    consent_checks = await asyncio.gather(
        *(consent.is_opted_in(uid) for uid in uids)
    )
    filtered_ids = [
        uid for uid, opted_in in zip(uids, consent_checks) if opted_in
    ]

    profiles: List[Dict[str, Any]] = (
        await asyncio.gather(*(user_profile(u) for u in filtered_ids))
        if filtered_ids
        else []
    )

    # ------- Semantic memory (vector search) -------
    cache = GLCache()
    memo_record = cache.get_memo_record(message.channel.id, message.id)

    # Grab all fragment contents for vector search
    frags_content = [
        content
        for fragment in memo_record.get("fragments", [])
        for content in [fragment.content_text()]
        if content.strip()
    ]

    logger.info("Fragment contents for vector search: %s", frags_content)

    # Perform vector search for each fragment across all configured channels
    # and aggregate results, ensuring no duplicates and excluding the current message
    per_fragment_candidates: list[list[dict[str, Any]]] = []
    for content in frags_content:
        fragment_candidates: list[dict[str, Any]] = []
        for c_id in core.CHANNEL_IDS:
            results = await vector_search(
                message.guild.id,
                c_id,
                content,
                k=prompt.VECTOR_SEARCH_K + 1,   # +1 to account for exclusion of the current message
            )
            filtered = [
                candidate
                for candidate in results
                if candidate.get("message_id") != message.id
            ]
            fragment_candidates.extend(filtered)
        per_fragment_candidates.append(fragment_candidates)

    semantic_candidates: list[dict[str, Any]] = []
    seen_candidate_ids: set[int] = set()
    indices = [0] * len(per_fragment_candidates)

    # Round-robin selection from each fragment's candidates to ensure diversity
    while len(semantic_candidates) < prompt.VECTOR_SEARCH_K and per_fragment_candidates:
        progress_made = False
        for frag_idx, candidates in enumerate(per_fragment_candidates):
            if len(semantic_candidates) >= prompt.VECTOR_SEARCH_K:
                break

            while indices[frag_idx] < len(candidates):
                candidate = candidates[indices[frag_idx]]
                indices[frag_idx] += 1
                candidate_id = candidate.get("id")

                if candidate_id is not None and candidate_id in seen_candidate_ids:
                    continue

                if candidate_id is not None:
                    seen_candidate_ids.add(candidate_id)

                semantic_candidates.append(candidate)
                progress_made = True
                break

        if not progress_made:
            break

    author_ids = {
        candidate.get("author_id")
        for candidate in semantic_candidates
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

    semantic_candidates = [
        {
            "author": author_names.get(
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

    return render_sys_prompt(
        channel_summary=chan_summary,
        user_profiles=profiles,
        semantic_memory=semantic_candidates,
    )


__all__ = ["build_sys_prompt"]
