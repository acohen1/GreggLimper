from __future__ import annotations
import asyncio, json
from typing import List, Dict, Any
from gregg_limper.clients import disc
from discord import Message
from gregg_limper.config import prompt, core
from gregg_limper.memory.cache import GLCache
from gregg_limper.memory.rag import (
    channel_summary as get_channel_summary,
    user_profile,
    vector_search,
)

import logging

logger = logging.getLogger(__name__)

# TODO: Better system prompt preface for the channel summary, user profiles, and semantic memory

MESSAGE_SCHEMA = """
## Message Schema
Cached conversation history is provided in JSON format. Each message has the form:

```json
{
  "author": "display_name",
  "fragments": [
    {"type": "text", "description": "Hello world!"},
    {"type": "image", "title": "sunset.jpg", "caption": "a red-orange sky"},
    {"type": "youtube", "title": "<title>", "description": "<video summary>",
     "thumbnail_url": "...", "thumbnail_caption": "..."},
    {"type": "link", "title": "<url>", "description": "<summary>"},
    {"type": "gif", "title": "<cleaned-title>", "caption": "<frame description>"}
  ]
}
```

Do not respond in this format unless explicitly instructed. This schema is only for interpreting cached messages.
"""
async def build_sys_prompt(message: Message) -> str:
    """
    Human-readable system prompt (Markdown-ish).
    """
    # RAG fetches (all async)

    # Channel summary
    chan_summary = await get_channel_summary(message.channel.id)

    # Profiles of users mentioned in the message (excluding the bot itself)
    user_mentions = [u.id for u in message.mentions if u.id != disc.client.user.id]
    profiles: List[Dict[str, Any]] = (
        await asyncio.gather(*(user_profile(u) for u in user_mentions))
        if user_mentions else []
    )

    # ------- Semantic memory (vector search) -------

    # Grab media fragments from the inbound message
    cache = GLCache()
    memo_record = cache.get_memo_record(message.channel.id, message.id)

    # From each media fragment, get its content_text (used for vector search query)
    frags_content = [
        content
        for fragment in memo_record.get("fragments", [])
        for content in [fragment.content_text()]
        if content.strip()
    ]

    # Log the fragment contents for debugging
    logger.info(f"Fragment contents for vector search: {frags_content}")

    # Perform vector search for each fragment content in all allowed channels
    per_fragment_candidates: list[list[dict[str, Any]]] = []
    for content in frags_content:
        fragment_candidates: list[dict[str, Any]] = []
        for c_id in core.CHANNEL_IDS:
            results = await vector_search(
                message.guild.id,
                c_id,
                content,
                k=prompt.VECTOR_SEARCH_K + 1,  # +1 to account for possible self-match
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

    # Round-robin merge candidates from each fragment until we reach k unique ones
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

    # We need only a subset of fields for prompt construction
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
            logger.warning("Failed to fetch author %s display name: %s", author_id, exc)
            continue
        author_names[author_id] = user.display_name

    semantic_candidates = [
        {
            "author": author_names.get(
                candidate.get("author_id"),
                "Unknown" if candidate.get("author_id") is None else str(candidate.get("author_id")),
            ),
            "title": candidate.get("title"),
            "content": candidate.get("content"),
            "type": candidate.get("type"),
            "url": candidate.get("url"),
        }
        for candidate in semantic_candidates
    ]

    # Build the system prompt
    parts: list[str] = []

    if chan_summary:
        parts.append(f"## Channel Summary\n{chan_summary.strip()}")

    if profiles:
        # Pretty-print profile dicts as JSON blocks
        profiles_json = "\n\n".join(
            json.dumps(p, ensure_ascii=False, indent=2) for p in profiles if p
        ).strip()
        if profiles_json:
            parts.append(f"## User Profiles\n```json\n{profiles_json}\n```")

    parts.append(MESSAGE_SCHEMA)

    if semantic_candidates:
        parts.append(
            "## Semantic Memory (top-k, JSON)\n"
            f"```json\n{json.dumps(semantic_candidates, ensure_ascii=False, indent=2)}\n```"
        )

    sys_prompt = "\n\n".join(parts).strip()
    return sys_prompt or "## Channel Summary\n(none)"


