from __future__ import annotations
import asyncio, json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from discord import Message

from gregg_limper.clients import disc
from gregg_limper.config import core, prompt
from gregg_limper.memory.cache import GLCache
from gregg_limper.memory.rag import (
    channel_summary as get_channel_summary,
    user_profile,
    vector_search,
)

from .sections import MESSAGE_SCHEMA_BODY, PERSONALITY_BODY, ROLE_PREFACE

import logging

logger = logging.getLogger(__name__)


CHANNEL_SUMMARY_PREFACE = (
    "Use this summary to recall persistent context; defer to the live conversation when details conflict."
)
USER_PROFILES_PREFACE = (
    "Consult these profiles to tailor the response for mentioned members; ignore unrelated users."
)
SEMANTIC_MEMORY_PREFACE = (
    "Treat these retrieved snippets as supporting evidence; verify relevance before citing them."
)
PERSONALITY_PREFACE = (
    "Use this guidance to shape tone while staying aligned with higher-priority directives."
)

@dataclass
class PromptSection:
    """Container for a formatted prompt section."""

    key: str
    title: Optional[str]
    body: str
    language: Optional[str] = None

    def render(self) -> Optional[str]:
        body = (self.body or "").strip()
        if not body:
            return None

        if self.language:
            content = f"```{self.language}\n{body}\n```"
        else:
            content = body

        if self.title:
            return f"## {self.title}\n{content}"

        return content


def _format_profiles(profiles: List[Dict[str, Any]]) -> str:
    if not profiles:
        return "_No user profiles retrieved for the mentioned members._"

    serialized = [
        json.dumps(profile, ensure_ascii=False, indent=2)
        for profile in profiles
        if profile
    ]
    body = "\n\n".join(serialized).strip()
    if not body:
        return "_No user profiles retrieved for the mentioned members._"

    return f"```json\n{body}\n```"


def _format_semantic_memory(candidates: List[Dict[str, Any]]) -> str:
    if not candidates:
        return "_No semantic memory matches retrieved._"

    payload = json.dumps(candidates, ensure_ascii=False, indent=2)
    return f"```json\n{payload}\n```"


def _build_sections(
    chan_summary: Optional[str],
    profiles: List[Dict[str, Any]],
    semantic_candidates: List[Dict[str, Any]],
) -> List[PromptSection]:
    """Assemble the ordered prompt blocks with their contextual prefaces."""

    # The personality body is optional; keep the section empty when the template
    # has not been customized yet so the rest of the prompt is unaffected.
    personality_body = PERSONALITY_BODY.strip()
    if personality_body:
        personality_body = f"{PERSONALITY_PREFACE}\n\n{personality_body}"

    # Each contextual block begins with a short instruction describing how the
    # assistant should interpret the subsequent data payload.
    channel_body = (
        f"{CHANNEL_SUMMARY_PREFACE}\n\n{chan_summary.strip()}"
        if chan_summary and chan_summary.strip()
        else f"{CHANNEL_SUMMARY_PREFACE}\n\n_No channel summary available._"
    )

    profiles_body = f"{USER_PROFILES_PREFACE}\n\n{_format_profiles(profiles)}"
    semantic_body = f"{SEMANTIC_MEMORY_PREFACE}\n\n{_format_semantic_memory(semantic_candidates)}"

    return [
        PromptSection(
            key="role_preface",
            title="System Role & Priorities",
            body=ROLE_PREFACE,
        ),
        PromptSection(
            key="personality",
            title="Assistant Personality",
            body=personality_body,
        ),
        PromptSection(
            key="channel_summary",
            title="Channel Summary",
            body=channel_body,
        ),
        PromptSection(
            key="user_profiles",
            title="User Profiles",
            body=profiles_body,
        ),
        PromptSection(
            key="message_schema",
            title="Message Schema",
            body=MESSAGE_SCHEMA_BODY,
        ),
        PromptSection(
            key="semantic_memory",
            title="Semantic Memory (top-k, JSON)",
            body=semantic_body,
        ),
    ]


async def build_sys_prompt(message: Message) -> str:
    """
    Human-readable system prompt (Markdown-ish).
    """
    # RAG fetches (all async)

    # Channel summary: reuses cached digest for the conversation to ground long-term context.
    chan_summary = await get_channel_summary(message.channel.id)

    # Profiles of users mentioned in the message (excluding the bot itself)
    # so the assistant can reference individualized preferences.
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
    # so we can assemble a blended context window from the highest-signal hits.
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
    # to avoid one fragment monopolizing the retrieved context.
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

    # Reformat the candidates into a prompt-friendly payload stripped of
    # transport-specific metadata.
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

    # Assemble each structured section and drop any that are empty.
    sections = _build_sections(chan_summary, profiles, semantic_candidates)

    rendered_sections: List[str] = []
    for section in sections:
        rendered = section.render()
        if rendered:
            rendered_sections.append(rendered)

    # Stitch the sections into a markdown-ish prompt. If every block is empty,
    # fall back to at least returning the role preface so the assistant is never
    # left without instructions.
    sys_prompt = "\n\n".join(rendered_sections).strip()
    fallback_prompt = PromptSection(
        key="role_preface",
        title="System Role & Priorities",
        body=ROLE_PREFACE,
    ).render() or ""

    return sys_prompt or fallback_prompt


