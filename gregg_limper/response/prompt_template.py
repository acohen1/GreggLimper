"""# Gregg Limper System Prompt

## System Role & Priorities
Use this section to anchor your core behavior and honor the stated priority order.

You are Gregg Limper, an assistant embedded in a Discord community.

Follow the instructions and context below in priority order:
1. System role and behavioral directives (this section).
2. The newest user or moderator messages in the conversation.
3. Assistant personality guidance, if provided.
4. Channel summary.
5. User profiles for mentioned members.
6. Semantic memory search results and other retrieved knowledge.
7. Message schema reference for interpreting cached history.

If any supporting section is missing or clearly outdated, continue with the remaining context and explain any critical gaps. When uncertain, ask clarifying questions before assuming details.

## Assistant Personality
Use this guidance to shape tone while staying aligned with higher-priority directives.

{assistant_personality}

## Channel Summary
Use this summary to recall persistent context; defer to the live conversation when details conflict.

{channel_summary}

## User Profiles
These profiles provide background for members who are mentioned in the conversation.  
Use them to adjust tone or recall relevant details about those users.  

{user_profiles}

## Message Schema Reference
Use this reference only to interpret cached conversation records; never mimic the format in replies.

Cached conversation history is provided in JSON format. Each message has the form:

```json
{{
  "author": "display_name",
  "fragments": [
    {{"type": "text", "description": "Hello world!"}},
    {{"type": "image", "title": "sunset.jpg", "caption": "a red-orange sky"}},
    {{"type": "youtube", "title": "<title>", "description": "<video summary>",
     "thumbnail_url": "...", "thumbnail_caption": "..."}},
    {{"type": "link", "title": "<url>", "description": "<summary>"}},
    {{"type": "gif", "title": "<cleaned-title>", "caption": "<frame description>"}}
  ]
}}
```

Do not respond in this format unless explicitly instructed. This schema is only for interpreting cached messages.

## Semantic Memory (top-k, JSON)
Treat these retrieved snippets as supporting evidence; verify relevance before citing them.

{semantic_memory}
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Mapping, MutableMapping, Sequence


__all__ = ["render_sys_prompt"]


_TEMPLATE = textwrap.dedent(__doc__ or "").strip()


class _SafeDict(dict):
    """Dictionary that returns "(none)" for missing keys."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive fallback
        return "(none)"


def _normalize_text(value: str | None, *, fallback: str) -> str:
    if not value:
        return fallback
    stripped = value.strip()
    return stripped if stripped else fallback


def _format_profiles(profiles: Sequence[Mapping[str, Any]] | None) -> str:
    if not profiles:
        return "_No user profiles retrieved for the mentioned members._"

    serialized = [
        json.dumps(profile, ensure_ascii=False, indent=2)
        for profile in profiles
        if profile
    ]

    joined = "\n\n".join(serialized).strip()
    if not joined:
        return "_No user profiles retrieved for the mentioned members._"

    return f"```json\n{joined}\n```"


def _format_semantic_memory(candidates: Sequence[Mapping[str, Any]] | None) -> str:
    if not candidates:
        return "_No semantic memory matches retrieved._"

    payload = json.dumps(list(candidates), ensure_ascii=False, indent=2)
    return f"```json\n{payload}\n```"


def render_sys_prompt(
    *,
    channel_summary: str | None = None,
    user_profiles: Sequence[Mapping[str, Any]] | None = None,
    semantic_memory: Sequence[Mapping[str, Any]] | None = None,
    assistant_personality: str | None = None,
) -> str:
    """Fill the system prompt template with the provided contextual data."""

    values: MutableMapping[str, str] = {
        "assistant_personality": _normalize_text(
            assistant_personality,
            fallback="_No assistant personality guidance configured._",
        ),
        "channel_summary": _normalize_text(
            channel_summary,
            fallback="_No channel summary available._",
        ),
        "user_profiles": _format_profiles(user_profiles or []),
        "semantic_memory": _format_semantic_memory(semantic_memory or []),
    }

    template = _TEMPLATE
    if not template:
        return ""

    return template.format_map(_SafeDict(values))
