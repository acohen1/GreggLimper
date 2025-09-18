from __future__ import annotations
import asyncio, json
from typing import List, Dict, Any
from gregg_limper.clients import disc
from discord import Message
from gregg_limper.config import prompt
from gregg_limper.memory.rag import (
    channel_summary as get_channel_summary,
    user_profile,
    vector_search,
)

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

    # Semantic memory (vector search)
    # TODO: We should really be comparing each fragment in the incoming message to the ones in semantic memory (vector_search).
    # At the moment, we simply take the message content as the vector search query, which is suboptimal if the message contains
    # images, links, or other non-text fragments.
    # We already have logic in the formatter/ package which parses incoming messages into media fragments (this pipeline is called when adding a msg to the cache),
    # but we can't easily reuse that here since adding the incoming message to the cache must happen *after* we build the system prompt
    # (otherwise the vector search will just return the new message itself as context).

    # Remove bot mention from content for vector search query (if present)
    content = message.content.replace(f"<@{disc.client.user.id}>", "").strip()

    # NOTE: we grab k+1 candidates since the incoming message may be returned as the top match;
    # we will filter it out in the prompt construction below.
    semantic_candidates = await vector_search(
        message.guild.id, message.channel.id, content, k=prompt.VECTOR_SEARCH_K + 1
    )

    # Filter out the incoming message itself (if present)
    semantic_candidates = [
        c for c in semantic_candidates if c.get("message_id") != message.id
    ][:prompt.VECTOR_SEARCH_K]  # limit to k results after filtering

    # We need only a subset of fields for prompt construction
    semantic_candidates = [
        {
            "author": (await disc.client.fetch_user(c.get("author_id"))).display_name,  # resolve author IDs to display names
            "title": c.get("title"),
            "content": c.get("content"),
            "type": c.get("type"),
            "url": c.get("url"),
        }
        for c in semantic_candidates
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


