from __future__ import annotations
import asyncio, json
from typing import List, Dict, Any
from gregg_limper.clients import disc
from discord import Message
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
    chan_summary = await get_channel_summary(message.channel.id)

    user_mentions = [u.id for u in message.mentions if u.id != disc.client.user.id]
    profiles: List[Dict[str, Any]] = (
        await asyncio.gather(*(user_profile(u) for u in user_mentions))
        if user_mentions else []
    )

    semantic_candidates = await vector_search(
        message.guild.id, message.channel.id, message.content, k=50
    )

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


