"""
TextHandler Pipeline
====================
1. Input  : str  (raw message text)
2. Output : List[TextFragment] with a single record, e.g.:
      ``TextFragment(description="<normalized text>")``

The handler normalizes mention tokens into display names and skips
fragments that consist only of a bot ping. Bare URLs have already been
removed by the classifier, so a remaining mention without additional text
is treated as noise.
"""

from __future__ import annotations

import re
from typing import List, Set
from discord import Message
from . import register
from ..model import TextFragment
from ...commands.handlers import all_commands  # explicit top-level import

# First slash-style command regex (anywhere in the text)
_COMMAND_RE = re.compile(r"/(\w+)(?:\s+(.*))?")

@register
class TextHandler:
    media_type = "text"
    needs_message = True

    @staticmethod
    async def handle(text: str, message: Message) -> List[TextFragment]:
        """
        Wrap non-empty message text in a :class:`TextFragment` after:
          - dropping pure bot pings
          - normalizing user mentions to display names,
          - dropping only if a slash command present in the message matches the registry.
        """
        if not message or not message.guild or not message.guild.me:
            raise ValueError("TextHandler expects a guild message with a bot user.")

        # Original text for token-based checks
        original = (text or "").strip()
        if not original:
            return []

        bot = message.guild.me
        bot_id_tokens: Set[str] = {f"<@{bot.id}>", f"<@!{bot.id}>"}

        # 1) Pure-ping check on the *original* content (before normalization)
        if original in bot_id_tokens:
            return []

        # 2) Normalize user mentions (including the bot's) to display names
        content = original
        for user in message.mentions:
            name = user.display_name
            content = content.replace(f"<@{user.id}>", name)
            content = content.replace(f"<@!{user.id}>", name)
        content = content.strip()
        if not content:
            return []
        
        # 3) Registry-backed slash-command detection (regex search anywhere)
        match = _COMMAND_RE.search(content)
        if match:
            candidate = (match.group(1) or "").lower()  # command name without '/'
            registry = {name.lstrip("/").lower() for name in all_commands().keys()}
            if candidate in registry:
                return []

        # 4) Emit normalized fragment
        return [TextFragment(description=content)]
