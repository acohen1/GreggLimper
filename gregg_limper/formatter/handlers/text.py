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

from typing import List, Set
from discord import Message
from . import register
from ..model import TextFragment

@register
class TextHandler:
    media_type = "text"
    needs_message = True

    @staticmethod
    async def handle(text: str, message: Message) -> List[TextFragment]:
        """
        Wrap non-empty message text in a :class:`TextFragment` after:
          - dropping pure bot pings when the bot identity is known
          - normalizing user mentions to display names.
        """
        if not message or not message.guild:
            raise ValueError("TextHandler expects a guild message.")

        # Original text for token-based checks
        original = (text or "").strip()
        if not original:
            return []

        bot = getattr(message.guild, "me", None)
        bot_id_tokens: Set[str] = set()
        if bot is not None:
            bot_id_tokens = {f"<@{bot.id}>", f"<@!{bot.id}>"}

        # 1) Pure-ping check on the *original* content (before normalization)
        if bot_id_tokens and original in bot_id_tokens:
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

        # Emit normalized fragment
        return [TextFragment(description=content)]
