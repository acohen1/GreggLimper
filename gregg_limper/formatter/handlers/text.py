"""
TextHandler Pipeline
====================
1. Input  : str  (raw message text)
2. Output : List[TextFragment] with a single record, e.g.:
      ``TextFragment(description="<normalized text>")``

The handler normalizes mention tokens into display names, removes bot
mentions, and skips fragments that consist only of a bot ping. Bare URLs
have already been removed by the classifier, so a remaining mention
without additional text is treated as noise.
"""

from __future__ import annotations

import re
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

        # 2) Normalize user mentions to display names and remove bot mentions
        content = original
        bot_id = getattr(bot, "id", None)
        for user in message.mentions:
            tokens = (f"<@{user.id}>", f"<@!{user.id}>")
            if bot_id is not None and user.id == bot_id:
                for token in tokens:
                    content = content.replace(token, "")
                continue

            name = user.display_name
            for token in tokens:
                content = content.replace(token, name)

        content = re.sub(r"\s+", " ", content).strip()
        if not content:
            return []

        # Emit normalized fragment
        return [TextFragment(description=content)]
