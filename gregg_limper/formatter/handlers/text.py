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
from typing import List
from discord import Message
from . import register
from ..model import TextFragment

@register
class TextHandler:
    media_type = "text"
    needs_message = True

    @staticmethod
    async def handle(text: str, message: Message | None = None) -> List[TextFragment]:
        """
        Wrap non-empty message text in a :class:`TextFragment`.

        :param text: Raw message content.
        :param message: Source Discord message containing metadata for mention handling.
        :returns: List containing a single fragment or empty list.
        """
        if message is None:
            raise ValueError("TextHandler requires the source Discord message")
        stripped = text.strip()
        if not stripped:
            return []

        content = stripped
        mention_display_map: dict[str, str] = {}

        for user in message.mentions:
            tokens = (f"<@{user.id}>", f"<@!{user.id}>")
            for token in tokens:
                mention_display_map[token] = user.display_name

        for token, display_name in mention_display_map.items():
            content = content.replace(token, display_name)

        content = content.strip()

        bot_user = None
        if getattr(message, "guild", None):
            bot_user = getattr(message.guild, "me", None)
        if not bot_user and getattr(message, "channel", None):
            bot_user = getattr(message.channel, "me", None)
        if not bot_user and hasattr(message, "_state"):
            bot_user = getattr(message._state, "user", None)

        if bot_user:
            bot_tokens = {f"<@{bot_user.id}>", f"<@!{bot_user.id}>"}
            if stripped in bot_tokens:
                return []

            display_name = getattr(bot_user, "display_name", None) or getattr(bot_user, "name", None)
            username = getattr(bot_user, "name", None)
            bot_names = {name.strip() for name in (display_name, username) if name}
            bot_names.update({f"@{name}" for name in list(bot_names)})
            if content in bot_names:
                return []

        return [TextFragment(description=content)] if content else []

