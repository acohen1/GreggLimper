from __future__ import annotations

from discord import Message

from .classifier import classify
from .composer import compose
from .model import (
    Fragment,
    TextFragment,
    ImageFragment,
    GIFFragment,
    YouTubeFragment,
    LinkFragment,
)

__all__ = [
    "format_message",
    "Fragment",
    "TextFragment",
    "ImageFragment",
    "GIFFragment",
    "YouTubeFragment",
    "LinkFragment",
]

# TODO: Make media handlers (img/gif/yt) output descriptions in the style of the bot.
# This should help us shape the model's personality.
async def format_message(msg: Message) -> dict:
    """
    Classify and compose message fragments.

    :param msg: Discord message to process.
    :returns: Dict with ``author`` and list of ``Fragment`` objects.

    Example
    -------
    .. code-block:: python

        {
            "author": "display_name",
            "fragments": [Fragment(...), ...]
        }
    """

    classified_msg = classify(msg)
    fragments = await compose(msg, classified_msg)
    return {
        "author": msg.author.display_name,
        "fragments": fragments,
    }


