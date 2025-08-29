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


async def format_message(msg: Message) -> dict:
    """
    Takes a Discord :class:`Message`, classifies it, and composes media fragments.

    Returns a dictionary where ``fragments`` is a list of ``Fragment`` objects.
    The objects can later be serialized via :meth:`Fragment.to_dict` for RAG
    ingestion or :meth:`Fragment.to_llm` for LLM context building.

        {
            "author": "display_name",
            "fragments": [Fragment(...), ...]
        }

    - ``author``: the display name of the message author.
    - ``fragments``: list of :class:`Fragment` instances (see ``model.py``).
    """

    classified_msg = classify(msg)
    fragments = await compose(msg, classified_msg)
    return {
        "author": msg.author.display_name,
        "fragments": fragments,
    }


