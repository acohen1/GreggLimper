from __future__ import annotations

import asyncio
from typing import Dict, List, Any
from discord import Message

from .handlers import get as get_handler
from ..memory.rag.media_id import stable_media_id
from .model import Fragment

import logging

logger = logging.getLogger(__name__)

ORDER = ["text", "image", "gif", "link", "youtube"]


async def compose(message: Message, classified: Dict[str, Any]) -> List[Fragment]:
    """
    Aggregate media-slice outputs into :class:`Fragment` objects.

    :param message: Source Discord message.
    :param classified: Media slices produced by :func:`classifier.classify`.
    :returns: List of fragments with stable ``id`` values.
    """

    coros: List[asyncio.Future] = [
        get_handler(mt).handle(classified[mt])
        for mt in ORDER
        if classified.get(mt) and get_handler(mt)
    ]

    # Execute slice handlers concurrently; absent types yield an empty list
    results = await asyncio.gather(*coros) if coros else []
    fragments: List[Fragment] = [rec for frag_list in results for rec in frag_list]

    for idx, frag in enumerate(fragments):
        frag.id = stable_media_id(
            cf=frag.to_dict(),
            server_id=message.guild.id if message.guild else 0,
            channel_id=message.channel.id,
            message_id=message.id,
            source_idx=idx,
        )

    return fragments


def serialize_fragments(fragments: List[Fragment]) -> List[dict]:
    """
    Convert ``Fragment`` objects into JSON-serializable dictionaries.

    :param fragments: List of fragment objects.
    :returns: Corresponding list of ``dict`` objects.
    """
    return [f.to_dict() for f in fragments]


