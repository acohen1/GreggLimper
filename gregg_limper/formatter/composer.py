import asyncio
from typing import Dict, List, Any
from .handlers import get as get_handler
from discord import User
from gregg_limper.config import Config
from discord import Message
import json

import logging
logger = logging.getLogger(__name__)

ORDER = ["text", "image", "gif", "link", "youtube"]

async def compose(message: Message, classified: Dict[str, Any]) -> str:
    """
    Aggregate media-slice outputs.

    1. Launch every handler (text, image, gif, link, youtube) whose slice exists.
    2. Await them concurrently.
    3. Flatten results into a single list of dicts.
    4. Return a JSON-serializable dict with author, channel_id, timestamp, and fragments.
    """

    # Kick off handler coroutines in ORDER for deterministic output.
    coros: List[asyncio.Future] = [
        get_handler(mt).handle(classified[mt])
        for mt in ORDER
        if classified.get(mt) and get_handler(mt)
    ]

    # Await all; results is List[List[dict|str]]
    results = await asyncio.gather(*coros) if coros else []

    flattened_fragments = [rec for frag_list in results for rec in frag_list]

    master_record = {
        "author": {
            "id": message.author.id,
            "name": message.author.display_name,
        },
        "channel_id": message.channel.id,
        "timestamp": message.created_at.isoformat(),
        "fragments": flattened_fragments
    }

    return json.dumps(master_record, ensure_ascii=False, separators=(',', ': '))