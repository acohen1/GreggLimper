import asyncio
from typing import Dict, List, Any
from .handlers import get as get_handler
from discord import User
from gregg_limper.config import Config
from gregg_limper.formatter.codec import dump

import logging
logger = logging.getLogger(__name__)

ORDER = ["text", "image", "gif", "link", "youtube"]

async def compose(author: User, classified: Dict[str, Any]) -> str:
    """
    Aggregate media-slice outputs.

    1. Launch every handler (text, image, gif, link, youtube) whose slice exists.
    2. Await them concurrently.
    3. Serialize dicts via formatter.codec.dump -> str.
    4. Join fragments with double new-lines.
    """

    # Kick off handler coroutines in ORDER for deterministic output.
    coros: List[asyncio.Future] = [
        get_handler(mt).handle(classified[mt])
        for mt in ORDER
        if classified.get(mt) and get_handler(mt)
    ]

    # Await all; results is List[List[dict|str]]
    results = await asyncio.gather(*coros) if coros else []

    fragments: List[str] = [
        dump(rec, fmt=Config.MEDIA_RECORD_FMT)
        for frag_list in results
        for rec in frag_list
    ]

    body = "\n\n".join(fragments).strip()
    
    # TODO: Should this occur here?
    # Prepend author name unless it's the bot itself.
    return f"{author}: {body}" if author.id != Config.BOT_USER_ID else body