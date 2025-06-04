import asyncio
from typing import Dict, List
from .handlers import get as get_handler
from discord import User
from config import Config

ORDER = ["text", "image", "gif", "link"]

async def compose(author: User, classified: Dict[str, any]) -> str:
    """
    - Launch every handler whose media slice exists.
    - Await them concurrently with asyncio.gather().
    - Join the returned fragment lists.
    """
    coros: List[asyncio.Future] = []

    for media_type in ORDER:
        data = classified.get(media_type)
        if not data:
            continue
        handler = get_handler(media_type)
        if handler:
            coros.append(handler.handle(data))

    # Run all handlers in parallel; each returns list[str]
    results = await asyncio.gather(*coros) if coros else []

    fragments: List[str] = []
    for frag_list in results:
        fragments.extend(frag_list)

    body = "\n\n".join(filter(None, fragments)).strip()
    
    # TODO: Do we want to author the prefix here? Composer might need to be agnostic.
    
    # If the author isn't the bot, include the author name in the output.
    if author.id != Config.BOT_USER_ID:
        return f"{author}: {body}"
    else:
        return body

