import asyncio
from typing import Dict, List
from .handlers import get as get_handler

ORDER = ["text", "image", "gif", "link"]

async def compose(author: str, classified: Dict[str, any]) -> str:
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
    
    # Add author prefix and return
    return f"{author}: {body}" if body else f"{author}:"
