# classifier.py
import re
from typing import Any, Dict, List
from urllib.parse import urlparse
from discord import Message

import logging
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------- #

_URL_RE = re.compile(r"https?://\S+")

_GIF_DOMAINS = {
    "tenor.com",
    "giphy.com",
    "media.tenor.com",
    "media.giphy.com",
}

def _is_gif_url(url: str) -> bool:
    """
    Heuristic:
    - ends with '.gif'   -> yes
    - hostname contains a known GIF service -> yes
    """
    url_lc = url.lower()
    if url_lc.endswith(".gif"):
        return True
    host = urlparse(url_lc).hostname or ""
    return any(dom in host for dom in _GIF_DOMAINS)

# --------------------------------------------------------------------- #
#  Main entry
# --------------------------------------------------------------------- #

def classify(msg: Message) -> Dict[str, Any]:
    """
    Returns dict media_type → slice_data.

    Example:
    {
        "text":  "Look at this",
        "gif":   ["https://tenor.com/view/..."],
        "image": [<discord.Attachment ...>],
        "link":  ["https://arxiv.org/abs/..."]
    }
    """
    result: Dict[str, Any] = {}

    # 1) plain text (keep full string for TextHandler to trim later)
    result["text"] = msg.content or ""

    # 2) attachments
    images, gifs = [], []
    for att in msg.attachments:
        if att.content_type == "image/gif":
            gifs.append(att.url)
        elif att.content_type and att.content_type.startswith("image/"):
            images.append(att)

    # 3) URLs in message content
    for url in _URL_RE.findall(msg.content or ""):
        if _is_gif_url(url):
            gifs.append(url)
        else:
            # generic link for LinkHandler
            result.setdefault("link", []).append(url)

    if images:
        result["image"] = images
    if gifs:
        result["gif"] = gifs

    # Log classification result
    # classify: 118235901246 [User#1234] -> types=['text', 'gif', 'link'] | 'Check this out https://tenor.com/view/...'
    msg_snippet = (msg.content[:50] + "...") if msg.content else "<empty>"
    logger.debug(f"classify: {msg.id} [{msg.author}] -> types={list(result.keys())} | '{msg_snippet}'")

    return result
