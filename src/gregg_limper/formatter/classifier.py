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

_YOUTUBE_DOMAINS = {
    "youtube.com",
    "youtu.be",
}

def _is_youtube_url(url: str) -> bool:
    url_lc = url.lower()
    host = urlparse(url_lc).hostname or ""
    return any(dom in host for dom in _YOUTUBE_DOMAINS)

def _is_gif_url(url: str) -> bool:
    url_lc = url.lower()
    if url_lc.endswith(".gif"):
        return True
    host = urlparse(url_lc).hostname or ""
    return any(dom in host for dom in _GIF_DOMAINS)

def _strip_urls(text: str) -> str:
    """Remove all URLs from the string."""
    return _URL_RE.sub("", text).strip()

# --------------------------------------------------------------------- #
#  Main entry
# --------------------------------------------------------------------- #

def classify(msg: Message) -> Dict[str, Any]:
    """
    Classify a message into media slices.

    :param msg: Discord message to analyze.
    :returns: Mapping of media type to slice data, e.g.:

    .. code-block:: python

        {
            "text":  "Look at this",
            "gif":   ["https://tenor.com/view/..."],
            "image": [<discord.Attachment ...>],
            "link":  ["https://arxiv.org/abs/..."]
        }
    """
    result: Dict[str, Any] = {}
    content = msg.content or ""

    # 1) text content -- only added if message includes text beyond just URLs
    if (stripped := _strip_urls(content)):
        result["text"] = stripped

    # 2) attachments -- note that GIFs are accumulated between both msg attachments and text URLs
    images, gifs = [], []
    for att in msg.attachments:
        if att.content_type == "image/gif":
            gifs.append(att.url)
        elif att.content_type and att.content_type.startswith("image/"):
            images.append(att)

    # 3) URLs in message content
    for url in _URL_RE.findall(content):
        if _is_gif_url(url):
            gifs.append(url)
        elif _is_youtube_url(url):
            result.setdefault("youtube", []).append(url)
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
    logger.info(f"classify: {msg.id} [{msg.author}] -> types={list(result.keys())} | '{msg_snippet}'")

    return result

