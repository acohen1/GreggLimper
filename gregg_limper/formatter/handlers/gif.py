"""
GIFHandler Pipeline
==========
1. Input slice  : List[str]  (GIF URLs — Tenor or Giphy, detected by domain)
2. For each URL
   a. Download raw GIF bytes  (≤ Config.MAX_GIF_MB)
   b. Extract FIRST frame -> PNG bytes (imageio.v3)
   c. await client_oai.describe_image_bytes(frame_png, mime="image/png")
   d. GET oEmbed metadata:
        - Tenor : https://tenor.com/oembed?url=<GIF_URL>
        - Giphy : https://giphy.com/services/oembed/?url=<GIF_URL>
      (Returns JSON with 'title', 'author_name', ...)
   e. Build fragment line:
        [gif] <title> — <OpenAI frame-description>
3. Return **list[str]** (one line per GIF)
"""

from __future__ import annotations
from typing import List
import aiohttp, asyncio, json, io
import imageio.v3 as iio
from urllib.parse import urlparse

from discord import Attachment         # not used but keeps type parity
from . import register
from ...config import Config
from ...client_oai import describe_image_bytes

TENOR_OEMBED = "https://tenor.com/oembed?url="
GIPHY_OEMBED = "https://giphy.com/services/oembed/?url="

@register
class GIFHandler:
    media_type = "gif"

    # ---------- low-level helpers -------------------------------------- #

    @staticmethod
    async def _fetch_bytes(url: str, cap_mb: int) -> bytes:
        async with aiohttp.ClientSession() as s, s.get(url) as r:
            r.raise_for_status()
            data = await r.read()
            if len(data) > cap_mb * 1024 * 1024:
                raise ValueError(f"GIF {url} exceeds {cap_mb} MB")
            return data

    @staticmethod
    async def _oembed(url: str) -> dict:
        dom = urlparse(url).hostname or ""
        if "tenor" in dom:
            endpoint = TENOR_OEMBED + url
        elif "giphy" in dom:
            endpoint = GIPHY_OEMBED + url
        else:
            return {}
        async with aiohttp.ClientSession() as s, s.get(endpoint) as r:
            if r.status != 200:
                return {}
            return await r.json()

    @staticmethod
    def _first_frame_png(gif_bytes: bytes) -> bytes:
        # Decode first frame → numpy → encode PNG in-memory
        frame = iio.imread(gif_bytes, index=0)     # ndarray
        png_bytes = iio.imwrite("<bytes>", frame, extension=".png")
        return png_bytes

    # ---------- public contract ---------------------------------------- #

    @staticmethod
    async def handle(urls: List[str]) -> List[str]:
        async def _process(url: str) -> str:
            try:
                gif_blob  = await GIFHandler._fetch_bytes(url, Config.MAX_GIF_MB)
                frame_png = GIFHandler._first_frame_png(gif_blob)
                vision    = await describe_image_bytes(frame_png, mime="image/png")
            except Exception as e:
                vision = f"(vision error: {e})"

            meta = await GIFHandler._oembed(url)
            title = meta.get("title") or meta.get("alt") or "Untitled GIF"

            # TODO: Remove TENOR/GIPHY-specific suffixes
            title = title.removesuffix(" GIF").strip()

            return f"[gif] {title} — {vision}"

        return await asyncio.gather(*(_process(u) for u in urls))
