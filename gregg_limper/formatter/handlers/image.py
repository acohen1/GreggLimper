"""
ImageHandler
============
1. Input slice  : List[discord.Attachment] (non-GIF images).
2. For each attachment
   a. Download bytes (≤ Config.MAX_IMAGE_MB).
   b. await client_oai.describe_image_bytes(...)
   c. Build :class:`ImageFragment`:
      ``ImageFragment(title="<filename>", caption="<vision caption>")``
3. Return ``list[ImageFragment]`` with one fragment per image.

NOTE: Vision errors keep the record; we substitute a placeholder caption.
"""

from __future__ import annotations
from typing import List
import aiohttp, asyncio
from discord import Attachment
from . import register
from ...config import Config
from ...clients.oai import describe_image_bytes
from ..model import ImageFragment

import logging
logger = logging.getLogger(__name__)

@register
class ImageHandler:
    media_type = "image"

    @staticmethod
    async def _fetch_bytes(url: str, max_mb: int) -> bytes:
        async with aiohttp.ClientSession() as s, s.get(url) as r:
            r.raise_for_status()
            data = await r.read()
            if len(data) > max_mb * 1024 * 1024:
                raise ValueError("image too large")
            return data

    @staticmethod
    async def handle(images: list[Attachment]) -> list[ImageFragment]:
        """
        Process a batch of image attachments and return ``ImageFragment`` objects.
        Each fragment contains ``title`` and a vision-generated ``caption``.
        """

        async def _process(att: Attachment) -> ImageFragment:
            blob = await ImageHandler._fetch_bytes(att.url, Config.MAX_IMAGE_MB)
            try:
                desc = await describe_image_bytes(blob, mime=att.content_type or "image/png")
            except Exception as e:
                logger.error(f"Failed to describe image {att.filename}: {e}")
                desc = f"(vision error: {e})"
            return ImageFragment(title=att.filename, url=att.url, caption=desc)

        return await asyncio.gather(*(_process(a) for a in images))


