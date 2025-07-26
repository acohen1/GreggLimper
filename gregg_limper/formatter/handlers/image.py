"""
ImageHandler
============
1. Input slice  : List[discord.Attachment] (non-GIF images).
2. For each attachment
   a. Download bytes (≤ Config.MAX_IMAGE_MB).
   b. await client_oai.describe_image_bytes(...)
   c. Build fragment dict:
      { "type": "image", "title": "<filename>", "description": "<vision description>" }
3. Return list[dict] with one dict per image

NOTE: Vision errors keep the record; we substitute a placeholder description.
"""

from __future__ import annotations
from typing import List
import aiohttp, asyncio
from discord import Attachment
from . import register
from ...config import Config
from ...client_oai import describe_image_bytes

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
    async def handle(images: list[Attachment]) -> list[dict]:
        """
        Process a batch of image attachments and return media-record dicts.
        Each dict contains:
        - "type": "image"
        - "title": filename
        - "description": text description from vision model
        """
        async def _process(att: Attachment) -> dict:
            blob = await ImageHandler._fetch_bytes(att.url, Config.MAX_IMAGE_MB)
            try:
                desc = await describe_image_bytes(blob, mime=att.content_type or "image/png")
            except Exception as e:
                logger.error(f"Failed to describe image {att.filename}: {e}")
                desc = f"(vision error: {e})"
            return {"type": "image", "title": att.filename, "description": desc}

        return await asyncio.gather(*(_process(a) for a in images))

