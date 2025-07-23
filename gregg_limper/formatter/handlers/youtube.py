"""
YouTubeHandler Pipeline
===========
1. Input slice : List[str] (URLs not already claimed by GIF / image logic)
2. For each URL
    a. Call YouTube API to get video details.
    b. Build fragment:  [youtube] <title> — <description>
3. Return list[str] with one line per YouTube video.
"""

# TODO: Implement YouTubeHandler to process YouTube URLs
from __future__ import annotations
import re
from typing import List
import asyncio
from . import register
from gregg_limper.client_oai import summarize_url
from gregg_limper.config import Config

import logging
logger = logging.getLogger(__name__)

@register
class YouTubeHandler:
    media_type = "youtube"

    @staticmethod
    async def handle(urls: List[str]) -> List[str]:
        async def _process(url: str) -> str:
            # TODO: Implement YouTube API call to get video details
            pass

        # Run all URL fetches concurrently
        return await asyncio.gather(*(_process(u) for u in urls))