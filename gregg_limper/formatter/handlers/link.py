"""
LinkHandler Pipeline
===========
1. Input slice : List[str] (URLs not already claimed by GIF / image logic)
2. For each URL
   a. await client_oai.summarize_url(url)
   b. Build fragment:  [link] <url> — <summary>
3. Return list[str] with one line per URL.
"""

from __future__ import annotations
import re
from typing import List
import asyncio
from . import register
from ...client_oai import summarize_url

import logging
logger = logging.getLogger(__name__)

@register
class LinkHandler:
    media_type = "link"

    @staticmethod
    async def handle(urls: List[str]) -> List[str]:
        async def _process(url: str) -> str:
            try:
                summary = await summarize_url(url, enable_citations=False)
            except Exception as e:
                logger.error(f"Failed to summarize URL {url}: {e}")
                summary = f"(link error: {e})"
            return f"[link] {url} — {summary}"

        # Run all URL fetches concurrently
        return await asyncio.gather(*(_process(u) for u in urls))
