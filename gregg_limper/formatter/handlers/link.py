"""
LinkHandler Pipeline
===========
1. Input slice : List[str] (URLs not already claimed by GIF / image logic)
2. For each URL
   a. await client_oai.summarize_url(url)
   b. Build fragment dict: 
      { "type": "link", "title": "<url>", "description": "<summary>" }
3. Return List[dict] with one dict per URL

NOTE: We tolerate summarization failures here and keep the URL.
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
    async def handle(urls: List[str]) -> List[dict]:
        """
        Process a batch of URLs and return media-record dicts.
        Each dict contains:
        - "type": "link"
        - "title": URL
        - "description": text summary from web search model
        """
        async def _process(url: str) -> dict:
            try:
                summary = await summarize_url(url, enable_citations=False)
            except Exception as e:
                logger.error(f"Failed to summarize URL {url}: {e}")
                summary = f"(link error: {e})"
            return {"type": "link", "title": url, "description": summary}

        # Run all URL fetches concurrently
        return await asyncio.gather(*(_process(u) for u in urls))
