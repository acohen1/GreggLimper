"""
LinkHandler Pipeline
===========
1. Input slice : List[str] (URLs not already claimed by GIF / image logic)
2. For each URL
   a. await client_oai.summarize_url(url)
   b. Build :class:`LinkFragment`:
      ``LinkFragment(title="<url>", description="<summary>")``
3. Return ``list[LinkFragment]`` with one fragment per URL.

NOTE: We tolerate summarization failures here and keep the URL.
"""

from __future__ import annotations
import re
from typing import List
import asyncio
from . import register
from ...clients.oai import summarize_url
from ..model import LinkFragment

import logging
logger = logging.getLogger(__name__)

@register
class LinkHandler:
    media_type = "link"

    @staticmethod
    async def handle(urls: List[str]) -> List[LinkFragment]:
        """
        Process URLs into :class:`LinkFragment` objects.

        :param urls: List of generic hyperlink URLs.
        :returns: Fragments containing the original ``url`` and summary
            ``description``.
        """
        async def _process(url: str) -> LinkFragment:
            try:
                summary = await summarize_url(url, enable_citations=False)
            except Exception as e:
                logger.error(f"Failed to summarize URL {url}: {e}")
                summary = f"(link error: {e})"
            return LinkFragment(title=url, url=url, description=summary)

        # Run all URL fetches concurrently
        return await asyncio.gather(*(_process(u) for u in urls))

