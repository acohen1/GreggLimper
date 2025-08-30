"""
GIFHandler Pipeline
===================
1. Input slice : List[str] (URLs not already claimed by link / image logic)
2. For each URL
   a. Extract <meta property="og:title"> for a title.
   b. Extract <meta property="og:image"> for the GIF URL.
   c. Download GIF -> first frame -> PNG -> vision model.
3. Return ``list[GIFFragment]`` like:
      ``GIFFragment(title="<cleaned-title>", caption="<frame description>")``

NOTE: Vision errors keep the record; we substitute a placeholder caption.
"""

from __future__ import annotations
from typing import List, Tuple
import aiohttp, asyncio
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image
import re, logging

from discord import Attachment
from . import register
from ...clients.oai import describe_image_bytes
from ..model import GIFFragment

logger = logging.getLogger(__name__)


@register
class GIFHandler:
    media_type = "gif"

    # ---------- low‑level helpers ------------------------------------ #

    @staticmethod
    def _clean_gif_title(raw_title: str) -> str:
        """
        Cleans up messy or repetitive GIF titles.

        - Removes suffixes like ' - Discover & Share GIFs'
        - Trims trailing 'GIF' token
        - Removes duplicated leading phrases (e.g. 'X - X Y')
        """
        title = raw_title.strip()
        for suffix in (
            " - Discover & Share GIFs",
            " | Giphy",
            " - Giphy",
        ):
            if title.endswith(suffix):
                title = title[: -len(suffix)].strip()
        return re.sub(r"\s+", " ", title)

    @staticmethod
    async def _parse_title_and_gif_url(session: aiohttp.ClientSession, url: str) -> Tuple[str, str]:
        """
        Scrapes the page to extract <meta property="og:title"> and <meta property="og:image">
        using a shared aiohttp session (defined in handle()).
        """
        async with session.get(url) as r:
            r.raise_for_status()
            html = await r.text()

        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("meta", property="og:title")
        image_tag = soup.find("meta", property="og:image")

        if not image_tag or not image_tag.get("content"):
            raise ValueError(f"GIF URL not found in page: {url}")

        raw_title = (
            title_tag["content"].strip()
            if title_tag and title_tag.get("content")
            else "Untitled GIF"
        )
        return GIFHandler._clean_gif_title(raw_title), image_tag["content"]

    @staticmethod
    async def _download_gif_and_extract_frame(
        session: aiohttp.ClientSession, gif_url: str
    ) -> bytes:
        """
        Downloads the .gif and extracts its first frame as PNG bytes.
        """
        async with session.get(gif_url) as r:
            r.raise_for_status()
            gif_bytes = await r.read()

        with Image.open(BytesIO(gif_bytes)) as im:
            im.seek(0)
            im = im.convert("RGBA")
            buf = BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()

    # ---------- public contract -------------------------------------- #

    @staticmethod
    async def handle(urls: List[str]) -> List[GIFFragment]:
        """Process GIF URLs into :class:`GIFFragment` objects.

        :param urls: List of candidate GIF URLs.
        :returns: Fragments with cleaned ``title`` and vision-generated ``caption``.
        """
        logger.info("Processing %d GIF URLs", len(urls))

        async with aiohttp.ClientSession() as session:

            async def _process(url: str) -> GIFFragment:
                try:
                    title, gif_url = await GIFHandler._parse_title_and_gif_url(session, url)
                    frame_png = await GIFHandler._download_gif_and_extract_frame(session, gif_url)
                    vision = await describe_image_bytes(frame_png, mime="image/png")
                    return GIFFragment(title=title, url=gif_url, caption=vision)
                except Exception as e:
                    logger.warning("Failed to process GIF at %s: %s", url, e)
                    return GIFFragment(title=url, url=url, caption=f"(vision error: {e})")

            # Process URLs concurrently
            return await asyncio.gather(*(_process(u) for u in urls))
