"""
GIFHandler Pipeline
===========
1. Input slice : List[str] (URLs not already claimed by link logic)
2. For each URL
    a. Scrape the page to extract <meta property="og:title"> and <meta property="og:image">.
    b. Download the actual .gif and extract the first frame as PNG bytes.
    c. await client_oai.describe_image_bytes(frame_png, mime="image/png")
    d. Build fragment:  [gif] <title> — <description>
3. Return list[str] with one line per GIF.
"""

from __future__ import annotations
from typing import List, Tuple
import aiohttp, asyncio
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image
import re

from discord import Attachment
from . import register
from gregg_limper.config import Config
from gregg_limper.client_oai import describe_image_bytes

import logging
logger = logging.getLogger(__name__)


@register
class GIFHandler:
    media_type = "gif"

    # ---------- low-level helpers -------------------------------------- #

    @staticmethod
    def _clean_gif_title(raw_title: str) -> str:
        """
        Cleans up messy or repetitive GIF titles.
        - Removes suffixes like ' - Discover & Share GIFs'
        - Trims trailing 'GIF' token
        - Removes duplicated leading phrases (e.g. 'X - X Y')
        """
        title = raw_title.strip()

        # Remove known suffixes
        for suffix in [
            " - Discover & Share GIFs",
            " | Giphy",
            " - Giphy",
        ]:
            if title.endswith(suffix):
                title = title[: -len(suffix)].strip()

        # Normalize spacing
        title = re.sub(r"\s+", " ", title)
        return title

    @staticmethod
    async def _parse_title_and_gif_url(url: str) -> Tuple[str, str]:
        """
        Scrapes the page to extract <meta property="og:title"> and <meta property="og:image">.
        Returns (title, gif_url).
        """
        async with aiohttp.ClientSession() as s, s.get(url) as r:
            if r.status != 200:
                raise ValueError(f"Failed to fetch page HTML: {url}")
            html = await r.text()

        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("meta", property="og:title")
        image_tag = soup.find("meta", property="og:image")

        if not image_tag or not image_tag.get("content"):
            raise ValueError(f"GIF URL not found in page: {url}")

        raw_title = title_tag["content"].strip() if title_tag and title_tag.get("content") else "Untitled GIF"
        title = GIFHandler._clean_gif_title(raw_title)

        return title, image_tag["content"]

    @staticmethod
    async def _download_gif_and_extract_frame(gif_url: str) -> bytes:
        """
        Downloads the actual .gif and extracts the first frame as PNG bytes using Pillow.
        """
        async with aiohttp.ClientSession() as s, s.get(gif_url) as r:
            r.raise_for_status()
            gif_bytes = await r.read()

        with Image.open(BytesIO(gif_bytes)) as im:
            im.seek(0)
            im = im.convert("RGBA")
            buf = BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()

    # ---------- public contract ---------------------------------------- #

    @staticmethod
    async def handle(urls: List[str]) -> List[str]:
        logger.info(f"Processing {len(urls)} GIF URLs")

        async def _process(url: str) -> str:
            try:
                title, gif_url = await GIFHandler._parse_title_and_gif_url(url)
                logger.info(f"GIF media URL for {url}: {gif_url}")
                frame_png = await GIFHandler._download_gif_and_extract_frame(gif_url)
                vision = await describe_image_bytes(frame_png, mime="image/png")
                return f"[gif] {title} — {vision}"

            except Exception as e:
                logger.warning(f"Failed to process GIF at {url}: {e}")
                return f"[gif] {url} — (vision error: {e})"

        return await asyncio.gather(*(_process(u) for u in urls))
