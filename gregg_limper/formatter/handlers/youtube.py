"""
YouTubeHandler Pipeline
===========
1. Input slice : List[str] (URLs not already claimed by GIF / image logic)
2. For each URL
    a. Call YouTube API to get video details.
    b. Build fragment:  [youtube] <title> — <description>
3. Return list[str] with one line per YouTube video.
"""

# TODO: Summarize the video description using a client_oai.py call--gathering all relevant context.
# TODO: Handle YouTube thumbnail extraction and run through vision pipeline for a description.

import asyncio
from typing import List, Tuple
from urllib.parse import urlparse, parse_qs
from . import register
import aiohttp
from gregg_limper.config import Config

import logging
logger = logging.getLogger(__name__)

@register
class YouTubeHandler:
    media_type = "youtube"

    # ---------- low‑level helpers ------------------------------------ #

    @staticmethod
    def _extract_video_id(url: str) -> str | None:
        """Return the YouTube video ID or None if the URL is invalid."""
        parsed = urlparse(url)
        if "youtu.be" in parsed.netloc:
            return parsed.path.lstrip("/")
        return parse_qs(parsed.query).get("v", [None])[0]

    @staticmethod
    async def _fetch_video_metadata(
        session: aiohttp.ClientSession, video_id: str
    ) -> Tuple[str, str]:
        """
        Query the YouTube Data API and return (title, description).
        """
        params = {
            "part": "snippet",
            "id": video_id,
            "maxResults": 1,
            "key": Config.GCLOUD_API_KEY,
        }
        api_url = "https://www.googleapis.com/youtube/v3/videos"

        async with session.get(api_url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        items = data.get("items", [])
        if not items:
            raise ValueError(f"No metadata returned for video ID {video_id}")

        snippet = items[0]["snippet"]
        return snippet.get("title", "Untitled"), snippet.get("description", "")

    # ---------- public contract -------------------------------------- #

    @staticmethod
    async def handle(urls: List[str]) -> List[str]:
        """
        Process a batch of YouTube URLs and return descriptive fragments.
        """
        logger.info("Processing %d YouTube URLs", len(urls))

        async with aiohttp.ClientSession() as session:

            async def _process(url: str) -> str:
                try:
                    video_id = YouTubeHandler._extract_video_id(url)
                    if not video_id:
                        raise ValueError("Invalid YouTube URL")

                    title, desc = await YouTubeHandler._fetch_video_metadata(
                        session, video_id
                    )
                    return f"[youtube] {title} — {desc}"

                except Exception as e:
                    logger.warning("Failed to process YouTube at %s: %s", url, e)
                    return f"[youtube] {url} — (error: {e})"

            return await asyncio.gather(*(_process(u) for u in urls))
