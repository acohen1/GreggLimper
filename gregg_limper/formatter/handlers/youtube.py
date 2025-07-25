"""
YouTubeHandler Pipeline
===========
1. Input slice : List[str] (URLs not already claimed by GIF / image logic)
2. For each URL
    a. Call YouTube API to get video details.
    b. Build fragment dict:
       { "type": "youtube", "title": "<title>", "description": "<description>", "thumbnail": "<thumbnail description>" }
3. Return List[dict] with one dict per URL

NOTE: We tolerate thumbnail-vision failures here and keep title/description.
"""

import asyncio
from typing import List, Tuple
from urllib.parse import urlparse, parse_qs
from . import register
import aiohttp
from gregg_limper.config import Config
from gregg_limper.client_oai import describe_image_bytes

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
        session: aiohttp.ClientSession, 
        video_id: str,
        thumbnail_size: str = "medium"  # "default", "medium", "high"
    ) -> Tuple[str, str, str]:
        """
        Query the YouTube Data API and return (title, description, thumbnail url).
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

        title = snippet.get("title", "Untitled")
        description = snippet.get("description", "")
        thumbnail_url = snippet.get("thumbnails", {}).get(thumbnail_size, {}).get("url", "")
        return title, description, thumbnail_url
    
    @staticmethod
    async def _download_image_bytes(
        session: aiohttp.ClientSession, url: str
    ) -> Tuple[bytes, str]:
        """Download thumbnail from url and return (bytes, mime-type)."""
        if not url:
            raise ValueError("Empty thumbnail URL")
        async with session.get(url) as resp:
            resp.raise_for_status()
            mime = (resp.headers.get("Content-Type") or "image/jpeg").lower().split(";")[0]
            return await resp.read(), mime

    # ---------- public contract -------------------------------------- #

    @staticmethod
    async def handle(urls: List[str]) -> List[dict]:
        """
        Process a batch of YouTube URLs and return media-record dicts.
        Each dict contains:
        - "type": "youtube"
        - "title": video title
        - "description": video description
        - "thumbnail": thumbnail description from vision model
        """
        logger.info("Processing %d YouTube URLs", len(urls))

        async with aiohttp.ClientSession() as session:

            async def _process(url: str) -> dict:
                try:
                    video_id = YouTubeHandler._extract_video_id(url)
                    if not video_id:
                        raise ValueError("Invalid YouTube URL")

                    # 1) Fetch video metadata
                    title, desc, thumbnail_url = await YouTubeHandler._fetch_video_metadata(
                        session, video_id, Config.YT_THUMBNAIL_SIZE
                    )
                    clean_desc = " ".join(desc.split())
                    max_len = Config.YT_DESC_MAX_LEN
                    desc = clean_desc[:max_len] + ("..." if len(clean_desc) > max_len else "")

                    # 2) Thumbnail -> vision model -> text description
                    try:
                        blob, mime = await YouTubeHandler._download_image_bytes(session, thumbnail_url)
                        thumb_desc = await describe_image_bytes(blob, mime=mime)
                    except Exception as e:
                        logger.warning("Failed to describe thumbnail for %s: %s", url, e)
                        thumb_desc = "(thumbnail unavailable)"

                    return {"type": "youtube", "title": title, "description": desc, "thumbnail": thumb_desc}

                except Exception as e:
                    logger.warning("Failed to process YouTube at %s: %s", url, e)
                    return {"type": "youtube", "title": url, "description": f"(error: {e})"}

            return await asyncio.gather(*(_process(u) for u in urls))
