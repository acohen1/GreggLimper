"""
YouTubeHandler Pipeline
===========
1. Input slice : List[str] (URLs not already claimed by GIF / image logic)
2. For each URL
    a. Call YouTube API to get video details.
    b. Build fragment:  [youtube] <title> — <description>
3. Return list[str] with one line per YouTube video.
"""

import asyncio
from typing import List
from urllib.parse import urlparse, parse_qs
import aiohttp
from gregg_limper.config import Config

# TODO: Grab thumbnail image and pass through image describer


class YouTubeHandler:
    # ---------- low-level helpers -------------------------------------- #

    @staticmethod
    def _extract_video_id(url: str) -> str | None:
        """
        Extracts the YouTube video ID from a URL.
        Supports both youtube.com and youtu.be formats.
        """
        parsed = urlparse(url)
        if "youtu.be" in parsed.netloc:
            return parsed.path.lstrip("/")
        return parse_qs(parsed.query).get("v", [None])[0]

    @staticmethod
    async def _fetch_video_metadata(
        session: aiohttp.ClientSession,
        video_id: str,
        original_url: str,
    ) -> str:
        """
        Queries the YouTube Data API for video metadata by video ID.
        Returns a formatted string for display.
        """
        params = {
            "part": "snippet",
            "id": video_id,
            "maxResults": 1,
            "key": Config.GCLOUD_API_KEY,
        }
        api_url = "https://www.googleapis.com/youtube/v3/videos"
        async with session.get(api_url, params=params) as resp:
            if resp.status != 200:
                return f"[youtube] Failed to fetch for {original_url}: {resp.status}"
            data = await resp.json()
            items = data.get("items", [])
            if not items:
                return f"[youtube] No metadata found for {original_url}"
            title = items[0]["snippet"].get("title", "Untitled")
            description = items[0]["snippet"].get("description", "")
            return f"[youtube] {title} — {description}"

    # ---------- public contract ---------------------------------------- #

    @staticmethod
    async def handle(urls: List[str]) -> List[str]:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in urls:
                video_id = YouTubeHandler._extract_video_id(url)
                if not video_id:
                    tasks.append(asyncio.sleep(0, result=f"[youtube] Invalid URL: {url}"))
                else:
                    tasks.append(
                        YouTubeHandler._fetch_video_metadata(session, video_id, url)
                    )
            return await asyncio.gather(*tasks)
