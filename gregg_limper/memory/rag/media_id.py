"""
media_id.py
===========

Stable identifiers for media fragments.

Functions here generate deterministic IDs for assets
(e.g. YouTube videos, Discord CDN attachments, generic URLs).
"""

from __future__ import annotations
from typing import Dict, Optional, Any
from urllib.parse import urlparse, parse_qs, parse_qsl, urlencode
import re
from .embeddings import blake16

_YT_RX = re.compile(r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|v/))([A-Za-z0-9_-]{6,})", re.I)

def _normalize_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    try:
        p = urlparse(u)
        netloc = p.netloc.lower()
        path = re.sub(r"/+", "/", p.path)
        norm = f"{p.scheme.lower()}://{netloc}{path}"

        qs = parse_qsl(p.query, keep_blank_values=True)
        if qs:
            qs = sorted(qs)
            query = urlencode(qs)
            return f"{norm}?{query}"
        return norm
    except Exception:
        return u

def _youtube_id(url: str) -> Optional[str]:
    m = _YT_RX.search(url or "")
    if m: return m.group(1)
    p = urlparse(url or "")
    if p.netloc.lower() in {"www.youtube.com", "youtube.com"}:
        return parse_qs(p.query).get("v", [None])[0]
    return None

def _discord_cdn_path(url: str) -> Optional[str]:
    if not url: return None
    p = urlparse(url)
    host = p.netloc.lower()
    if host.endswith("discordapp.com") or host.endswith("discord.com"):
        return re.sub(r"^/+", "", p.path)
    return None

def stable_media_id(
    *,
    cf: Dict[str, Any],
    server_id: int,
    channel_id: int,
    message_id: int,
    source_idx: int,
) -> str:
    """
    Deterministic, namespaced media id:

    Priority:
      1) youtube:<video_id>          (if type=youtube and URL present)
      2) dc:<attachments/...>        (Discord CDN path)
      3) url:<blake16(normalized_url)>
      4) msg:<message_id>:<idx>:<type> (for text/no-url fragments)
      5) fallback:<blake16(all provenance)>

    This is stable across reposts when the underlying asset (e.g., YouTube ID or CDN path) is the same.
    """
    typ = (cf.get("type") or "").strip()
    url = _normalize_url(cf.get("url"))

    # YouTube explicit ID
    if typ == "youtube" and url:
        yid = _youtube_id(url)
        if yid:
            return f"yt:{yid}"

    # Discord CDN attachments (works for images/gifs/files posted in Discord)
    if url:
        dc_path = _discord_cdn_path(url)
        if dc_path:
            return f"dc:{dc_path}"

        # Generic URL-stable id
        return f"url:{blake16(url)}"

    # No URL (e.g., raw text fragments)
    if typ == "text":
        return f"msg:{message_id}:{source_idx}:{typ}"

    # Absolute fallback (should rarely fire)
    prov = f"{server_id}:{channel_id}:{message_id}:{source_idx}:{typ}:{cf.get('title') or ''}"
    return f"fallback:{blake16(prov)}"
