"""
TextHandler Pipeline
====================
1. Input  : str  (raw message text)
2. Output : List[dict] with a single record, e.g.:
      { "type": "text", "content": "<original text>" }

NOTE: Plain passthrough for now. Future steps might include username processing or other transformations.
"""

from __future__ import annotations
from typing import List, Dict
from . import register

@register
class TextHandler:
    media_type = "text"

    @staticmethod
    async def handle(text: str) -> List[Dict[str, str]]:
        """
        Wrap non-empty message text in a media-record dict.
        """
        stripped = text.strip()
        return [{"type": "text", "content": stripped}] if stripped else []
