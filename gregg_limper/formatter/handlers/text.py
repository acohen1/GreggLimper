"""
TextHandler Pipeline
===========
1. Input: str (plain text message)
2. Output: List[str] (single-element list if input is non-empty)
"""

from __future__ import annotations
from typing import List
from . import register

@register
class TextHandler:
    media_type = "text"

    @staticmethod
    async def handle(text: str) -> List[str]:
        """
        Simple passthrough for plain message text.
        Returns a single-element list if text is non-empty.
        """
        stripped = text.strip()
        return [stripped] if stripped else []
