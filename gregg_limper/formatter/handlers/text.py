"""
TextHandler Pipeline
====================
1. Input  : str  (raw message text)
2. Output : List[TextFragment] with a single record, e.g.:
      ``TextFragment(description="<original text>")``

NOTE: Plain passthrough for now. Future steps might include username
processing or other transformations.
"""

from __future__ import annotations
from typing import List
from . import register
from ..model import TextFragment

@register
class TextHandler:
    media_type = "text"

    @staticmethod
    async def handle(text: str) -> List[TextFragment]:
        """Wrap non-empty message text in a :class:`TextFragment`.

        :param text: Raw message content.
        :returns: List containing a single fragment or empty list.
        """
        stripped = text.strip()
        return [TextFragment(description=stripped)] if stripped else []

