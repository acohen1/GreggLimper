# handlers/__init__.py
"""
Auto-discovery & registry for media-slice handlers.

Any file inside formatter/handlers/ that defines:

    from . import register

    @register
    class MyHandler:
        media_type = "video"
        @staticmethod
        def handle(slice_data): ...

is picked up automatically at import-time.

If adding a new handler, ensure:
1. It has a unique `media_type` string, added to the ORDER list in composer.py.
2. It implements the `SliceHandler` protocol (see below).
3. It is placed in this directory (formatter/handlers/).
"""

from __future__ import annotations
from typing import Protocol, Dict, List, Any
from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path

# ------------------------------------------------------------------ #
# 1.  Registry contract + decorator
# ------------------------------------------------------------------ #

class SliceHandler(Protocol):
    media_type: str                          # unique key, e.g. "gif"
    @staticmethod
    async def handle(slice_data) -> list[str]:
        """
        Coroutine -> returns **a list of one-line strings** (“fragments”).

        - Each string will be separated from the next with *two* new-lines by the composer.  Don't add blank lines yourself.
        - Don't include the author prefix.
        - If you have nothing to contribute, return an empty list (not None).
        """

_REGISTRY: Dict[str, SliceHandler] = {}

def register(cls: SliceHandler):
    """Decorator that stores the handler in the global registry."""
    _REGISTRY[cls.media_type] = cls
    return cls

def get(media_type: str) -> SliceHandler | None:
    return _REGISTRY.get(media_type)

# ------------------------------------------------------------------ #
# 2.  Auto-import every sibling module (plug-n-play)
# ------------------------------------------------------------------ #

_pkg_path = Path(__file__).resolve().parent
for _, modname, _ in iter_modules([str(_pkg_path)]):
    if modname != "__init__":
        import_module(f"{__name__}.{modname}")
