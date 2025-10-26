"""
Auto-discovery & registry for media-slice handlers.

Any file inside ``formatter/handlers/`` that defines::

    from . import register

    @register
    class MyHandler:
        media_type = "video"
        needs_message = False  # set to True to receive the Discord message
        @staticmethod
        async def handle(slice_data): ...

is picked up automatically at import-time.

Adding a new handler requires changes across the app:

1. Give it a unique ``media_type`` and place the module in this directory.
2. Append that type to ``ORDER`` in ``formatter/composer.py`` so it runs.
3. Teach ``formatter/classifier.py`` to populate the new slice from messages.
4. Define a matching ``Fragment`` subclass and extend ``FragmentType`` in
   ``formatter/model.py`` (export from ``formatter/__init__.py`` if needed).
5. Update ``response/prompt.py``'s ``MESSAGE_SCHEMA`` to document the new
   fragment shape.
6. Update the RAG pipeline: implement ``content_text`` on the fragment class so
   embeddings capture its content, and adjust ``memory/rag/media_id.py`` if the
   fragment needs special stable-id logic.
"""

from __future__ import annotations
from typing import Protocol, Dict, Any, ClassVar
from discord import Message
from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path
from ..model import Fragment

# ------------------------------------------------------------------ #
# 1.  Registry contract + decorator
# ------------------------------------------------------------------ #


class SliceHandler(Protocol):
    media_type: ClassVar[str]  # unique key, e.g. "gif"
    needs_message: ClassVar[bool]

    @staticmethod
    async def handle(slice_data, message: Message | None = ...) -> list[Fragment]:
        """Coroutine returning ``Fragment`` instances."""


_REGISTRY: Dict[str, SliceHandler] = {}


def register(cls: SliceHandler):
    """
    Decorator that stores the handler in the global registry.

    :param cls: Handler class to register.
    :returns: The class unchanged.
    """
    _REGISTRY[cls.media_type] = cls
    return cls


def get(media_type: str) -> SliceHandler | None:
    """Return handler class for ``media_type`` or ``None``."""
    return _REGISTRY.get(media_type)


# ------------------------------------------------------------------ #
# 2.  Auto-import every sibling module (plug-n-play)
# ------------------------------------------------------------------ #

_pkg_path = Path(__file__).resolve().parent
for _, modname, _ in iter_modules([str(_pkg_path)]):
    if modname != "__init__":
        import_module(f"{__name__}.{modname}")


