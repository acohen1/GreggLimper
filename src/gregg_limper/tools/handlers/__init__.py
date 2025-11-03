"""
Import side-effects for tool handlers.

Every module imported here should register itself with the tool registry using
the :func:`gregg_limper.tools.register_tool` decorator.
"""

from __future__ import annotations

# Import concrete tools so module-level decorators execute on import.
from . import rag  # noqa: F401
