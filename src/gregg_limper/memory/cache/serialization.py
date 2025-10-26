"""
Memo serialization helpers.

Expose utilities for converting memo payloads into structures consumed by
callers. 

:func:`serialize` transforms memo fragments into the compact LLM view
or full-fidelity dictionaries, while :func:`copy_memo_entry` returns a shallow
copy suitable for mutation by callers without affecting the memo store.
"""

from __future__ import annotations

from typing import Literal

Mode = Literal["llm", "full", "markdown"]


def serialize(cache_msg: dict, mode: Mode) -> dict:
    """Serialize ``cache_msg`` for the requested ``mode``."""

    fragments = cache_msg.get("fragments", [])
    if mode == "llm":
        # Compact form strips fragment metadata to what downstream prompting needs.
        return {
            "author": cache_msg.get("author"),
            "fragments": [frag.to_llm() for frag in fragments],
        }
    if mode == "markdown":
        return {
            "author": cache_msg.get("author"),
            "fragments": [frag.to_markdown() for frag in fragments],
        }
    # Full form keeps every field so operators can inspect the formatter output verbatim.
    return {
        "author": cache_msg.get("author"),
        "fragments": [frag.to_dict() for frag in fragments],
    }


def copy_memo_entry(cache_msg: dict | None) -> dict:
    """Return a shallow copy of ``cache_msg`` safe for callers."""

    if cache_msg is None:
        # Provide a predictable empty skeleton for callers expecting fragment-like structures.
        return {"author": None, "fragments": []}
    return {
        "author": cache_msg.get("author"),
        "fragments": list(cache_msg.get("fragments", [])),
    }
