"""Memo serialization helpers."""

from __future__ import annotations

from typing import Literal

Mode = Literal["llm", "full"]


def serialize(cache_msg: dict, mode: Mode) -> dict:
    """Serialize ``cache_msg`` for the requested ``mode``."""

    fragments = cache_msg.get("fragments", [])
    if mode == "llm":
        return {
            "author": cache_msg.get("author"),
            "fragments": [frag.to_llm() for frag in fragments],
        }
    return {
        "author": cache_msg.get("author"),
        "fragments": [frag.to_dict() for frag in fragments],
    }


def copy_memo_entry(cache_msg: dict | None) -> dict:
    """Return a shallow copy of ``cache_msg`` safe for callers."""

    if cache_msg is None:
        return {"author": None, "fragments": []}
    return {
        "author": cache_msg.get("author"),
        "fragments": list(cache_msg.get("fragments", [])),
    }
