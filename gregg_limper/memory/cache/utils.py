"""
Internal helpers shared within the cache package.

This module currently provides logging utilities that summarize fragment
payloads when the cache records memo activity. The functions are intentionally
prefixed with underscores to signal that they are not part of the public API.
"""

import textwrap

def _frag_summary(frag, *, width: int = 20) -> str:
    """Return a compact one-line summary for logs: e.g., text:'Hello…'."""
    d = frag.to_llm()  # lean dict
    t = d.get("type", "?")
    val = d.get("description") or d.get("caption") or d.get("title") or ""
    if not val:
        return t
    return f"{t}:'{textwrap.shorten(str(val), width=width, placeholder='…')}'"


def _frags_preview(frags, *, width_each: int = 20, max_total_chars: int = 200) -> str:
    """Join multiple summaries and cap total length to avoid noisy logs."""
    parts = []
    total = 0
    for f in frags:
        s = _frag_summary(f, width=width_each)
        # Once the preview budget is spent, bail early with an ellipsis marker.
        if total + len(s) + (2 if parts else 0) > max_total_chars:
            parts.append("…")
            break
        parts.append(s)
        total += len(s) + (2 if parts else 0)
    return ", ".join(parts)
