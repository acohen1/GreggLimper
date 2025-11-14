from __future__ import annotations

import json
import logging
import re
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)

TRIGGER_KEYWORDS = (
    "remember",
    "remind",
    "last time",
    "you said",
    "you told",
    "as you said",
    "back when",
    "previously",
)


def inject_synthetic_rag_blocks(messages: Sequence[dict]) -> Tuple[List[dict], int]:
    """
    Examine relabeled messages and inject training-only retrieve_context tool calls.

    This function must:
        - detect lore callbacks worthy of the memory tool
        - insert an assistant message with "tool_calls" metadata
        - append the fake tool response (role="tool") that mirrors production schema
    """

    augmented: list[dict] = []
    pending_query: dict | None = None
    synthetic_count = 0

    for entry in messages:
        role = entry.get("role")
        content = entry.get("content", "")

        if role == "user":
            pending_query = _maybe_trigger_query(content)
        elif role == "assistant" and pending_query:
            tool_id = f"synth-call-{synthetic_count + 1}"
            augmented.extend(_build_tool_sequence(tool_id, pending_query))
            entry = dict(entry)
            entry["content"] = _append_archive_note(content, pending_query["summary"])
            synthetic_count += 1
            pending_query = None

        augmented.append(entry)

    return augmented, synthetic_count


def _maybe_trigger_query(user_content: str) -> dict | None:
    lowered = user_content.lower()
    if not any(keyword in lowered for keyword in TRIGGER_KEYWORDS):
        return None

    name, body = _split_user_message(user_content)
    summary = body.splitlines()[0].strip() if body else "earlier conversation"
    summary = summary[:140]

    query = f"memory lookup for {name}: {summary}"
    return {"query": query, "summary": summary, "user": name}


def _split_user_message(content: str) -> tuple[str, str]:
    marker = " said:\n"
    if marker in content:
        name, rest = content.split(marker, 1)
        return name.strip() or "user", rest.strip()
    return "user", content.strip()


def _build_tool_sequence(tool_call_id: str, payload: dict) -> List[dict]:
    query = payload["query"]
    summary = payload["summary"]
    arguments = json.dumps({"query": query, "k": 3}, ensure_ascii=False)

    assistant_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "retrieve_context",
                    "arguments": arguments,
                },
            }
        ],
    }

    tool_reply = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": "retrieve_context",
        "content": f"1. {summary} (author: {payload['user']}, message: synthetic-memory)",
    }
    return [assistant_call, tool_reply]


def _append_archive_note(content: str, summary: str) -> str:
    note = f"Checked the archive—here's what I found: {summary}"
    if not content:
        return note
    if note in content:
        return content
    separator = "\n\n" if not content.endswith("\n") else "\n"
    return f"{content}{separator}{note}"


__all__ = ["inject_synthetic_rag_blocks"]
