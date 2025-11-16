from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Sequence, Tuple

from gregg_limper.clients import oai

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

ToolTriggerDecider = Callable[["SyntheticQuery"], Awaitable[bool]]


@dataclass(slots=True)
class SyntheticQuery:
    user: str
    summary: str
    query: str
    raw_content: str


async def inject_synthetic_rag_blocks(
    messages: Sequence[dict],
    *,
    decider: ToolTriggerDecider | None = None,
) -> Tuple[List[dict], int]:
    """
    Examine relabeled messages and inject training-only retrieve_context tool calls.

    This function must:
        - detect lore callbacks worthy of the memory tool
        - optionally confirm intent with an LLM decider
        - insert an assistant message with "tool_calls" metadata
        - append the fake tool response (role="tool") that mirrors production schema
    """

    augmented: list[dict] = []
    pending_query: SyntheticQuery | None = None
    synthetic_count = 0

    for entry in messages:
        role = entry.get("role")
        content = entry.get("content", "")

        if role == "user":
            require_specific = decider is None
            candidate = _maybe_trigger_query(
                content, require_specific=require_specific
            )
            if candidate and decider:
                try:
                    should_use = await decider(candidate)
                except Exception:
                    logger.warning(
                        "Tool trigger decider failed; falling back to heuristics.",
                        exc_info=True,
                    )
                    should_use = True
                if not should_use:
                    candidate = None
            pending_query = candidate
        elif role == "assistant" and pending_query:
            tool_id = f"synth-call-{synthetic_count + 1}"
            augmented.extend(_build_tool_sequence(tool_id, pending_query))
            entry = dict(entry)
            entry["content"] = _append_archive_note(content, pending_query.summary)
            synthetic_count += 1
            pending_query = None

        augmented.append(entry)

    return augmented, synthetic_count


def build_llm_tool_trigger_decider(model_id: str) -> ToolTriggerDecider:
    async def _decide(candidate: SyntheticQuery) -> bool:
        prompt = (
            "Judge whether a Discord user is referencing a concrete past event that "
            "Gregg Limper should look up via the retrieve_context tool.\n"
            "Return ONLY 'yes' or 'no'."
        )
        response = await oai.chat(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "User message:\n"
                        f"{candidate.raw_content}\n\n"
                        "If the user is recalling lore or an earlier conversation with "
                        "specific details, answer 'yes'. Otherwise answer 'no'."
                    ),
                },
            ],
            model=model_id,
        )
        return response.strip().lower().startswith("y")

    return _decide


def _maybe_trigger_query(
    user_content: str, *, require_specific: bool = True
) -> SyntheticQuery | None:
    lowered = user_content.lower()
    if not any(keyword in lowered for keyword in TRIGGER_KEYWORDS):
        return None

    name, body = _split_user_message(user_content)
    if require_specific and not _looks_specific(body):
        return None

    summary = body.splitlines()[0].strip() if body else "earlier conversation"
    summary = summary[:140]
    query = f"memory lookup for {name}: {summary}"
    return SyntheticQuery(
        user=name,
        summary=summary,
        query=query,
        raw_content=user_content.strip(),
    )


def _split_user_message(content: str) -> tuple[str, str]:
    marker = " said:\n"
    if marker in content:
        name, rest = content.split(marker, 1)
        return name.strip() or "user", rest.strip()
    return "user", content.strip()


def _looks_specific(text: str) -> bool:
    if not text:
        return False
    normalized = text.strip()
    tokens = normalized.split()
    if len(tokens) >= 9:
        return True
    if re.search(r"\b\d+\b", normalized):
        return True
    if re.search(r'"[^"]{3,}"', normalized):
        return True
    proper_nouns = sum(
        1 for token in tokens if token[:1].isupper() and len(token) > 2
    )
    return proper_nouns >= 2


def _build_tool_sequence(tool_call_id: str, payload: SyntheticQuery) -> List[dict]:
    arguments = json.dumps({"query": payload.query, "k": 3}, ensure_ascii=False)

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
        "content": f"1. {payload.summary} (author: {payload.user}, message: synthetic-memory)",
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


__all__ = ["inject_synthetic_rag_blocks", "build_llm_tool_trigger_decider"]
