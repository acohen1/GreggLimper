from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Awaitable, Callable, Dict, Iterable, List, Sequence, Set, cast

from discord import Message

from gregg_limper.clients import oai

from ..config import DatasetBuildConfig
from .types import RawConversation, SegmentCandidate, SegmentedConversation

logger = logging.getLogger(__name__)

MAX_GAP_SECONDS = 180
MAX_SEGMENT_MESSAGES = 40
MIN_SEGMENT_MESSAGES = 4
MAX_PARTICIPANTS = 5
_URL_RE = re.compile(r"https?://\S+")
_CUSTOM_EMOJI_RE = re.compile(r"<a?:([\w\d_]+):\d+>")


def propose_segments(conversations: Iterable[RawConversation]) -> List[SegmentCandidate]:
    """
    Generate deterministic segment candidates using timestamp gaps and participant overlap.

    This pass should be conservative: it never merges across large gaps or channel switches
    and serves purely as input to the LLM boundary verifier.
    """

    candidates: list[SegmentCandidate] = []
    for convo in conversations:
        if not convo.messages:
            continue

        current_ids: list[int] = []
        participants: set[int] = set()
        last_timestamp: datetime | None = None
        segment_count_before = len(candidates)

        for message in convo.messages:
            created = _ensure_timezone(message.created_at)
            author_id = getattr(message.author, "id", None)

            should_break = False
            if last_timestamp is not None:
                gap = (created - last_timestamp).total_seconds()
                if gap > MAX_GAP_SECONDS:
                    should_break = True
                elif len(current_ids) >= MAX_SEGMENT_MESSAGES:
                    should_break = True
                elif (
                    author_id is not None
                    and author_id not in participants
                    and len(participants) >= MAX_PARTICIPANTS
                ):
                    should_break = True

            if should_break and len(current_ids) >= MIN_SEGMENT_MESSAGES:
                candidates.append(
                    SegmentCandidate(
                        channel_id=convo.channel_id,
                        message_ids=current_ids,
                    )
                )
                current_ids = []
                participants = set()

            current_ids.append(message.id)
            if author_id is not None:
                participants.add(author_id)
            last_timestamp = created

        if len(current_ids) >= MIN_SEGMENT_MESSAGES:
            candidates.append(
                SegmentCandidate(
                    channel_id=convo.channel_id,
                    message_ids=current_ids,
                )
            )

        produced = len(candidates) - segment_count_before
        if produced:
            logger.info(
                "Segmenter: channel %s produced %d candidates (total %d)",
                convo.channel_id,
                produced,
                len(candidates),
            )

    logger.info("Proposed %s raw segments", len(candidates))
    return candidates


def drop_ineligible_candidates(
    candidates: Iterable[SegmentCandidate],
    *,
    message_lookup: Dict[int, Message],
    config: DatasetBuildConfig,
) -> tuple[list[SegmentCandidate], int]:
    """
    Remove segment candidates that cannot possibly yield a valid assistant.

    Ensures we do not send clearly invalid spans to the LLM by applying the
    same deterministic eligibility checks used inside refinement.
    """

    allowed = set(getattr(config, "allowed_user_ids", set()))
    whitelist = config.allowed_assistant_custom_emojis

    filtered: list[SegmentCandidate] = []
    pruned = 0

    for candidate in candidates:
        records = [
            message_lookup[mid]
            for mid in candidate.message_ids
            if mid in message_lookup
        ]
        if len(records) < MIN_SEGMENT_MESSAGES:
            pruned += 1
            continue

        eligible = _eligible_assistant_ids(
            records,
            assigned_ids=allowed,
            emoji_whitelist=whitelist,
        )
        if not eligible:
            pruned += 1
            continue

        filtered.append(candidate)

    return filtered, pruned


SegmentDecider = Callable[
    [Sequence[Message], Sequence[int]],
    Awaitable["_SegmentDecision | None"],
]


async def refine_segments_with_llm(
    candidates: Iterable[SegmentCandidate],
    *,
    message_lookup: Dict[int, Message],
    config: DatasetBuildConfig,
    decide_segment: SegmentDecider | None = None,
) -> List[SegmentedConversation]:
    """
    Send each candidate to an LLM that confirms/adjusts boundaries and selects the proxy assistant.

    The LLM is required to:
        - enforce alternating user/assistant turns
        - drop low-signal chunks
        - pick a speaker from the allowed list to represent Gregg
    """

    refined: list[SegmentedConversation] = []
    allowed = set(getattr(config, "allowed_user_ids", set()))
    default_decider: SegmentDecider | None = None
    model_id = getattr(config, "segment_decider_model", None)
    leniency = getattr(config, "segment_decider_leniency", "default") or "default"
    if model_id:
        default_decider = cast(
            SegmentDecider,
            partial(_llm_decide, model_id=model_id, leniency=leniency),
        )

    candidates = list(candidates)
    total_candidates = len(candidates)
    log_interval = max(1, total_candidates // 20) if total_candidates else 1
    active_decider = decide_segment or default_decider
    allow_fallback = decide_segment is None and default_decider is not None

    if active_decider:
        concurrency = max(1, getattr(config, "segment_decider_concurrency", 4))
        semaphore = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        processed = 0

        async def mark_progress() -> None:
            nonlocal processed
            async with lock:
                processed += 1
                if processed % log_interval == 0 or processed == total_candidates:
                    logger.info(
                        "Segmenter: refined %d/%d candidates (approved %d)",
                        processed,
                        total_candidates,
                        len(refined),
                    )

        async def evaluate(candidate: SegmentCandidate) -> None:
            records = [
                message_lookup[mid]
                for mid in candidate.message_ids
                if mid in message_lookup
            ]
            if len(records) < MIN_SEGMENT_MESSAGES:
                await mark_progress()
                return

            eligible_assistants = _eligible_assistant_ids(
                records,
                assigned_ids=allowed,
                emoji_whitelist=config.allowed_assistant_custom_emojis,
            )
            if not eligible_assistants:
                await mark_progress()
                return

            decision = None
            async with semaphore:
                try:
                    decision = await active_decider(records, eligible_assistants)
                except Exception:
                    logger.warning(
                        "LLM boundary evaluation failed for channel %s; skipping.",
                        candidate.channel_id,
                        exc_info=True,
                    )
                    decision = None

            if decision is None and allow_fallback:
                decision = _heuristic_decide(records, eligible_assistants, leniency=leniency)

            if decision is not None:
                refined.append(
                    SegmentedConversation(
                        channel_id=candidate.channel_id,
                        message_ids=decision.message_ids,
                        assigned_assistant_id=decision.assistant_id,
                    )
                )

            await mark_progress()

        await asyncio.gather(*(evaluate(candidate) for candidate in candidates))

    else:
        processed = 0
        for candidate in candidates:
            records = [
                message_lookup[mid]
                for mid in candidate.message_ids
                if mid in message_lookup
            ]
            if len(records) < MIN_SEGMENT_MESSAGES:
                processed += 1
                continue

            eligible_assistants = _eligible_assistant_ids(
                records,
                assigned_ids=allowed,
                emoji_whitelist=config.allowed_assistant_custom_emojis,
            )
            if not eligible_assistants:
                processed += 1
                continue

            decision = _heuristic_decide(records, eligible_assistants, leniency=leniency)
            if decision is None:
                processed += 1
                continue

            refined.append(
                SegmentedConversation(
                    channel_id=candidate.channel_id,
                    message_ids=decision.message_ids,
                    assigned_assistant_id=decision.assistant_id,
                )
            )

            processed += 1
            if processed % log_interval == 0 or processed == total_candidates:
                logger.info(
                    "Segmenter: refined %d/%d candidates (approved %d)",
                    processed,
                    total_candidates,
                    len(refined),
                )

    logger.info("LLM approved %s segments", len(refined))
    return refined


@dataclass(slots=True)
class _SegmentDecision:
    message_ids: List[int]
    assistant_id: int


async def _llm_decide(
    messages: Sequence[Message],
    allowed_user_ids: Sequence[int],
    *,
    model_id: str,
    leniency: str = "default",
) -> _SegmentDecision | None:
    allowed_text = ", ".join(str(uid) for uid in sorted(allowed_user_ids)) or "ANY"
    transcript = _format_for_llm(messages)

    if leniency == "strict":
        prompt = (
            "You are curating Discord snippets for supervised finetuning.\n"
            "Goal:\n"
            "1. Keep only spans that read like a coherent conversation between a user cohort and one assistant persona.\n"
            "2. The assistant must be one of the allowed user IDs and should speak at least three times with meaningful replies.\n"
            "3. Preserve chronological order of the kept message IDs.\n"
            "4. Drop the snippet if it contains reaction spam, single-word replies, or lacks clear turns.\n"
            "\nRespond ONLY with minified JSON matching:\n"
            '{"keep": bool, "assistant_id": int, "message_ids": [int, ...]}.\n'
            "If you reject the snippet, set keep=false and leave the other fields empty arrays/defaults."
        )
    elif leniency == "lenient":
        prompt = (
            "You are curating Discord snippets for supervised finetuning.\n"
            "Goal:\n"
            "1. Prefer spans that read like a coherent conversation between a user cohort and one assistant persona.\n"
            "2. The assistant must be one of the allowed user IDs and should speak at least once; a single assistant reply is acceptable, ideally as the final turn.\n"
            "3. Preserve chronological order of the kept message IDs.\n"
            "4. Allow short or reaction-like replies unless the entire snippet is reaction spam with no real conversation.\n"
            "\nRespond ONLY with minified JSON matching:\n"
            '{"keep": bool, "assistant_id": int, "message_ids": [int, ...]}.\n'
            "If you reject the snippet, set keep=false and leave the other fields empty arrays/defaults."
        )
    else:
        prompt = (
            "You are curating Discord snippets for supervised finetuning.\n"
            "Goal:\n"
            "1. Keep only spans that read like a coherent conversation between a user cohort and one assistant persona.\n"
            "2. The assistant must be one of the allowed user IDs and should speak at least twice with meaningful replies.\n"
            "3. Preserve chronological order of the kept message IDs.\n"
            "4. Drop the snippet entirely if it devolves into single-word replies, pure reaction spam, or lacks clear turns.\n"
            "\nRespond ONLY with minified JSON matching:\n"
            '{"keep": bool, "assistant_id": int, "message_ids": [int, ...]}.\n'
            "If you reject the snippet, set keep=false and leave the other fields empty arrays/defaults."
        )
    response = await oai.chat(
        [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Allowed assistant user IDs: {allowed_text}\n"
                    f"Conversation:\n{transcript}"
                ),
            },
        ],
        model=model_id,
    )

    payload = _parse_json_response(response)
    if not payload or not payload.get("keep"):
        return None

    assistant_id = int(payload.get("assistant_id", 0))
    if allowed_user_ids and assistant_id not in allowed_user_ids:
        return None

    included_ids = payload.get("message_ids") or []
    allowed_ids = {msg.id for msg in messages}
    filtered_ids = [mid for mid in included_ids if mid in allowed_ids]
    if len(filtered_ids) < MIN_SEGMENT_MESSAGES:
        return None

    return _SegmentDecision(message_ids=filtered_ids, assistant_id=assistant_id)


def _heuristic_decide(
    messages: Sequence[Message],
    allowed_user_ids: Sequence[int],
    *,
    leniency: str = "default",
) -> _SegmentDecision | None:
    min_assistant_turns = {"strict": 3, "lenient": 1}.get(leniency, 2)
    allowed = set(allowed_user_ids)
    counts: Counter[int] = Counter()
    for msg in messages:
        speaker = getattr(msg.author, "id", None)
        if speaker is None:
            continue
        if allowed and speaker not in allowed:
            continue
        counts[speaker] += 1

    if not counts:
        return None

    assistant_id, turns = counts.most_common(1)[0]
    if turns < min_assistant_turns:
        return None

    message_ids = [msg.id for msg in messages]
    return _SegmentDecision(message_ids=message_ids, assistant_id=assistant_id)


def _eligible_assistant_ids(
    messages: Sequence[Message],
    *,
    assigned_ids: Sequence[int],
    emoji_whitelist: Sequence[str],
) -> set[int]:
    assigned_set = set(assigned_ids or [])
    whitelist = {
        _normalize_custom_emoji_token(token)
        for token in (emoji_whitelist or [])
        if _normalize_custom_emoji_token(token)
    }

    seen_authors: set[int] = set()
    disqualified_media: set[int] = set()
    disqualified_emoji: set[int] = set()
    max_consecutive: dict[int, int] = {}

    last_author: int | None = None
    streak = 0

    for message in messages:
        author_id = getattr(message.author, "id", None)
        if author_id is None or (assigned_set and author_id not in assigned_set):
            last_author = None
            streak = 0
            continue

        seen_authors.add(author_id)

        if author_id == last_author:
            streak += 1
        else:
            last_author = author_id
            streak = 1
        max_consecutive[author_id] = max(
            streak, max_consecutive.get(author_id, 0)
        )

        if _message_has_media(message):
            disqualified_media.add(author_id)

        if _contains_disallowed_custom_emoji(message, whitelist):
            disqualified_emoji.add(author_id)

    eligible = {
        author_id
        for author_id in seen_authors
        if max_consecutive.get(author_id, 0) <= 1
        and author_id not in disqualified_media
        and author_id not in disqualified_emoji
    }
    return eligible


def _message_has_media(message: Message) -> bool:
    attachments = getattr(message, "attachments", None) or []
    embeds = getattr(message, "embeds", None) or []
    stickers = getattr(message, "stickers", None) or []
    if attachments or embeds or stickers:
        return True

    fragments = getattr(message, "fragments", None)
    if fragments:
        for frag in fragments:
            frag_type = (frag.get("type") or "text").lower()
            if frag_type != "text":
                return True

    classified = getattr(message, "classified_fragments", None)
    if isinstance(classified, dict):
        for key, bucket in classified.items():
            if key.lower() != "text" and bucket:
                return True

    content = (getattr(message, "clean_content", None) or message.content or "").strip()
    if not content:
        return False

    if _URL_RE.search(content):
        return True
    return False


def _contains_disallowed_custom_emoji(
    message: Message, whitelist: set[str]
) -> bool:
    content = (getattr(message, "clean_content", None) or message.content or "") or ""
    matches = _CUSTOM_EMOJI_RE.findall(content)
    if not matches:
        return False

    if not whitelist:
        return True

    for token in matches:
        if _normalize_custom_emoji_token(token) not in whitelist:
            return True
    return False


def _normalize_custom_emoji_token(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return ""
    match = _CUSTOM_EMOJI_RE.fullmatch(token)
    if match:
        return match.group(1).lower()

    trimmed = token.strip("<>")
    parts = [part for part in trimmed.split(":") if part]
    if len(parts) >= 2:
        # Handles strings like "a:emoji:123456"
        candidate = parts[1]
    else:
        candidate = parts[0]
    return candidate.strip(":").lower()


def _format_for_llm(messages: Sequence[Message]) -> str:
    lines: list[str] = []
    for idx, msg in enumerate(messages, start=1):
        author = getattr(msg.author, "display_name", None) or getattr(
            msg.author, "name", f"user-{msg.author.id}"
        )
        content = (msg.clean_content or msg.content or "").strip()
        if not content:
            content = "(non-text message)"
        created = _ensure_timezone(msg.created_at).isoformat()
        lines.append(
            f"{idx}. {msg.id} | {created} | {author} ({msg.author.id}): {content}"
        )
    return "\n".join(lines)


def _parse_json_response(raw: str) -> dict | None:
    text = raw.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def _ensure_timezone(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = ["propose_segments", "refine_segments_with_llm", "drop_ineligible_candidates"]
