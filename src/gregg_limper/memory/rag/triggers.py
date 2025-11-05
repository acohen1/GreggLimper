"""
Helpers for identifying emoji reactions that should trigger RAG ingestion.

The trigger set is configurable via the ``RAG_REACTION_EMOJIS`` environment
variable, which accepts a comma-separated list of emoji descriptors. Each entry
may be one of:

* A standard Unicode emoji literal, e.g. ``🧠``.
* A Discord custom emoji spec like ``<:brain:1234567890>`` or ``<a:brain:123...>``.
* A ``name:id`` pair (``brain:1234567890``) to improve readability.
* A bare numeric ID (``1234567890``) or bare name (``brain``) as a fallback.

This module normalizes those descriptors into lookup sets that can be safely
matched against ``discord.Reaction.emoji`` payloads in both live hooks and
offline backfill routines.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Iterable, Sequence

import discord

from gregg_limper.config import rag as rag_cfg

logger = logging.getLogger(__name__)

_CUSTOM_FULL_RE = re.compile(r"^<a?:(?P<name>[\w~\-]+):(?P<id>\d+)>$")
_NAMED_ID_RE = re.compile(r"^(?P<name>[\w~\-]+):(?P<id>\d+)$")
_NUMERIC_ID_RE = re.compile(r"^\d+$")
_NAME_RE = re.compile(r"^[\w~\-]+$")


@dataclass(frozen=True)
class TriggerSet:
    """Normalized emoji triggers split across Unicode glyphs and custom ids/names."""

    unicode_emojis: frozenset[str]
    custom_ids: frozenset[int]
    custom_names: frozenset[str]

    def is_empty(self) -> bool:
        return not (self.unicode_emojis or self.custom_ids or self.custom_names)


def _parse_trigger_entry(entry: str) -> tuple[str | None, int | None, str | None]:
    text = entry.strip()
    if not text:
        return None, None, None

    match = _CUSTOM_FULL_RE.match(text)
    if match:
        return None, int(match.group("id")), match.group("name")

    match = _NAMED_ID_RE.match(text)
    if match:
        return None, int(match.group("id")), match.group("name")

    if _NUMERIC_ID_RE.match(text):
        return None, int(text), None

    if text.startswith(":") and text.endswith(":") and len(text) > 2:
        return None, None, text[1:-1]

    if _NAME_RE.match(text):
        return None, None, text

    # Fallback: treat as Unicode emoji literal (multi-codepoint safe)
    return text, None, None


def build_trigger_set(entries: Sequence[str]) -> TriggerSet:
    unicode_values: set[str] = set()
    custom_ids: set[int] = set()
    custom_names: set[str] = set()

    for raw in entries:
        unicode_val, emoji_id, name = _parse_trigger_entry(raw)
        if unicode_val:
            unicode_values.add(unicode_val)
        if emoji_id is not None:
            custom_ids.add(emoji_id)
        if name:
            custom_names.add(name)

    return TriggerSet(
        unicode_emojis=frozenset(unicode_values),
        custom_ids=frozenset(custom_ids),
        custom_names=frozenset(custom_names),
    )


_CACHED_TRIGGERS: TriggerSet | None = None


def get_trigger_set(force_refresh: bool = False) -> TriggerSet:
    """
    Return the configured trigger set, refreshing from config when requested.
    """

    global _CACHED_TRIGGERS
    if _CACHED_TRIGGERS is None or force_refresh:
        _CACHED_TRIGGERS = build_trigger_set(rag_cfg.REACTION_TRIGGERS)
        if _CACHED_TRIGGERS.is_empty():
            logger.info("RAG reaction triggers not configured; reaction ingestion disabled.")
    return _CACHED_TRIGGERS


def emoji_matches_trigger(
    emoji: discord.PartialEmoji | discord.Emoji | str,
    triggers: TriggerSet | None = None,
) -> bool:
    """
    Return True if ``emoji`` matches the configured trigger set.
    """

    triggers = triggers or get_trigger_set()
    if triggers.is_empty():
        return False

    if isinstance(emoji, str):
        return emoji in triggers.unicode_emojis

    emoji_id = getattr(emoji, "id", None)
    if emoji_id is not None and emoji_id in triggers.custom_ids:
        return True

    emoji_name = getattr(emoji, "name", None)
    if emoji_name and emoji_name in triggers.custom_names:
        return True

    # PartialEmoji.__str__ returns "<:name:id>" which may appear in config entries.
    emoji_repr = str(emoji)
    return emoji_repr in triggers.unicode_emojis


def message_has_trigger_reaction(
    message: discord.Message,
    *,
    triggers: TriggerSet | None = None,
) -> bool:
    """
    Check whether ``message`` already has a reaction that matches the trigger set.
    """

    triggers = triggers or get_trigger_set()
    if triggers.is_empty():
        return False

    reactions: Iterable[discord.Reaction] = getattr(message, "reactions", [])
    for reaction in reactions:
        if emoji_matches_trigger(reaction.emoji, triggers):
            return True
    return False


__all__ = [
    "TriggerSet",
    "build_trigger_set",
    "get_trigger_set",
    "emoji_matches_trigger",
    "message_has_trigger_reaction",
]
