from __future__ import annotations

import hashlib
import re
from typing import Dict
from weakref import WeakKeyDictionary

ADJECTIVES = [
    "Amber",
    "Brisk",
    "Crimson",
    "Dusky",
    "Emerald",
    "Fuzzy",
    "Golden",
    "Hazel",
    "Ivory",
    "Jolly",
    "Kindly",
    "Lunar",
    "Mellow",
    "Nimble",
    "Obsidian",
    "Plush",
    "Quantum",
    "Rusty",
    "Sunny",
    "Teal",
    "Umber",
    "Velvet",
    "Witty",
    "Xenial",
    "Young",
    "Zephyr",
]

NOUNS = [
    "Anchor",
    "Badger",
    "Comet",
    "Drift",
    "Ember",
    "Falcon",
    "Garden",
    "Harbor",
    "Iris",
    "Jungle",
    "Koala",
    "Lagoon",
    "Meadow",
    "Nebula",
    "Otter",
    "Prairie",
    "Quill",
    "Raven",
    "Sprout",
    "Turtle",
    "Vale",
    "Willow",
    "Xylophone",
    "Yonder",
    "Zebra",
]


_PII_META_WEAK = WeakKeyDictionary()
_PII_META_STRONG: Dict[int, dict] = {}


class AliasGenerator:
    def __init__(self) -> None:
        self._cache: Dict[int, str] = {}

    def alias(self, user_id: int | None) -> str:
        if user_id is None:
            return "SolarScribe"
        if user_id in self._cache:
            return self._cache[user_id]

        digest = hashlib.sha256(str(user_id).encode("utf-8")).digest()
        adj = ADJECTIVES[digest[0] % len(ADJECTIVES)]
        noun = NOUNS[digest[1] % len(NOUNS)]
        label = f"{adj}{noun}"
        self._cache[user_id] = label
        return label


MENTION_RE = re.compile(r"<@!?([0-9]+)>")


def scrub_text(text: str, alias_fn) -> str:
    def repl(match: re.Match[str]) -> str:
        user_id = int(match.group(1))
        return alias_fn(user_id)

    return MENTION_RE.sub(repl, text)


def ensure_meta(message):
    meta = getattr(message, "_pii_meta", None)
    if isinstance(meta, dict):
        return meta
    if hasattr(message, "__dict__"):
        meta = {}
        setattr(message, "_pii_meta", meta)
        return meta
    try:
        meta = _PII_META_WEAK.get(message)
    except TypeError:
        meta = _PII_META_STRONG.get(id(message))
        if meta is None:
            meta = {}
            _PII_META_STRONG[id(message)] = meta
        return meta
    if meta is None:
        meta = {}
        _PII_META_WEAK[message] = meta
    return meta


def get_meta(message):
    meta = getattr(message, "_pii_meta", None)
    if isinstance(meta, dict):
        return meta
    try:
        return _PII_META_WEAK.get(message)
    except TypeError:
        return _PII_META_STRONG.get(id(message))


__all__ = ["AliasGenerator", "scrub_text", "ensure_meta", "get_meta"]
