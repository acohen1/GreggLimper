from __future__ import annotations

import hashlib
import re
from typing import Dict


class AliasGenerator:
    def __init__(self, salt: str, prefix: str = "User") -> None:
        self.salt = salt
        self.prefix = prefix
        self._cache: Dict[int, str] = {}

    def alias(self, user_id: int | None) -> str:
        if user_id is None:
            return f"{self.prefix}_anon"
        if user_id in self._cache:
            return self._cache[user_id]

        digest = hashlib.sha256(f"{self.salt}:{user_id}".encode("utf-8")).hexdigest()
        label = f"{self.prefix}_{digest[:8]}"
        self._cache[user_id] = label
        return label


MENTION_RE = re.compile(r"<@!?([0-9]+)>")


def scrub_text(text: str, alias_fn) -> str:
    def repl(match: re.Match[str]) -> str:
        user_id = int(match.group(1))
        return alias_fn(user_id)

    return MENTION_RE.sub(repl, text)


__all__ = ["AliasGenerator", "scrub_text"]


__all__ = ["AliasGenerator"]
