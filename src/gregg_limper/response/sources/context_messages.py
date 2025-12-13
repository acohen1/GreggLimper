"""Render dynamic context into chat-completion-friendly messages."""

from __future__ import annotations

import json
from typing import Iterable

from .context import ConversationContext

__all__ = ["build_context_messages"]


def build_context_messages(context: ConversationContext) -> list[dict[str, str]]:
    """Convert context into assistant messages for the LLM."""

    sections: list[str] = []

    sections.append(
        _format_section(
            "Channel Summary",
            context.channel_summary
            or "_No persistent channel summary was retrieved._",
        )
    )

    sections.append(
        _format_section(
            "User Profiles",
            _format_user_profiles(context.user_profiles)
            if context.user_profiles
            else "_No opted-in user profiles were available._",
        )
    )

    content = "\n\n".join(sections).strip()
    if not content:
        return []

    return [
        {
            "role": "assistant",
            "content": "### Context\n" + content,
        }
    ]


def _format_section(title: str, body: str) -> str:
    return f"#### {title}\n{body.strip()}" if body else f"#### {title}\n"


def _format_user_profiles(profiles: Iterable[dict]) -> str:
    serialized = [
        f"- {json.dumps(profile, ensure_ascii=False, indent=2)}" for profile in profiles
    ]
    return "\n".join(serialized)

