from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

from discord import Client, Message, TextChannel

from ..config import DatasetBuildConfig
from .types import RawConversation

logger = logging.getLogger(__name__)


async def collect_history(
    *,
    client: Client,
    config: DatasetBuildConfig,
) -> List[RawConversation]:
    """
    Fetch and filter Discord messages based on the dataset configuration.

    Responsibilities:
        - hydrate channel history up to config.max_messages
        - drop messages older than config.earliest_timestamp
        - keep only authors in config.allowed_user_ids
        - preserve real discord.Message instances for downstream processing
    """

    cutoff = _parse_timestamp(config.earliest_timestamp)
    allowed = set(config.allowed_user_ids)

    conversations: list[RawConversation] = []
    for channel_id in config.channel_ids:
        channel = await _resolve_channel(client, channel_id)
        if channel is None:
            continue

        messages = await _fetch_channel_history(
            channel,
            limit=config.max_messages,
            cutoff=cutoff,
            allowed_user_ids=allowed,
        )
        conversations.append(
            RawConversation(
                channel_id=channel_id,
                guild_id=getattr(channel.guild, "id", None),
                messages=messages,
            )
        )

    return conversations


def persist_raw_conversations(
    conversations: Iterable[RawConversation],
    *,
    destination: Path,
) -> None:
    """
    Write a JSONL transcript for each channel to the destination directory.

    Each line contains a serializable subset of the Discord message for
    auditability (ids, timestamps, display names, normalized content).
    """

    destination.mkdir(parents=True, exist_ok=True)
    for convo in conversations:
        path = destination / f"{convo.channel_id}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for message in convo.messages:
                handle.write(json.dumps(_serialize_message(message), ensure_ascii=False))
                handle.write("\n")
        logger.info("Wrote raw transcript: %s", path)


async def _resolve_channel(client: Client, channel_id: int) -> TextChannel | None:
    channel = client.get_channel(channel_id)
    if channel is None:
        try:
            channel = await client.fetch_channel(channel_id)
        except Exception:
            logger.warning("Failed to fetch channel %s", channel_id, exc_info=True)
            return None

    if not isinstance(channel, TextChannel):
        logger.warning("Channel %s is not a text channel. Skipping.", channel_id)
        return None
    return channel


async def _fetch_channel_history(
    channel: TextChannel,
    *,
    limit: int,
    cutoff: datetime,
    allowed_user_ids: Sequence[int],
) -> List[Message]:
    keep_all = not allowed_user_ids
    rows: list[Message] = []
    async for message in channel.history(limit=limit, oldest_first=True):
        created_at = _ensure_timezone(message.created_at)
        if created_at < cutoff:
            continue
        author_id = getattr(message.author, "id", None)
        if not keep_all and author_id not in allowed_user_ids:
            continue
        rows.append(message)
    logger.info(
        "Collected %s messages from #%s (%s)",
        len(rows),
        getattr(channel, "name", channel.id),
        channel.id,
    )
    return rows


def _serialize_message(message: Message) -> dict:
    author = getattr(message, "author", None)
    display_name = getattr(author, "display_name", None) or getattr(author, "name", "")
    return {
        "id": getattr(message, "id", None),
        "author_id": getattr(author, "id", None),
        "author": display_name,
        "created_at": _ensure_timezone(message.created_at).isoformat(),
        "content": (message.clean_content or message.content or "").strip(),
        "attachments": [getattr(att, "url", "") for att in getattr(message, "attachments", [])],
    }


def _parse_timestamp(raw: str) -> datetime:
    if not raw:
        raise ValueError("earliest timestamp must be provided")

    text = raw.strip()
    text = text.replace("Z", "+00:00")
    candidates = [text, f"{text}T00:00:00", f"{text}T00:00:00+00:00"]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            return _ensure_timezone(dt)
        except ValueError as exc:
            last_error = exc
            continue
    raise ValueError(f"Could not parse earliest timestamp '{raw}'") from last_error


def _ensure_timezone(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = ["collect_history", "persist_raw_conversations"]
