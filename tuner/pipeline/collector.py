from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

from discord import Client, Message, TextChannel
from types import SimpleNamespace

from ..config import DatasetBuildConfig
from .types import RawConversation

logger = logging.getLogger(__name__)


async def collect_history(
    *,
    client: Client,
    config: DatasetBuildConfig,
    reuse_raw: bool = False,
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
    raw_dir = config.raw_dump_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    reuse_raw = reuse_raw or config.reuse_raw
    for channel_id in config.channel_ids:
        if reuse_raw:
            cached = _load_cached_channel(raw_dir, channel_id)
            if cached is not None:
                conversations.append(cached)
                continue

        channel = await _resolve_channel(client, channel_id)
        if channel is None:
            continue

        channel_name = getattr(channel, "name", channel_id)
        logger.info(
            "Collecting channel %s (cutoff %s, limit %s)",
            channel_name,
            cutoff.isoformat(),
            config.max_messages,
        )
        messages = await _fetch_channel_history(
            channel,
            limit=config.max_messages,
            cutoff=cutoff,
            allowed_user_ids=allowed,
            channel_label=str(channel_name),
        )
        logger.info(
            "Collected %s messages from #%s (%s)",
            len(messages),
            channel_name,
            channel.id,
        )
        convo = RawConversation(
            channel_id=channel_id,
            guild_id=getattr(channel.guild, "id", None),
            messages=messages,
        )
        conversations.append(convo)
        _persist_single_raw(convo, raw_dir)

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
        _persist_single_raw(convo, destination)


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
    channel_label: str,
    heartbeat_interval: int = 1000,
) -> List[Message]:
    keep_all = not allowed_user_ids
    rows: list[Message] = []
    total_window = max(
        (datetime.now(timezone.utc) - cutoff).total_seconds(), 1.0
    )
    async for message in channel.history(
        limit=limit,
        oldest_first=True,
        after=cutoff,
    ):
        created_at = _ensure_timezone(message.created_at)
        author_id = getattr(message.author, "id", None)
        if not keep_all and author_id not in allowed_user_ids:
            continue
        rows.append(message)
        if heartbeat_interval and len(rows) % heartbeat_interval == 0:
            progress = min(
                max((created_at - cutoff).total_seconds() / total_window, 0.0), 1.0
            )
            logger.info(
                "Channel %s: fetched %d messages (~%d%% of cutoff window)",
                channel_label,
                len(rows),
                int(progress * 100),
            )
    return rows


def _serialize_message(message: Message) -> dict:
    author = getattr(message, "author", None)
    display_name = getattr(author, "display_name", None) or getattr(author, "name", "")
    guild = getattr(message, "guild", None)
    channel = getattr(message, "channel", None)
    return {
        "id": getattr(message, "id", None),
        "author_id": getattr(author, "id", None),
        "author": display_name,
        "channel_id": getattr(channel, "id", None),
        "channel_name": getattr(channel, "name", None),
        "guild_id": getattr(guild, "id", None),
        "guild_name": getattr(guild, "name", None),
        "created_at": _ensure_timezone(message.created_at).isoformat(),
        "content": (message.clean_content or message.content or "").strip(),
        "attachments": [
            getattr(att, "url", "")
            for att in getattr(message, "attachments", [])
        ],
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


def _persist_single_raw(convo: RawConversation, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{convo.channel_id}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for message in convo.messages:
            handle.write(json.dumps(_serialize_message(message), ensure_ascii=False))
            handle.write("\n")
    logger.info("Wrote raw transcript: %s", path)


def _load_cached_channel(directory: Path, channel_id: int) -> RawConversation | None:
    path = directory / f"{channel_id}.jsonl"
    if not path.is_file():
        return None
    try:
        messages: list[Message] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                author = SimpleNamespace(
                    id=payload.get("author_id"),
                    display_name=payload.get("author"),
                    name=payload.get("author"),
                )
                guild_stub = SimpleNamespace(
                    id=payload.get("guild_id"),
                    name=payload.get("guild_name"),
                ) if payload.get("guild_id") is not None else None
                channel_stub = SimpleNamespace(
                    id=payload.get("channel_id", channel_id),
                    name=payload.get("channel_name"),
                    guild=guild_stub,
                )
                stub = SimpleNamespace(
                    id=payload.get("id"),
                    author=author,
                    channel=channel_stub,
                    guild=guild_stub,
                    created_at=_ensure_timezone(
                        datetime.fromisoformat(payload["created_at"])
                    ),
                    clean_content=payload.get("content"),
                    content=payload.get("content"),
                    attachments=[
                        SimpleNamespace(url=url)
                        for url in payload.get("attachments", [])
                    ],
                    embeds=[],
                )
                messages.append(stub)
    except Exception:
        logger.warning("Failed to load cached raw channel %s", channel_id, exc_info=True)
        return None

    return RawConversation(channel_id=channel_id, guild_id=None, messages=messages)


__all__ = ["collect_history", "persist_raw_conversations"]
