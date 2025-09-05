from __future__ import annotations

import asyncio
import datetime
import logging

import discord

from . import register
from ...formatter import format_message
from ...memory import rag
from ...memory.rag import consent, purge_user
from gregg_limper.config import rag as rag_cfg
from gregg_limper.config import core as core_cfg

logger = logging.getLogger(__name__)

# ----------------------------- Backfill Helper ----------------------------- #

async def _backfill_user_messages(
    user: discord.User, guild: discord.Guild | None
) -> int:
    """
    Backfill a user's past messages into RAG.

    :param user: Discord user being processed.
    :param guild: Guild context or ``None`` in DMs.
    :returns: Number of ingested messages.
    """
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=rag_cfg.OPT_IN_LOOKBACK_DAYS)
    )

    processed = 0

    # Only process channels allowed by the config
    channels = (
        [c for c in guild.text_channels if c.id in core_cfg.CHANNEL_IDS]
        if guild
        else []
    )

    # Parallelize *formatting* with a bounded semaphore
    sem = asyncio.Semaphore(rag_cfg.BACKFILL_CONCURRENCY)
    tasks: list[asyncio.Task[tuple[discord.Message, int, dict | None]]] = []

    async def _format_bounded(
        msg: discord.Message, channel_id: int
    ) -> tuple[discord.Message, int, dict | None]:
        """Format one message while respecting concurrency limits."""
        async with sem:
            try:
                cache_msg = await format_message(msg)
            except Exception:
                logger.exception("Failed to format message %s during backfill", msg.id)
                cache_msg = None
        return msg, channel_id, cache_msg

    # Ingest semaphore and task list prepared after formatting completes
    ingest_sem = asyncio.Semaphore(rag_cfg.BACKFILL_CONCURRENCY)

    async def _ingest_bounded(
        msg: discord.Message, channel_id: int, cache_msg: dict
    ) -> bool:
        """Ingest one message while respecting concurrency limits."""
        async with ingest_sem:
            try:
                created_at = msg.created_at
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=datetime.timezone.utc)
                await rag.ingest_cache_message(
                    server_id=msg.guild.id if msg.guild else 0,
                    channel_id=channel_id,
                    message_id=msg.id,
                    author_id=msg.author.id,
                    ts=created_at.timestamp(),
                    cache_message=cache_msg,
                )
                return True
            except Exception:
                logger.exception("Failed to ingest message %s during backfill", msg.id)
                return False

    # Enumerate candidate messages (opted-in user only, within lookback)
    for channel in channels:
        try:
            async for msg in channel.history(limit=None, after=cutoff, oldest_first=True):
                if msg.author.id != user.id:
                    continue
                # Skip if any fragment already exists for this message_id (idempotent)
                if await rag.message_exists(msg.id):
                    continue
                tasks.append(asyncio.create_task(_format_bounded(msg, channel.id)))
        except Exception:
            logger.exception("Failed to iterate channel %s during backfill", channel.id)

    # Consume formatter results, then ingest in parallel with bounded concurrency
    formatted: list[tuple[discord.Message, int, dict]] = []
    if tasks:
        for coro in asyncio.as_completed(tasks):
            msg, channel_id, cache_msg = await coro
            if cache_msg is not None:
                formatted.append((msg, channel_id, cache_msg))

    if formatted:
        ingest_tasks = [
            _ingest_bounded(msg, channel_id, cache_msg)
            for msg, channel_id, cache_msg in formatted
        ]
        results = await asyncio.gather(*ingest_tasks)
        processed += sum(1 for r in results if r)

    return processed

# ----------------------------- Command Definitions ----------------------------- #

@register
class RagOptInCommand:
    """
    Slash command: ``/rag_opt_in``

    Effect
    ------
    - Adds the caller to the RAG consent registry.
    - Queues a non-blocking historical backfill across allowed channels.
    - Notifies the user when the backfill completes with a processed count.
    """
    command_str = "rag_opt_in"

    @staticmethod
    async def handle(
        client: discord.Client, message: discord.Message, args: str
    ) -> None:
        """
        Opt the caller into RAG and trigger historical backfill.

        :param client: Discord client instance (unused).
        :param message: Command message invoking the opt-in.
        :param args: Raw argument string (unused).
        """
        added = await consent.add_user(message.author.id)
        if added:
            await message.channel.send("Opted in to RAG. Backfill queued.")

            async def _backfill_and_notify() -> None:
                processed = await _backfill_user_messages(message.author, message.guild)
                await message.channel.send(
                    f"Backfill complete. Ingested {processed} messages."
                )

            asyncio.create_task(_backfill_and_notify())
        else:
            await message.channel.send("Already opted in.")


@register
class RagOptOutCommand:
    """
    Slash command: ``/rag_opt_out``

    Effect
    ------
    - Removes the caller from the RAG consent registry.
    - Purges all of the caller's data from both SQL and the vector index.
    """
    command_str = "rag_opt_out"

    @staticmethod
    async def handle(
        client: discord.Client, message: discord.Message, args: str
    ) -> None:
        """
        Remove caller from consent registry and purge their data.

        :param client: Discord client instance (unused).
        :param message: Command message invoking the opt-out.
        :param args: Raw argument string (unused).
        """
        await consent.remove_user(message.author.id)
        await purge_user(message.author.id)
        await message.channel.send("Opted out and data purged from RAG.")


@register
class RagStatusCommand:
    """
    Slash command: ``/rag_status``

    Effect
    ------
    - Reports whether the caller is currently opted in to RAG ingestion.
    """
    command_str = "rag_status"

    @staticmethod
    async def handle(
        client: discord.Client, message: discord.Message, args: str
    ) -> None:
        """
        Report the caller's current RAG consent state.

        :param client: Discord client instance (unused).
        :param message: Command message invoking the status check.
        :param args: Raw argument string (unused).
        """
        opted = await consent.is_opted_in(message.author.id)
        msg = "You are opted in." if opted else "You are not opted in."
        await message.channel.send(msg)
