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

# ----------------------------- Ingest & Backfill Helpers ----------------------------- #

async def _ingest_message(msg: discord.Message, channel_id: int) -> bool:
    """Return True if successfully ingested, False otherwise."""
    try:
        if await rag.message_exists(msg.id):
            return False
        cache_msg = await format_message(msg)
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


async def _backfill_user_messages(
    user: discord.User, guild: discord.Guild | None
) -> int:
    """Iterate channels and ingest messages for a user, returns count processed."""
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=rag_cfg.OPT_IN_LOOKBACK_DAYS)
    )
    # Some message sources (e.g., tests) may use naive datetimes. Convert our cutoff
    # to naive form as well so comparisons don't raise TypeError.
    cutoff_naive = cutoff.replace(tzinfo=None)
    processed = 0
    # Only process channels allowed by the config
    channels = (
        [c for c in guild.text_channels if c.id in core_cfg.CHANNEL_IDS]
        if guild
        else []
    )
    sem = asyncio.Semaphore(rag_cfg.BACKFILL_CONCURRENCY)
    tasks: list[asyncio.Task[bool]] = []

    async def _bounded_ingest(msg: discord.Message, channel_id: int) -> bool:
        async with sem:
            return await _ingest_message(msg, channel_id)

    for channel in channels:
        try:
            async for msg in channel.history(
                limit=None, after=cutoff_naive, oldest_first=True
            ):
                if msg.author.id != user.id:
                    continue
                if await rag.message_exists(msg.id):
                    continue
                tasks.append(asyncio.create_task(_bounded_ingest(msg, channel.id)))
        except Exception:
            logger.exception(
                "Failed to iterate channel %s during backfill", channel.id
            )

    if tasks:
        for coro in asyncio.as_completed(tasks):
            if await coro:
                processed += 1

    return processed

# ----------------------------- Command Definitions ----------------------------- #

@register
class RagOptInCommand:
    command_str = "rag_opt_in"

    @staticmethod
    async def handle(client: discord.Client, message: discord.Message, args: str) -> None:
        added = await consent.add_user(message.author.id)
        if added:
            await message.channel.send("Opted in to RAG. Backfill queued.")

            async def _backfill_and_notify() -> None:
                processed = await _backfill_user_messages(message.author, message.guild)
                await message.channel.send(f"Backfill complete. Ingested {processed} messages.")

            asyncio.create_task(_backfill_and_notify())
        else:
            await message.channel.send("Already opted in.")


@register
class RagOptOutCommand:
    command_str = "rag_opt_out"

    @staticmethod
    async def handle(client: discord.Client, message: discord.Message, args: str) -> None:
        await consent.remove_user(message.author.id)
        await purge_user(message.author.id)
        await message.channel.send("Opted out and data purged from RAG.")


@register
class RagStatusCommand:
    command_str = "rag_status"

    @staticmethod
    async def handle(client: discord.Client, message: discord.Message, args: str) -> None:
        opted = await consent.is_opted_in(message.author.id)
        msg = "You are opted in." if opted else "You are not opted in."
        await message.channel.send(msg)
