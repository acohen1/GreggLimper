from __future__ import annotations

import asyncio
import datetime
import logging

import discord

from . import register
from ...memory.cache.core import process_message_for_rag
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
    bot_user = getattr(guild, "me", None) if guild else None

    # Only process channels allowed by the config
    channels = (
        [c for c in guild.text_channels if c.id in core_cfg.CHANNEL_IDS]
        if guild
        else []
    )

    # Bound concurrent formatting + ingestion so opt-in backfill remains cooperative
    sem = asyncio.Semaphore(rag_cfg.BACKFILL_CONCURRENCY)
    tasks: list[asyncio.Task[bool]] = []

    async def _process_bounded(msg: discord.Message, channel_id: int) -> bool:
        """Format and ingest one message while respecting concurrency limits."""
        async with sem:
            _, did_ingest = await process_message_for_rag(
                msg,
                channel_id,
                ingest=True,
                cache_msg=None,
                memo={},
                bot_user=bot_user,
            )
            return did_ingest

    # Enumerate candidate messages (opted-in user only, within lookback)
    for channel in channels:
        try:
            async for msg in channel.history(limit=None, after=cutoff, oldest_first=True):
                if msg.author.id != user.id:
                    continue
                tasks.append(asyncio.create_task(_process_bounded(msg, channel.id)))
        except Exception:
            logger.exception("Failed to iterate channel %s during backfill", channel.id)

    if tasks:
        for coro in asyncio.as_completed(tasks):
            if await coro:
                processed += 1

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
    def match_feedback(message: discord.Message) -> bool:
        """Identify canned feedback emitted by the opt-in command."""

        content = (message.content or "").strip()
        return (
            content == "Opted in to RAG. Backfill queued."
            or content == "Already opted in."
            or content.startswith("Backfill complete.")
        )

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
    def match_feedback(message: discord.Message) -> bool:
        """Identify canned feedback emitted by the opt-out command."""

        content = (message.content or "").strip()
        return content == "Opted out and data purged from RAG."

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
    def match_feedback(message: discord.Message) -> bool:
        """Identify canned feedback emitted by the status command."""

        content = (message.content or "").strip()
        return content in {"You are opted in.", "You are not opted in."}

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
