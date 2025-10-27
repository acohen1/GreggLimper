from __future__ import annotations

import asyncio
import datetime
import logging

import discord
from discord import app_commands
from discord.ext import commands

from .. import register_cog
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

def _log_background_task(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except Exception:
        logger.exception("RAG backfill task failed")


# ----------------------------- Command Definitions ----------------------------- #


@register_cog
class RagOpt(commands.Cog):
    """
    Slash commands for RAG consent management.

    Provides opt-in, opt-out, and status commands that coordinate with the
    consent registry and storage backends.
    """

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="optin",
        description="Enable or disable knowledge retention for your messages.",
    )
    @app_commands.describe(
        enabled="Set to True to opt in (with backfill) or False to opt out and purge."
    )
    async def optin(
        self, interaction: discord.Interaction, enabled: bool
    ) -> None:
        """Toggle the caller's RAG opt-in state and coordinate related work."""

        if enabled:
            added = await consent.add_user(interaction.user.id)
            if not added:
                await interaction.response.send_message(
                    "Already opted in.", ephemeral=True
                )
                return

            await interaction.response.send_message(
                "Opted in to RAG. Backfill queued.", ephemeral=True
            )

            async def _backfill_and_notify() -> None:
                processed = await _backfill_user_messages(
                    interaction.user, interaction.guild
                )
                await interaction.followup.send(
                    f"Backfill complete. Ingested {processed} messages.",
                    ephemeral=True,
                )

            task = asyncio.create_task(_backfill_and_notify())
            task.add_done_callback(_log_background_task)
            return

        await consent.remove_user(interaction.user.id)
        await purge_user(interaction.user.id)
        await interaction.response.send_message(
            "Opted out and data purged from RAG.", ephemeral=True
        )

    @app_commands.command(
        name="rag_status",
        description="Report whether you are currently opted in to RAG ingestion.",
    )
    async def rag_status(self, interaction: discord.Interaction) -> None:
        """Report the caller's current RAG consent state."""

        opted = await consent.is_opted_in(interaction.user.id)
        msg = "You are opted in." if opted else "You are not opted in."
        await interaction.response.send_message(msg, ephemeral=True)
