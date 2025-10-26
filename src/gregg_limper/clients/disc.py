"""Discord bot bootstrap utilities."""

from __future__ import annotations

import logging

import discord
from discord.ext import commands as discord_commands

from gregg_limper import commands as gl_commands
from gregg_limper.config import core
from gregg_limper.event_hooks import message_hook, reaction_hook, ready_hook
from gregg_limper.memory.rag import scheduler

logger = logging.getLogger(__name__)

# --- Intents --------------------------------------------------------------- #
intents = discord.Intents.all()


class GLBot(discord_commands.Bot):
    """Primary Discord bot implementation with slash command support."""

    def __init__(self) -> None:
        super().__init__(command_prefix=discord_commands.when_mentioned, intents=intents)

    async def setup_hook(self) -> None:
        """Register slash commands and synchronise with Discord."""

        await gl_commands.setup(self)

        try:
            synced = await self.tree.sync()
            logger.info("Synced %d application command(s)", len(synced))
        except Exception:
            logger.exception("Failed to sync application commands")

    async def close(self) -> None:
        await scheduler.stop()
        await super().close()


bot = GLBot()


@bot.event
async def on_ready() -> None:
    await ready_hook.handle(bot)


@bot.event
async def on_message(message: discord.Message) -> None:
    await message_hook.handle(bot, message)


@bot.event
async def on_reaction_add(reaction: discord.Reaction, user: discord.User) -> None:
    await reaction_hook.handle(bot, reaction, user)


def run() -> None:
    """Start the Discord bot using configuration from the environment."""

    if not core.DISCORD_API_TOKEN:
        logger.error("No DISCORD_API_TOKEN configured. Cannot run client.")
        return

    try:
        bot.run(core.DISCORD_API_TOKEN)
    except discord.LoginFailure as exc:
        logger.error("Login failed: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error while running client: %s", exc)
