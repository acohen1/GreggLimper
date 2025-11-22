import asyncio

import discord
from discord.ext import commands as discord_commands

from gregg_limper import commands as gl_commands


async def _collect_cogs():
    bot = discord_commands.Bot(command_prefix="!", intents=discord.Intents.none())
    try:
        await gl_commands.setup(bot)
        return set(bot.cogs.keys())
    finally:
        await bot.close()


def test_setup_registers_known_cogs():
    cogs = asyncio.run(_collect_cogs())
    assert {"Help", "Lobotomy", "RagOpt"}.issubset(cogs)
