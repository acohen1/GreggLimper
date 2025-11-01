from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from .. import register_cog


@register_cog
class Help(commands.Cog):
    """List available slash commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="help", description="List available slash commands.")
    async def help(self, interaction: discord.Interaction) -> None:
        """
        Send a comma-separated list of registered slash commands to the caller.
        """

        command_names = sorted(cmd.name for cmd in self.bot.tree.get_commands())
        listing = ", ".join(command_names) if command_names else "None registered"
        await interaction.response.send_message(
            f"Available commands: {listing}", ephemeral=True
        )
