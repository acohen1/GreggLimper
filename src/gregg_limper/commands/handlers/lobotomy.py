from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from .. import register_cog


@register_cog
class Lobotomy(commands.Cog):
    """Utility command for demo purposes."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="lobotomy", description="Trigger the lobotomy sequence.")
    async def lobotomy(self, interaction: discord.Interaction) -> None:
        """Respond with the lobotomy confirmation message."""

        await interaction.response.send_message(
            "Initiating lobotomy sequence...", ephemeral=True
        )
