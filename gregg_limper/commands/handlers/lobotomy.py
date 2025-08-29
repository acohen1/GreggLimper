from __future__ import annotations
import discord
from . import register


@register
class LobotomyCommand:
    command_str = "lobotomy"

    @staticmethod
    async def handle(client: discord.Client, message: discord.Message, args: str) -> None:
        await message.channel.send("Initiating lobotomy sequence...")
