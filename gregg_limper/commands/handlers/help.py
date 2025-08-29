from __future__ import annotations
import discord
from . import register, all_commands


@register
class HelpCommand:
    command_str = "help"

    @staticmethod
    async def handle(client: discord.Client, message: discord.Message, args: str) -> None:
        cmds = ", ".join(sorted(all_commands().keys()))
        await message.channel.send(f"Available commands: {cmds}")
