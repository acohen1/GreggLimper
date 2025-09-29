from __future__ import annotations
import discord
from . import register


@register
class LobotomyCommand:
    command_str = "lobotomy"

    @staticmethod
    def match_feedback(message: discord.Message) -> bool:
        """Identify canned feedback emitted by the lobotomy command."""

        content = (message.content or "").strip()
        return content == "Initiating lobotomy sequence..."

    @staticmethod
    async def handle(
        client: discord.Client, message: discord.Message, args: str
    ) -> None:
        await message.channel.send("Initiating lobotomy sequence...")
