from __future__ import annotations
import discord
from . import register, all_commands


@register
class HelpCommand:
    """List available slash commands."""

    command_str = "help"

    @staticmethod
    def match_feedback(message: discord.Message) -> bool:
        """Identify canned feedback emitted by the help command."""

        content = (message.content or "").strip()
        return content.startswith("Available commands:")

    @staticmethod
    async def handle(
        client: discord.Client, message: discord.Message, args: str
    ) -> None:
        """
        Send a comma-separated list of registered commands.

        :param client: Discord client instance (unused).
        :param message: Incoming command message.
        :param args: Raw argument string (unused).
        """
        cmds = ", ".join(sorted(all_commands().keys()))
        await message.channel.send(f"Available commands: {cmds}")
