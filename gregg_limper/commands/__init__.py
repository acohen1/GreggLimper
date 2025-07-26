"""Command dispatch utilities."""

from __future__ import annotations
import discord
from .handlers import get as get_handler


async def dispatch(client: discord.Client, message: discord.Message) -> bool:
    """
    Parse and execute a slash-style command.
    Returns True if a command was handled.
    """
    content = message.content or ""
    if not content.startswith("/"):
        return False

    parts = content.lstrip("/").split(None, 1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    handler = get_handler(command)
    if not handler:
        return False

    await handler.handle(client, message, args)
    return True