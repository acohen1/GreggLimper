"""Command dispatch utilities."""

from __future__ import annotations
import discord
from .handlers import get as get_handler
import re

import logging
logger = logging.getLogger(__name__)

_COMMAND_RE = re.compile(r"/(\w+)(?:\s+(.*))?")

async def dispatch(client: discord.Client, message: discord.Message) -> bool:
    """
    Parse and execute the first slash-style command in the message.
    Returns True if a command was handled.
    """
    content = message.content or ""

    match = _COMMAND_RE.search(content)
    if not match:
        return False

    command, args = match.groups()
    command = command.lower()
    args = args or ""

    handler = get_handler(command)
    if not handler:
        return False

    logger.info(f"Dispatching command '{command}' with args: {args}")
    await handler.handle(client, message, args)
    return True
