"""Command dispatch utilities."""

from __future__ import annotations

import logging
import re
from typing import NamedTuple

import discord
from discord.abc import User

from .handlers import CommandHandler, get as get_handler, feedback_matchers

logger = logging.getLogger(__name__)

_COMMAND_RE = re.compile(r"/(\w+)(?:\s+(.*))?")


class CommandInvocation(NamedTuple):
    """Resolved command data for downstream consumers."""

    handler: CommandHandler
    name: str
    args: str


def _resolve_command(content: str) -> CommandInvocation | None:
    """Return the handler, command name, and args if ``content`` matches."""

    match = _COMMAND_RE.search(content)
    if not match:
        return None

    command, args = match.groups()
    command = (command or "").lower()
    handler = get_handler(command)
    if not handler:
        return None

    return CommandInvocation(handler=handler, name=command, args=(args or ""))


async def dispatch(client: discord.Client, message: discord.Message) -> bool:
    """
    Parse and execute the first slash-style command in the message.
    Returns True if a command was handled.
    """

    invocation = _resolve_command(message.content or "")
    if not invocation:
        return False

    handler, command, args = invocation
    logger.info("Dispatching command '%s' with args: %s", command, args)
    await handler.handle(client, message, args)
    return True


def is_command_message(
    message: discord.Message,
    bot_user: User | None = None,
    mentioned: bool | None = None,
) -> bool:
    """Return ``True`` when ``message`` targets the bot with a registered command."""

    if not message:
        return False

    if mentioned is None:
        if bot_user is None:
            guild = getattr(message, "guild", None)
            bot_user = getattr(guild, "me", None) if guild else None
        if bot_user is None:
            mentioned = False
        else:
            mentions = getattr(message, "mentions", []) or []
            mentioned = bot_user in mentions

    if not mentioned:
        return False

    return _resolve_command(message.content or "") is not None


def is_command_feedback(
    message: discord.Message, bot_user: User | None = None
) -> bool:
    """Return ``True`` when ``message`` is canned command feedback from the bot."""

    if not message:
        return False

    author = getattr(message, "author", None)
    if author is None:
        return False

    if bot_user is None:
        guild = getattr(message, "guild", None)
        bot_user = getattr(guild, "me", None) if guild else None

    if bot_user is not None:
        if getattr(author, "id", None) != getattr(bot_user, "id", None):
            return False
    elif not getattr(author, "bot", False):
        return False

    content = getattr(message, "content", None)
    if not content:
        return False

    for command_name, matcher in feedback_matchers().items():
        try:
            if matcher(message):
                return True
        except Exception:  # pragma: no cover - defensive guard
            logger.exception(
                "Feedback matcher for command '%s' raised an exception", command_name
            )

    return False
