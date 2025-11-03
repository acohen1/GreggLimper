import json

import logging

import discord

from gregg_limper.config import core
from gregg_limper.memory.cache import GLCache
from gregg_limper import response

logger = logging.getLogger(__name__)

async def handle(client: discord.Client, message: discord.Message):
    """Handle incoming Discord messages."""

    # Slash command follow-ups (and true DMs) arrive without guild context; skip them.
    if message.guild is None or getattr(getattr(message, "flags", None), "ephemeral", False):
        logger.debug("Skipping non-guild or ephemeral message %s", message.id)
        return

    # 1) Ignore channels that are not configured for processing
    if message.channel.id not in core.CHANNEL_IDS:
        return

    channel_name = getattr(message.channel, "name", None) or getattr(
        message.channel, "recipient", None
    )
    if channel_name is None:
        channel_name = message.channel.__class__.__name__

    logger.info(
        "New message received in channel %s (ID: %s)",
        channel_name,
        getattr(message.channel, "id", "unknown"),
    )
    bot_user = client.user
    bot_mentioned = bot_user in message.mentions if bot_user else False

    # 2) Add message to cache
    # NOTE: The add message pipeline automatically handles formatting and ingestion.
    # It will skip commands and feedback messages.
    cache = GLCache()  # Singleton instance
    try:
        await cache.add_message(message.channel.id, message, bot_user=bot_user)
    except KeyError as e:
        logger.error(f"Failed to cache message {message.id}: {e}")


    # DEBUGGING
    recent_messages = cache.list_formatted_messages(message.channel.id, "llm", n=5)
    for m in recent_messages:
        m_str = json.dumps(m, ensure_ascii=False, separators=(",", ": "))
        logger.info(
            f"Cached message: {m_str[:100]}..."
        )  # Log first 100 chars for brevity

    # 3) Start response pipeline if bot is mentioned
    if not bot_mentioned:
        return

    response_text = await response.handle(message)
    # await message.channel.send(response_text)
