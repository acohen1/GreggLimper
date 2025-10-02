import discord
from gregg_limper.memory.cache import GLCache
from gregg_limper import commands
from gregg_limper.config import core
from gregg_limper import response
import json

import logging

logger = logging.getLogger(__name__)

# TODO: SEE TODO IN response/prompt.py ABOUT BETTER HANDLING SEMANTIC SEARCH

async def handle(client: discord.Client, message: discord.Message):
    """
    Handle incoming discord messages.
    - client: Discord bot client instance
    - message: The incoming message object
    """
    # 1) Is message in allowed channel?
    if message.channel.id not in core.CHANNEL_IDS:
        return

    logger.info(
        f"New message received in channel {message.channel.name} (ID: {message.channel.id})"
    )
    bot_user = client.user
    bot_mentioned = bot_user in message.mentions if bot_user else False

    # 2) Parse the message for commands (e.g. @bot /lobotomy, /help, etc.)
    if bot_mentioned and await commands.dispatch(client, message):
        return
    
    # 3) Add message to cache
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

    # 4) Start response pipeline if bot is mentioned
    if not bot_mentioned:
        return

    response_text = await response.handle(message)
    await message.channel.send(response_text)
