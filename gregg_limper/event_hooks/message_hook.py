import discord
from gregg_limper.cache import GLCache
from gregg_limper import commands
from gregg_limper.config import Config

import logging
logger = logging.getLogger(__name__)

async def handle(client: discord.Client, message: discord.Message):
    """
    Handle incoming discord messages.
    - client: Discord bot client instance
    - message: The incoming message object
    """
    # 1) Is message in allowed channel?
    if message.channel.id not in Config.CHANNEL_IDS:
        return
    
    logger.info(f"New message received in channel {message.channel.name} (ID: {message.channel.id})")
    bot_mentioned = client.user in message.mentions

    # 2) Parse the message for commands (e.g. @bot /lobotomy, /help, etc.)
    if bot_mentioned:
        if await commands.dispatch(client, message):
            return
    
    # 3) Add message to cache
    cache = GLCache()   # Singleton instance
    try:
        await cache.add_message(message.channel.id, message)
        logger.info(f"Message {message.id} cached successfully.")
    except KeyError as e:
        logger.error(f"Failed to cache message {message.id}: {e}")


    # 4) Start response pipeline if bot is mentioned
    if not bot_mentioned:
        return


