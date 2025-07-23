import discord
from gregg_limper.cache import GLCache

import logging
logger = logging.getLogger(__name__)

async def handle(client: discord.Client, message: discord.Message):
    logger.info(f"New message received in channel {message.channel.name} (ID: {message.channel.id})")
    
    # 1) Add message to cache
    cache = GLCache()   # Singleton instance
    try:
        await cache.add_message(message.channel.id, message)
        logger.info(f"Message {message.id} cached successfully.")
    except KeyError as e:
        logger.error(f"Failed to cache message {message.id}: {e}")

