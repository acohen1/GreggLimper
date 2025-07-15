import discord
from gregg_limper.cache import GLCache
from gregg_limper.config import Config

import logging
logger = logging.getLogger(__name__)

async def handle(client: discord.Client):
    """Cache initialization on client ready event."""
    logger.info(f"Logged in as {client.user.name} (ID: {client.user.id})")

    logger.info(f"Initializing GLCache with configured channel IDs: {Config.CHANNEL_IDS}")
    cache = GLCache()
    await cache.initialize(client, Config.CHANNEL_IDS)
