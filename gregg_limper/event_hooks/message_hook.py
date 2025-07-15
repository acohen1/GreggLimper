import discord

import logging
logger = logging.getLogger(__name__)

async def handle(client: discord.Client, message: discord.Message):
    logger.info(f"New message received in channel {message.channel.name} (ID: {message.channel.id})")