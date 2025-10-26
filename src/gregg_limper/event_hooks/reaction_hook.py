import discord

import logging
logger = logging.getLogger(__name__)

async def handle(client: discord.Client, reaction: discord.Reaction, user: discord.User):
    logger.info(f"User {user.name} reacted with {reaction.emoji} in channel {reaction.message.channel.name} (ID: {reaction.message.channel.id})")
