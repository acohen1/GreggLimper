# client_disc.py
import discord
from gregg_limper.config import Config
from gregg_limper.event_hooks import message_hook, reaction_hook, ready_hook

import logging
logger = logging.getLogger(__name__)

# --- Intents --------------
intents = discord.Intents.all()
# intents.message_content = True
# intents.guilds = True
# intents.members = True
# intents.reactions = True
# intents.presences = True
# intents.emojis = True
# intents.typing = True

# --- Client --------------
client = discord.Client(intents=intents)

# --- Event Handlers --------
@client.event
async def on_ready():
    await ready_hook.handle(client)

@client.event
async def on_message(message: discord.Message):
    await message_hook.handle(client, message)

@client.event
async def on_reaction_add(reaction: discord.Reaction, user: discord.User):
    await reaction_hook.handle(client, reaction, user)

def run():
    """
    Start the Discord client with configured token.
    """
    if not Config.DISCORD_API_TOKEN:
        logger.error("No DISCORD_TOKEN configured. Cannot run client.")
        return

    try:
        client.run(Config.DISCORD_API_TOKEN)
    except discord.LoginFailure as e:
        logger.error(f"Login failed: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error while running client: {e}")