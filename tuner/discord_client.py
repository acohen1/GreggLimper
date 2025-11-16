from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import discord

logger = logging.getLogger(__name__)

INTENTS = discord.Intents.none()
INTENTS.guilds = True
INTENTS.members = True
INTENTS.messages = True
INTENTS.message_content = True


class TunerDiscordClient(discord.Client):
    """Minimal Discord client for offline dataset hydration."""

    def __init__(self) -> None:
        super().__init__(intents=INTENTS)


@asynccontextmanager
async def connect_tuner_client(token: str) -> AsyncIterator[discord.Client]:
    """
    Authenticate a lightweight Discord client for the tuner.

    The client only exposes the HTTP APIs required for channel history fetches.
    """

    client = TunerDiscordClient()
    try:
        await client.login(token)
        yield client
    finally:
        await client.close()


__all__ = ["connect_tuner_client", "TunerDiscordClient"]
