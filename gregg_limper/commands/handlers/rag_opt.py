from __future__ import annotations
import discord
import asyncio
from . import register
from ...memory.rag import consent, purge_user

@register
class RagOptInCommand:
    command_str = "rag_opt_in"

    @staticmethod
    async def handle(client: discord.Client, message: discord.Message, args: str) -> None:
        added = await consent.add_user(message.author.id)
        if added:
            await message.channel.send("Opted in to RAG. Backfill queued.")
            # Placeholder for backfill trigger
            asyncio.create_task(asyncio.sleep(0))
        else:
            await message.channel.send("Already opted in.")


@register
class RagOptOutCommand:
    command_str = "rag_opt_out"

    @staticmethod
    async def handle(client: discord.Client, message: discord.Message, args: str) -> None:
        await consent.remove_user(message.author.id)
        await purge_user(message.author.id)
        await message.channel.send("Opted out and data purged from RAG.")


@register
class RagStatusCommand:
    command_str = "rag_status"

    @staticmethod
    async def handle(client: discord.Client, message: discord.Message, args: str) -> None:
        opted = await consent.is_opted_in(message.author.id)
        msg = "You are opted in." if opted else "You are not opted in."
        await message.channel.send(msg)
