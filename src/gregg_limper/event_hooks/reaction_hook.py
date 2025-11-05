"""
Handle reaction events that should trigger RAG ingestion.
"""

from __future__ import annotations

import logging
from typing import Any

import discord

from gregg_limper.config import core
from gregg_limper.memory.cache import GLCache
from gregg_limper.memory.cache import formatting as cache_formatting
from gregg_limper.memory.cache.ingestion import evaluate_ingestion, ingest_message
from gregg_limper.memory.rag.triggers import (
    emoji_matches_trigger,
    get_trigger_set,
)

logger = logging.getLogger(__name__)


async def handle(
    client: discord.Client, reaction: discord.Reaction, user: discord.User
) -> None:
    """
    Ingest the reacted message when both the emoji and author qualify.
    """

    message = reaction.message
    channel = getattr(message, "channel", None)
    guild = getattr(message, "guild", None)

    # Skip DM messages and other contexts without guild/channel.
    if guild is None or channel is None:
        logger.debug(
            "Ignoring reaction %s on message %s without guild/channel context",
            reaction.emoji,
            getattr(message, "id", "unknown"),
        )
        return

    # Skip channels not in the configured set.
    channel_id = getattr(channel, "id", None)
    if channel_id not in core.CHANNEL_IDS:
        return

    # Skip reactions that don't match the emoji trigger set.
    triggers = get_trigger_set()
    if triggers.is_empty():
        logger.debug("Reaction ingestion skipped: no trigger emojis configured.")
        return

    if not emoji_matches_trigger(reaction.emoji, triggers):
        return

    cache = GLCache()

    try:
        cache_record = cache.get_memo_record(channel_id, message.id)
        memo_present = True
    except KeyError:
        cache_record = None
        memo_present = False

    # Evaluate whether to ingest the message based on consent and existing RAG state.
    should_ingest, resources = await evaluate_ingestion(
        message,
        ingest_requested=True,
        memo_present=memo_present,
        bot_user=getattr(client, "user", None),
    )

    if not should_ingest:
        logger.debug(
            "Reaction ingestion vetoed for message %s (likely missing consent).",
            message.id,
        )
        return

    if resources.sqlite:
        logger.debug(
            "Reaction ingestion skipped for message %s; already present in RAG.",
            message.id,
        )
        return

    if cache_record is None:
        cache_record = await cache_formatting.format_for_cache(message)

    # Ingest the message into the RAG stores.
    try:
        await ingest_message(channel_id, message, cache_record)
    except Exception:
        logger.exception("Failed to ingest message %s from reaction trigger", message.id)
        return

    actor: Any = (
        getattr(user, "name", None)
        or getattr(user, "display_name", None)
        or getattr(user, "id", None)
        or user
    )
    logger.info(
        "Ingested message %s via reaction %s by %s in channel %s",
        message.id,
        reaction.emoji,
        actor,
        channel_id,
    )
