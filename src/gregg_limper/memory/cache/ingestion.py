"""
Bridge cached memos into long-term RAG stores.

The cache defers to this module for deciding if messages should be pushed into
downstream retrieval stores (SQLite + vector indices) and for performing the
ingestion. External callers should use :func:`evaluate_ingestion` to combine
user consent checks with duplicate detection and then invoke
:func:`ingest_message` to persist the memo payload. Both functions are resilient
to downstream errors and log failures without raising so the cache can continue
operating.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

from discord import Message
from discord.abc import User

from .. import rag
from ..rag import consent

logger = logging.getLogger(__name__)


@dataclass
class ResourceState:
    """Represents cache memo availability and downstream RAG readiness."""

    memo: bool
    sqlite: bool = False
    vector: bool = False


async def evaluate_ingestion(
    message: Message,
    ingest_requested: bool,
    memo_present: bool,
    bot_user: User | None = None,
) -> tuple[bool, ResourceState]:
    """Evaluate whether ``message`` should be ingested into the RAG stores."""

    resources = ResourceState(memo=memo_present)
    if not ingest_requested:
        return False, resources

    try:
        # Respect user consent before touching downstream stores.
        if not await consent.is_opted_in(message.author.id):
            return False, resources
        # We only fan out ingest work when storage lacks the message snapshot.
        exists = await rag.message_exists(message.id)
        resources.sqlite = exists
        resources.vector = exists
        return True, resources
    except Exception:
        # Downstream checks should not break caching; log and tell the caller to skip ingest.
        logger.exception("Failed to evaluate ingestion state for message %s", message.id)
        resources.sqlite = resources.vector = False
        return False, resources


async def ingest_message(channel_id: int, message: Message, cache_message: dict) -> None:
    """Persist ``message`` and its memoized payload into the RAG stores."""

    try:
        created_at = message.created_at
        if created_at.tzinfo is None:
            # Normalize naive timestamps so cross-region persistence is consistent.
            created_at = created_at.replace(tzinfo=datetime.timezone.utc)
        await rag.ingest_cache_message(
            server_id=message.guild.id if message.guild else 0,
            channel_id=channel_id,
            message_id=message.id,
            author_id=message.author.id,
            ts=created_at.timestamp(),
            cache_message=cache_message,
        )
    except Exception:
        # Ingestion is best-effort—failures should not stop the cache from moving forward.
        logger.exception("RAG ingestion failed for message %s", message.id)
