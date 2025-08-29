"""
Public façade for RAG memory
===========================

Stable, async API for medium/long-term recall. Import from here::

    from gregg_limper.memory.rag import ingest_cache_message, vector_search, ...
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import asyncio

from .sql import db as _db
from .sql.repositories import FragmentsRepo as _FragmentsRepo, MetaRepo as _MetaRepo
from .vector.search import vector_search as _vector_search
from .sql.admin import retention_prune as _retention_prune, vacuum as _vacuum

# Limit the public surface (keeps star-imports clean)
__all__ = [
    "ingest_cache_message",
    "vector_search",
    "channel_summary",
    "user_profile",
    "server_stylesheet",
    "set_user_profile",
    "set_server_stylesheet",
    "set_channel_summary",
    "retention_prune",
    "vacuum",
]

# --- Internals -------------------------------------------------------------

# Singleton connection + repositories
_conn = _db.connect()
_db.migrate(_conn)
_db_lock = asyncio.Lock()

_frag_repo = _FragmentsRepo(_conn, _db_lock)
_meta_repo = _MetaRepo(_conn, _db_lock)

# --- Public async-friendly API ----------------------------------------------

async def ingest_cache_message(
    server_id: int,
    channel_id: int,
    message_id: int,
    author_id: int,
    ts: float,
    cache_message: Dict[str, Any],
) -> None:
    """
    Ingest one cache-formatted message into RAG memory.

    :param server_id: Discord server (guild) id.
    :param channel_id: Channel id.
    :param message_id: Message id (source of fragments).
    :param author_id: Author id for provenance.
    :param ts: Unix timestamp (seconds).
    :param cache_message: Cache message dict with ``fragments`` list containing
        :class:`Fragment` objects.
    """
    from .ingest import project_and_upsert

    await project_and_upsert(
        repo=_frag_repo,
        server_id=server_id,
        channel_id=channel_id,
        message_id=message_id,
        author_id=author_id,
        ts=ts,
        cache_message=cache_message,
    )


async def vector_search(
    server_id: int,
    channel_id: int,
    query: str,
    k: int = 50,
) -> List[Dict[str, Any]]:
    """Similarity search over stored fragments."""
    return await _vector_search(
        repo=_frag_repo,
        server_id=server_id,
        channel_id=channel_id,
        query=query,
        k=k,
    )


# --- Metadata ---------------------------------------------------------------

async def channel_summary(channel_id: int) -> str:
    """
    Get the stored summary for a channel.

    :param channel_id: Channel id.
    :returns: Summary text or empty string.
    """
    return await _meta_repo.get_channel_summary(channel_id)


async def user_profile(user_id: int) -> Dict[str, Any]:
    """
    Get a stored user profile blob.

    :param user_id: User id.
    :returns: Decoded JSON dict or empty dict.
    """
    s = await _meta_repo.get_user_profile(user_id)
    return json.loads(s) if s else {}


async def server_stylesheet(server_id: int) -> Optional[Dict[str, Any]]:
    """
    Get the server style/config blob.

    :param server_id: Server (guild) id.
    :returns: Decoded JSON dict or None.
    """
    s = await _meta_repo.get_server_style(server_id)
    return json.loads(s) if s else None


async def set_user_profile(user_id: int, blob: Dict[str, Any]) -> None:
    """
    Upsert a user profile blob.

    :param user_id: User id.
    :param blob: JSON-serializable dict.
    """
    await _meta_repo.set_user_profile(user_id, json.dumps(blob, ensure_ascii=False))


async def set_server_stylesheet(server_id: int, blob: Dict[str, Any]) -> None:
    """
    Upsert a server style/config blob.

    :param server_id: Server (guild) id.
    :param blob: JSON-serializable dict.
    """
    await _meta_repo.set_server_style(server_id, json.dumps(blob, ensure_ascii=False))


async def set_channel_summary(channel_id: int, summary: str) -> None:
    """
    Upsert a channel summary.

    :param channel_id: Channel id.
    :param summary: Plaintext summary.
    """
    await _meta_repo.set_channel_summary(channel_id, summary)


# --- Maintenance ------------------------------------------------------------

async def retention_prune(older_than_seconds: float) -> int:
    """
    Delete fragments older than ``now - older_than_seconds``.

    :param older_than_seconds: Age threshold in seconds.
    :returns: Number of rows deleted.
    """
    return await _retention_prune(_conn, _db_lock, older_than_seconds)


async def vacuum() -> None:
    """
    Run ``VACUUM`` and ``ANALYZE`` on the database.
    """
    await _vacuum(_conn, _db_lock)

