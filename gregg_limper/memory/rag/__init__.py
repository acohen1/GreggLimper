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
import time

from gregg_limper.config import core as core_cfg

from .sql import db as _db
from .sql.repositories import FragmentsRepo as _FragmentsRepo, MetaRepo as _MetaRepo
from .vector.search import vector_search as _vector_search
from .sql.admin import retention_prune as _retention_prune, vacuum as _vacuum
from .vector import vector_index as _vector_index

# Limit the public surface (keeps star-imports clean)
__all__ = [
    "ingest_cache_message",
    "message_exists",
    "vector_search",
    "fetch_vectors_for_index",
    "channel_summary",
    "user_profile",
    "server_stylesheet",
    "set_user_profile",
    "set_server_stylesheet",
    "set_channel_summary",
    "retention_prune",
    "vacuum",
    "purge_user",
]

# --- Internals -------------------------------------------------------------

# Singleton connection + repositories
_conn = _db.connect()
_db.migrate(_conn)
with _conn:
    _conn.execute(
        "INSERT OR IGNORE INTO rag_consent(user_id, ts) VALUES(?, ?)",
        (core_cfg.BOT_USER_ID, time.time()),
    )
_db_lock = asyncio.Lock()

_frag_repo = _FragmentsRepo(_conn, _db_lock)
_meta_repo = _MetaRepo(_conn, _db_lock)

# --- Public async-friendly API ----------------------------------------------

async def message_exists(message_id: int) -> bool:
    """
    Return True if a given message exists in the SQL database.
    """
    return await _frag_repo.message_exists(message_id)

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
    """Similarity search over stored fragments.

    :param server_id: Discord server (guild) id to scope the search.
    :param channel_id: Channel id to scope the search.
    :param query: Natural language query to embed and search with.
    :param k: Maximum number of nearest fragments to return.
    :returns: Ordered list of fragment dictionaries containing metadata such as
        ``id``, ``message_id``, ``author_id`` and ``content``.
    """
    return await _vector_search(
        repo=_frag_repo,
        server_id=server_id,
        channel_id=channel_id,
        query=query,
        k=k,
    )


async def fetch_vectors_for_index(*, conn=None, lock=None):
    """
    Fetch all stored fragment vectors.

    Optional ``conn`` and ``lock`` parameters allow callers to supply their own
    database connection and lock, primarily for testing.
    """
    repo = _frag_repo if conn is None or lock is None else _FragmentsRepo(conn, lock)
    return await repo.fetch_vectors_for_index()


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


# --- Purge -------------------------------------------------------------------

async def purge_user(author_id: int) -> int:
    """
    Delete all fragments for ``author_id`` from SQL and vector index.

    Returns number of fragments removed from the SQL store.
    """

    def _run():
        with _conn:
            rows = _conn.execute(
                "SELECT id FROM fragments WHERE author_id=?", (author_id,)
            ).fetchall()
            ids = [int(r[0]) for r in rows]
            _conn.execute(
                "DELETE FROM fragments WHERE rowid IN (SELECT id FROM fragments WHERE author_id=?)",
                (author_id,),
            )
            cur = _conn.execute(
                "DELETE FROM fragments WHERE author_id=?", (author_id,)
            )
        return ids, cur.rowcount

    async with _db_lock:
        ids, count = await asyncio.to_thread(_run)

    try:
        await _vector_index.delete_many(ids)
    except Exception:
        pass

    return count

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

