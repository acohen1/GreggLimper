"""Vector index maintenance tasks.

The Milvus vector index can drift out of sync with the fragments table if
inserts fail or if the index accumulates many deleted rows.  This module
exposes a ``run`` coroutine that performs one maintenance cycle consisting of
index synchronization and optional collection compaction.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable

from ..embeddings import from_bytes
from . import vector_index

logger = logging.getLogger(__name__)

async def _all_fragment_vectors(conn, lock) -> Iterable[tuple[int, int, int, bytes]]:
    """Fetch all fragment vectors from the SQL store."""

    def _rows():
        sql = "SELECT id, server_id, channel_id, embedding FROM fragments"
        return conn.execute(sql).fetchall()

    async with lock:
        return await asyncio.to_thread(_rows)


async def _sync_index(conn, lock) -> None:
    """Ensure fragment vectors are in sync with Milvus."""

    rows = await _all_fragment_vectors(conn, lock)
    existing = await vector_index.existing_ids()

    items = []
    seen: set[int] = set()
    for row in rows:
        rid = int(row["id"])
        seen.add(rid)
        blob = row["embedding"]
        if not blob or rid in existing:
            continue
        vec = from_bytes(blob)
        items.append((rid, int(row["server_id"]), int(row["channel_id"]), vec))

    if items:
        await vector_index.upsert_many(items)

    missing = existing - seen
    if missing:
        await vector_index.delete_many(list(missing))


async def _compact() -> None:
    """Request compaction on the underlying Milvus collection."""
    try:
        col = vector_index._get_collection()
    except Exception as e:  # pragma: no cover - connection errors
        logger.warning("Milvus collection unavailable for compaction: %s", e)
        return

    try:
        await asyncio.to_thread(col.compact)
    except Exception as e:  # pragma: no cover - compaction errors
        logger.error("Vector compaction failed: %s", e)


async def run(conn, lock) -> None:
    """Perform one maintenance cycle for the vector index."""
    await _sync_index(conn, lock)
    await _compact()
