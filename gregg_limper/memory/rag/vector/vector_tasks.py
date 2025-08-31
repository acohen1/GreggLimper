"""
Vector index maintenance tasks.

The Milvus vector index can drift out of sync with the fragments table if
inserts fail or if the index accumulates many deleted rows.  This module
exposes a ``run`` coroutine that performs one maintenance cycle consisting of
index synchronization and optional collection compaction.
"""

from __future__ import annotations

import asyncio
import logging
from ..embeddings import from_bytes
from .. import fetch_vectors_for_index
from . import vector_index
from gregg_limper.config import milvus

logger = logging.getLogger(__name__)


async def _sync_index(conn, lock) -> None:
    """Ensure fragment vectors are in sync with Milvus."""

    rows = await fetch_vectors_for_index(conn=conn, lock=lock)
    existing = await vector_index.existing_ids()

    items = []
    seen: set[int] = set()
    for rid, server_id, channel_id, blob in rows:
        rid = int(rid)
        seen.add(rid)
        if not blob or rid in existing:
            continue
        vec = from_bytes(blob)
        items.append((rid, int(server_id), int(channel_id), vec))

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
    if not milvus.ENABLE_MILVUS:
        logger.info("ENABLE_MILVUS is false; skipping vector maintenance run")
        return
    await _sync_index(conn, lock)
    await _compact()
