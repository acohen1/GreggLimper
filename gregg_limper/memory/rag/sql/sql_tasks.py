"""
Embedding maintenance tasks
==========================

Ensures fragments table embeddings conform to the configured
spec (model + dimension) and are non-zero. Intended to run on
startup and periodically as a background task.
"""

from __future__ import annotations

import asyncio
import logging
import time

from gregg_limper.config import Config
from ..embeddings import embed, to_bytes

logger = logging.getLogger(__name__)

_last_run_ts: float = 0.0


async def _enforce_spec(conn, lock) -> None:
    """Re-embed fragments that are out of date or wrong."""

    global _last_run_ts
    now = time.time()

    def _rows():
        sql = (
            "SELECT id, content, embedding, emb_model, emb_dim, last_embedded_ts "
            "FROM fragments WHERE last_embedded_ts <= ? "
            "OR emb_model != ? OR emb_dim != ? OR length(embedding) != ?"
        )
        return conn.execute(
            sql,
            (_last_run_ts, Config.EMB_MODEL_ID, Config.EMB_DIM, Config.EMB_DIM * 4),
        ).fetchall()

    async with lock:
        rows = await asyncio.to_thread(_rows)

    for row in rows:
        rid = row["id"]
        content = row["content"]
        emb = row["embedding"]
        model = row["emb_model"]
        dim = row["emb_dim"]
        last_ts = row["last_embedded_ts"]

        needs_update = last_ts <= _last_run_ts
        if model != Config.EMB_MODEL_ID or dim != Config.EMB_DIM:
            needs_update = True
        elif len(emb) != Config.EMB_DIM * 4:
            needs_update = True
        elif not emb or not any(emb):
            needs_update = True

        if not needs_update:
            continue

        try:
            vec = await embed(content or "")
            if vec.size != Config.EMB_DIM:
                logger.warning(
                    "Re-embed produced %s dims for id=%s (expected %s)",
                    vec.size,
                    rid,
                    Config.EMB_DIM,
                )
                continue
            new_blob = to_bytes(vec)
        except Exception as e:  # pragma: no cover - network/embedding errors
            logger.error("Embedding refresh failed for id=%s err=%s", rid, e)
            new_blob = b"\x00" * (Config.EMB_DIM * 4)

        def _update():
            with conn:
                conn.execute(
                    "UPDATE fragments SET embedding=?, emb_model=?, emb_dim=?, last_embedded_ts=? WHERE id=?",
                    (new_blob, Config.EMB_MODEL_ID, Config.EMB_DIM, now, rid),
                )

        async with lock:
            await asyncio.to_thread(_update)

    _last_run_ts = now

async def run(conn, lock) -> None:
    """Run one maintenance pass over stored embeddings."""
    await _enforce_spec(conn, lock)
