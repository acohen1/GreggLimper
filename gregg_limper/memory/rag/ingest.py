"""
Ingestion: cache message -> fragments
=====================================

Projects a cache-formatted message dict into normalized fragments
and upserts via the repositories.
"""
from __future__ import annotations
from typing import Any, Dict
from gregg_limper.config import Config
from .embeddings import embed, to_bytes, blake16
from .media_id import stable_media_id
from .vector import vector_index
import numpy as np
import asyncio
import time

import logging
logger = logging.getLogger(__name__)

async def project_and_upsert(
    *,
    repo,
    server_id: int,
    channel_id: int,
    message_id: int,
    author_id: int,
    ts: float,
    cache_message: Dict[str, Any],
) -> None:
    """
    Project a cache-formatted message into normalized fragments and upsert.

    Clean storage:
      - No modality tokens in `content` (store text only).
      - Modality is captured by the `type` column.
      - title/url are optional (NULL when unknown).
    """
    frags = cache_message.get("fragments") or []

    # Collect fragment data for embedding
    prep: list[Dict] = []

    for i, cf in enumerate(frags):
        typ = (cf.type or "").strip()
        if not typ:
            continue
        content = cf.content_text()
        if not content:
            continue

        content_h = blake16(f"{typ}:{content}")   # dedupe per message/modality/payload
        prep.append({
            "i": i,
            "cf": cf,
            "typ": typ,
            "content": content,
            "content_h": content_h,
        })

    tasks = [embed(p["content"]) for p in prep]
    results = []
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for p, vec in zip(prep, results):
        embed_ts = time.time()
        if isinstance(vec, Exception):
            logger.error(
                "Embed failed (message_id=%s idx=%s type=%s err=%s); using zero-vector",
                message_id, p["i"], p["typ"], vec,
            )
            vec = np.zeros(Config.EMB_DIM, dtype=np.float32)
            embed_ts = 0.0

        emb = to_bytes(vec)
        cf = p["cf"]
        cf_dict = cf.to_dict()
        media_id = stable_media_id(
            cf=cf_dict,
            server_id=server_id,
            channel_id=channel_id,
            message_id=message_id,
            source_idx=p["i"],
        )

        await repo.insert_or_update_fragment((
            server_id, channel_id, message_id, author_id, ts,
            p["content"], p["typ"],
            (cf.title or None),
            (cf.url or None),
            media_id,
            emb, Config.EMB_MODEL_ID, Config.EMB_DIM,
            p["i"], p["content_h"], embed_ts,
        ))
        rid = await repo.lookup_fragment_id(message_id, p["i"], p["typ"], p["content_h"])
        if rid is not None:
            try:
                await vector_index.upsert(rid, server_id, channel_id, vec)
            except Exception as e:
                logger.error(
                    "Vector index upsert failed (message_id=%s idx=%s type=%s err=%s)",
                    message_id,
                    p["i"],
                    p["typ"],
                    e,
                )
        else:
            logger.warning(
                "Fragment id lookup failed after insert (message_id=%s idx=%s type=%s)",
                message_id,
                p["i"],
                p["typ"],
            )

