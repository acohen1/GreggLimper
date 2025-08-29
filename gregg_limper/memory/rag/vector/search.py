import logging
from typing import Optional, List, Dict, Any
from ..embeddings import embed
from . import vector_index

import numpy as np

logger = logging.getLogger(__name__)


async def vector_search(
    *,
    repo,
    server_id: int,
    channel_id: int,
    query: str,
    k: int = 50,
) -> List[Dict[str, Any]]:
    """Embed the query and search the Milvus vector index."""

    qvec = await embed(query)

    # If the embedding failed (returned a zero vector), log and return empty results
    if not np.any(qvec):
        logger.info("Empty/degenerate query embedding (server_id=%d channel_id=%d)", server_id, channel_id)
        return []

    try:
        id_scores = await vector_index.search(server_id, channel_id, qvec, k)
    except Exception as e:
        logger.warning("Vector search failed (err=%s)", e)
        return []

    if not id_scores:
        logger.info("Vector search returned no results (server_id=%d channel_id=%d)", server_id, channel_id)
        return []

    ids = [rid for rid, _ in id_scores]
    rows = await repo.rows_by_ids(ids)
    if not rows:
        logger.info("No rows found for vector search results (server_id=%d channel_id=%d)", server_id, channel_id)
        return []

    rows_by_id = {r[0]: r for r in rows}
    ordered: List[Dict[str, Any]] = []
    for rid, _ in id_scores:
        row = rows_by_id.get(rid)
        if row is None:
            continue
        (rid, sid, cid, mid, aid, ts_val, content, typ, title, url, media_id, source_idx) = row
        ordered.append({
            "id": rid,
            "server_id": sid,
            "channel_id": cid,
            "message_id": mid,
            "author_id": aid,
            "ts": ts_val,
            "content": content,
            "type": typ,
            "title": title,
            "url": url,
            "media_id": media_id,
            "source_idx": source_idx,
        })
    return ordered
