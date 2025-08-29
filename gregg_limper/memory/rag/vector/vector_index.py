from __future__ import annotations
import asyncio
import threading
from typing import Any, List, Tuple, Set

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from gregg_limper.config import Config

_collection: Collection | None = None
_collection_loaded = False
_collection_lock = threading.Lock()

def _normalize(v) -> list[float]:
    v = np.asarray(v, dtype=np.float32)
    if v.shape != (Config.EMB_DIM,):
        v = v.reshape(-1)
        if v.shape[0] != Config.EMB_DIM:
            raise ValueError(
                f"Expected embedding of dim {Config.EMB_DIM}, got {v.shape[0]}"
            )
    np.nan_to_num(v, copy=False)
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v.tolist()


def _get_collection() -> Collection:
    global _collection, _collection_loaded

    if _collection is not None and _collection_loaded:
        return _collection

    with _collection_lock:
        if _collection is not None and _collection_loaded:
            return _collection

        connections.connect(
            alias="default", uri=f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}"
        )
        name = Config.MILVUS_COLLECTION

        index_params = {
            "index_type": "IVF_FLAT",  # GPU controlled server-side
            "metric_type": "IP",
            "params": {"nlist": Config.MILVUS_NLIST or 1024},
        }

        if not utility.has_collection(name):
            fields = [
                FieldSchema(
                    name="rid", dtype=DataType.INT64, is_primary=True, auto_id=False
                ),
                FieldSchema(name="server_id", dtype=DataType.INT64),
                FieldSchema(name="channel_id", dtype=DataType.INT64),
                FieldSchema(
                    name="embedding", dtype=DataType.FLOAT_VECTOR, dim=Config.EMB_DIM
                ),
            ]
            schema = CollectionSchema(fields, description="Fragment embeddings")
            _collection = Collection(name, schema)
            _collection.create_index("embedding", index_params)
            _collection_loaded = False
        else:
            _collection = Collection(name)
            if not _collection.has_index():
                _collection.create_index("embedding", index_params)
                _collection_loaded = False

        if not _collection_loaded:
            _collection.load()
            _collection_loaded = True

        return _collection


async def upsert(rid: int, server_id: int, channel_id: int, embedding) -> None:
    vec = _normalize(embedding)

    def _run() -> None:
        col = _get_collection()
        col.delete(f"rid == {int(rid)}")  # idempotent upsert
        col.insert([[int(rid)], [int(server_id)], [int(channel_id)], [vec]])

    await asyncio.to_thread(_run)


async def upsert_many(items: list[tuple[int, int, int, Any]]) -> None:
    if not items:
        return

    rids: list[int] = []
    server_ids: list[int] = []
    channel_ids: list[int] = []
    embeddings: list[list[float]] = []

    for rid, server_id, channel_id, emb in items:
        rids.append(int(rid))
        server_ids.append(int(server_id))
        channel_ids.append(int(channel_id))
        embeddings.append(_normalize(emb))

    def _run() -> None:
        col = _get_collection()
        # Delete existing rows in batches to avoid overly long expressions
        chunk = Config.MILVUS_DELETE_CHUNK
        for i in range(0, len(rids), chunk):
            ids = rids[i : i + chunk]
            col.delete(f"rid in {ids}")
        col.insert([rids, server_ids, channel_ids, embeddings])
    await asyncio.to_thread(_run)
    await flush()


async def delete_many(ids: list[int]) -> None:
    if not ids:
        return

    def _run() -> None:
        col = _get_collection()
        chunk = Config.MILVUS_DELETE_CHUNK
        for i in range(0, len(ids), chunk):
            col.delete(f"rid in {ids[i:i+chunk]}")

    await asyncio.to_thread(_run)
    await flush()


async def existing_ids() -> Set[int]:
    def _run() -> Set[int]:
        col = _get_collection()
        chunk = max(1, Config.MILVUS_DELETE_CHUNK)
        offset = 0
        ids: Set[int] = set()
        while True:
            try:
                rows = col.query(
                    expr="",
                    output_fields=["rid"],
                    limit=chunk,
                    offset=offset,
                )
            except Exception:
                return set()
            if not rows:
                break
            ids.update(int(r["rid"]) for r in rows)
            if len(rows) < chunk:
                break
            offset += chunk
        return ids

    return await asyncio.to_thread(_run)


async def flush() -> None:
    def _run() -> None:
        col = _get_collection()
        col.flush()

    await asyncio.to_thread(_run)


async def search(
    server_id: int, channel_id: int, query_vec, k: int
) -> List[Tuple[int, float]]:
    vec = _normalize(query_vec)

    def _run() -> List[Tuple[int, float]]:
        col = _get_collection()
        expr = f"server_id == {int(server_id)} and channel_id == {int(channel_id)}"
        res = col.search(
            data=[vec],
            anns_field="embedding",
            search_params={"metric_type": "IP", "params": {"nprobe": Config.MILVUS_NPROBE}},
            limit=k,
            expr=expr,
            consistency_level="Strong",
        )
        hits = res[0] if res else []
        return [(int(h.id), float(h.score)) for h in hits]


    return await asyncio.to_thread(_run)

