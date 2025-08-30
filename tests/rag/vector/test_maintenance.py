import asyncio
import numpy as np
import sqlite3
from pathlib import Path
import types, sys

from gregg_limper.config import rag

# Provide a stub pymilvus module before importing vector components
pymilvus_stub = types.SimpleNamespace(
    Collection=object,
    CollectionSchema=object,
    DataType=types.SimpleNamespace(INT64=0, FLOAT_VECTOR=1),
    FieldSchema=object,
    connections=types.SimpleNamespace(connect=lambda **k: None),
    utility=types.SimpleNamespace(has_collection=lambda name: False),
)
sys.modules.setdefault("pymilvus", pymilvus_stub)

from gregg_limper.memory.rag import scheduler
from gregg_limper.memory.rag.vector import vector_index
from gregg_limper.memory.rag.sql import sql_tasks


def _insert_fragment(conn, content: str) -> None:
    """Insert a fragment row with the given content."""
    vec = np.arange(rag.EMB_DIM, dtype=np.float32)
    blob = vec.tobytes()
    row = (
        1,
        1,
        hash(content) & 0xFFFFFFFF,
        1,
        0.0,
        content,
        "text",
        "",
        "",
        content,
        blob,
          rag.EMB_MODEL_ID,
          rag.EMB_DIM,
        0,
        content,
        0.0,
    )
    with conn:
        conn.execute(
            """
            INSERT INTO fragments (
                server_id, channel_id, message_id, author_id, ts,
                content, type, title, url, media_id,
                embedding, emb_model, emb_dim,
                source_idx, content_hash, last_embedded_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )


def _make_conn():
    conn = sqlite3.connect(
        ":memory:", isolation_level=None, check_same_thread=False,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )
    conn.row_factory = sqlite3.Row
    schema = (Path(__file__).resolve().parents[3] / "gregg_limper/memory/rag/sql/schema.sql").read_text()
    conn.executescript(schema)
    return conn


def test_vector_maintenance_startup_and_periodic(monkeypatch):
    conn = _make_conn()
    lock = asyncio.Lock()

    # Insert initial fragment
    _insert_fragment(conn, "alpha")

    calls = []
    flushes = []
    existing = set()
    deletes = []

    async def fake_existing_ids():
        return set(existing)

    async def fake_upsert_many(items):
        calls.append(list(items))
        for rid, *_ in items:
            existing.add(rid)
        await fake_flush()

    async def fake_delete_many(ids):
        deletes.append(list(ids))
        for rid in ids:
            existing.discard(rid)
        await fake_flush()

    async def fake_flush():
        flushes.append(True)

    class FakeCollection:
        def __init__(self):
            self.compacts = 0

        def compact(self):
            self.compacts += 1

    fake_col = FakeCollection()

    monkeypatch.setattr(vector_index, "existing_ids", fake_existing_ids)
    monkeypatch.setattr(vector_index, "upsert_many", fake_upsert_many)
    monkeypatch.setattr(vector_index, "delete_many", fake_delete_many)
    monkeypatch.setattr(vector_index, "flush", fake_flush)
    monkeypatch.setattr(vector_index, "_get_collection", lambda: fake_col)

    async def _no_sql(conn, lock):
        return None

    monkeypatch.setattr(sql_tasks, "run", _no_sql)

    async def run_test():
        await scheduler.start(interval=0.1, conn=conn, lock=lock)
        await asyncio.sleep(0.05)
        # Insert another fragment to be picked up by the periodic cycle
        _insert_fragment(conn, "beta")
        await asyncio.sleep(0.25)
        await scheduler.stop()

    asyncio.run(run_test())

    # Startup upserts 1 item; periodic cycle upserts only the new item
    assert len(calls) >= 2
    assert len(calls[0]) == 1
    assert len(calls[-1]) == 1
    assert flushes  # flush called
    assert not deletes  # no deletions in this scenario
    assert fake_col.compacts > 0  # compaction performed

