import asyncio
from types import SimpleNamespace
import numpy as np

from gregg_limper.memory.rag import embeddings
from gregg_limper.config import rag
from gregg_limper.memory.rag.vector import vector_index


class FakeCollection:
    def __init__(self):
        self.store = {}
    def delete(self, expr: str):
        expr = expr.strip()
        if " in " in expr:
            ids = eval(expr.split(" in ")[1])  # simple parsing for tests
            for rid in ids:
                self.store.pop(int(rid), None)
        else:
            rid = int(expr.split("==")[1])
            self.store.pop(rid, None)
    def insert(self, cols):
        rids, servers, channels, vecs = cols
        for rid, server, channel, vec in zip(rids, servers, channels, vecs):
            self.store[int(rid)] = (int(server), int(channel), np.array(vec, dtype=np.float32))
    def search(
        self,
        data,
        anns_field=None,
        param=None,
        limit=10,
        expr=None,
        output_fields=None,
        consistency_level=None,
    ):
        vec = np.array(data[0], dtype=np.float32)
        server_id = int(expr.split("server_id ==")[1].split("and")[0])
        channel_id = int(expr.split("channel_id ==")[1])
        hits = []
        for rid, (srv, chan, emb) in self.store.items():
            if srv == server_id and chan == channel_id:
                dist = float(np.dot(vec, emb))
                hits.append(SimpleNamespace(id=rid, score=dist))
        hits.sort(key=lambda h: h.score, reverse=True)
        return [hits[:limit]]
    def load(self):
        pass
    def has_index(self):
        return True
    def create_index(self, *a, **k):
        pass
    def flush(self):
        pass


async def fake_embed_text(text: str, model: str | None = None) -> np.ndarray:
    base = np.arange(rag.EMB_DIM, dtype=np.float32)
    return base + len(text)


def test_vector_upsert_and_search(monkeypatch):
    fake = FakeCollection()
    monkeypatch.setattr(vector_index, "_collection", fake)
    monkeypatch.setattr(vector_index, "_get_collection", lambda: fake)
    monkeypatch.setattr(embeddings, "embed_text", fake_embed_text)

    vec = asyncio.run(embeddings.embed("hello"))
    asyncio.run(vector_index.upsert(5, 1, 2, vec))
    results = asyncio.run(vector_index.search(1, 2, vec, k=1))
    assert results and results[0][0] == 5
    assert results[0][1] > 0


def test_upsert_many(monkeypatch):
    fake = FakeCollection()
    monkeypatch.setattr(vector_index, "_collection", fake)
    monkeypatch.setattr(vector_index, "_get_collection", lambda: fake)
    monkeypatch.setattr(embeddings, "embed_text", fake_embed_text)

    vec = asyncio.run(embeddings.embed("hello"))
    items = [(1, 1, 2, vec), (2, 1, 2, vec)]
    asyncio.run(vector_index.upsert_many(items))
    results = asyncio.run(vector_index.search(1, 2, vec, k=2))
    assert {r for r, _ in results} == {1, 2}

