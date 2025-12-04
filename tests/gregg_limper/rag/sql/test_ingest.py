import asyncio
from types import SimpleNamespace
import numpy as np

from gregg_limper.memory.rag import ingest
from gregg_limper.formatter.model import TextFragment, ImageFragment, GIFFragment, LinkFragment, YouTubeFragment
from gregg_limper.config import rag, milvus


class DummyRepo:
    def __init__(self):
        self.rows = []
    async def insert_or_update_fragment(self, row):
        self.rows.append(row)
    async def lookup_fragment_id(self, message_id, source_idx, typ, content_hash):
        return 1


async def fake_embed(text: str) -> np.ndarray:
    return np.arange(rag.EMB_DIM, dtype=np.float32)


async def fake_upsert(rid, server_id, channel_id, vec):
    fake_upsert.called = True
    fake_upsert.args = (rid, server_id, channel_id, vec)


fake_upsert.called = False

def test_fragment_content_text():
    assert TextFragment(description="hi").content_text() == "hi"
    assert ImageFragment(caption="cap").content_text() == "cap"
    assert GIFFragment(caption="gif").content_text() == "gif"
    assert LinkFragment(description="lnk").content_text() == "lnk"
    assert YouTubeFragment(description="yt").content_text() == "yt"


def test_project_and_upsert(monkeypatch):
    repo = DummyRepo()
    msg = {
        "author": "Tester",
        "fragments": [TextFragment(description="hello")],
    }
    monkeypatch.setattr(ingest, "embed", fake_embed)
    monkeypatch.setattr(ingest.vector_index, "upsert", fake_upsert)
    monkeypatch.setattr(milvus, "ENABLE_MILVUS", True, raising=False)

    asyncio.run(
        ingest.project_and_upsert(
            repo=repo,
            server_id=1,
            channel_id=2,
            message_id=3,
            author_id=4,
            ts=0.0,
            cache_message=msg,
        )
    )

    assert len(repo.rows) == 1
    row = repo.rows[0]
    assert row[0] == 1 and row[1] == 2 and row[2] == 3
    assert row[5] == "hello" and row[6] == "text"
    # embedding stored as bytes of correct length
    assert isinstance(row[10], (bytes, bytearray)) and len(row[10]) == rag.EMB_DIM * 4
    assert row[-1] > 0
    assert fake_upsert.called
    rid, server, channel, vec = fake_upsert.args
    assert rid == 1 and server == 1 and channel == 2
    assert np.array_equal(vec, np.arange(rag.EMB_DIM, dtype=np.float32))
