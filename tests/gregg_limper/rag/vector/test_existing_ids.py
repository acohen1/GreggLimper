import asyncio

from gregg_limper.config import milvus
from gregg_limper.memory.rag.vector import vector_index


def test_existing_ids_queries_with_limit(monkeypatch):
    calls = []

    class FakeCollection:
        def query(self, expr="", output_fields=None, limit=None, offset=0):
            calls.append((expr, limit, offset))
            # Return a small fixed number of rows based on the requested slice
            total = 5
            start = offset
            end = min(offset + (limit or total), total)
            return [{"rid": i} for i in range(start, end)]

    fake = FakeCollection()
    monkeypatch.setattr(vector_index, "_get_collection", lambda: fake)
    # Force a small chunk size to exercise pagination
    monkeypatch.setattr(milvus, "MILVUS_DELETE_CHUNK", 2, raising=False)
    monkeypatch.setattr(milvus, "ENABLE_MILVUS", True, raising=False)

    ids = asyncio.run(vector_index.existing_ids())
    assert ids == {0, 1, 2, 3, 4}

    # With chunk size 2 we should have paginated across three calls
    assert len(calls) == 3
    for expr, limit, _ in calls:
        assert expr == ""
        assert limit == 2
