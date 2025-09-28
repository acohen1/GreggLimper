import types

from gregg_limper.memory.cache.manager import GLCache
from gregg_limper.memory.cache.channel_state import ChannelCacheState
from gregg_limper.memory.cache.memo_store import MemoStore


class _FakeFragment:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def to_llm(self) -> str:
        return self._payload

    def to_dict(self) -> dict:
        return {"text": self._payload}


def test_list_formatted_messages_skips_missing_memos(monkeypatch):
    monkeypatch.setattr(GLCache, "_instance", None)
    cache = GLCache()
    cache._states = {1: ChannelCacheState(1, 10)}
    cache._memo_store = MemoStore()

    state = cache._states[1]
    missing = types.SimpleNamespace(id=1)
    present = types.SimpleNamespace(id=2)

    state.append(missing)
    state.append(present)

    cache._memo_store.set(
        present.id,
        {"author": "cached", "fragments": [_FakeFragment("hello")]},
    )

    formatted = cache.list_formatted_messages(1, "llm")

    assert formatted == [{"author": "cached", "fragments": ["hello"]}]

    memo_copies = cache.list_memo_records(1)
    assert memo_copies == [
        {"author": "cached", "fragments": cache._memo_store.get(present.id)["fragments"]}
    ]
    assert memo_copies[0]["fragments"] is not cache._memo_store.get(present.id)["fragments"]
