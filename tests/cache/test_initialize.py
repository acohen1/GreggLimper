import asyncio
import datetime
from types import SimpleNamespace

from gregg_limper.config import cache as cache_cfg
from gregg_limper.memory.cache import GLCache
from gregg_limper.memory.cache import core as cache_core


class FakeMessage(SimpleNamespace):
    pass


class FakeChannel:
    def __init__(self, cid, messages):
        self.id = cid
        self._messages = messages
        self.last_history = None

    def history(self, *, limit=None, after=None, oldest_first=False):
        self.last_history = {
            "limit": limit,
            "after": after,
            "oldest_first": oldest_first,
        }

        async def gen():
            msgs = list(self._messages)
            msgs.sort(key=lambda m: m.created_at, reverse=not oldest_first)
            if limit is not None:
                msgs = msgs[:limit]
            for m in msgs:
                yield m

        return gen()


class FakeClient:
    def __init__(self, channel):
        self._channel = channel

    def get_channel(self, cid):
        assert cid == self._channel.id
        return self._channel


def test_initialize_hydrates_recent_history(monkeypatch):
    monkeypatch.setattr(cache_cfg, "CACHE_LENGTH", 10)
    monkeypatch.setattr(cache_core, "TextChannel", FakeChannel)

    async def fake_is_opted_in(uid):
        return False

    async def fake_format_message(msg):
        return {"author": msg.author.display_name, "fragments": []}

    monkeypatch.setattr(cache_core.consent, "is_opted_in", fake_is_opted_in)
    monkeypatch.setattr(cache_core, "format_message", fake_format_message)
    monkeypatch.setattr(cache_core.memo, "exists", lambda cid: False)
    monkeypatch.setattr(cache_core.memo, "load", lambda cid: {})
    monkeypatch.setattr(cache_core.memo, "prune", lambda cid, d: d)
    monkeypatch.setattr(cache_core.memo, "save", lambda cid, d: None)

    now = datetime.datetime.now(datetime.timezone.utc)
    messages = [
        FakeMessage(
            id=i,
            author=SimpleNamespace(id=1, display_name="u"),
            created_at=now + datetime.timedelta(seconds=i),
            channel=SimpleNamespace(id=1),
            guild=SimpleNamespace(id=1),
        )
        for i in range(20)
    ]
    channel = FakeChannel(1, messages)
    client = FakeClient(channel)

    cache_inst = GLCache()
    cache_inst._caches = {}
    cache_inst._memo = {}
    cache_inst._index = {}

    asyncio.run(cache_inst.initialize(client, [1]))

    stored = list(cache_inst._caches[1])
    assert len(stored) == cache_cfg.CACHE_LENGTH
    expected_ids = [m.id for m in messages[-cache_cfg.CACHE_LENGTH:]]
    assert [m.id for m in stored] == expected_ids
    assert channel.last_history["limit"] == cache_cfg.CACHE_LENGTH
    assert channel.last_history["oldest_first"] is False


def test_initialize_preserves_order_with_slow_formatter(monkeypatch):
    monkeypatch.setattr(cache_cfg, "CACHE_LENGTH", 5)
    monkeypatch.setattr(cache_cfg, "INIT_CONCURRENCY", 3)
    monkeypatch.setattr(cache_core, "TextChannel", FakeChannel)

    async def fake_is_opted_in(uid):
        return False

    async def fake_format_message(msg):
        # Sleep inversely proportional to id so completion order differs
        await asyncio.sleep(0.01 * (5 - msg.id))
        return {"author": msg.author.display_name, "fragments": []}

    monkeypatch.setattr(cache_core.consent, "is_opted_in", fake_is_opted_in)
    monkeypatch.setattr(cache_core, "format_message", fake_format_message)
    monkeypatch.setattr(cache_core.memo, "exists", lambda cid: False)
    monkeypatch.setattr(cache_core.memo, "load", lambda cid: {})
    monkeypatch.setattr(cache_core.memo, "prune", lambda cid, d: d)
    monkeypatch.setattr(cache_core.memo, "save", lambda cid, d: None)

    now = datetime.datetime.now(datetime.timezone.utc)
    messages = [
        FakeMessage(
            id=i,
            author=SimpleNamespace(id=1, display_name="u"),
            created_at=now + datetime.timedelta(seconds=i),
            channel=SimpleNamespace(id=1),
            guild=SimpleNamespace(id=1),
        )
        for i in range(5)
    ]
    channel = FakeChannel(1, messages)
    client = FakeClient(channel)

    cache_inst = GLCache()
    cache_inst._caches = {}
    cache_inst._memo = {}
    cache_inst._index = {}

    asyncio.run(cache_inst.initialize(client, [1]))

    stored = list(cache_inst._caches[1])
    assert [m.id for m in stored] == [m.id for m in messages]


def test_initialize_formats_only_missing_payloads(monkeypatch):
    monkeypatch.setattr(cache_cfg, "CACHE_LENGTH", 10)
    monkeypatch.setattr(cache_core, "TextChannel", FakeChannel)

    async def fake_is_opted_in(uid):
        return False

    async def fake_format_message(msg):
        return {"author": msg.author.display_name, "fragments": []}

    monkeypatch.setattr(cache_core.consent, "is_opted_in", fake_is_opted_in)
    monkeypatch.setattr(cache_core, "format_message", fake_format_message)
    monkeypatch.setattr(cache_core.memo, "exists", lambda cid: True)
    monkeypatch.setattr(
        cache_core.memo,
        "load",
        lambda cid: {1: {"author": "u", "fragments": []}},
    )
    monkeypatch.setattr(cache_core.memo, "prune", lambda cid, d: d)
    monkeypatch.setattr(cache_core.memo, "save", lambda cid, d: None)

    created: list[int] = []
    orig_create_task = asyncio.create_task

    def fake_create_task(coro, *args, **kwargs):
        msg = getattr(coro, "cr_frame", None)
        if msg is not None:
            m = coro.cr_frame.f_locals.get("msg")
            if m is not None:
                created.append(m.id)
        return orig_create_task(coro, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    now = datetime.datetime.now(datetime.timezone.utc)
    messages = [
        FakeMessage(
            id=i,
            author=SimpleNamespace(id=1, display_name="u"),
            created_at=now + datetime.timedelta(seconds=i),
            channel=SimpleNamespace(id=1),
            guild=SimpleNamespace(id=1),
        )
        for i in range(1, 4)
    ]
    channel = FakeChannel(1, messages)
    client = FakeClient(channel)

    cache_inst = GLCache()
    cache_inst._caches = {}
    cache_inst._memo = {}
    cache_inst._index = {}

    asyncio.run(cache_inst.initialize(client, [1]))

    assert created == [2, 3]
