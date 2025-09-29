import asyncio
import datetime
from types import SimpleNamespace

from gregg_limper.memory.cache import GLCache
from gregg_limper.memory.cache.channel_state import ChannelCacheState
from gregg_limper.commands.handlers.rag_opt import _backfill_user_messages
from gregg_limper.config import cache as cache_cfg
from gregg_limper.config import core as core_cfg
from gregg_limper.memory.cache import memo, formatting, ingestion


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_backfill_matches_live_ingestion(monkeypatch):
    """Ensure backfill and live caching ingest the same opted-in messages."""

    monkeypatch.setattr(core_cfg, "CHANNEL_IDS", [1])
    monkeypatch.setattr(memo, "save", lambda *_, **__: None)
    monkeypatch.setattr(memo, "prune", lambda _channel_id, memo_dict: memo_dict)
    monkeypatch.setattr("gregg_limper.commands.is_command_message", lambda *_, **__: False)
    monkeypatch.setattr("gregg_limper.commands.is_command_feedback", lambda *_, **__: False)

    async def fake_is_opted_in(user_id: int) -> bool:
        return user_id == 1

    async def fake_message_exists(_message_id: int) -> bool:
        return False

    async def fake_format_for_cache(msg):
        return {"author": msg.author.display_name, "fragments": []}

    monkeypatch.setattr(ingestion.consent, "is_opted_in", fake_is_opted_in)
    monkeypatch.setattr(ingestion.rag, "message_exists", fake_message_exists)
    monkeypatch.setattr(formatting, "format_for_cache", fake_format_for_cache)

    now = datetime.datetime.now(datetime.timezone.utc)
    guild = SimpleNamespace(id=1, text_channels=[])
    user = SimpleNamespace(id=1, display_name="user")
    other = SimpleNamespace(id=2, display_name="other")

    def _make_message(mid: int, author):
        return SimpleNamespace(
            id=mid,
            author=author,
            guild=guild,
            created_at=now - datetime.timedelta(days=1),
            content="",
        )

    m1 = _make_message(101, user)
    m2 = _make_message(102, user)
    m3 = _make_message(103, other)
    messages = [m1, m2, m3]

    class FakeChannel:
        def __init__(self, cid: int, history_messages):
            self.id = cid
            self._messages = history_messages
            self.guild = guild

        def history(self, *, limit=None, after=None, oldest_first=False):
            async def _gen():
                ordered = sorted(self._messages, key=lambda item: item.created_at)
                for msg in ordered:
                    if after is not None and msg.created_at <= after:
                        continue
                    yield msg

            return _gen()

    channel = FakeChannel(1, messages)
    guild.text_channels = [channel]

    GLCache._instance = None
    cache = GLCache()
    cache._states = {1: ChannelCacheState(1, cache_cfg.CACHE_LENGTH)}
    cache._memo_store.reset()

    ingested_live: list[int] = []

    async def fake_ingest_live(**kwargs):
        ingested_live.append(kwargs["message_id"])

    monkeypatch.setattr(ingestion.rag, "ingest_cache_message", fake_ingest_live)

    for msg in messages:
        _run(cache.add_message(1, msg, ingest=True))

    ingested_backfill: list[int] = []

    async def fake_ingest_backfill(**kwargs):
        ingested_backfill.append(kwargs["message_id"])

    monkeypatch.setattr(ingestion.rag, "ingest_cache_message", fake_ingest_backfill)

    _run(_backfill_user_messages(user, guild))

    assert sorted(ingested_live) == sorted(ingested_backfill)
