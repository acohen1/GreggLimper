import asyncio
from types import SimpleNamespace
import datetime

from gregg_limper.memory.rag import consent
from gregg_limper.memory.rag import consent as consent_mod
from gregg_limper.memory.cache import GLCache
from gregg_limper.memory.cache.channel_state import ChannelCacheState
from gregg_limper.memory.cache.memo_store import MemoStore
from gregg_limper.config import cache as cache_cfg
from gregg_limper.config import core as core_cfg
from gregg_limper.memory import rag


def test_consent_registry():
    uid = 9999
    asyncio.run(consent.remove_user(uid))
    assert asyncio.run(consent.is_opted_in(uid)) is False
    added = asyncio.run(consent.add_user(uid))
    assert added
    assert asyncio.run(consent.is_opted_in(uid)) is True
    asyncio.run(consent.remove_user(uid))
    assert asyncio.run(consent.is_opted_in(uid)) is False


def test_bot_whitelisted_on_init():
    assert asyncio.run(consent.is_opted_in(core_cfg.BOT_USER_ID)) is True


def test_cache_ingestion_gate(monkeypatch):
    gc = GLCache()
    gc._states = {1: ChannelCacheState(1, cache_cfg.CACHE_LENGTH)}
    gc._memo_store = MemoStore()

    msg = SimpleNamespace(
        id=1,
        author=SimpleNamespace(id=123, display_name="u"),
        guild=SimpleNamespace(id=1),
        created_at=datetime.datetime.utcfromtimestamp(0),
        content="",
    )

    async def fake_format_message(m):
        return {"author": "u", "fragments": []}

    async def fake_message_exists(mid):
        return False

    async def fake_ingest(**k):
        fake_ingest.called = True

    fake_ingest.called = False

    monkeypatch.setattr(
        "gregg_limper.memory.cache.formatting.format_for_cache", fake_format_message
    )
    monkeypatch.setattr(rag, "message_exists", fake_message_exists)
    monkeypatch.setattr(rag, "ingest_cache_message", fake_ingest)
    monkeypatch.setattr("gregg_limper.memory.cache.memo.save", lambda cid, data: None)
    monkeypatch.setattr("gregg_limper.memory.cache.memo.prune", lambda cid, data: data)

    async def consent_false(uid):
        return False

    async def consent_true(uid):
        return True

    monkeypatch.setattr(consent_mod, "is_opted_in", consent_false)
    asyncio.run(gc.add_message(1, msg, ingest=True))
    assert fake_ingest.called is False

    fake_ingest.called = False
    monkeypatch.setattr(consent_mod, "is_opted_in", consent_true)
    asyncio.run(gc.add_message(1, msg, ingest=True))
    assert fake_ingest.called is True


def test_commands_skipped_from_ingest(monkeypatch):
    gc = GLCache()
    gc._states = {1: ChannelCacheState(1, cache_cfg.CACHE_LENGTH)}
    gc._memo_store = MemoStore()

    bot_user = SimpleNamespace(id=4242, bot=True)

    async def fake_format_message(m):
        fake_format_message.called = True
        return {"author": "u", "fragments": []}

    fake_format_message.called = False

    async def fake_ingest(**k):
        fake_ingest.called = True

    fake_ingest.called = False

    monkeypatch.setattr(
        "gregg_limper.memory.cache.formatting.format_for_cache", fake_format_message
    )
    monkeypatch.setattr(rag, "ingest_cache_message", fake_ingest)
    monkeypatch.setattr("gregg_limper.memory.cache.memo.save", lambda cid, data: None)
    monkeypatch.setattr("gregg_limper.memory.cache.memo.prune", lambda cid, data: data)

    message_exists_calls = []

    async def fake_message_exists(mid):
        message_exists_calls.append(mid)
        return False

    monkeypatch.setattr(rag, "message_exists", fake_message_exists)

    user = SimpleNamespace(id=123, display_name="user", bot=False)
    command_msg = SimpleNamespace(
        id=10,
        author=user,
        guild=SimpleNamespace(id=1),
        created_at=datetime.datetime.utcfromtimestamp(0),
        content="/rag_opt_in",
        mentions=[bot_user],
    )

    asyncio.run(gc.add_message(1, command_msg, ingest=True, bot_user=bot_user))

    assert fake_format_message.called is False
    assert fake_ingest.called is False
    assert message_exists_calls == []
    assert gc._memo_store.has(command_msg.id) is False

    fake_format_message.called = False
    fake_ingest.called = False

    feedback_msg = SimpleNamespace(
        id=11,
        author=bot_user,
        guild=SimpleNamespace(id=1),
        created_at=datetime.datetime.utcfromtimestamp(0),
        content="Opted in to RAG. Backfill queued.",
        mentions=[],
    )

    asyncio.run(gc.add_message(1, feedback_msg, ingest=True, bot_user=bot_user))

    assert fake_format_message.called is False
    assert fake_ingest.called is False
    assert message_exists_calls == []
    assert gc._memo_store.has(feedback_msg.id) is False
