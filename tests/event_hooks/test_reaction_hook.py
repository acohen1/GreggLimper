import asyncio
from types import SimpleNamespace

from gregg_limper.event_hooks import reaction_hook
from gregg_limper.memory.cache.ingestion import ResourceState
from gregg_limper.memory.rag.triggers import TriggerSet


def _setup_trigger_patches(monkeypatch, *, should_match: bool):
    triggers = TriggerSet(frozenset({"🧠"}), frozenset(), frozenset())

    monkeypatch.setattr(reaction_hook, "get_trigger_set", lambda: triggers)
    monkeypatch.setattr(
        reaction_hook,
        "emoji_matches_trigger",
        lambda emoji, trig: should_match and emoji == "🧠",
    )


def test_reaction_hook_ingests_when_trigger_matches(monkeypatch):
    ingested = []

    async def fake_evaluate(*args, **kwargs):
        return True, ResourceState(memo=False, sqlite=False)

    async def fake_ingest(channel_id, message, cache_message):
        ingested.append((channel_id, message.id, cache_message))

    async def fake_format_for_cache(message):
        return {"message_id": message.id}

    _setup_trigger_patches(monkeypatch, should_match=True)

    monkeypatch.setattr(reaction_hook, "evaluate_ingestion", fake_evaluate)
    monkeypatch.setattr(reaction_hook, "ingest_message", fake_ingest)
    monkeypatch.setattr(
        reaction_hook.cache_formatting, "format_for_cache", fake_format_for_cache
    )

    def _raise_key_error(cid, mid):
        raise KeyError(mid)

    cache_stub = SimpleNamespace(get_memo_record=_raise_key_error)
    monkeypatch.setattr(reaction_hook, "GLCache", lambda: cache_stub)

    monkeypatch.setattr(reaction_hook.core, "CHANNEL_IDS", [1])

    message = SimpleNamespace(
        id=42,
        author=SimpleNamespace(id=10, bot=False),
        guild=SimpleNamespace(id=7),
        channel=SimpleNamespace(id=1),
        reactions=[],
    )
    reaction = SimpleNamespace(message=message, emoji="🧠")
    user = SimpleNamespace(name="tester")
    client = SimpleNamespace(user=SimpleNamespace(id=999))

    asyncio.run(reaction_hook.handle(client, reaction, user))

    assert ingested == [(1, 42, {"message_id": 42})]


def test_reaction_hook_ignores_non_trigger(monkeypatch):
    ingested = []

    async def fake_evaluate(*args, **kwargs):
        raise AssertionError("evaluate_ingestion should not be called")

    monkeypatch.setattr(reaction_hook, "evaluate_ingestion", fake_evaluate)
    monkeypatch.setattr(reaction_hook, "ingest_message", lambda *args, **kwargs: ingested.append(args))
    monkeypatch.setattr(reaction_hook, "GLCache", lambda: SimpleNamespace(get_memo_record=lambda *_: None))
    monkeypatch.setattr(reaction_hook.core, "CHANNEL_IDS", [1])

    _setup_trigger_patches(monkeypatch, should_match=False)

    message = SimpleNamespace(
        id=43,
        author=SimpleNamespace(id=10, bot=False),
        guild=SimpleNamespace(id=7),
        channel=SimpleNamespace(id=1),
        reactions=[],
    )
    reaction = SimpleNamespace(message=message, emoji="💡")
    user = SimpleNamespace(name="tester")
    client = SimpleNamespace(user=SimpleNamespace(id=999))

    asyncio.run(reaction_hook.handle(client, reaction, user))

    assert ingested == []
