import asyncio
import datetime
from types import SimpleNamespace

from gregg_limper.commands.handlers.rag_opt import RagOptInCommand
from gregg_limper.config import core as core_cfg


class FakeMessage(SimpleNamespace):
    pass


class FakeChannel:
    def __init__(self, cid, guild, messages=None):
        self.id = cid
        self.guild = guild
        self._messages = messages or []
        self.sent = []

    async def send(self, content):
        self.sent.append(content)

    def history(self, *, limit=None, after=None, oldest_first=False):
        async def gen():
            msgs = [m for m in self._messages if not after or m.created_at > after]
            msgs.sort(key=lambda m: m.created_at, reverse=not oldest_first)
            if limit is not None:
                msgs = msgs[:limit]
            for m in msgs:
                yield m

        return gen()


def run(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(coro)
        while True:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            if not pending:
                break
            loop.run_until_complete(asyncio.gather(*pending))
    finally:
        loop.close()


def test_backfill(monkeypatch):
    user = SimpleNamespace(id=1, display_name="u")
    other = SimpleNamespace(id=2, display_name="o")
    monkeypatch.setattr(core_cfg, "CHANNEL_IDS", [1, 2])

    now = datetime.datetime.utcnow()

    guild = SimpleNamespace(id=10, text_channels=[])

    m1 = FakeMessage(
        id=101,
        author=user,
        guild=guild,
        created_at=now - datetime.timedelta(days=1),
    )
    m2 = FakeMessage(
        id=102,
        author=other,
        guild=guild,
        created_at=now - datetime.timedelta(days=1),
    )
    m3 = FakeMessage(
        id=103,
        author=user,
        guild=guild,
        created_at=now - datetime.timedelta(days=190),
    )
    m4 = FakeMessage(
        id=104,
        author=user,
        guild=guild,
        created_at=now - datetime.timedelta(days=1),
    )

    ch1 = FakeChannel(1, guild, [m1, m2, m3])
    ch2 = FakeChannel(2, guild, [m4])
    guild.text_channels = [ch1, ch2]

    cmd_channel = FakeChannel(99, guild)
    cmd_msg = FakeMessage(id=999, author=user, guild=guild, channel=cmd_channel)

    existing = {104}
    ingested = []

    async def fake_add_user(uid):
        return True

    async def fake_message_exists(mid):
        return mid in existing

    async def fake_ingest_cache_message(**kwargs):
        ingested.append(kwargs["message_id"])

    async def fake_format_message(msg):
        return {"author": msg.author.display_name, "fragments": []}

    monkeypatch.setattr(
        "gregg_limper.memory.rag.consent.add_user", fake_add_user
    )
    monkeypatch.setattr(
        "gregg_limper.memory.rag.message_exists", fake_message_exists
    )
    monkeypatch.setattr(
        "gregg_limper.memory.rag.ingest_cache_message", fake_ingest_cache_message
    )
    monkeypatch.setattr(
        "gregg_limper.commands.handlers.rag_opt.format_message", fake_format_message
    )

    run(RagOptInCommand.handle(None, cmd_msg, ""))

    assert ingested == [101]
    assert cmd_channel.sent[0] == "Opted in to RAG. Backfill queued."
    assert cmd_channel.sent[-1].startswith("Backfill complete")

