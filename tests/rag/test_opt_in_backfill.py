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

    now = datetime.datetime.now(datetime.timezone.utc)

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
    processed = []
    ingested = []

    async def fake_add_user(uid):
        return True

    async def fake_process_message_for_rag(message, channel_id, **kwargs):
        processed.append((message.id, channel_id))
        did_ingest = message.id not in existing
        if did_ingest:
            ingested.append(message.id)
        return {"message_id": message.id}, did_ingest

    monkeypatch.setattr(
        "gregg_limper.memory.rag.consent.add_user", fake_add_user
    )
    monkeypatch.setattr(
        "gregg_limper.commands.handlers.rag_opt.process_message_for_rag",
        fake_process_message_for_rag,
    )

    run(RagOptInCommand.handle(None, cmd_msg, ""))

    assert processed == [(101, 1), (104, 2)]
    assert ingested == [101]
    assert cmd_channel.sent[0] == "Opted in to RAG. Backfill queued."
    assert cmd_channel.sent[-1].startswith("Backfill complete")


def test_backfill_skips_command_messages(monkeypatch):
    user = SimpleNamespace(id=1, display_name="u")
    bot_user = SimpleNamespace(id=50, bot=True)
    monkeypatch.setattr(core_cfg, "CHANNEL_IDS", [1])

    now = datetime.datetime.now(datetime.timezone.utc)

    guild = SimpleNamespace(id=10, text_channels=[], me=bot_user)

    command_history_msg = FakeMessage(
        id=201,
        author=user,
        guild=guild,
        created_at=now - datetime.timedelta(hours=1),
        content="/rag_opt_in",
        mentions=[bot_user],
    )

    ch1 = FakeChannel(1, guild, [command_history_msg])
    command_history_msg.channel = ch1
    guild.text_channels = [ch1]

    cmd_channel = FakeChannel(99, guild)
    cmd_msg = FakeMessage(
        id=999,
        author=user,
        guild=guild,
        channel=cmd_channel,
        mentions=[bot_user],
        content="/rag_opt_in",
    )

    processed = []

    async def fake_add_user(uid):
        return True

    async def fake_process_message_for_rag(message, channel_id, **kwargs):
        processed.append((message.id, channel_id))
        return {"message_id": message.id}, False

    monkeypatch.setattr(
        "gregg_limper.memory.rag.consent.add_user", fake_add_user
    )
    monkeypatch.setattr(
        "gregg_limper.commands.handlers.rag_opt.process_message_for_rag",
        fake_process_message_for_rag,
    )

    run(RagOptInCommand.handle(None, cmd_msg, ""))

    assert processed == [(201, 1)]
    assert cmd_channel.sent[0] == "Opted in to RAG. Backfill queued."
    assert cmd_channel.sent[-1].startswith("Backfill complete")

