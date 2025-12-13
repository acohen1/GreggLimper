import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from tuner.pipeline.collector import persist_raw_conversations
from tuner.pipeline.types import RawConversation


def _make_msg(mid: int, content: str) -> SimpleNamespace:
    author = SimpleNamespace(id=mid * 10, display_name=f"user-{mid}", name=f"user-{mid}")
    return SimpleNamespace(
        id=mid,
        author=author,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        clean_content=content,
        content=content,
        attachments=[],
    )


def test_persist_raw_conversations(tmp_path: Path):
    convo = RawConversation(
        channel_id=123,
        guild_id=1,
        messages=[_make_msg(1, "hello"), _make_msg(2, "world")],
    )

    persist_raw_conversations([convo], destination=tmp_path)

    dump = tmp_path / "123.jsonl"
    assert dump.exists()
    lines = dump.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["author"] == "user-1"
