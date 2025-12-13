from types import SimpleNamespace

import pytest

from tuner.pipeline.relabel import relabel_segment
from tuner.pipeline.types import SegmentedConversation


class DummyAuthor(SimpleNamespace):
    pass


class DummyMessage(SimpleNamespace):
    pass


@pytest.mark.asyncio
async def test_relabel_segment_falls_back_on_formatter_error(monkeypatch):
    segment = SegmentedConversation(
        channel_id=1,
        message_ids=[111],
        assigned_assistant_id=42,
    )
    author = DummyAuthor(id=99, display_name="alex", name="alex")
    message = DummyMessage(
        id=111,
        author=author,
        clean_content="hello world",
        content="hello world",
    )

    async def fake_formatter(_message):  # pragma: no cover - patched behaviour
        raise ValueError("image too large")

    monkeypatch.setattr("tuner.pipeline.relabel.format_for_cache", fake_formatter)

    results = await relabel_segment(
        segment,
        message_lookup={111: message},
    )

    assert len(results) == 1
    entry = results[0]
    assert entry["role"] == "user"
    assert "hello world" in entry["content"]
    assert "Formatter error" in entry["content"]
