from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tuner.pipeline.segmenter import propose_segments, refine_segments_with_llm
from tuner.pipeline.types import RawConversation, SegmentCandidate


def _make_message(mid: int, author_id: int, minutes_offset: int) -> SimpleNamespace:
    created = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minutes_offset)
    author = SimpleNamespace(id=author_id, display_name=f"user-{author_id}", name=f"user-{author_id}")
    return SimpleNamespace(
        id=mid,
        author=author,
        created_at=created,
        clean_content=f"message {mid}",
        content=f"message {mid}",
    )


def test_propose_segments_splits_on_large_gap():
    convo = RawConversation(
        channel_id=10,
        guild_id=1,
        messages=[
            _make_message(1, 100, 0),
            _make_message(2, 101, 1),
            _make_message(3, 100, 2),
            _make_message(4, 101, 3),
            _make_message(5, 102, 15),  # > MAX_GAP_SECONDS triggers split
            _make_message(6, 103, 16),
            _make_message(7, 102, 17),
            _make_message(8, 103, 18),
        ],
    )

    segments = propose_segments([convo])

    assert len(segments) == 2
    assert segments[0].message_ids == [1, 2, 3, 4]
    assert segments[1].message_ids == [5, 6, 7, 8]


@pytest.mark.asyncio
async def test_refine_segments_with_llm_uses_custom_decider():
    messages = [
        _make_message(1, 100, 0),
        _make_message(2, 200, 1),
        _make_message(3, 200, 2),
        _make_message(4, 100, 3),
    ]
    message_lookup = {msg.id: msg for msg in messages}
    candidate = SegmentCandidate(channel_id=99, message_ids=[msg.id for msg in messages])

    async def fake_decider(records, allowed):
        return SimpleNamespace(message_ids=[1, 2, 3, 4], assistant_id=200)

    refined = await refine_segments_with_llm(
        [candidate],
        message_lookup=message_lookup,
        config=SimpleNamespace(allowed_user_ids={200}),
        decide_segment=fake_decider,
    )

    assert len(refined) == 1
    assert refined[0].assigned_assistant_id == 200
    assert refined[0].message_ids == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_refine_segments_drops_when_decider_rejects():
    messages = [
        _make_message(1, 100, 0),
        _make_message(2, 200, 1),
        _make_message(3, 200, 2),
        _make_message(4, 100, 3),
    ]
    message_lookup = {msg.id: msg for msg in messages}
    candidate = SegmentCandidate(channel_id=99, message_ids=[msg.id for msg in messages])

    async def fake_decider(records, allowed):
        return None

    refined = await refine_segments_with_llm(
        [candidate],
        message_lookup=message_lookup,
        config=SimpleNamespace(allowed_user_ids={200}),
        decide_segment=fake_decider,
    )

    assert refined == []


@pytest.mark.asyncio
async def test_refine_segments_falls_back_to_heuristics_without_decider():
    messages = [
        _make_message(1, 100, 0),
        _make_message(2, 100, 1),
        _make_message(3, 200, 2),
        _make_message(4, 100, 3),
        _make_message(5, 200, 4),
    ]
    message_lookup = {msg.id: msg for msg in messages}
    candidate = SegmentCandidate(channel_id=1, message_ids=[msg.id for msg in messages])

    refined = await refine_segments_with_llm(
        [candidate],
        message_lookup=message_lookup,
        config=SimpleNamespace(allowed_user_ids=set(), segment_decider_model=None),
        decide_segment=None,
    )

    assert len(refined) == 1
    assert refined[0].assigned_assistant_id in {100, 200}
