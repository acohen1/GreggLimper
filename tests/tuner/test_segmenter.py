from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tuner.pipeline.segmenter import (
    propose_segments,
    refine_segments_with_llm,
    _eligible_assistant_ids,
)
from tuner.pipeline.types import RawConversation, SegmentCandidate


def _make_message(
    mid: int,
    author_id: int,
    minutes_offset: int,
    *,
    content: str | None = None,
    attachments: list | None = None,
    embeds: list | None = None,
    fragments: list | None = None,
    classified: dict | None = None,
    stickers: list | None = None,
) -> SimpleNamespace:
    created = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minutes_offset)
    author = SimpleNamespace(id=author_id, display_name=f"user-{author_id}", name=f"user-{author_id}")
    body = content if content is not None else f"message {mid}"
    return SimpleNamespace(
        id=mid,
        author=author,
        created_at=created,
        clean_content=body,
        content=body,
        attachments=attachments or [],
        embeds=embeds or [],
        fragments=fragments,
        classified_fragments=classified or {},
        stickers=stickers or [],
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
        _make_message(3, 100, 2),
        _make_message(4, 200, 3),
    ]
    message_lookup = {msg.id: msg for msg in messages}
    candidate = SegmentCandidate(channel_id=99, message_ids=[msg.id for msg in messages])

    async def fake_decider(records, allowed):
        return SimpleNamespace(message_ids=[1, 2, 3, 4], assistant_id=200)

    refined = await refine_segments_with_llm(
        [candidate],
        message_lookup=message_lookup,
        config=SimpleNamespace(
            allowed_user_ids={200},
            allowed_assistant_custom_emojis=set(),
        ),
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
        config=SimpleNamespace(
            allowed_user_ids={200},
            allowed_assistant_custom_emojis=set(),
        ),
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
        config=SimpleNamespace(
            allowed_user_ids=set(),
            segment_decider_model=None,
            allowed_assistant_custom_emojis=set(),
        ),
        decide_segment=None,
    )

    assert len(refined) == 1
    assert refined[0].assigned_assistant_id in {100, 200}


def test_eligible_assistant_ids_requires_plaintext():
    media_author = 100
    plain_author = 200
    messages = [
        _make_message(
            1,
            media_author,
            0,
            attachments=[SimpleNamespace(url="https://cdn.discordapp.com/image.png")],
        ),
        _make_message(2, plain_author, 1, content="hello there"),
        _make_message(3, media_author, 2, content="still text but previous attachment disqualifies me"),
    ]

    eligible = _eligible_assistant_ids(
        messages,
        assigned_ids=set(),
        emoji_whitelist=[],
    )

    assert plain_author in eligible
    assert media_author not in eligible


def test_eligible_assistant_ids_blocks_consecutive_turns():
    author = 300
    other = 301
    messages = [
        _make_message(1, author, 0),
        _make_message(2, author, 1),
        _make_message(3, other, 2),
    ]

    eligible = _eligible_assistant_ids(
        messages,
        assigned_ids=set(),
        emoji_whitelist=[],
    )

    assert other in eligible
    assert author not in eligible


def test_eligible_assistant_ids_enforces_custom_emoji_whitelist():
    allowed = 400
    blocked = 401
    messages = [
        _make_message(1, allowed, 0, content="thanks <:SoloPog:123456>"),
        _make_message(2, blocked, 1, content="wow <:OtherEmoji:654321> neat"),
    ]

    eligible = _eligible_assistant_ids(
        messages,
        assigned_ids=set(),
        emoji_whitelist=[":SoloPog:"],
    )

    assert allowed in eligible
    assert blocked not in eligible
