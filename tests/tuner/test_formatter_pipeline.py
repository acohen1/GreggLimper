import pytest

from tuner.pipeline.formatter import build_prompt_shaped_sample
from tuner.pipeline.types import SegmentedConversation


@pytest.mark.asyncio
async def test_build_prompt_shaped_sample_keeps_dialogue_only():
    segment = SegmentedConversation(
        channel_id=111,
        message_ids=[1, 2],
        assigned_assistant_id=42,
    )
    history = [
        {"role": "user", "content": "alex said:\nhello", "message_id": 1},
        {"role": "assistant", "content": "hey", "message_id": 2},
    ]

    sample = await build_prompt_shaped_sample(
        segment=segment,
        relabeled_history=history,
    )

    assert sample is not None
    assert sample.messages == [
        {"role": "user", "content": "alex said:\nhello"},
        {"role": "assistant", "content": "hey"},
    ]
    assert all(entry["role"] in ("user", "assistant") for entry in sample.messages)
    assert sample.messages[-1]["content"] == "hey"
    assert sample.metadata["channel_id"] == 111
    assert sample.metadata["target_message_id"] == 2


@pytest.mark.asyncio
async def test_build_prompt_shaped_sample_requires_assistant():
    segment = SegmentedConversation(
        channel_id=5,
        message_ids=[1],
        assigned_assistant_id=99,
    )
    history = [{"role": "user", "content": "alex said:\nhello"}]

    sample = await build_prompt_shaped_sample(
        segment=segment,
        relabeled_history=history,
    )

    assert sample is None


@pytest.mark.asyncio
async def test_build_prompt_shaped_sample_skips_when_last_turn_user():
    segment = SegmentedConversation(
        channel_id=77,
        message_ids=[1, 2, 3],
        assigned_assistant_id=10,
    )
    history = [
        {"role": "assistant", "content": "first reply", "message_id": 1},
        {"role": "user", "content": "follow-up", "message_id": 2},
        {"role": "user", "content": "final user", "message_id": 3},
    ]

    sample = await build_prompt_shaped_sample(
        segment=segment,
        relabeled_history=history,
    )

    assert sample is None


@pytest.mark.asyncio
async def test_build_prompt_shaped_sample_drops_tool_entries():
    segment = SegmentedConversation(
        channel_id=555,
        message_ids=[1, 2, 3],
        assigned_assistant_id=1,
    )
    history = [
        {"role": "user", "content": "hi", "message_id": 1},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"foo": "bar"}],
            "message_id": 2,
        },
        {"role": "assistant", "content": "reply", "message_id": 3},
    ]

    sample = await build_prompt_shaped_sample(
        segment=segment,
        relabeled_history=history,
    )

    assert sample is not None
    assert sample.messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "reply"},
    ]
