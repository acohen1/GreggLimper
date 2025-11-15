import pytest

from tuner.pipeline.formatter import build_prompt_shaped_sample
from tuner.pipeline.types import SegmentedConversation


@pytest.mark.asyncio
async def test_build_prompt_shaped_sample_adds_headers():
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
        synthetic_tool_uses=0,
        context_messages=[{"role": "assistant", "content": "### Context\nfoo"}],
    )

    assert sample is not None
    assert sample.messages[0]["role"] == "system"
    assert sample.messages[1]["content"].startswith("### Tools")
    assert sample.messages[2]["content"].startswith("### Context")
    assert sample.messages[-1]["content"] == "hey"
    assert sample.metadata["channel_id"] == 111
    assert sample.metadata["target_message_id"] == 2
    assert isinstance(sample.tools, list)
    assert sample.parallel_tool_calls is False


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
        synthetic_tool_uses=0,
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
        synthetic_tool_uses=0,
    )

    assert sample is None
