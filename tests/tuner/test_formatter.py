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
        {"role": "user", "content": "alex said:\nhello"},
        {"role": "assistant", "content": "hey"}
    ]

    sample = await build_prompt_shaped_sample(
        segment=segment,
        relabeled_history=history,
        synthetic_tool_uses=0,
        context_messages=[{"role": "assistant", "content": "### Context\nfoo"}],
    )

    assert sample.messages[0]["role"] == "system"
    assert sample.messages[1]["content"].startswith("### Tools")
    assert sample.messages[2]["content"].startswith("### Context")
    assert sample.messages[-1]["content"] == "hey"
    assert sample.metadata["channel_id"] == 111
