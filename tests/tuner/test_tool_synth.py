import pytest

from tuner.pipeline import tool_synth


@pytest.mark.asyncio
async def test_inject_synthetic_rag_blocks_is_noop():
    messages = [
        {"role": "user", "content": "alex said:\nremember when we queued?"},
        {"role": "assistant", "content": "I remember."},
    ]

    augmented, count = await tool_synth.inject_synthetic_rag_blocks(messages)

    assert count == 0
    assert augmented == messages


def test_synthetic_query_structure():
    payload = tool_synth.SyntheticQuery(
        user="alex",
        summary="summary",
        query="lookup",
        raw_content="full text",
    )

    assert payload.user == "alex"
    assert payload.query == "lookup"
