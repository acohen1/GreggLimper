import pytest

from tuner.pipeline import tool_synth


@pytest.mark.asyncio
async def test_inject_synthetic_rag_blocks_inserts_sequence():
    messages = [
        {
            "role": "user",
            "content": "alex said:\nremember when I aced dust 2 in the finals?",
        },
        {"role": "assistant", "content": "I do."},
    ]

    augmented, count = await tool_synth.inject_synthetic_rag_blocks(messages)

    assert count == 1
    assert len(augmented) == 4
    assert augmented[1]["role"] == "assistant" and "tool_calls" in augmented[1]
    assert augmented[2]["role"] == "tool"
    assert "Checked the archive" in augmented[-1]["content"]


@pytest.mark.asyncio
async def test_inject_synthetic_rag_blocks_respects_decider():
    messages = [
        {
            "role": "user",
            "content": "alex said:\nremember when I aced dust 2 in the finals?",
        },
        {"role": "assistant", "content": "I do."},
    ]

    async def deny(_candidate):
        return False

    augmented, count = await tool_synth.inject_synthetic_rag_blocks(
        messages, decider=deny
    )

    assert count == 0
    assert augmented == messages


@pytest.mark.asyncio
async def test_inject_synthetic_rag_blocks_no_trigger():
    messages = [
        {"role": "user", "content": "alex said:\nhello there"},
        {"role": "assistant", "content": "hey"},
    ]

    augmented, count = await tool_synth.inject_synthetic_rag_blocks(messages)

    assert count == 0
    assert augmented == messages


@pytest.mark.asyncio
async def test_build_llm_tool_trigger_decider_invokes_oai(monkeypatch):
    captured = {}

    async def fake_chat(messages, model):
        captured["messages"] = messages
        captured["model"] = model
        return "Yes"

    monkeypatch.setattr(tool_synth.oai, "chat", fake_chat)
    decider = tool_synth.build_llm_tool_trigger_decider("model-x")
    candidate = tool_synth.SyntheticQuery(
        user="alex",
        summary="remember when",
        query="q",
        raw_content="alex said:\nremember when we aced dust 2?",
    )

    assert await decider(candidate) is True
    assert captured["model"] == "model-x"
    assert captured["messages"][1]["role"] == "user"
