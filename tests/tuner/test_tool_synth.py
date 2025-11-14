from tuner.pipeline.tool_synth import inject_synthetic_rag_blocks


def test_inject_synthetic_rag_blocks_inserts_sequence():
    messages = [
        {"role": "user", "content": "alex said:\nremember when I aced dust 2?"},
        {"role": "assistant", "content": "I do."},
    ]

    augmented, count = inject_synthetic_rag_blocks(messages)

    assert count == 1
    assert len(augmented) == 4
    assert augmented[1]["role"] == "assistant" and "tool_calls" in augmented[1]
    assert augmented[2]["role"] == "tool"
    assert "Checked the archive" in augmented[-1]["content"]


def test_inject_synthetic_rag_blocks_no_trigger():
    messages = [
        {"role": "user", "content": "alex said:\nhello there"},
        {"role": "assistant", "content": "hey"},
    ]

    augmented, count = inject_synthetic_rag_blocks(messages)

    assert count == 0
    assert augmented == messages
