import json

from tuner.pipeline.audit_view import (
    build_human_records_from_files,
    extract_human_turns,
    write_human_records_from_samples,
)
from tuner.pipeline.types import TrainingSample


def test_extract_human_turns_filters_prompts_and_tools():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "### Tools\ncall these things"},
        {"role": "assistant", "content": "### Context\ninfo"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "tool reply"},
        {"role": "user", "content": "alex said:\nhello", "message_id": 1},
        {"role": "assistant", "content": "hey", "message_id": 2},
    ]

    turns = extract_human_turns(messages)

    assert turns == [
        {"role": "user", "content": "alex said:\nhello", "message_id": 1},
        {"role": "assistant", "content": "hey", "message_id": 2},
    ]


def test_write_human_records_from_samples_numbers_segments(tmp_path):
    samples = [
        TrainingSample(messages=[{"role": "user", "content": "u"}]),
        TrainingSample(messages=[{"role": "assistant", "content": "a"}]),
    ]
    out_path = tmp_path / "records.audit.json"

    written = write_human_records_from_samples(samples, out_path)

    assert written == 2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload[0]["index"] == 1
    assert payload[1]["index"] == 2


def test_build_human_records_from_files(tmp_path):
    records_path = tmp_path / "records.jsonl"
    meta_path = tmp_path / "records.metadata.jsonl"
    out_path = tmp_path / "records.audit.json"

    record = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "alex said:\nHello", "message_id": 1},
            {"role": "assistant", "content": "Hey back", "message_id": 2},
        ]
    }
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    meta_path.write_text(json.dumps({"channel_id": 5}) + "\n", encoding="utf-8")

    written = build_human_records_from_files(records_path)

    assert written == 1
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    audit_row = payload[0]
    assert audit_row["index"] == 1
    assert audit_row["metadata"]["channel_id"] == 5
    assert audit_row["turns"][0]["content"] == "alex said:\nHello"
    assert audit_row["turns"][1]["role"] == "assistant"
