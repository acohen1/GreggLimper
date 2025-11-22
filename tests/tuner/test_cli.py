from pathlib import Path

import pytest

from tuner.cli import build_parser, _resolve_dataset_config, _run_audit_command


def test_resolve_dataset_config_from_toml(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[dataset]
channels = [123, 456]
allowed_users = [10, 11]
earliest = "2024-01-01"
max_messages = 50
max_samples = 5
output_path = "custom.jsonl"
raw_dump_dir = "raw"
dry_run = true
print_stats = true
stats_path = "stats.json"

[models]
segment = "seg-model"
tool_trigger = "tool-model"

[discord]
token_env = "TEST_TOKEN"
"""
    )
    monkeypatch.setenv("TEST_TOKEN", "disc-token")

    parser = build_parser()
    args = parser.parse_args(
        [
            "build-dataset",
            "--config",
            str(cfg),
            "--max-messages",
            "75",
            "--segment-concurrency",
            "12",
            "--max-samples",
            "12",
            "--dry-run",
        ]
    )

    config = _resolve_dataset_config(args, parser)

    assert config.channel_ids == [123, 456]
    assert config.allowed_user_ids == {10, 11}
    assert config.earliest_timestamp == "2024-01-01"
    assert config.max_messages == 75
    assert config.max_samples == 12
    assert config.dry_run is True
    assert config.print_stats is True
    assert config.output_path == Path("custom.jsonl")
    assert config.raw_dump_dir == Path("raw")
    assert config.stats_path == Path("stats.json")
    assert config.segment_decider_model == "seg-model"
    assert config.tool_trigger_model == "tool-model"
    assert config.discord_token == "disc-token"
    assert config.segment_decider_concurrency == 12


def test_resolve_dataset_config_missing_config(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "build-dataset",
        "--config",
        str(tmp_path / "missing.toml"),
    ])

    with pytest.raises(SystemExit):
        _resolve_dataset_config(args, parser)


def test_audit_records_subcommand(tmp_path, capsys):
    parser = build_parser()
    records_path = tmp_path / "records.jsonl"
    meta_path = tmp_path / "records.metadata.jsonl"
    audit_path = tmp_path / "records.audit.json"
    final_path = tmp_path / "records.final.jsonl"
    final_meta_path = tmp_path / "records.final.metadata.jsonl"

    records_path.write_text(
        '{"messages": [{"role": "user", "content": "alex said:\\nhello"}, {"role": "assistant", "content": "sup"}]}\n',
        encoding="utf-8",
    )
    meta_path.write_text('{"channel_id": 1}\n', encoding="utf-8")

    args = parser.parse_args(["audit-records", "--records", str(records_path)])

    # Keep only first segment
    input_values = iter(["1"])
    capsys.readouterr()
    import builtins

    original_input = builtins.input
    builtins.input = lambda _: next(input_values)
    try:
        _run_audit_command(args, parser)
    finally:
        builtins.input = original_input

    assert not audit_path.exists()
    assert final_path.is_file()
    assert final_meta_path.is_file()
    payload = final_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(payload) == 1
    assert '"role": "user"' in payload[0]
    assert '"channel_id": 1' in final_meta_path.read_text(encoding="utf-8")
