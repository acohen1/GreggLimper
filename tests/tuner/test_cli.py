from pathlib import Path

import pytest

from tuner.cli import build_parser, _resolve_dataset_config


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
segment_leniency = "default"

[models]
segment = "seg-model"

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
    assert config.segment_decider_leniency == "default"
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
