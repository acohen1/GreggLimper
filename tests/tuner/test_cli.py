from pathlib import Path

from tuner.cli import build_parser


def test_build_parser_parses_dataset_args(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "build-dataset",
        "--channels", "123", "456",
        "--earliest", "2024-01-01",
        "--allowed-users", "10", "11",
        "--output", str(tmp_path / "out.jsonl"),
        "--max-messages", "50",
        "--raw-dump-dir", str(tmp_path / "raw"),
        "--dry-run",
    ])

    assert args.command == "build-dataset"
    assert args.channels == [123, 456]
    assert args.earliest == "2024-01-01"
    assert args.allowed_users == ["10", "11"]
    assert Path(args.output) == tmp_path / "out.jsonl"
    assert Path(args.raw_dump_dir) == tmp_path / "raw"
    assert args.max_messages == 50
    assert args.dry_run is True
