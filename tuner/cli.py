from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Iterable

from .config import DatasetBuildConfig
from .runner import build_dataset


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return ivalue


def _parse_allowed_users(raw: Iterable[str] | None) -> set[int]:
    if not raw:
        return set()
    values: set[int] = set()
    for chunk in raw:
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.add(int(chunk))
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise argparse.ArgumentTypeError(f"invalid user id: {chunk}") from exc
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tuner",
        description="Offline finetuning helpers for Gregg Limper.",
    )
    subparsers = parser.add_subparsers(
        dest="command", metavar="<command>", required=True
    )

    build_cmd = subparsers.add_parser(
        "build-dataset",
        help="Collect, segment, and format training data from Discord history.",
    )
    build_cmd.add_argument(
        "--channels",
        "-c",
        nargs="+",
        type=int,
        required=True,
        help="Discord channel IDs to ingest.",
    )
    build_cmd.add_argument(
        "--earliest",
        type=str,
        required=True,
        help="ISO8601 timestamp or YYYY-MM-DD for the earliest message to include.",
    )
    build_cmd.add_argument(
        "--allowed-users",
        "-u",
        nargs="+",
        type=str,
        help="Discord user IDs permitted in the dataset (space separated).",
    )
    build_cmd.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("finetune-data/records.jsonl"),
        help="Destination JSONL file for the dataset.",
    )
    build_cmd.add_argument(
        "--raw-dump-dir",
        type=Path,
        default=Path("finetune-data/raw"),
        help="Directory to store raw channel transcripts for auditing.",
    )
    build_cmd.add_argument(
        "--max-messages",
        type=_positive_int,
        default=10000,
        help="Safety ceiling for raw messages fetched per channel.",
    )
    build_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect stats without writing any dataset artifacts.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-dataset":
        config = DatasetBuildConfig(
            channel_ids=args.channels,
            earliest_timestamp=args.earliest,
            allowed_user_ids=_parse_allowed_users(args.allowed_users),
            output_path=args.output,
            raw_dump_dir=args.raw_dump_dir,
            max_messages=args.max_messages,
            dry_run=args.dry_run,
        )
        asyncio.run(build_dataset(config))
        return

    parser.print_help()


__all__ = ["main", "build_parser"]
