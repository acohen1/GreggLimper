from __future__ import annotations

import argparse
import asyncio
import os
import tomllib
from pathlib import Path
from typing import Any, Iterable, List

from .config import DatasetBuildConfig
from .runner import build_dataset

DEFAULT_OUTPUT_PATH = Path("data/finetune/records.jsonl")
DEFAULT_RAW_DUMP_DIR = Path("data/finetune/raw")
DEFAULT_SEGMENT_DIR = Path("data/finetune/segments")
DEFAULT_STATS_PATH = Path("data/finetune/stats.json")
DEFAULT_CONFIG_PATH = Path("config.toml")
DEFAULT_MAX_MESSAGES = 10000
DEFAULT_SEGMENT_CONCURRENCY = 4
DEFAULT_MAX_SAMPLES = None


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
        "--config",
        type=Path,
        default=None,
        help="Path to tuner config TOML (defaults to tuner/config.toml when present).",
    )
    build_cmd.add_argument(
        "--channels",
        "-c",
        nargs="+",
        type=int,
        default=None,
        help="Discord channel IDs to ingest (overrides config).",
    )
    build_cmd.add_argument(
        "--earliest",
        type=str,
        default=None,
        help="ISO8601 timestamp or YYYY-MM-DD for the earliest message to include (overrides config).",
    )
    build_cmd.add_argument(
        "--allowed-users",
        "-u",
        nargs="+",
        type=str,
        default=None,
        help="Discord user IDs permitted in the dataset (space separated, overrides config).",
    )
    build_cmd.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Destination JSONL file for the dataset (overrides config).",
    )
    build_cmd.add_argument(
        "--raw-dump-dir",
        type=Path,
        default=None,
        help="Directory to store raw channel transcripts for auditing (overrides config).",
    )
    build_cmd.add_argument(
        "--max-messages",
        type=_positive_int,
        default=None,
        help="Safety ceiling for raw messages fetched per channel (overrides config).",
    )
    build_cmd.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        help="Limit the number of formatted samples written (overrides config).",
    )
    build_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect stats without writing any dataset artifacts (overrides config).",
    )
    build_cmd.add_argument(
        "--segment-model",
        type=str,
        default=None,
        help="Model ID to use for segment boundary LLM refinement (overrides config).",
    )
    build_cmd.add_argument(
        "--segment-concurrency",
        type=_positive_int,
        default=None,
        help="Maximum concurrent LLM refinement calls (overrides config).",
    )
    build_cmd.add_argument(
        "--moderation-model",
        type=str,
        default=None,
        help="OpenAI model ID used to screen formatted samples (e.g., omni-moderation-2024-09-26).",
    )
    build_cmd.add_argument(
        "--relevance-model",
        type=str,
        default=None,
        help="OpenAI model ID used to confirm assistant replies are relevant to the last user turn.",
    )
    build_cmd.add_argument(
        "--segment-dir",
        type=Path,
        default=None,
        help="Directory to cache refined segments (overrides config).",
    )
    build_cmd.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Reuse previously persisted raw Discord dumps when available.",
    )
    build_cmd.add_argument(
        "--reuse-segments",
        action="store_true",
        help="Reuse cached refined segments instead of re-running the LLM.",
    )
    build_cmd.add_argument(
        "--assistant-emojis",
        nargs="+",
        default=None,
        help="Custom emoji tokens permitted in assistant turns (overrides config).",
    )
    build_cmd.add_argument(
        "--scrub-pii",
        action="store_true",
        help="Anonymize user identifiers in the exported dataset.",
    )
    build_cmd.add_argument(
        "--print-stats",
        action="store_true",
        help="Print aggregated run statistics after completion (overrides config).",
    )
    build_cmd.add_argument(
        "--stats-file",
        type=Path,
        default=None,
        help="JSON file to write aggregated run statistics (overrides config).",
    )
    build_cmd.add_argument(
        "--discord-token",
        type=str,
        default=None,
        help="Discord API token used to hydrate history (overrides discord.token_env).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-dataset":
        config = _resolve_dataset_config(args, parser)
        asyncio.run(build_dataset(config))
        return

    parser.print_help()


def _resolve_dataset_config(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> DatasetBuildConfig:
    file_config = _load_file_config(args.config, parser)
    finetune_cfg = file_config.get("finetune", {}) if isinstance(file_config, dict) else {}
    dataset_cfg = finetune_cfg.get("dataset", {}) or file_config.get("dataset", {})
    models_cfg = finetune_cfg.get("models", {}) or file_config.get("models", {})
    discord_cfg = finetune_cfg.get("discord", {}) or file_config.get("discord", {})

    try:
        channels = args.channels or _coerce_int_list(
            dataset_cfg.get("channels"), "dataset.channels"
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not channels:
        parser.error(
            "Missing channel IDs. Supply --channels or set dataset.channels in config.toml."
        )

    earliest = args.earliest or dataset_cfg.get("earliest")
    if not earliest:
        parser.error("Missing earliest timestamp. Supply --earliest or dataset.earliest in config.toml.")

    if args.allowed_users:
        allowed_user_ids = _parse_allowed_users(args.allowed_users)
    else:
        try:
            allowed_user_ids = set(
                _coerce_int_list(dataset_cfg.get("allowed_users"), "dataset.allowed_users")
            )
        except ValueError as exc:
            parser.error(str(exc))

    max_messages = (
        args.max_messages
        if args.max_messages is not None
        else int(dataset_cfg.get("max_messages", DEFAULT_MAX_MESSAGES))
    )
    max_samples_cfg = dataset_cfg.get("max_samples")
    max_samples = args.max_samples
    if max_samples is None and max_samples_cfg is not None:
        try:
            max_samples = int(max_samples_cfg)
        except (TypeError, ValueError) as exc:
            parser.error("dataset.max_samples must be a positive integer")
    if max_samples is not None and max_samples <= 0:
        parser.error("max samples must be greater than zero")

    seg_concurrency_cfg = dataset_cfg.get("segment_concurrency")
    segment_concurrency = args.segment_concurrency
    if segment_concurrency is None and seg_concurrency_cfg is not None:
        try:
            segment_concurrency = int(seg_concurrency_cfg)
        except (TypeError, ValueError) as exc:
            parser.error("segment_concurrency must be a positive integer")
    if segment_concurrency is None:
        segment_concurrency = DEFAULT_SEGMENT_CONCURRENCY
    if segment_concurrency <= 0:
        parser.error("segment concurrency must be greater than zero")
    output_path = _coerce_path(
        args.output or dataset_cfg.get("output_path"), DEFAULT_OUTPUT_PATH
    )
    raw_dump_dir = _coerce_path(
        args.raw_dump_dir or dataset_cfg.get("raw_dump_dir"), DEFAULT_RAW_DUMP_DIR
    )
    stats_path = _coerce_path(
        args.stats_file or dataset_cfg.get("stats_path"), DEFAULT_STATS_PATH
    )
    segment_dump_dir = _coerce_path(
        args.segment_dir or dataset_cfg.get("segment_dir"), DEFAULT_SEGMENT_DIR
    )
    dry_run = args.dry_run or bool(dataset_cfg.get("dry_run", False))
    print_stats = args.print_stats or bool(dataset_cfg.get("print_stats", False))
    reuse_raw = bool(args.reuse_raw or dataset_cfg.get("reuse_raw", False))
    reuse_segments = bool(args.reuse_segments or dataset_cfg.get("reuse_segments", False))
    if args.assistant_emojis is not None:
        assistant_emojis = {emoji for emoji in args.assistant_emojis if emoji}
    else:
        assistant_cfg = dataset_cfg.get("assistant_custom_emojis") or []
        assistant_emojis = {str(item) for item in assistant_cfg if str(item)}

    segment_model = args.segment_model or models_cfg.get("segment")
    moderation_model = args.moderation_model or models_cfg.get("moderation")
    relevance_model = args.relevance_model or models_cfg.get("relevance") or segment_model
    if not segment_model:
        parser.error("Missing segment model. Provide --segment-model or models.segment in config.toml.")

    scrub_pii = args.scrub_pii or bool(dataset_cfg.get("scrub_pii", False))

    discord_token = _resolve_discord_token(args.discord_token, discord_cfg)
    if not discord_token:
        parser.error(
            "Missing Discord token. Provide --discord-token, discord.token, or ensure discord.token_env points to a populated env var."
        )

    return DatasetBuildConfig(
        channel_ids=channels,
        earliest_timestamp=earliest,
        allowed_user_ids=allowed_user_ids,
        output_path=output_path,
        raw_dump_dir=raw_dump_dir,
        max_messages=max_messages,
        max_samples=max_samples,
        dry_run=dry_run,
        segment_decider_model=segment_model,
        moderation_model=moderation_model,
        relevance_model=relevance_model,
        scrub_pii=scrub_pii,
        segment_decider_concurrency=segment_concurrency,
        allowed_assistant_custom_emojis=assistant_emojis,
        segment_dump_dir=segment_dump_dir,
        reuse_raw=reuse_raw,
        reuse_segments=reuse_segments,
        print_stats=print_stats,
        stats_path=stats_path,
        discord_token=discord_token,
    )


def _coerce_int_list(value: Any, label: str) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        result: list[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{label} contains invalid integer value '{item}'"
                ) from exc
        return result
    raise ValueError(f"{label} must be a list of integers in config")


def _coerce_path(value: Any, default: Path) -> Path:
    if value is None:
        return default
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _load_file_config(
    path: Path | None, parser: argparse.ArgumentParser
) -> dict[str, Any]:
    candidates = []
    if path:
        candidates.append(Path(path))
    else:
        candidates.append(DEFAULT_CONFIG_PATH)
        candidates.append(Path("tuner/config.toml"))

    for candidate in candidates:
        if candidate.is_file():
            return _read_toml(candidate)

    if path:
        parser.error(f"Config file {path} does not exist.")
    return {}


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return data


def _resolve_discord_token(cli_token: str | None, discord_cfg: dict[str, Any]) -> str | None:
    if cli_token:
        return cli_token
    token_env = discord_cfg.get("token_env", "DISCORD_API_TOKEN")
    if token_env:
        return os.getenv(str(token_env))
    return None


__all__ = ["main", "build_parser", "_resolve_dataset_config"]
