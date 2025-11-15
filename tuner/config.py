from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set


@dataclass(slots=True)
class DatasetBuildConfig:
    """
    High-level settings for the finetuning dataset builder.

    Attributes:
        channel_ids: Discord channel IDs to hydrate.
        earliest_timestamp: ISO8601 or YYYY-MM-DD cutoff; older messages are ignored.
        allowed_user_ids: Whitelisted Discord user IDs permitted in samples.
        output_path: JSONL destination for the formatted dataset.
        max_messages: Upper bound of raw messages to fetch per channel.
        dry_run: When True, only report stats without writing files.
        segment_decider_model: Override model for segment LLM refinement.
        tool_trigger_model: Override model for synthetic tool trigger checks.
        print_stats: When True, emit run statistics to stdout/logs.
        stats_path: Optional destination for structured run statistics.
        discord_token: API token used by the standalone tuner client.
    """

    channel_ids: List[int]
    earliest_timestamp: str
    allowed_user_ids: Set[int] = field(default_factory=set)
    output_path: Path = field(default_factory=lambda: Path("data/finetune/records.jsonl"))
    raw_dump_dir: Path = field(default_factory=lambda: Path("data/finetune"))
    max_messages: int = 10000
    max_samples: int | None = None
    dry_run: bool = False
    segment_decider_model: str | None = None
    tool_trigger_model: str | None = None
    print_stats: bool = False
    stats_path: Path | None = None
    discord_token: str | None = None


__all__ = ["DatasetBuildConfig"]
