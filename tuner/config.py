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
    """

    channel_ids: List[int]
    earliest_timestamp: str
    allowed_user_ids: Set[int] = field(default_factory=set)
    output_path: Path = field(default_factory=lambda: Path("finetune-data/records.jsonl"))
    raw_dump_dir: Path = field(default_factory=lambda: Path("finetune-data/raw"))
    max_messages: int = 10000
    dry_run: bool = False


__all__ = ["DatasetBuildConfig"]
