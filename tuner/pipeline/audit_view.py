from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

from .types import TrainingSample

PROMPT_PREFIXES = ("### Tools", "### Context")


def extract_human_turns(messages: Sequence[dict]) -> list[dict]:
    turns: list[dict] = []
    for entry in messages:
        role = entry.get("role")
        if role not in ("user", "assistant"):
            continue
        if role == "assistant" and entry.get("tool_calls"):
            continue

        content = _normalize_content(entry.get("content"))
        if role == "assistant" and _looks_like_prompt_header(content):
            continue
        if not content.strip():
            continue

        clean = {"role": role, "content": content}
        for key in ("message_id", "author_id"):
            if key in entry and entry[key] is not None:
                clean[key] = entry[key]
        turns.append(clean)
    return turns


def build_human_record(
    messages: Sequence[dict], metadata: dict | None = None, *, index: int | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "turns": extract_human_turns(messages),
        "metadata": metadata or {},
    }
    if index is not None:
        payload["index"] = index
    return payload


def write_human_records_from_samples(
    samples: Iterable[TrainingSample], destination: Path
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for idx, sample in enumerate(samples, start=1):
        records.append(build_human_record(sample.messages, sample.metadata, index=idx))

    destination.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(records)


def build_human_records_from_files(
    records_path: Path,
    *,
    output_path: Path | None = None,
    metadata_path: Path | None = None,
) -> int:
    output_path = output_path or records_path.with_name(f"{records_path.stem}.audit.json")
    meta_path = metadata_path or records_path.with_name(
        f"{records_path.stem}.metadata.jsonl"
    )
    if not records_path.is_file():
        raise FileNotFoundError(f"Records file not found: {records_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    meta_lines = []
    if meta_path and meta_path.is_file():
        meta_lines = meta_path.read_text(encoding="utf-8").splitlines()

    with records_path.open("r", encoding="utf-8") as record_handle:
        for idx, record_line in enumerate(record_handle, start=1):
            record = json.loads(record_line)
            metadata = {}
            if idx - 1 < len(meta_lines):
                try:
                    metadata = json.loads(meta_lines[idx - 1])
                except json.JSONDecodeError:
                    metadata = {}
            records.append(
                build_human_record(record.get("messages", []), metadata, index=idx)
            )

    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(records)


def _normalize_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                parts.append(chunk["text"])
            else:
                parts.append(str(chunk))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _looks_like_prompt_header(content: str) -> bool:
    stripped = content.lstrip()
    return any(stripped.startswith(prefix) for prefix in PROMPT_PREFIXES)


def parse_keep_tokens(raw: str, total: int) -> tuple[bool, set[int]]:
    raw = raw.strip()
    if not raw:
        return True, set()

    keep_set: set[int] = set()
    for chunk in re.split(r"[,\s]+", raw):
        if not chunk:
            continue
        try:
            value = int(chunk)
            if 1 <= value <= total:
                keep_set.add(value)
        except ValueError:
            continue
    return False, keep_set


__all__ = [
    "build_human_record",
    "build_human_records_from_files",
    "extract_human_turns",
    "write_human_records_from_samples",
    "parse_keep_tokens",
]
