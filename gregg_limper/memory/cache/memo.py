"""Persistent memo store for cached message fragments.

Each Discord channel gets a gzipped JSON file:
    {msg_id: {"author": str, "fragments": [Fragment-as-dict, ...]}, ...}

Helpers:
    - exists(channel_id)
    - load(channel_id)
    - prune(channel_id, memo_dict)
    - save(channel_id, memo_dict)
"""
from __future__ import annotations

from pathlib import Path
import gzip
import json
import os
from typing import Dict

from gregg_limper.config import cache
from gregg_limper.formatter.model import fragment_from_dict, Fragment


def _memo_dir() -> Path:
    return Path(cache.MEMO_DIR)


def _path(channel_id: int) -> Path:
    d = _memo_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{channel_id}.json.gz"


def exists(channel_id: int) -> bool:
    return _path(channel_id).exists()


def load(channel_id: int) -> Dict[int, dict]:
    p = _path(channel_id)
    if not p.exists():
        return {}
    with gzip.open(p, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[int, dict] = {}
    for k, v in raw.items():
        frags = [fragment_from_dict(fd) for fd in v.get("fragments", [])]
        out[int(k)] = {"author": v.get("author"), "fragments": frags}
    return out


def prune(channel_id: int, memo_dict: Dict[int, dict]) -> Dict[int, dict]:
    if len(memo_dict) <= cache.CACHE_LENGTH:
        return memo_dict
    items = list(memo_dict.items())[-cache.CACHE_LENGTH:]
    return dict(items)


def save(channel_id: int, memo_dict: Dict[int, dict]) -> None:
    p = _path(channel_id)
    tmp = p.with_suffix(".tmp")
    serializable = {
        str(k): {
            "author": v.get("author"),
            "fragments": [f.to_dict() for f in v.get("fragments", [])],
        }
        for k, v in memo_dict.items()
    }
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(serializable, f)
    os.replace(tmp, p)
