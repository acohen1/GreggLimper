from dataclasses import dataclass
import os
from pathlib import Path

_DEFAULT_MEMO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


@dataclass
class Cache:
    CACHE_LENGTH: int = int(os.getenv("CACHE_LENGTH", "200"))
    MEMO_DIR: str = os.getenv("MEMO_DIR", str(_DEFAULT_MEMO_DIR))
    INIT_CONCURRENCY: int = int(os.getenv("CACHE_INIT_CONCURRENCY", "20"))
    INGEST_CONCURRENCY: int = int(os.getenv("CACHE_INGEST_CONCURRENCY", "20"))
