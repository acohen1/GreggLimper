from dataclasses import dataclass
import os


@dataclass
class Cache:
    CACHE_LENGTH: int = int(os.getenv("CACHE_LENGTH", "200"))
    MEMO_DIR: str = os.getenv("MEMO_DIR", "gregg_limper/memory/cache/data")
    INIT_CONCURRENCY: int = int(os.getenv("CACHE_INIT_CONCURRENCY", "20"))
    INGEST_CONCURRENCY: int = int(os.getenv("CACHE_INGEST_CONCURRENCY", "20"))
