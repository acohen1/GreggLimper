import os
from pathlib import Path

_DEFAULT_MEMO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


class Cache:
    def __init__(self, config: dict | None = None) -> None:
        cache_cfg = (config or {}).get("gregglimper", {}).get("cache", {})
        self.CACHE_LENGTH: int = int(cache_cfg.get("cache_length", os.getenv("CACHE_LENGTH", "200")))
        self.MEMO_DIR: str = str(cache_cfg.get("memo_dir", os.getenv("MEMO_DIR", str(_DEFAULT_MEMO_DIR))))
        self.INIT_CONCURRENCY: int = int(cache_cfg.get("cache_init_concurrency", os.getenv("CACHE_INIT_CONCURRENCY", "20")))
        self.INGEST_CONCURRENCY: int = int(
            cache_cfg.get("cache_ingest_concurrency", os.getenv("CACHE_INGEST_CONCURRENCY", "20"))
        )
