from dataclasses import dataclass
import os


@dataclass
class Cache:
    CACHE_LENGTH: int = int(os.getenv("CACHE_LENGTH", "1000"))
    MEMO_DIR: str = os.getenv("MEMO_DIR", "gregg_limper/memory/cache/data")
