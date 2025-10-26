from dataclasses import dataclass
import os
from pathlib import Path

_DEFAULT_SQLITE_PATH = (
    Path(__file__).resolve().parent.parent / "memory" / "rag" / "sql" / "memory.db"
)


@dataclass
class Rag:
    SQL_DB_DIR: str = os.getenv("SQL_DB_DIR", str(_DEFAULT_SQLITE_PATH))               # SQLite file path
    EMB_MODEL_ID: str = os.getenv("EMB_MODEL_ID", "text-embedding-3-small")             # OpenAI embedding model
    EMB_DIM: int = int(os.getenv("EMB_DIM", "1536"))                                    # Embedding dimension (OpenAI text-embedding-3-small)
    MAINTENANCE_INTERVAL: int = int(os.getenv("MAINTENANCE_INTERVAL", "3600"))          # Seconds between maintenance tasks
    OPT_IN_LOOKBACK_DAYS: int = int(os.getenv("RAG_OPT_IN_LOOKBACK_DAYS", "180"))       # How far back to backfill user messages when they opt in to RAG
    BACKFILL_BATCH_SIZE: int = int(os.getenv("RAG_BACKFILL_BATCH_SIZE", "100"))         # Number of messages per batch when fetching from Discord for backfilling
    BACKFILL_RATE_LIMIT: float = float(os.getenv("RAG_BACKFILL_RATE_LIMIT", "1"))       # Rate limit for backfilling (requests per second)
    BACKFILL_CONCURRENCY: int = int(os.getenv("RAG_BACKFILL_CONCURRENCY", "20"))        # Number of concurrent backfill tasks
