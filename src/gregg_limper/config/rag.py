from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List

_DEFAULT_SQLITE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "memory.db"
)


def _split_triggers(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


@dataclass
class Rag:
    SQL_DB_DIR: str = os.getenv("SQL_DB_DIR", str(_DEFAULT_SQLITE_PATH))               # SQLite file path
    EMB_MODEL_ID: str = os.getenv("EMB_MODEL_ID", "text-embedding-3-small")             # OpenAI embedding model
    EMB_DIM: int = int(os.getenv("EMB_DIM", "1536"))                                    # Embedding dimension (OpenAI text-embedding-3-small)
    MAINTENANCE_INTERVAL: int = int(os.getenv("MAINTENANCE_INTERVAL", "3600"))          # Seconds between maintenance tasks
    OPT_IN_LOOKBACK_DAYS: int = int(os.getenv("RAG_OPT_IN_LOOKBACK_DAYS", "180"))       # How far back to backfill user messages when they opt in to RAG
    BACKFILL_CONCURRENCY: int = int(os.getenv("RAG_BACKFILL_CONCURRENCY", "20"))        # Number of concurrent backfill tasks
    REACTION_TRIGGERS: List[str] = field(
        default_factory=lambda: _split_triggers(os.getenv("RAG_REACTION_EMOJIS", ""))
    )                                                                                   # Emoji strings that trigger ingestion
