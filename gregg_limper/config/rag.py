from dataclasses import dataclass
import os


@dataclass
class Rag:
    SQL_DB_DIR: str = os.getenv("SQL_DB_DIR", "gregg_limper/memory/rag/sql/memory.db")
    EMB_MODEL_ID: str = os.getenv("EMB_MODEL_ID", "text-embedding-3-small")
    EMB_DIM: int = int(os.getenv("EMB_DIM", "1536"))
    MAINTENANCE_INTERVAL: int = int(os.getenv("MAINTENANCE_INTERVAL", "3600"))
    OPT_IN_LOOKBACK_DAYS: int = int(os.getenv("RAG_OPT_IN_LOOKBACK_DAYS", "180"))
    BACKFILL_BATCH_SIZE: int = int(os.getenv("RAG_BACKFILL_BATCH_SIZE", "100"))
    BACKFILL_RATE_LIMIT: float = float(os.getenv("RAG_BACKFILL_RATE_LIMIT", "1"))
