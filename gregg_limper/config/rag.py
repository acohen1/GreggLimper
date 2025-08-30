from dataclasses import dataclass
import os


@dataclass
class Rag:
    SQL_DB_DIR: str = os.getenv("SQL_DB_DIR", "gregg_limper/memory/rag/sql/memory.db")
    EMB_MODEL_ID: str = os.getenv("EMB_MODEL_ID", "text-embedding-3-small")
    EMB_DIM: int = int(os.getenv("EMB_DIM", "1536"))
    MAINTENANCE_INTERVAL: int = int(os.getenv("MAINTENANCE_INTERVAL", "3600"))
