import os
from pathlib import Path
from typing import List

_DEFAULT_SQLITE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "memory.db"
)


def _split_triggers(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


class Rag:
    def __init__(self, config: dict | None = None) -> None:
        rag_cfg = (config or {}).get("gregglimper", {}).get("retrieval", {})
        self.SQL_DB_DIR: str = str(rag_cfg.get("sql_db_dir", os.getenv("SQL_DB_DIR", str(_DEFAULT_SQLITE_PATH))))
        self.EMB_MODEL_ID: str = str(rag_cfg.get("emb_model_id", os.getenv("EMB_MODEL_ID", "text-embedding-3-small")))
        self.EMB_DIM: int = int(rag_cfg.get("emb_dim", os.getenv("EMB_DIM", "1536")))
        self.MAINTENANCE_INTERVAL: int = int(
            rag_cfg.get("maintenance_interval", os.getenv("MAINTENANCE_INTERVAL", "3600"))
        )
        self.OPT_IN_LOOKBACK_DAYS: int = int(
            rag_cfg.get("rag_opt_in_lookback_days", os.getenv("RAG_OPT_IN_LOOKBACK_DAYS", "180"))
        )
        self.BACKFILL_CONCURRENCY: int = int(
            rag_cfg.get("rag_backfill_concurrency", os.getenv("RAG_BACKFILL_CONCURRENCY", "20"))
        )
        triggers_raw = rag_cfg.get("rag_reaction_emojis", os.getenv("RAG_REACTION_EMOJIS", "")) or ""
        self.REACTION_TRIGGERS: List[str] = _split_triggers(triggers_raw)
