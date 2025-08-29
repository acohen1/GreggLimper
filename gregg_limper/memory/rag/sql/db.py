"""
SQLite bootstrap and connection helpers
=======================================

- Path resolution pinned to this package directory.
- WAL + pragmatic PRAGMAs for decent concurrent read perf.
"""

from __future__ import annotations
from gregg_limper.config import Config
import pathlib
import sqlite3
from typing import Optional


def db_path() -> str:
    here = pathlib.Path(__file__).parent
    return str(here / Config.DB_NAME)


def connect(path: Optional[str] = None) -> sqlite3.Connection:
    # Autocommit; we use explicit `with conn:` blocks in worker threads.
    conn = sqlite3.connect(
        path or db_path(),
        isolation_level=None,
        check_same_thread=False,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )

    # Pragmas: order matters a bit; set WAL first, then tuning.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-65536;")    # ~64 MiB page cache
    conn.execute("PRAGMA mmap_size=268435456;")  # 256 MiB
    # Reduce SQLITE_BUSY errors under contention
    conn.execute("PRAGMA busy_timeout=3000;")    # 3s

    # dict-like rows
    conn.row_factory = sqlite3.Row

    return conn


def migrate(conn: sqlite3.Connection) -> None:
    """
    Execute schema.sql (idempotent). Ensure your schema.sql uses IF NOT EXISTS
    for tables, indexes, and virtual tables.
    """
    schema_file = pathlib.Path(__file__).with_name("schema.sql")
    sql = schema_file.read_text(encoding="utf-8")
    with conn:  # single transaction for the whole migration
        conn.executescript(sql)


def wal_checkpoint_truncate(conn: sqlite3.Connection) -> None:
    """Run a WAL checkpoint + truncate to keep WAL from growing unbounded."""
    # Safe to run periodically (e.g., on shutdown or a background maintenance task)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

