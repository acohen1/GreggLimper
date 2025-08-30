"""
Repositories (SQL-only)
=======================
- No embedding logic here; pure CRUD and selects.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple
import asyncio
import sqlite3
import time


class FragmentsRepo:
    def __init__(self, conn: sqlite3.Connection, lock: asyncio.Lock):
        self.conn = conn
        self._lock = lock

    async def insert_or_update_fragment(self, row: tuple[Any, ...]) -> None:
        sql = """
            INSERT INTO fragments (
              server_id, channel_id, message_id, author_id, ts,
              content, type, title, url, media_id,
              embedding, emb_model, emb_dim,
              source_idx, content_hash, last_embedded_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id, source_idx, type, content_hash) DO UPDATE SET
              content=excluded.content,
              title=excluded.title,
              url=excluded.url,
              media_id=excluded.media_id,
              embedding=excluded.embedding,
              emb_model=excluded.emb_model,
              emb_dim=excluded.emb_dim,
              ts=excluded.ts,
              last_embedded_ts=excluded.last_embedded_ts
        """
        def _run():
            with self.conn:
                self.conn.execute(sql, row)

        async with self._lock:
            await asyncio.to_thread(_run)

    async def update_embedding(self, rid: int, emb: bytes, model: str, dim: int) -> None:
        sql = """
            UPDATE fragments
            SET embedding=?, emb_model=?, emb_dim=?, last_embedded_ts=?
            WHERE id=?
        """
        def _run():
            with self.conn:
                self.conn.execute(sql, (emb, model, dim, time.time(), rid))
        
        async with self._lock:
            await asyncio.to_thread(_run)

    async def lookup_fragment_id(
        self,
        message_id: int,
        source_idx: int,
        typ: str,
        content_hash: str,
    ) -> Optional[int]:
        sql = """
            SELECT id FROM fragments
            WHERE message_id=? AND source_idx=? AND type=? AND content_hash=?
        """

        def _query() -> Optional[int]:
            row = self.conn.execute(sql, (message_id, source_idx, typ, content_hash)).fetchone()
            return row[0] if row else None

        async with self._lock:
            return await asyncio.to_thread(_query)
        
    async def message_exists(self, message_id: int) -> bool:
        """Return True if any fragment exists for the given message id."""
        sql = "SELECT 1 FROM fragments WHERE message_id=? LIMIT 1"

        def _query() -> bool:
            return self.conn.execute(sql, (message_id,)).fetchone() is not None
        
        async with self._lock:
            return await asyncio.to_thread(_query)
        
    async def rows_recent(
        self,
        server_id: int,
        channel_id: int,
        time_min: float,
        limit: int = 300,
    ) -> Sequence[Tuple]:
        sql = """
            SELECT id, content, type, ts, embedding
            FROM fragments
            WHERE server_id=? AND channel_id=? AND ts>=?
            ORDER BY ts DESC LIMIT ?
        """
        def _query():
            return self.conn.execute(sql, (server_id, channel_id, time_min, limit)).fetchall()

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def rows_by_ids(self, ids: list[int]) -> Sequence[Tuple]:
        if not ids:
            return []
        ph = ",".join(["?"] * len(ids))
        sql = f"""
            SELECT id, server_id, channel_id, message_id, author_id, ts,
                   content, type, title, url, media_id, source_idx
            FROM fragments
            WHERE id IN ({ph})
        """
        def _query():
            return self.conn.execute(sql, ids).fetchall()

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def fetch_vectors_for_index(self) -> Sequence[Tuple[int, int, int, bytes]]:
        sql = "SELECT id, server_id, channel_id, embedding FROM fragments"

        def _query() -> list[tuple[int, int, int, bytes]]:
            rows = self.conn.execute(sql).fetchall()
            return [
                (int(r["id"]), int(r["server_id"]), int(r["channel_id"]), r["embedding"])
                for r in rows
            ]

        async with self._lock:
            return await asyncio.to_thread(_query)


class MetaRepo:
    def __init__(self, conn: sqlite3.Connection, lock: asyncio.Lock):
        self.conn = conn
        self._lock = lock

    async def set_user_profile(self, user_id: int, blob: str) -> None:
        sql = """
            INSERT INTO user_profiles(user_id, blob) VALUES(?, ?)
            ON CONFLICT(user_id) DO UPDATE SET blob=excluded.blob
        """
        def _run():
            with self.conn:
                self.conn.execute(sql, (user_id, blob))

        async with self._lock:
            await asyncio.to_thread(_run)

    async def get_user_profile(self, user_id: int) -> Optional[str]:
        def _query() -> Optional[str]:
            row = self.conn.execute(
                "SELECT blob FROM user_profiles WHERE user_id=?", (user_id,)
            ).fetchone()
            return row[0] if row else None

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def set_server_style(self, server_id: int, blob: str) -> None:
        sql = """
            INSERT INTO server_styles(server_id, blob) VALUES(?, ?)
            ON CONFLICT(server_id) DO UPDATE SET blob=excluded.blob
        """
        def _run():
            with self.conn:
                self.conn.execute(sql, (server_id, blob))

        async with self._lock:
            await asyncio.to_thread(_run)

    async def get_server_style(self, server_id: int) -> Optional[str]:
        def _query() -> Optional[str]:
            row = self.conn.execute(
                "SELECT blob FROM server_styles WHERE server_id=?", (server_id,)
            ).fetchone()
            return row[0] if row else None

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def set_channel_summary(self, channel_id: int, summary: str) -> None:
        sql = """
            INSERT INTO channel_summaries(channel_id, summary) VALUES(?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET summary=excluded.summary
        """
        def _run():
            with self.conn:
                self.conn.execute(sql, (channel_id, summary))

        async with self._lock:
            await asyncio.to_thread(_run)

    async def get_channel_summary(self, channel_id: int) -> str:
        def _query() -> str:
            row = self.conn.execute(
                "SELECT summary FROM channel_summaries WHERE channel_id=?", (channel_id,)
            ).fetchone()
            return row[0] if row else ""

        async with self._lock:
            return await asyncio.to_thread(_query)


class ConsentRepo:
    def __init__(self, conn: sqlite3.Connection, lock: asyncio.Lock):
        self.conn = conn
        self._lock = lock

    async def is_opted_in(self, user_id: int) -> bool:
        sql = "SELECT 1 FROM rag_consent WHERE user_id=? LIMIT 1"

        def _query() -> bool:
            return self.conn.execute(sql, (user_id,)).fetchone() is not None

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def add_user(self, user_id: int) -> bool:
        sql = "INSERT OR IGNORE INTO rag_consent(user_id, ts) VALUES(?, ?)"

        def _run() -> bool:
            with self.conn:
                cur = self.conn.execute(sql, (user_id, time.time()))
            return cur.rowcount > 0

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def remove_user(self, user_id: int) -> None:
        sql = "DELETE FROM rag_consent WHERE user_id=?"

        def _run() -> None:
            with self.conn:
                self.conn.execute(sql, (user_id,))

        async with self._lock:
            await asyncio.to_thread(_run)
