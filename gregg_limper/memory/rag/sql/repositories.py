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
    """Async CRUD helpers for the ``fragments`` table."""

    def __init__(self, conn: sqlite3.Connection, lock: asyncio.Lock):
        self.conn = conn
        self._lock = lock

    async def insert_or_update_fragment(self, row: tuple[Any, ...]) -> None:
        """
        Insert or update a fragment row.

        :param row: Column values matching the table schema.
        """
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
            await asyncio.to_thread(_run)  # blocking sqlite call

    async def update_embedding(self, rid: int, emb: bytes, model: str, dim: int) -> None:
        """
        Update embedding fields for a fragment.

        :param rid: Fragment row id.
        :param emb: Serialized embedding blob.
        :param model: Embedding model id.
        :param dim: Embedding dimension.
        """
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
        """Return fragment id for a (message, index, type, hash) tuple."""
        sql = """
            SELECT id FROM fragments
            WHERE message_id=? AND source_idx=? AND type=? AND content_hash=?
        """

        def _query() -> Optional[int]:
            row = self.conn.execute(sql, (message_id, source_idx, typ, content_hash)).fetchone()
            return row[0] if row else None

        async with self._lock:
            return await asyncio.to_thread(_query)  # blocking sqlite call
        
    async def message_exists(self, message_id: int) -> bool:
        """Return True if any fragment exists for the given message id."""
        sql = "SELECT 1 FROM fragments WHERE message_id=? LIMIT 1"

        def _query() -> bool:
            return self.conn.execute(sql, (message_id,)).fetchone() is not None
        
        async with self._lock:
            return await asyncio.to_thread(_query)  # blocking sqlite call
        
    async def rows_recent(
        self,
        server_id: int,
        channel_id: int,
        time_min: float,
        limit: int = 300,
    ) -> Sequence[Tuple]:
        """Return rows newer than ``time_min`` for a channel."""
        sql = """
            SELECT id, content, type, ts, embedding
            FROM fragments
            WHERE server_id=? AND channel_id=? AND ts>=?
            ORDER BY ts DESC LIMIT ?
        """
        def _query():
            return self.conn.execute(sql, (server_id, channel_id, time_min, limit)).fetchall()

        async with self._lock:
            return await asyncio.to_thread(_query)  # blocking sqlite call

    async def rows_by_ids(self, ids: list[int]) -> Sequence[Tuple]:
        """Fetch rows by primary key list."""
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
            return await asyncio.to_thread(_query)  # blocking sqlite call

    async def fetch_vectors_for_index(self) -> Sequence[Tuple[int, int, int, bytes]]:
        """Return ``(id, server_id, channel_id, embedding)`` rows for vector sync."""
        sql = "SELECT id, server_id, channel_id, embedding FROM fragments"

        def _query() -> list[tuple[int, int, int, bytes]]:
            rows = self.conn.execute(sql).fetchall()
            return [
                (int(r["id"]), int(r["server_id"]), int(r["channel_id"]), r["embedding"])
                for r in rows
            ]

        async with self._lock:
            return await asyncio.to_thread(_query)  # blocking sqlite call


class MetaRepo:
    """Repository for miscellaneous metadata tables."""

    def __init__(self, conn: sqlite3.Connection, lock: asyncio.Lock):
        self.conn = conn
        self._lock = lock

    async def set_user_profile(self, user_id: int, blob: str) -> None:
        """
        Upsert a JSON profile for ``user_id``.

        :param user_id: Discord user id.
        :param blob: JSON string representing the profile.
        :returns: ``None``.
        """
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
        """
        Return JSON profile blob for ``user_id`` or ``None``.

        :param user_id: Discord user id.
        :returns: Stored JSON string or ``None``.
        """
        def _query() -> Optional[str]:
            row = self.conn.execute(
                "SELECT blob FROM user_profiles WHERE user_id=?", (user_id,)
            ).fetchone()
            return row[0] if row else None

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def set_server_style(self, server_id: int, blob: str) -> None:
        """
        Upsert a style/config blob for ``server_id``.

        :param server_id: Discord server id.
        :param blob: JSON string representing the style.
        :returns: ``None``.
        """
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
        """
        Return style/config blob for ``server_id`` or ``None``.

        :param server_id: Discord server id.
        :returns: Stored JSON string or ``None``.
        """
        def _query() -> Optional[str]:
            row = self.conn.execute(
                "SELECT blob FROM server_styles WHERE server_id=?", (server_id,)
            ).fetchone()
            return row[0] if row else None

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def set_channel_summary(self, channel_id: int, summary: str) -> None:
        """
        Upsert summary text for a channel.

        :param channel_id: Channel id.
        :param summary: Summary text to store.
        :returns: ``None``.
        """
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
        """
        Return summary text for a channel or empty string.

        :param channel_id: Channel id.
        :returns: Stored summary text.
        """
        def _query() -> str:
            row = self.conn.execute(
                "SELECT summary FROM channel_summaries WHERE channel_id=?", (channel_id,)
            ).fetchone()
            return row[0] if row else ""

        async with self._lock:
            return await asyncio.to_thread(_query)


class ConsentRepo:
    """Simple opt-in/opt-out registry."""

    def __init__(self, conn: sqlite3.Connection, lock: asyncio.Lock):
        self.conn = conn
        self._lock = lock

    async def is_opted_in(self, user_id: int) -> bool:
        """
        Return ``True`` if ``user_id`` is present in consent table.

        :param user_id: Discord user id.
        :returns: ``True`` if user has opted in.
        """
        sql = "SELECT 1 FROM rag_consent WHERE user_id=? LIMIT 1"

        def _query() -> bool:
            return self.conn.execute(sql, (user_id,)).fetchone() is not None

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def add_user(self, user_id: int) -> bool:
        """
        Insert ``user_id`` into consent table.

        :param user_id: Discord user id.
        :returns: ``True`` if a new row was added.
        """
        sql = "INSERT OR IGNORE INTO rag_consent(user_id, ts) VALUES(?, ?)"

        def _run() -> bool:
            with self.conn:
                cur = self.conn.execute(sql, (user_id, time.time()))
            return cur.rowcount > 0

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def remove_user(self, user_id: int) -> None:
        """
        Delete ``user_id`` from consent table.

        :param user_id: Discord user id.
        :returns: ``None``.
        """
        sql = "DELETE FROM rag_consent WHERE user_id=?"

        def _run() -> None:
            with self.conn:
                self.conn.execute(sql, (user_id,))

        async with self._lock:
            await asyncio.to_thread(_run)
