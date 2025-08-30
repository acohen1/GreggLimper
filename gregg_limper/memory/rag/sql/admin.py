"""
Maintenance helpers: retention and vacuum
=========================================
"""

from __future__ import annotations
import time
import asyncio

async def retention_prune(
    conn,
    lock: asyncio.Lock,
    older_than_seconds: float, 
    vacuum: bool = False
) -> int:
    def _run():
        cutoff = time.time() - older_than_seconds
        with conn:
            conn.execute(
                "DELETE FROM fragments WHERE rowid IN (SELECT id FROM fragments WHERE ts < ?)",
                (cutoff,),
            )
            cur = conn.execute("DELETE FROM fragments WHERE ts < ?", (cutoff,))
            if vacuum:
                conn.execute("INSERT INTO fragments(fragments) VALUES('optimize')")

        return cur.rowcount
    
    async with lock:
        return await asyncio.to_thread(_run)

async def vacuum(conn, lock: asyncio.Lock) -> None:
    def _run():
        conn.execute("VACUUM")
        conn.execute("ANALYZE")
    
    async with lock:
        await asyncio.to_thread(_run)


