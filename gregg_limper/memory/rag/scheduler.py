"""Schedule SQL and vector maintenance tasks."""

from __future__ import annotations

import asyncio
from typing import Tuple

from gregg_limper.maintenance import startup as _startup, shutdown as _shutdown
from gregg_limper.config import rag
from .sql import sql_tasks
from .vector import vector_tasks
from . import _conn as _default_conn, _db_lock as _default_lock
import logging

logger = logging.getLogger(__name__)

_sql_task: asyncio.Task | None = None
_vec_task: asyncio.Task | None = None


async def start(interval: float = 3600, *, conn=None, lock=None) -> Tuple[asyncio.Task, asyncio.Task]:
    """Run maintenance once and schedule periodic cycles.

    Optional ``conn`` and ``lock`` parameters allow tests to supply their own
    database connection; otherwise the package-level defaults are used.
    Returns a tuple of ``(sql_task, vector_task)`` handles.
    """
    global _sql_task, _vec_task

    conn = conn or _default_conn
    lock = lock or _default_lock

    # Run one cycle immediately
    logger.info("Starting SQL maintenance (interval=%ds)", rag.MAINTENANCE_INTERVAL)
    await sql_tasks.run(conn, lock)

    logger.info("Starting vector maintenance (interval=%ds)", rag.MAINTENANCE_INTERVAL)
    await vector_tasks.run(conn, lock)

    if not _sql_task or _sql_task.done():
        async def _sql_loop():
            logger.info("Starting SQL maintenance (interval=%ds)", rag.MAINTENANCE_INTERVAL)
            await sql_tasks.run(conn, lock)
        _sql_task = await _startup(_sql_loop, interval)

    if not _vec_task or _vec_task.done():
        async def _vec_loop():
            logger.info("Starting vector maintenance (interval=%ds)", rag.MAINTENANCE_INTERVAL)
            await vector_tasks.run(conn, lock)
        _vec_task = await _startup(_vec_loop, interval)

    return _sql_task, _vec_task


async def stop() -> None:
    """Cancel scheduled maintenance tasks if running."""
    global _sql_task, _vec_task

    if _sql_task:
        await _shutdown(_sql_task)
        _sql_task = None
    if _vec_task:
        await _shutdown(_vec_task)
        _vec_task = None

