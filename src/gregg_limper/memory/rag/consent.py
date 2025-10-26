from __future__ import annotations
from .sql.repositories import ConsentRepo as _ConsentRepo
from . import _conn, _db_lock

_repo = _ConsentRepo(_conn, _db_lock)

async def is_opted_in(user_id: int) -> bool:
    return await _repo.is_opted_in(user_id)

async def add_user(user_id: int) -> bool:
    return await _repo.add_user(user_id)

async def remove_user(user_id: int) -> None:
    await _repo.remove_user(user_id)
