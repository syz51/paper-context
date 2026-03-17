from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from paper_context.config import get_settings


def make_engine(database_url: str) -> Engine:
    return create_engine(database_url, future=True, pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine(get_settings().database.url)


def database_is_ready() -> bool:
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def dispose_engine() -> None:
    try:
        get_engine().dispose()
    finally:
        get_engine.cache_clear()
