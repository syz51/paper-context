from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session, sessionmaker

from .engine import get_engine


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(
        bind=engine or get_engine(),
        autoflush=False,
        expire_on_commit=False,
        future=True,
    )


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    session = get_session_factory(engine)()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def connection_scope(engine: Engine | None = None) -> Iterator[Connection]:
    with (engine or get_engine()).begin() as connection:
        yield connection
