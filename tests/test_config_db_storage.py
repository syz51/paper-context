from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text

from paper_context.config import AppSettings, get_settings
from paper_context.config.settings import DatabaseSettings
from paper_context.db import engine as db_engine
from paper_context.db.engine import (
    database_is_ready,
    dispose_engine,
    get_engine,
    make_engine,
)
from paper_context.db.session import connection_scope, get_session_factory, session_scope
from paper_context.logging import configure_logging
from paper_context.storage.local_fs import LocalFilesystemStorage


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    assert first.queue.name == "document_ingest"


def test_configure_logging_sets_level() -> None:
    logging.getLogger().handlers.clear()
    configure_logging("warning")
    assert logging.getLogger().level == logging.WARNING


def test_local_filesystem_storage(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    storage = LocalFilesystemStorage(root)
    storage.ensure_root()
    artifact = storage.store_bytes("subdir/file.bin", b"abc")
    target = root / "subdir" / "file.bin"
    assert target.read_bytes() == b"abc"
    assert artifact.storage_ref == "subdir/file.bin"
    assert artifact.checksum == hashlib.sha256(b"abc").hexdigest()
    assert artifact.size_bytes == 3
    assert storage.resolve(artifact.storage_ref) == target


def test_make_engine_connects() -> None:
    engine = make_engine("sqlite+pysqlite:///:memory:")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar_one() == 1
    engine.dispose()
    engine.dispose()


def test_get_engine_returns_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings(
        database=DatabaseSettings(url="sqlite+pysqlite:///:memory:"),
    )
    monkeypatch.setattr(db_engine, "get_settings", lambda: settings)
    get_engine.cache_clear()
    first = get_engine()
    second = get_engine()
    assert first is second


def test_database_is_ready_true(monkeypatch: pytest.MonkeyPatch) -> None:
    sqlite = create_engine("sqlite+pysqlite:///:memory:", future=True)
    monkeypatch.setattr(db_engine, "get_engine", lambda: sqlite)
    assert database_is_ready()
    sqlite.dispose()
    sqlite.dispose()


def test_database_is_ready_false(monkeypatch: pytest.MonkeyPatch) -> None:
    class BadEngine:
        def connect(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(db_engine, "get_engine", lambda: BadEngine())
    assert not database_is_ready()


def test_dispose_engine_clears_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = MagicMock()
    stub.dispose = MagicMock()
    mock_get_engine = MagicMock(return_value=stub)
    mock_get_engine.cache_clear = MagicMock()
    monkeypatch.setattr(db_engine, "get_engine", mock_get_engine)
    dispose_engine()
    stub.dispose.assert_called_once()
    mock_get_engine.cache_clear.assert_called_once()


def test_get_session_factory_binding() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    factory = get_session_factory(engine)
    session = factory()
    assert session.get_bind() is engine
    session.close()
    engine.dispose()
    engine.dispose()


def test_connection_scope_yields_connection() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    with connection_scope(engine) as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar_one() == 1
    engine.dispose()


def test_session_scope_closes_session(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MagicMock()
    factory = MagicMock(return_value=session)
    monkeypatch.setattr("paper_context.db.session.get_session_factory", lambda engine=None: factory)

    with session_scope() as yielded:
        assert yielded is session

    session.close.assert_called_once_with()
