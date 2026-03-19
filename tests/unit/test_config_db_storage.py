from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text

from paper_context.config import AppSettings, get_settings
from paper_context.config.settings import DatabaseSettings, RuntimeSettings
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

pytestmark = pytest.mark.unit


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    assert first.queue.name == "document_ingest"


def test_runtime_settings_default_to_local_bind_when_env_is_ignored() -> None:
    assert RuntimeSettings().app_host == "127.0.0.1"


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


def test_make_engine_applies_postgres_hardening(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_engine(url: str, **kwargs: object) -> MagicMock:
        captured["url"] = url
        captured["kwargs"] = kwargs
        engine = MagicMock()
        engine.dispose = MagicMock()
        return engine

    monkeypatch.setattr(db_engine, "create_engine", fake_create_engine)

    make_engine(
        "postgresql+psycopg://paper_context:secret@db/paper_context",
        database_settings=DatabaseSettings(
            url="postgresql+psycopg://paper_context:secret@db/paper_context",
            ssl_mode="require",
            connect_timeout_seconds=10,
            statement_timeout_ms=30_000,
            lock_timeout_ms=5_000,
            idle_in_transaction_session_timeout_ms=15_000,
            application_name="paper-context-worker",
            pool_size=7,
            max_overflow=3,
            pool_timeout_seconds=12,
            pool_recycle_seconds=90,
        ),
        app_name="paper-context",
        environment="production",
    )

    assert captured["url"] == "postgresql+psycopg://paper_context:secret@db/paper_context"
    assert captured["kwargs"] == {
        "future": True,
        "pool_pre_ping": True,
        "connect_args": {
            "application_name": "paper-context-worker",
            "connect_timeout": 10,
            "sslmode": "require",
            "options": (
                "-c statement_timeout=30000 "
                "-c lock_timeout=5000 "
                "-c idle_in_transaction_session_timeout=15000"
            ),
        },
        "pool_size": 7,
        "max_overflow": 3,
        "pool_timeout": 12,
        "pool_recycle": 90,
    }


def test_make_engine_rejects_incomplete_production_settings() -> None:
    with pytest.raises(ValueError, match="production database settings are incomplete"):
        make_engine(
            "postgresql+psycopg://paper_context:secret@db/paper_context",
            database_settings=DatabaseSettings(
                url="postgresql+psycopg://paper_context:secret@db/paper_context",
            ),
            environment="production",
        )


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
