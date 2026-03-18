from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

from paper_context.config import get_settings
from paper_context.config.settings import DatabaseSettings


def _postgres_connect_args(
    database_settings: DatabaseSettings,
    *,
    application_name: str,
) -> dict[str, object]:
    connect_args: dict[str, object] = {
        "application_name": database_settings.effective_application_name(application_name),
    }
    if database_settings.connect_timeout_seconds is not None:
        connect_args["connect_timeout"] = database_settings.connect_timeout_seconds
    if database_settings.ssl_mode is not None:
        connect_args["sslmode"] = database_settings.ssl_mode
    if database_settings.ssl_root_cert is not None:
        connect_args["sslrootcert"] = str(database_settings.ssl_root_cert)
    if database_settings.ssl_cert is not None:
        connect_args["sslcert"] = str(database_settings.ssl_cert)
    if database_settings.ssl_key is not None:
        connect_args["sslkey"] = str(database_settings.ssl_key)

    session_options: list[str] = []
    if database_settings.statement_timeout_ms is not None:
        session_options.append(f"-c statement_timeout={database_settings.statement_timeout_ms}")
    if database_settings.lock_timeout_ms is not None:
        session_options.append(f"-c lock_timeout={database_settings.lock_timeout_ms}")
    if database_settings.idle_in_transaction_session_timeout_ms is not None:
        session_options.append(
            "-c idle_in_transaction_session_timeout="
            f"{database_settings.idle_in_transaction_session_timeout_ms}"
        )
    if session_options:
        connect_args["options"] = " ".join(session_options)
    return connect_args


def make_engine(
    database_url: str,
    *,
    database_settings: DatabaseSettings | None = None,
    app_name: str = "paper-context",
    environment: str = "development",
) -> Engine:
    settings = database_settings or DatabaseSettings(url=database_url)
    settings.validate_runtime(environment=environment, default_app_name=app_name)

    engine_kwargs: dict[str, object] = {
        "future": True,
        "pool_pre_ping": True,
    }
    backend_name = make_url(database_url).get_backend_name()
    if backend_name != "sqlite":
        engine_kwargs["connect_args"] = _postgres_connect_args(
            settings,
            application_name=app_name,
        )
        if settings.pool_size is not None:
            engine_kwargs["pool_size"] = settings.pool_size
        if settings.max_overflow is not None:
            engine_kwargs["max_overflow"] = settings.max_overflow
        if settings.pool_timeout_seconds is not None:
            engine_kwargs["pool_timeout"] = settings.pool_timeout_seconds
        if settings.pool_recycle_seconds is not None:
            engine_kwargs["pool_recycle"] = settings.pool_recycle_seconds
    return create_engine(database_url, **engine_kwargs)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    return make_engine(
        settings.database.url,
        database_settings=settings.database,
        app_name=settings.app_name,
        environment=settings.environment,
    )


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
