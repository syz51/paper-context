from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

from alembic import command

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DATABASE_URL = (
    "postgresql+psycopg://paper_context:paper_context@localhost:5433/paper_context"
)
STAGING_ENV_FLAG = "PAPER_CONTEXT_RUN_STAGING_TESTS"
TEST_DATABASE_URL_ENV = "PAPER_CONTEXT_TEST_DATABASE_URL"
POSTGRES_FIXTURE_LOCK_PATH = REPO_ROOT / ".pytest_cache" / "postgres-fixture.lock"
POSTGRES_FIXTURE_LEASES_DIR = REPO_ROOT / ".pytest_cache" / "postgres-fixture-leases"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    run_staging = os.environ.get(STAGING_ENV_FLAG, "").lower() in {"1", "true", "yes", "on"}
    if run_staging:
        return

    skip_marker = pytest.mark.skip(
        reason=(
            "staging_only tests are disabled by default; set "
            "PAPER_CONTEXT_RUN_STAGING_TESTS=1 to enable them."
        )
    )
    for item in items:
        if item.get_closest_marker("staging_only"):
            item.add_marker(skip_marker)


def _run_compose(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _wait_for_database(database_url: str, timeout_seconds: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        engine = create_engine(database_url, future=True, pool_pre_ping=True)
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return
        except Exception as exc:  # pragma: no cover - only exercised on startup race
            last_error = exc
            time.sleep(1)
        finally:
            engine.dispose()
    raise RuntimeError(f"database did not become ready in time: {last_error}")


def _with_database_url_env(database_url: str, callback: Callable[[], None]) -> None:
    previous = os.environ.get("PAPER_CONTEXT_DATABASE__URL")
    os.environ["PAPER_CONTEXT_DATABASE__URL"] = database_url
    try:
        callback()
    finally:
        if previous is None:
            os.environ.pop("PAPER_CONTEXT_DATABASE__URL", None)
        else:
            os.environ["PAPER_CONTEXT_DATABASE__URL"] = previous


def _quoted_identifier(identifier: str) -> str:
    return identifier.replace('"', '""')


@contextmanager
def _postgres_fixture_lock() -> Iterator[None]:
    POSTGRES_FIXTURE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POSTGRES_FIXTURE_LOCK_PATH.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _prune_dead_postgres_fixture_leases() -> list[Path]:
    POSTGRES_FIXTURE_LEASES_DIR.mkdir(parents=True, exist_ok=True)
    active_leases: list[Path] = []
    for lease_path in POSTGRES_FIXTURE_LEASES_DIR.glob("*.lease"):
        try:
            pid = int(lease_path.read_text(encoding="utf-8").strip())
        except OSError, ValueError:
            lease_path.unlink(missing_ok=True)
            continue

        if _process_is_alive(pid):
            active_leases.append(lease_path)
            continue

        lease_path.unlink(missing_ok=True)

    return active_leases


@contextmanager
def _postgres_fixture_lease() -> Iterator[None]:
    lease_path = POSTGRES_FIXTURE_LEASES_DIR / f"{os.getpid()}-{uuid.uuid4().hex}.lease"
    should_start_compose = False

    with _postgres_fixture_lock():
        active_leases = _prune_dead_postgres_fixture_leases()
        should_start_compose = not active_leases
        lease_path.write_text(str(os.getpid()), encoding="utf-8")

        if should_start_compose:
            _run_compose("up", "-d", "db")

    try:
        yield
    finally:
        with _postgres_fixture_lock():
            lease_path.unlink(missing_ok=True)
            active_leases = _prune_dead_postgres_fixture_leases()
            if not active_leases:
                _run_compose("down", "-v", check=False)


@pytest.fixture(scope="session")
def postgres_server_url() -> Iterator[str]:
    database_url = os.environ.get(TEST_DATABASE_URL_ENV, DEFAULT_TEST_DATABASE_URL)
    if os.environ.get(TEST_DATABASE_URL_ENV):
        _wait_for_database(database_url)
        yield database_url
        return

    if shutil.which("docker") is None:
        pytest.skip(
            "requires_postgres tests need Docker or "
            f"{TEST_DATABASE_URL_ENV} pointing at a prepared Postgres instance."
        )

    with _postgres_fixture_lease():
        _wait_for_database(database_url)
        yield database_url


@pytest.fixture(scope="session")
def alembic_config() -> Config:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    return config


@pytest.fixture(scope="session")
def run_alembic_upgrade(alembic_config: Config) -> Callable[[str], None]:
    def _run(database_url: str) -> None:
        _with_database_url_env(database_url, lambda: command.upgrade(alembic_config, "head"))

    return _run


@pytest.fixture
def postgres_test_database_url(postgres_server_url: str) -> Iterator[str]:
    server_url = make_url(postgres_server_url)
    database_name = f"paper_context_test_{uuid.uuid4().hex}"
    admin_url = server_url.set(database="postgres")
    admin_engine = create_engine(
        admin_url.render_as_string(hide_password=False),
        future=True,
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
    )

    quoted_name = _quoted_identifier(database_name)
    with admin_engine.connect() as connection:
        connection.exec_driver_sql(f'DROP DATABASE IF EXISTS "{quoted_name}" WITH (FORCE)')
        connection.exec_driver_sql(f'CREATE DATABASE "{quoted_name}"')

    try:
        yield server_url.set(database=database_name).render_as_string(hide_password=False)
    finally:
        with admin_engine.connect() as connection:
            connection.exec_driver_sql(f'DROP DATABASE IF EXISTS "{quoted_name}" WITH (FORCE)')
        admin_engine.dispose()


@pytest.fixture
def migrated_postgres_url(
    postgres_test_database_url: str, run_alembic_upgrade: Callable[[str], None]
) -> str:
    run_alembic_upgrade(postgres_test_database_url)
    return postgres_test_database_url


@pytest.fixture
def migrated_postgres_engine(migrated_postgres_url: str) -> Iterator[Engine]:
    engine = create_engine(migrated_postgres_url, future=True, pool_pre_ping=True)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def unique_queue_name() -> str:
    return f"document_ingest_{uuid.uuid4().hex[:12]}"
