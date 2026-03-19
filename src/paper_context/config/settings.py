from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    url: str = "postgresql+psycopg://paper_context:paper_context@localhost:5433/paper_context"
    ssl_mode: str | None = "disable"
    ssl_root_cert: Path | None = None
    ssl_cert: Path | None = None
    ssl_key: Path | None = None
    connect_timeout_seconds: int | None = 10
    statement_timeout_ms: int | None = 30_000
    lock_timeout_ms: int | None = 10_000
    idle_in_transaction_session_timeout_ms: int | None = 30_000
    application_name: str | None = None
    pool_size: int | None = 5
    max_overflow: int | None = 10
    pool_timeout_seconds: int | None = 30
    pool_recycle_seconds: int | None = 1_800

    def effective_application_name(self, default_app_name: str) -> str:
        return self.application_name or default_app_name

    def validate_runtime(self, *, environment: str, default_app_name: str) -> None:
        if environment.lower() != "production":
            return

        missing: list[str] = []
        if self.ssl_mode not in {"require", "verify-ca", "verify-full"}:
            missing.append("database.ssl_mode")
        for field_name in (
            "connect_timeout_seconds",
            "statement_timeout_ms",
            "lock_timeout_ms",
            "idle_in_transaction_session_timeout_ms",
            "pool_size",
            "max_overflow",
            "pool_timeout_seconds",
            "pool_recycle_seconds",
        ):
            value = getattr(self, field_name)
            if value is None or value <= 0:
                missing.append(f"database.{field_name}")

        application_name = self.effective_application_name(default_app_name)
        if not application_name.strip():
            missing.append("database.application_name")

        if missing:
            missing_fields = ", ".join(missing)
            raise ValueError(
                f"production database settings are incomplete or insecure: {missing_fields}"
            )


class StorageSettings(BaseModel):
    root_path: Path = Path("./var/artifacts")


class UploadSettings(BaseModel):
    max_bytes: int = 25 * 1024 * 1024


class QueueSettings(BaseModel):
    name: str = "document_ingest"
    visibility_timeout_seconds: int = 300
    max_poll_seconds: int = 5
    poll_interval_ms: int = 250


class ProviderSettings(BaseModel):
    voyage_api_key: str | None = None
    zero_entropy_api_key: str | None = None
    openalex_api_key: str | None = None
    semantic_scholar_api_key: str | None = None
    voyage_model: str = "voyage-4-large"
    reranker_model: str = "zerank-2"
    index_version: str = "mvp-v1"


class ParserSettings(BaseModel):
    primary: str = "docling"
    fallback: str = "pdfplumber"
    execution_mode: str = "subprocess"
    timeout_seconds: int = 120
    memory_limit_mb: int = 2_048
    output_limit_mb: int = 32


class ChunkingSettings(BaseModel):
    version: str = "phase0"
    min_tokens: int = 300
    max_tokens: int = 700
    overlap_fraction: float = 0.15


class RuntimeSettings(BaseModel):
    # Bind locally by default; container and remote deployments should opt in explicitly.
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    worker_idle_sleep_seconds: float = 1.0


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PAPER_CONTEXT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "paper-context"
    environment: str = "development"
    log_level: str = "INFO"
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    upload: UploadSettings = Field(default_factory=UploadSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    parser: ParserSettings = Field(default_factory=ParserSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
