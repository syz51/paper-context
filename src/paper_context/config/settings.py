from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    url: str = "postgresql+psycopg://paper_context:paper_context@localhost:5433/paper_context"


class StorageSettings(BaseModel):
    root_path: Path = Path("./var/artifacts")


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


class ChunkingSettings(BaseModel):
    version: str = "phase0"
    min_tokens: int = 300
    max_tokens: int = 700
    overlap_fraction: float = 0.15


class RuntimeSettings(BaseModel):
    # These defaults are intentional so local/container deployments are reachable.
    api_host: str = "0.0.0.0"  # nosec B104
    api_port: int = 8000
    mcp_host: str = "0.0.0.0"  # nosec B104
    mcp_port: int = 8001
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
    queue: QueueSettings = Field(default_factory=QueueSettings)
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    parser: ParserSettings = Field(default_factory=ParserSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
