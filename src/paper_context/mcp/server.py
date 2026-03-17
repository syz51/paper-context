from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastmcp import FastMCP

from paper_context import __version__
from paper_context.config import get_settings
from paper_context.db.engine import database_is_ready, dispose_engine
from paper_context.logging import configure_logging
from paper_context.schemas.common import HealthResponse, ReadinessResponse
from paper_context.storage.local_fs import LocalFilesystemStorage


def create_server() -> FastMCP:
    return FastMCP(name="paper-context")


def create_app() -> FastAPI:
    settings = get_settings()
    mcp = create_server()
    mcp_app = mcp.http_app(path="/", transport="streamable-http")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        storage = LocalFilesystemStorage(settings.storage.root_path)
        storage.ensure_root()
        async with mcp_app.lifespan(app):
            yield
        dispose_engine()

    app = FastAPI(title="Paper Context MCP", lifespan=lifespan)

    @app.get("/healthz", response_model=HealthResponse)
    def healthcheck() -> HealthResponse:
        return HealthResponse(service="mcp", status="ok", version=__version__)

    @app.get("/readyz", response_model=ReadinessResponse)
    def readiness() -> ReadinessResponse:
        db_ready = database_is_ready()
        return ReadinessResponse(
            service="mcp",
            status="ready" if db_ready else "degraded",
            version=__version__,
            database_ready=db_ready,
            storage_root=settings.storage.root_path,
            queue_name=settings.queue.name,
        )

    app.mount("/mcp", mcp_app)
    return app
