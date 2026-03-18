from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from paper_context.config import get_settings
from paper_context.db.engine import dispose_engine
from paper_context.logging import configure_logging
from paper_context.mcp import create_http_app
from paper_context.storage.local_fs import LocalFilesystemStorage

from .routes.documents import router as documents_router
from .routes.health import router as health_router


def create_app() -> FastAPI:
    mcp_app = create_http_app()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        configure_logging(settings.log_level)
        storage = LocalFilesystemStorage(settings.storage.root_path)
        storage.ensure_root()
        try:
            async with AsyncExitStack() as stack:
                await stack.enter_async_context(mcp_app.lifespan(app))
                yield
        finally:
            dispose_engine()

    app = FastAPI(title="Paper Context App", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.mount("/mcp", mcp_app)
    return app
