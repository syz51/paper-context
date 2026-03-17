from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from paper_context.config import get_settings
from paper_context.db.engine import dispose_engine
from paper_context.logging import configure_logging
from paper_context.storage.local_fs import LocalFilesystemStorage

from .routes.health import router as health_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    storage = LocalFilesystemStorage(settings.storage.root_path)
    storage.ensure_root()
    yield
    dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(title="Paper Context API", lifespan=lifespan)
    app.include_router(health_router)
    return app
