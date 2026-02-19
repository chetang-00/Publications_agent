"""FastAPI application factory. Run with `uvicorn app.main:app`."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import catalog, conversations, documents, health, runs
from app.api.errors import RequestContextMiddleware, install_error_handlers
from app.config import get_settings
from app.container import Container, build_container
from app.db import repo
from app.db.session import run_migrations
from app.logging import configure_logging

log = logging.getLogger(__name__)


async def run_startup_recovery(container: Container) -> None:
    """Work interrupted by a restart can never finish; record that instead of leaving it 'in progress'."""
    documents_failed = await container.ingestion.mark_interrupted()
    runs_failed = await repo.fail_interrupted_runs(container.db)
    if documents_failed or runs_failed:
        log.warning("Recovered interrupted work", extra={"documents": documents_failed, "runs": runs_failed})


def create_app(container: Container | None = None) -> FastAPI:
    """With a container (tests), services are used as given; otherwise they are built at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = container is None
        if owned:
            settings = get_settings()
            configure_logging(settings.log_level)
            await asyncio.to_thread(run_migrations, settings.database_url)
            app.state.container = build_container(settings)
        await run_startup_recovery(app.state.container)
        log.info("Application started")
        try:
            yield
        finally:
            if owned:
                await app.state.container.aclose()

    app = FastAPI(
        title="Research Publications Agent",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    if container is not None:
        app.state.container = container
        origins = container.settings.cors_origins
    else:
        origins = get_settings().cors_origins
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID"],
        )
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)
    for router in (health.router, conversations.router, runs.router, documents.router, catalog.router):
        app.include_router(router, prefix="/api")
    return app


def __getattr__(name: str) -> FastAPI:
    # `uvicorn app.main:app` builds the real app lazily, so importing this module (tests) needs no .env.
    if name == "app":
        return create_app()
    raise AttributeError(name)
