"""FastAPI lifecycle composition without concrete infrastructure ownership."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypeVar

from fastapi import FastAPI

from riskforge.api.app import create_app
from riskforge.authentication.protocols import AuthenticationService
from riskforge.runtime.api_adapter import RuntimeReadinessProvider
from riskforge.runtime.contracts import DependencyComposer, RuntimeSettings
from riskforge.runtime.lifecycle import RuntimeManager

T = TypeVar("T")


def create_managed_app(
    settings: RuntimeSettings,
    composer: DependencyComposer[T],
    *,
    auth_service: AuthenticationService | None = None,
    enable_metrics: bool = True,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Compose dependencies once and bind their lifecycle to the ASGI app.

    Concrete database, encoder, and inference construction remains in the
    injected composer.  The HTTP layer only receives the assembled application
    facade and a transport-safe readiness adapter.
    """

    assembly = composer.compose(settings)
    manager = RuntimeManager(
        assembly,
        shutdown_timeout_s=settings.shutdown_timeout_seconds,
    )
    readiness = RuntimeReadinessProvider(manager)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        manager.start()
        try:
            yield
        finally:
            manager.stop()

    app = create_app(
        manager.application,
        readiness,
        auth_service=auth_service,
        enable_metrics=enable_metrics,
        cors_origins=cors_origins,
        max_request_bytes=settings.max_request_bytes,
        request_timeout_seconds=settings.request_timeout_seconds,
        max_batch_size=settings.max_batch_size,
        lifespan=lifespan,
    )
    app.state.runtime_manager = manager
    app.state.runtime_settings = settings
    return app
