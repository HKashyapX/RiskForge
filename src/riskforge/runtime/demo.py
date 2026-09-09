"""Explicit, non-inferencing ASGI composition for demonstrations."""

from __future__ import annotations

import os
from datetime import UTC, datetime

from fastapi import FastAPI

from riskforge.api.app import create_app
from riskforge.api.dependencies import ReadinessSnapshot
from riskforge.application.exceptions import (
    InferenceApplicationError,
    MetricsApplicationError,
    QueryApplicationError,
    ReviewApplicationError,
)


class _DemoReadiness:
    def snapshot(self) -> ReadinessSnapshot:
        return ReadinessSnapshot(
            ready=False,
            state="not_ready",
            checked_at=datetime.now(UTC),
            components=("model",),
        )


class _UnavailableApplication:
    """Reject business operations instead of manufacturing demo predictions."""

    def process_incident(self, _record: object) -> None:
        raise InferenceApplicationError("model artifact is not installed")

    def process_batch(self, _records: object) -> None:
        raise InferenceApplicationError("model artifact is not installed")

    def get_asset_summary(self, _asset_id: str) -> None:
        raise MetricsApplicationError("demo runtime has no analytics store")

    def get_incident(self, _log_id: str) -> None:
        raise QueryApplicationError("demo runtime has no incident store")

    def list_incidents(self, _query: object, _page: object) -> None:
        raise QueryApplicationError("demo runtime has no incident store")

    def list_audit_events(self, _log_id: str, _page: object) -> None:
        raise QueryApplicationError("demo runtime has no audit store")

    def decide_review(self, _command: object) -> None:
        raise ReviewApplicationError("demo runtime has no review store")


def create_demo_app() -> FastAPI:
    """Create a health/API-documentation demo that is never marked ready."""
    mode = os.environ.get("RISKFORGE_MODE", "").strip().lower()
    environment = os.environ.get("RISKFORGE_ENV", "development").strip().lower()
    if mode != "demo":
        raise RuntimeError("set RISKFORGE_MODE=demo to start the non-inferencing demo service")
    if environment in {"staging", "production"}:
        raise RuntimeError("demo mode is forbidden in staging and production")
    return create_app(
        _UnavailableApplication(),  # type: ignore[arg-type]
        _DemoReadiness(),
        enable_metrics=False,
        cors_origins=[],
    )
