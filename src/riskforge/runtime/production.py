"""Environment-driven production dependency composition.

Assembles the complete backend: persistence (SQLite by default, PostgreSQL
when ``RISKFORGE_PERSISTENCE=postgres``), the inference engine (ONNX artifact
when configured, deterministic heuristic engine otherwise), workflow adapters,
review service, and metrics.  Implements the ``DependencyComposer`` protocol
consumed by ``riskforge.runtime.asgi.create_managed_app``.

This module contains no business logic: it only wires established subsystem
boundaries together.  The engine mode is surfaced honestly through readiness
so deployments never silently claim model-backed inference they do not have.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from riskforge.application.backend_service import BackendApplicationService
from riskforge.application.protocols import InferenceEngine
from riskforge.application.service import ApplicationService
from riskforge.application.workflow_models import (
    AuditEventView,
    IncidentQuery,
    IncidentView,
    Page,
    PageRequest,
    ReviewCommand,
    ReviewDecisionView,
)
from riskforge.core.contracts import (
    AssetRiskSummary,
    IncidentNormalizedRecord,
    ModelInferenceResult,
)
from riskforge.ingestion.pipeline import IngestionPipeline
from riskforge.metrics.aggregator import MetricsAggregator
from riskforge.persistence.exceptions import PersistenceConflictError
from riskforge.persistence.models import IncidentResultFilter, StoredIncidentResult
from riskforge.persistence.models import PageRequest as StorePageRequest
from riskforge.persistence.protocols import (
    AuditEventRepository,
    IncidentResultRepository,
)
from riskforge.persistence.sqlite.repository import (
    SQLiteAuditEventRepository,
    SQLiteIncidentResultRepository,
)
from riskforge.persistence.sqlite.review_audit import SQLiteReviewAuditWriter
from riskforge.review.authorizer import AllowAllReviewerAuthorizer
from riskforge.review.protocols import ReviewerAuthorizer
from riskforge.review.service import ReviewService
from riskforge.runtime.contracts import (
    ComponentReadiness,
    LifecycleComponent,
    RuntimeAssembly,
    RuntimeSettings,
)
from riskforge.runtime.workflow_adapters import (
    PersistenceWorkflowReader,
    ReviewServiceAdapter,
)
from riskforge.serving.heuristic_engine import HeuristicRuleEngine

if TYPE_CHECKING:
    from fastapi import FastAPI

from riskforge.normalization.gazetteer import SpanPreservingGazetteer

logger = logging.getLogger("riskforge.runtime.production")

_ENGINE_MODE_COMPONENT = "inference-engine"
_METRICS_SCAN_LIMIT = 5_000


def _sqlite_database_path() -> Path:
    return Path(os.environ.get("RISKFORGE_SQLITE_PATH", "data/riskforge.db"))


def _build_engine(settings: RuntimeSettings) -> InferenceEngine:
    """Build the inference engine the deployment actually has.

    When a model artifact and manifest are configured AND exist, use the real
    ONNX engine.  When neither is configured, fall back to the deterministic
    heuristic engine (logged loudly, surfaced via readiness).  A configured-
    but-missing artifact is a hard startup error rather than a silent
    downgrade, and half-configuration is rejected.
    """
    model_path = os.environ.get("RISKFORGE_MODEL_PATH", "").strip()
    manifest_path = os.environ.get("RISKFORGE_MANIFEST_PATH", "").strip()
    if model_path and manifest_path:
        if not Path(model_path).is_file() or not Path(manifest_path).is_file():
            raise RuntimeError(
                f"configured model artifact not found: {model_path!s}, {manifest_path!s}"
            )
        from riskforge.serving.engine import ONNXInferenceEngine

        engine = ONNXInferenceEngine.from_artifact(
            model_path,
            manifest_path,
            max_batch_size=settings.max_batch_size,
            inference_timeout_s=settings.request_timeout_seconds,
        )
        logger.info("inference engine: ONNX model artifact (%s)", model_path)
        return engine
    if model_path or manifest_path:
        raise RuntimeError(
            "RISKFORGE_MODEL_PATH and RISKFORGE_MANIFEST_PATH must be configured together"
        )
    logger.warning(
        "RISKFORGE_MODEL_PATH/RISKFORGE_MANIFEST_PATH not configured; using the "
        "deterministic heuristic rule engine (no trained model artifact)"
    )
    return HeuristicRuleEngine()


def _authorizer_from_env() -> ReviewerAuthorizer:
    level = os.environ.get("RISKFORGE_REVIEW_AUTHORIZATION", "allow_all").strip().lower()
    if level == "allow_all":
        return AllowAllReviewerAuthorizer()
    raise RuntimeError(f"unsupported RISKFORGE_REVIEW_AUTHORIZATION policy: {level!r}")


def _build_persistence() -> tuple[
    IncidentResultRepository,
    AuditEventRepository,
    SQLiteReviewAuditWriter,
    list[LifecycleComponent],
]:
    """Build persistence repositories plus any lifecycle components."""
    backend = os.environ.get("RISKFORGE_PERSISTENCE", "sqlite").strip().lower()
    if backend == "postgres":
        from riskforge.persistence.postgres import (
            PostgresAuditEventRepository,
            PostgresConfig,
            PostgresConnectionPool,
            PostgresIncidentResultRepository,
            PostgresReviewAuditWriter,
        )
        from riskforge.persistence.postgres.migrate import run_migrations
        from riskforge.runtime.persistence_adapter import PostgresPoolLifecycle

        config = PostgresConfig()
        applied = run_migrations()
        logger.info("postgres migrations applied: %s", applied)
        pool = PostgresConnectionPool(config)
        return (
            PostgresIncidentResultRepository(pool),
            PostgresAuditEventRepository(pool),
            PostgresReviewAuditWriter(pool),
            [PostgresPoolLifecycle(pool)],
        )
    if backend != "sqlite":
        raise RuntimeError(f"unsupported RISKFORGE_PERSISTENCE backend: {backend!r}")

    db_path = _sqlite_database_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    incidents: IncidentResultRepository = SQLiteIncidentResultRepository(db_path)
    audit: AuditEventRepository = SQLiteAuditEventRepository(db_path)
    review_writer = SQLiteReviewAuditWriter(db_path)
    return incidents, audit, review_writer, []


class _StoreMetricsService:
    """MetricsService adapter computing per-asset summaries from persistence.

    Aggregates every stored result for an asset (bounded by a scan limit) so
    analytics reflect the full operational history rather than a single call.
    """

    def __init__(self, incidents: IncidentResultRepository) -> None:
        self._incidents = incidents

    def asset_summary(self, asset_id: str) -> AssetRiskSummary:
        stored = self._incidents.list(
            IncidentResultFilter(asset_id=asset_id),
            page=StorePageRequest(offset=0, limit=_METRICS_SCAN_LIMIT),
        )
        if not stored.items:
            raise KeyError(asset_id)
        results = [item.result for item in stored.items]
        mapping = {
            item.incident.log_id: (item.incident.asset_id, item.incident.asset_type)
            for item in stored.items
        }
        summaries = MetricsAggregator(min_barrier_recurrence=2).compute(results, mapping)
        if not summaries:
            raise KeyError(asset_id)
        return summaries[0]


class _PersistingApplicationService:
    """Persist every scored result so read routes see the same data.

    ``POST /v1/inference`` is a scored write: the workflow read routes and the
    review service read exclusively from persistence, so results must be
    stored idempotently at scoring time.
    """

    def __init__(
        self,
        core: BackendApplicationService,
        incidents: IncidentResultRepository,
    ) -> None:
        self._core = core
        self._incidents = incidents

    def process_incident(self, record: IncidentNormalizedRecord) -> ModelInferenceResult:
        result = self._core.process_incident(record)
        self._store(record, result)
        return result

    def process_batch(
        self, records: Sequence[IncidentNormalizedRecord]
    ) -> tuple[ModelInferenceResult, ...]:
        results = self._core.process_batch(records)
        for record, result in zip(records, results, strict=True):
            self._store(record, result)
        return tuple(results)

    def _store(self, record: IncidentNormalizedRecord, result: ModelInferenceResult) -> None:
        try:
            self._incidents.create_idempotent(StoredIncidentResult(incident=record, result=result))
        except PersistenceConflictError:
            # Re-submitting the same report is idempotent only when the stored
            # decision matches the fresh one; latency_ms is timing noise and
            # must not make identical rescoring look like a conflict.
            stored = self._incidents.get(record.log_id)
            if stored is not None and _same_decision(record, result, stored):
                return
            raise

    def get_asset_summary(self, asset_id: str) -> AssetRiskSummary:
        return self._core.get_asset_summary(asset_id)

    def get_incident(self, log_id: str) -> IncidentView:
        return self._core.get_incident(log_id)

    def list_incidents(self, query: IncidentQuery, page: PageRequest) -> Page[IncidentView]:
        return self._core.list_incidents(query, page)

    def list_audit_events(self, log_id: str, page: PageRequest) -> Page[AuditEventView]:
        return self._core.list_audit_events(log_id, page)

    def decide_review(self, command: ReviewCommand) -> ReviewDecisionView:
        return self._core.decide_review(command)


def _same_decision(
    record: IncidentNormalizedRecord,
    result: ModelInferenceResult,
    stored: StoredIncidentResult,
) -> bool:
    if stored.incident != record:
        return False
    fresh = result.model_copy(update={"latency_ms": 0.0})
    existing = stored.result.model_copy(update={"latency_ms": 0.0})
    return fresh == existing


class _EngineModeLifecycle:
    """Expose the active inference-engine mode through readiness."""

    def __init__(self, engine_mode: str) -> None:
        self._engine_mode = engine_mode

    @property
    def name(self) -> str:
        return _ENGINE_MODE_COMPONENT

    def start(self) -> None:
        logger.info("inference engine mode: %s", self._engine_mode)

    def stop(self) -> None:
        return None

    def readiness(self) -> ComponentReadiness:
        return ComponentReadiness(name=self.name, ready=True, detail=self._engine_mode)


class _ScoringIngestionPipeline:
    """Ingestion pipeline bound to normalization, scoring, and persistence."""

    def __init__(
        self,
        pipeline: IngestionPipeline,
        incidents: IncidentResultRepository,
    ) -> None:
        self._pipeline = pipeline
        self._incidents = incidents

    def ingest(self, data: bytes, fmt: str) -> object:
        outcome = self._pipeline.run(data, fmt)
        persisted = 0
        for item in outcome.items:
            if item.status == "normalized" and item.normalized is not None and item.result is not None:
                try:
                    self._incidents.create_idempotent(
                        StoredIncidentResult(incident=item.normalized, result=item.result)
                    )
                    persisted += 1
                except PersistenceConflictError:
                    stored = self._incidents.get(item.normalized.log_id)
                    if stored is None or not _same_decision(
                        item.normalized, item.result, stored
                    ):
                        raise
        return outcome


class ProductionComposer:
    """Assemble the complete application facade and lifecycle components.

    Composition order: persistence first (independent of the engine), then
    metrics over the store, then scoring, then the transport facade.
    """

    def compose(self, settings: RuntimeSettings) -> RuntimeAssembly:
        engine = _build_engine(settings)
        engine_mode = getattr(engine, "mode", "onnx-model")

        incidents, audit, review_writer, components = _build_persistence()
        metrics = _StoreMetricsService(incidents)
        core = ApplicationService(engine, metrics)
        reader = PersistenceWorkflowReader(incidents, audit)
        review_service = ReviewService(incidents, review_writer, _authorizer_from_env())
        backend = BackendApplicationService(core, reader, ReviewServiceAdapter(review_service))
        scoring = _PersistingApplicationService(backend, incidents)
        ingestion = _ScoringIngestionPipeline(
            IngestionPipeline(SpanPreservingGazetteer().process, engine),
            incidents,
        )
        scoring.ingest = ingestion.ingest  # type: ignore[method-assign]

        components.append(_EngineModeLifecycle(engine_mode))
        return RuntimeAssembly(application=scoring, components=tuple(components))


def create_app() -> FastAPI:
    """Uvicorn-compatible ASGI factory for production composition."""
    from riskforge.runtime.asgi import create_managed_app

    settings = RuntimeSettings(
        model_path=Path(os.environ.get("RISKFORGE_MODEL_PATH", "unused.onnx")),
        manifest_path=Path(os.environ.get("RISKFORGE_MANIFEST_PATH", "unused.yaml")),
        api_host=os.environ.get("RISKFORGE_API_HOST", "0.0.0.0"),
        api_port=int(os.environ.get("RISKFORGE_API_PORT", "8000")),
    )
    cors = os.environ.get("RISKFORGE_CORS_ORIGINS", "").strip()
    origins = None
    if cors and cors != "*":
        origins = [origin.strip() for origin in cors.split(",") if origin.strip()]
    auth_service = None
    if os.environ.get("RISKFORGE_AUTH_ENABLED", "").strip().lower() in {"1", "true", "yes"}:
        from riskforge.authentication.jwt_service import JwtAuthenticationService

        auth_service = JwtAuthenticationService.from_env()
    return create_managed_app(
        settings,
        ProductionComposer(),
        auth_service=auth_service,
        cors_origins=origins,
    )
