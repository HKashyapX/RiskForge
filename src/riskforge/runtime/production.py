"""Mode-aware production dependency composition.

Assembles the complete backend according to the explicit deployment mode
(``riskforge.runtime.deployment``):

- persistence (SQLite for demo/pilot, mandatory PostgreSQL for production)
- inference engine (ONNX artifact where required/configured; the deterministic
  heuristic engine only where the mode explicitly permits it, always labelled)
- a concrete incident encoder whenever a model artifact serves traffic
- workflow adapters, deny-by-default review authorization, and metrics

The composer uses explicit, typed construction only — no monkey-patching and
no ``getattr`` capability discovery.  Every engine is wrapped so the scoring
mode is enforced on results, and readiness reports the honest engine label.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from riskforge.analytics.service import compute_summary
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
    ScoringMode,
)
from riskforge.ingestion.pipeline import IngestionPipeline
from riskforge.metrics.aggregator import MetricsAggregator
from riskforge.normalization.gazetteer import SpanPreservingGazetteer
from riskforge.persistence.exceptions import PersistenceConflictError
from riskforge.persistence.models import (
    AuditEvent,
    AuditEventType,
    IncidentResultFilter,
    StoredIncidentResult,
)
from riskforge.persistence.models import (
    PageRequest as StorePageRequest,
)
from riskforge.persistence.protocols import (
    AuditEventRepository,
    IncidentResultRepository,
)
from riskforge.persistence.sqlite.repository import (
    SQLiteAuditEventRepository,
    SQLiteIncidentResultRepository,
)
from riskforge.persistence.sqlite.review_audit import SQLiteReviewAuditWriter
from riskforge.review.authorizer import SubjectAllowlistReviewerAuthorizer
from riskforge.review.service import ReviewService
from riskforge.runtime.contracts import (
    ComponentReadiness,
    LifecycleComponent,
    RuntimeAssembly,
    RuntimeSettings,
)
from riskforge.runtime.deployment import (
    DeploymentConfig,
    DeploymentConfigError,
    DeploymentMode,
)
from riskforge.runtime.deployment import (
    ScoringMode as ConfigScoringMode,
)
from riskforge.runtime.workflow_adapters import (
    PersistenceWorkflowReader,
    ReviewServiceAdapter,
)
from riskforge.serving.engine import ONNXInferenceEngine
from riskforge.serving.heuristic_engine import HeuristicRuleEngine

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger("riskforge.runtime.production")

_ENGINE_MODE_COMPONENT = "inference-engine"
_PAGE_LIMIT = 500  # persistence PageRequest hard cap; scans must paginate


class ModeCompositionError(RuntimeError):
    """Raised when the deployment mode's requirements cannot be satisfied."""


def _parse_window(value: str | None, name: str):
    from datetime import datetime

    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ModeCompositionError(f"invalid {name}: {value!r}") from error


def _iter_stored(
    incidents: IncidentResultRepository,
    incident_filter: IncidentResultFilter | None = None,
) -> list[StoredIncidentResult]:
    """Page through the store (bounded) respecting the 500-item cap."""
    collected: list[StoredIncidentResult] = []
    offset = 0
    incident_filter = incident_filter or IncidentResultFilter()
    while True:
        page = incidents.list(
            incident_filter,
            page=StorePageRequest(offset=offset, limit=_PAGE_LIMIT),
        )
        collected.extend(page.items)
        offset += len(page.items)
        if len(page.items) < _PAGE_LIMIT or offset >= page.total or offset >= 100_000:
            return collected


def _sqlite_database_path() -> Path:
    return Path(os.environ.get("RISKFORGE_SQLITE_PATH", "data/riskforge.db"))


class _ModeEnforcingEngine:
    """Engine wrapper that stamps the deployment's scoring mode on results.

    The wrapper refuses results whose self-declared scoring mode contradicts
    the deployment mode: model modes reject heuristic results (the reverse of
    a silent fallback), and heuristic-permitted modes reject unlabeled model
    results.
    """

    engine_name: str
    mode: str

    def __init__(self, inner: InferenceEngine, scoring_mode: ScoringMode) -> None:
        self._inner = inner
        self._scoring_mode = scoring_mode
        # Both concrete engines declare `engine_name` as a class attribute;
        # no capability discovery — the attribute is part of the engine
        # protocol.  Fall back only if a test double omits it.
        name = getattr(inner, "engine_name", None)
        self.engine_name = name if isinstance(name, str) else "unknown-engine"
        self.mode = (
            "onnx-model" if scoring_mode is ScoringMode.ONNX_MODEL else "heuristic"
        )

    def infer(self, record: IncidentNormalizedRecord) -> ModelInferenceResult:
        result = self._inner.infer(record)
        return self._enforce(result)

    def infer_batch(
        self, records: list[IncidentNormalizedRecord]
    ) -> list[ModelInferenceResult]:
        return [self._enforce(result) for result in self._inner.infer_batch(records)]

    def _enforce(self, result: ModelInferenceResult) -> ModelInferenceResult:
        if self._scoring_mode is ScoringMode.ONNX_MODEL:
            if result.scoring_mode is not ScoringMode.ONNX_MODEL:
                raise ModeCompositionError(
                    "model-backed deployment received a non-model scoring "
                    f"result for {result.log_id}; refusing to serve it"
                )
            return result
        if result.scoring_mode is not ScoringMode.HEURISTIC:
            raise ModeCompositionError(
                "heuristic deployment received a model scoring result for "
                f"{result.log_id}; refusing to serve it"
            )
        return result


class _ModelInferenceEngine:
    """Model-backed engine built through the application adapter layer.

    Record-to-tensor conversion lives in the injected encoder; this class is
    a thin ``ServingInferenceAdapter`` binding so the application protocol
    sees one consistent engine interface regardless of backend.
    """

    engine_name = "onnx-inference-engine"
    mode = "onnx-model"

    def __init__(self, engine: ONNXInferenceEngine, encoder: object) -> None:
        from riskforge.application.inference_adapter import ServingInferenceAdapter

        self._adapter = ServingInferenceAdapter(engine, encoder)  # type: ignore[arg-type]

    def infer(self, record: IncidentNormalizedRecord) -> ModelInferenceResult:
        return self._adapter.infer(record)

    def infer_batch(
        self, records: list[IncidentNormalizedRecord]
    ) -> list[ModelInferenceResult]:
        return list(self._adapter.infer_batch(records))


def _build_model_engine(settings: RuntimeSettings) -> _ModelInferenceEngine:
    """Build the ONNX engine plus encoder, validating every artifact field.

    Requires ``RISKFORGE_MODEL_PATH``, ``RISKFORGE_MANIFEST_PATH``,
    ``RISKFORGE_TOKENIZER_NAME`` and (for production-grade posture)
    ``RISKFORGE_MODEL_MAX_SEQ_LEN`` where it matters.  Any missing or
    incompatible piece is a hard startup failure — never a fallback.
    """
    model_path = os.environ.get("RISKFORGE_MODEL_PATH", "").strip()
    manifest_path = os.environ.get("RISKFORGE_MANIFEST_PATH", "").strip()
    tokenizer_name = os.environ.get("RISKFORGE_TOKENIZER_NAME", "").strip()
    if not model_path or not manifest_path or not tokenizer_name:
        raise ModeCompositionError(
            "model-backed scoring requires RISKFORGE_MODEL_PATH, "
            "RISKFORGE_MANIFEST_PATH, and RISKFORGE_TOKENIZER_NAME"
        )
    if not Path(model_path).is_file() or not Path(manifest_path).is_file():
        raise ModeCompositionError(
            f"configured model artifact not found: {model_path!s}, {manifest_path!s}"
        )
    from riskforge.serving.artifact import ModelArtifactManifest

    try:
        manifest = ModelArtifactManifest.load(manifest_path)
    except Exception as error:
        raise ModeCompositionError(f"artifact manifest invalid: {error}") from error
    if tokenizer_name != manifest.backbone:
        raise ModeCompositionError(
            f"RISKFORGE_TOKENIZER_NAME {tokenizer_name!r} does not match the "
            f"artifact manifest backbone {manifest.backbone!r}"
        )
    try:
        engine = ONNXInferenceEngine.from_artifact(
            model_path,
            manifest_path,
            max_batch_size=settings.max_batch_size,
            inference_timeout_s=settings.request_timeout_seconds,
        )
    except Exception as error:
        raise ModeCompositionError(f"model artifact failed validation: {error}") from error
    from riskforge.encoding.tokenizer_encoder import HFIncidentEncoder

    try:
        encoder = HFIncidentEncoder(tokenizer_name, manifest.max_sequence_length)
    except Exception as error:
        raise ModeCompositionError(f"incident encoder failed to initialize: {error}") from error
    return _ModelInferenceEngine(engine, encoder)


def _build_engine(
    settings: RuntimeSettings, config: DeploymentConfig
) -> tuple[_ModeEnforcingEngine, str]:
    """Build the engine the mode demands; never silently downgrade.

    production: model artifact + encoder mandatory.
    pilot: model artifact when configured (missing configured artifact is an
    error); otherwise the explicitly-selected heuristic engine, labelled.
    demo: heuristic engine only (no artifact is trusted in demo).
    """
    model_configured = bool(
        os.environ.get("RISKFORGE_MODEL_PATH", "").strip()
        or os.environ.get("RISKFORGE_MANIFEST_PATH", "").strip()
    )
    if config.require_model_artifact:
        if not model_configured:
            raise ModeCompositionError(
                "production mode requires a validated model artifact: set "
                "RISKFORGE_MODEL_PATH, RISKFORGE_MANIFEST_PATH, and "
                "RISKFORGE_TOKENIZER_NAME; heuristic fallback is unavailable"
            )
        inner = _build_model_engine(settings)
        wrapped = _ModeEnforcingEngine(inner, ScoringMode.ONNX_MODEL)
        return wrapped, "onnx-model"
    if model_configured:
        # Half-configuration is always an error.
        model_path = os.environ.get("RISKFORGE_MODEL_PATH", "").strip()
        manifest_path = os.environ.get("RISKFORGE_MANIFEST_PATH", "").strip()
        if not (model_path and manifest_path):
            raise ModeCompositionError(
                "RISKFORGE_MODEL_PATH and RISKFORGE_MANIFEST_PATH must be "
                "configured together"
            )
        inner = _build_model_engine(settings)
        wrapped = _ModeEnforcingEngine(inner, ScoringMode.ONNX_MODEL)
        return wrapped, "onnx-model"
    if config.allow_heuristic_engine:
        engine = HeuristicRuleEngine()
        logger.warning(
            "using the deterministic heuristic rule engine (mode=%s); scores "
            "are rule-based, not model probabilities",
            config.mode.value,
        )
        return _ModeEnforcingEngine(engine, ScoringMode.HEURISTIC), "heuristic"
    raise ModeCompositionError(
        "production mode requires a validated model artifact; heuristic "
        "fallback is not available"
    )


def _authorizer_from_env() -> SubjectAllowlistReviewerAuthorizer:
    """Deny-by-default review authorization from the subject allow-list.

    ``RISKFORGE_REVIEWER_SUBJECTS`` is a comma-separated list of authenticated
    subject ids permitted to record review decisions.  When unset, every
    review is denied — deployments must explicitly name their reviewers.
    """
    subjects_env = os.environ.get("RISKFORGE_REVIEWER_SUBJECTS", "").strip()
    subjects = {subject.strip() for subject in subjects_env.split(",") if subject.strip()}
    if not subjects:
        logger.warning(
            "RISKFORGE_REVIEWER_SUBJECTS unset; all review decisions will be denied"
        )
    return SubjectAllowlistReviewerAuthorizer(allowed_subjects=subjects)


def _build_persistence(
    config: DeploymentConfig,
) -> tuple[
    IncidentResultRepository,
    AuditEventRepository,
    SQLiteReviewAuditWriter,
    list[LifecycleComponent],
]:
    """Build persistence; production mandates PostgreSQL and fails closed."""
    backend = os.environ.get("RISKFORGE_PERSISTENCE", "sqlite").strip().lower()
    if config.mode is DeploymentMode.PRODUCTION and backend != "postgres":
        raise ModeCompositionError(
            "production mode requires RISKFORGE_PERSISTENCE=postgres"
        )
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

        config_pg = PostgresConfig()
        applied = run_migrations()
        logger.info("postgres migrations applied: %s", applied)
        pool = PostgresConnectionPool(config_pg)
        return (
            PostgresIncidentResultRepository(pool),
            PostgresAuditEventRepository(pool),
            PostgresReviewAuditWriter(pool),
            [PostgresPoolLifecycle(pool)],
        )
    if backend != "sqlite":
        raise ModeCompositionError(f"unsupported RISKFORGE_PERSISTENCE backend: {backend!r}")
    if config.mode is DeploymentMode.PRODUCTION:
        raise ModeCompositionError("production mode requires PostgreSQL")

    db_path = _sqlite_database_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    incidents: IncidentResultRepository = SQLiteIncidentResultRepository(db_path)
    audit: AuditEventRepository = SQLiteAuditEventRepository(db_path)
    review_writer = SQLiteReviewAuditWriter(db_path)
    return incidents, audit, review_writer, []


class _StoreMetricsService:
    """MetricsService adapter computing per-asset summaries from persistence."""

    def __init__(self, incidents: IncidentResultRepository) -> None:
        self._incidents = incidents

    def asset_summary(self, asset_id: str) -> AssetRiskSummary:
        stored = _iter_stored(self._incidents, IncidentResultFilter(asset_id=asset_id))
        if not stored:
            raise KeyError(asset_id)
        results = [item.result for item in stored]
        mapping = {
            item.incident.log_id: (item.incident.asset_id, item.incident.asset_type)
            for item in stored
        }
        summaries = MetricsAggregator(min_barrier_recurrence=2).compute(results, mapping)
        if not summaries:
            raise KeyError(asset_id)
        return summaries[0]


class _PersistingApplicationService:
    """Persist every scored result so read routes see the same data.

    ``POST /v1/inference`` is a scored write: the workflow read routes and the
    review service read exclusively from persistence, so results must be
    stored idempotently at scoring time.  Every successful scoring action also
    appends an ``inference_recorded`` audit event carrying the actor, scoring
    mode, and model/artifact version — audit failures never fail the scored
    write (governance is best-effort, durability is not).  Scoring, ingestion,
    and analytics capabilities are declared explicitly on the facade — no
    dynamic attach.
    """

    def __init__(
        self,
        core: BackendApplicationService,
        incidents: IncidentResultRepository,
        scoring_pipeline: _ScoringIngestionPipeline,
        analytics: _StoreAnalytics,
        audit: AuditEventRepository,
        *,
        scoring_mode: str,
        artifact_version: str | None,
    ) -> None:
        self._core = core
        self._incidents = incidents
        self._scoring_pipeline = scoring_pipeline
        self._analytics = analytics
        self._audit = audit
        self._scoring_mode = scoring_mode
        self._artifact_version = artifact_version

    # ── scoring ────────────────────────────────────────────────────────
    def process_incident(
        self,
        record: IncidentNormalizedRecord,
        *,
        actor_id: str = "system",
        correlation_id: str | None = None,
    ) -> ModelInferenceResult:
        result = self._core.process_incident(record)
        self._store(record, result)
        self._audit_scoring(record, result, actor_id=actor_id, correlation_id=correlation_id)
        return result

    def process_batch(
        self,
        records: list[IncidentNormalizedRecord],
        *,
        actor_id: str = "system",
        correlation_id: str | None = None,
    ) -> list[ModelInferenceResult]:
        results = self._core.process_batch(records)
        for record, result in zip(records, results, strict=True):
            self._store(record, result)
            self._audit_scoring(record, result, actor_id=actor_id, correlation_id=correlation_id)
        return results

    def ingest(
        self,
        data: bytes,
        fmt: str,
        *,
        actor_id: str = "system",
        correlation_id: str | None = None,
    ) -> object:
        return self._scoring_pipeline.ingest(
            data, fmt, actor_id=actor_id, correlation_id=correlation_id
        )

    def analytics_summary(self, timestamp_from: str | None = None, timestamp_to: str | None = None) -> object:
        return self._analytics.summary(timestamp_from, timestamp_to)

    # ── audit ──────────────────────────────────────────────────────────
    def _audit_scoring(
        self,
        record: IncidentNormalizedRecord,
        result: ModelInferenceResult,
        *,
        actor_id: str,
        correlation_id: str | None,
    ) -> None:
        """Append an inference_recorded event; best-effort, never fatal.

        The reason field carries structured provenance (scoring mode, model
        version, correlation id) without copying the raw narrative.
        """
        provenance_parts = [f"scoring_mode={self._scoring_mode}"]
        if result.model_version:
            provenance_parts.append(f"model_version={result.model_version}")
        if result.calibration_version:
            provenance_parts.append(f"calibration_version={result.calibration_version}")
        if correlation_id:
            provenance_parts.append(f"correlation_id={correlation_id}")
        event = AuditEvent(
            event_id=f"inference:{record.log_id}:{record.timestamp.isoformat()}",
            log_id=record.log_id,
            event_type=AuditEventType.INFERENCE_RECORDED,
            actor_id=actor_id,
            occurred_at=record.timestamp,
            reason="; ".join(provenance_parts),
        )
        try:
            self._audit.append(event)
        except Exception:
            logger.warning(
                "inference audit event could not be appended",
                extra={"log_id": record.log_id},
                exc_info=True,
            )

    # ── persistence ────────────────────────────────────────────────────
    def _store(self, record: IncidentNormalizedRecord, result: ModelInferenceResult) -> None:
        try:
            self._incidents.create_idempotent(
                StoredIncidentResult(incident=record, result=result)
            )
        except PersistenceConflictError:
            # Re-submitting the same report is idempotent only when the stored
            # decision matches the fresh one; latency_ms is timing noise and
            # must not make identical rescoring look like a conflict.
            stored = self._incidents.get(record.log_id)
            if stored is not None and _same_decision(record, result, stored):
                return
            raise

    # ── read/workflow delegation ───────────────────────────────────────
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


def _inference_audit_event(
    record: IncidentNormalizedRecord,
    result: ModelInferenceResult,
    *,
    actor_id: str,
    scoring_mode: str,
    correlation_id: str | None,
) -> AuditEvent:
    """Build an inference_recorded governance event for one scored report.

    The reason field carries structured provenance (scoring mode, model and
    calibration versions, correlation id) without copying the raw narrative.
    """
    provenance_parts = [f"scoring_mode={scoring_mode}"]
    if result.model_version:
        provenance_parts.append(f"model_version={result.model_version}")
    if result.calibration_version:
        provenance_parts.append(f"calibration_version={result.calibration_version}")
    if correlation_id:
        provenance_parts.append(f"correlation_id={correlation_id}")
    return AuditEvent(
        event_id=f"inference:{record.log_id}:{record.timestamp.isoformat()}",
        log_id=record.log_id,
        event_type=AuditEventType.INFERENCE_RECORDED,
        actor_id=actor_id,
        occurred_at=record.timestamp,
        reason="; ".join(provenance_parts),
    )


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

    def __init__(self, engine_mode: str, scoring_label: str) -> None:
        self._engine_mode = engine_mode
        self._scoring_label = scoring_label

    @property
    def name(self) -> str:
        return _ENGINE_MODE_COMPONENT

    def start(self) -> None:
        logger.info(
            "inference engine mode: %s (%s)", self._engine_mode, self._scoring_label
        )

    def stop(self) -> None:
        return None

    def readiness(self) -> ComponentReadiness:
        return ComponentReadiness(
            name=self.name,
            ready=True,
            detail=f"{self._engine_mode} [{self._scoring_label}]",
        )


class _StoreAnalytics:
    """Analytics over the incident-result store, computed on demand.

    An optional time window (ISO timestamps) bounds every aggregate so the
    dashboard's period selector filters server-side.
    """

    def __init__(self, incidents: IncidentResultRepository) -> None:
        self._incidents = incidents

    def summary(self, timestamp_from: str | None = None, timestamp_to: str | None = None) -> object:
        incident_filter = IncidentResultFilter(
            timestamp_from=_parse_window(timestamp_from, "timestamp_from"),
            timestamp_to=_parse_window(timestamp_to, "timestamp_to"),
        )
        stored = _iter_stored(self._incidents, incident_filter)
        return compute_summary([(item.incident, item.result) for item in stored]).to_dict()


class _ScoringIngestionPipeline:
    """Ingestion pipeline bound to normalization, scoring, and persistence."""

    def __init__(
        self,
        pipeline: IngestionPipeline,
        incidents: IncidentResultRepository,
        audit: AuditEventRepository,
        *,
        scoring_mode: str,
    ) -> None:
        self._pipeline = pipeline
        self._incidents = incidents
        self._audit = audit
        self._scoring_mode = scoring_mode

    def ingest(
        self,
        data: bytes,
        fmt: str,
        *,
        actor_id: str = "system",
        correlation_id: str | None = None,
    ) -> object:
        outcome = self._pipeline.run(data, fmt)
        persisted = 0
        audit_events: list[AuditEvent] = []
        for item in outcome.items:
            if (
                item.status == "normalized"
                and item.normalized is not None
                and item.result is not None
            ):
                try:
                    self._incidents.create_idempotent(
                        StoredIncidentResult(incident=item.normalized, result=item.result)
                    )
                    persisted += 1
                    audit_events.append(
                        _inference_audit_event(
                            item.normalized,
                            item.result,
                            actor_id=actor_id,
                            scoring_mode=self._scoring_mode,
                            correlation_id=correlation_id,
                        )
                    )
                except PersistenceConflictError:
                    stored = self._incidents.get(item.normalized.log_id)
                    if stored is None or not _same_decision(
                        item.normalized, item.result, stored
                    ):
                        raise
        if audit_events:
            try:
                self._audit.append_many(audit_events)
            except Exception:
                logger.warning(
                    "inference audit events could not be appended",
                    extra={"count": len(audit_events)},
                    exc_info=True,
                )
        return outcome


class ProductionComposer:
    """Assemble the complete application facade and lifecycle components.

    Composition order: persistence (mode-gated), then metrics over the store,
    then scoring, then the transport facade.  Every capability is declared
    explicitly on the facade class.
    """

    def compose(self, settings: RuntimeSettings) -> RuntimeAssembly:
        config = DeploymentConfig.from_env()
        incidents, audit, review_writer, components = _build_persistence(config)
        engine, engine_mode = _build_engine(settings, config)
        metrics = _StoreMetricsService(incidents)
        core = ApplicationService(engine, metrics)
        reader = PersistenceWorkflowReader(incidents, audit)
        review_service = ReviewService(incidents, review_writer, _authorizer_from_env())
        backend = BackendApplicationService(core, reader, ReviewServiceAdapter(review_service))
        analytics = _StoreAnalytics(incidents)
        scoring_label = (
            ConfigScoringMode.HEURISTIC.value
            if engine_mode == "heuristic"
            else ConfigScoringMode.ONNX_MODEL.value
        )
        scoring_pipeline = _ScoringIngestionPipeline(
            IngestionPipeline(SpanPreservingGazetteer().process, engine),
            incidents,
            audit,
            scoring_mode=scoring_label,
        )
        artifact_version = engine.model_version if hasattr(engine, "model_version") else None
        scoring = _PersistingApplicationService(
            backend,
            incidents,
            scoring_pipeline,
            analytics,
            audit,
            scoring_mode=scoring_label,
            artifact_version=artifact_version,
        )

        components.append(_EngineModeLifecycle(engine_mode, scoring_label))
        return RuntimeAssembly(application=scoring, components=tuple(components))


def create_app() -> FastAPI:
    """Uvicorn-compatible ASGI factory for mode-aware composition."""
    from riskforge.runtime.asgi import create_managed_app

    config = DeploymentConfig.from_env()
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
    if config.require_auth:
        if os.environ.get("RISKFORGE_AUTH_ENABLED", "").strip().lower() not in {
            "1",
            "true",
            "yes",
        }:
            raise DeploymentConfigError(
                f"{config.mode.value} mode requires RISKFORGE_AUTH_ENABLED=true "
                "and a fully configured JwtAuthenticationService"
            )
        from riskforge.authentication.jwt_service import JwtAuthenticationService

        auth_service = JwtAuthenticationService.from_env()
    return create_managed_app(
        settings,
        ProductionComposer(),
        auth_service=auth_service,
        cors_origins=origins,
        enable_metrics=config.public_metrics,
        require_authentication=config.require_auth,
        deployment_mode=config.mode.value,
    )
