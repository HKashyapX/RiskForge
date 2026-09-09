from datetime import UTC, datetime

from riskforge.application.workflow_models import (
    IncidentQuery,
    PageRequest,
    ReviewAction,
    ReviewCommand,
)
from riskforge.core.contracts import (
    AssetType,
    IncidentNormalizedRecord,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)
from riskforge.persistence.models import (
    AuditEvent,
    AuditEventType,
    StoredIncidentResult,
)
from riskforge.persistence.models import (
    ReviewAction as PersistenceReviewAction,
)
from riskforge.persistence.sqlite.repository import (
    SQLiteAuditEventRepository,
    SQLiteIncidentResultRepository,
)
from riskforge.persistence.sqlite.review_audit import SQLiteReviewAuditWriter
from riskforge.review.service import ReviewService
from riskforge.runtime.workflow_adapters import PersistenceWorkflowReader, ReviewServiceAdapter

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _stored(log_id: str, routing: RoutingBucket) -> StoredIncidentResult:
    return StoredIncidentResult(
        incident=IncidentNormalizedRecord(
            log_id=log_id,
            timestamp=NOW,
            asset_id="RIG_01",
            asset_type=AssetType.DRILLING_RIG,
            raw_narrative="Synthetic incident.",
            spans=[],
        ),
        result=ModelInferenceResult(
            log_id=log_id,
            raw_sif_p_score=0.8,
            calibrated_sif_p_score=0.8,
            deterministic_override=False,
            routing=routing,
            matched_iogp_rules=[],
            triad=OperationalTriad(),
            latency_ms=1.0,
        ),
    )


class AllowAll:
    def can_decide(
        self, reviewer_id: str, log_id: str, action: PersistenceReviewAction
    ) -> bool:
        return True


def test_workflow_adapters_integrate_existing_sqlite_and_review_services(tmp_path) -> None:
    database = tmp_path / "riskforge.db"
    incidents = SQLiteIncidentResultRepository(database)
    audit = SQLiteAuditEventRepository(database)
    incidents.create_idempotent(_stored("LOG_2", RoutingBucket.AUTO_DISMISS))
    incidents.create_idempotent(_stored("LOG_1", RoutingBucket.CRITICAL_ESCALATION))
    audit.append(
        AuditEvent(
            event_id="inference:LOG_1",
            log_id="LOG_1",
            event_type=AuditEventType.INFERENCE_RECORDED,
            actor_id="system",
            occurred_at=NOW,
        )
    )

    reader = PersistenceWorkflowReader(incidents, audit)
    page = reader.list_incidents(
        IncidentQuery(routing=RoutingBucket.CRITICAL_ESCALATION),
        PageRequest(limit=10),
    )
    assert [item.result.log_id for item in page.items] == ["LOG_1"]
    assert reader.get_incident("LOG_1").incident.raw_narrative == "Synthetic incident."
    assert reader.list_audit_events("LOG_1", PageRequest()).items[0].event_id == "inference:LOG_1"

    reviews = ReviewServiceAdapter(
        ReviewService(incidents, SQLiteReviewAuditWriter(database), AllowAll())
    )
    decision = reviews.decide(
        ReviewCommand(
            log_id="LOG_1",
            decision_id="decision-1",
            reviewer_id="reviewer-1",
            action=ReviewAction.CONFIRM,
            reason="verified",
            decided_at=NOW,
        )
    )
    assert decision.action is ReviewAction.CONFIRM
    assert reader.list_audit_events("LOG_1", PageRequest()).total == 2
