from datetime import datetime, timezone
from riskforge.core.contracts import (
    AssetRiskSummary,
    AssetType,
    EntitySpan,
    IncidentNormalizedRecord,
    IncidentRawRecord,
    LifeSavingRule,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)

def test_mock_end_to_end_contract_flow():
    raw = IncidentRawRecord(
        log_id="MOCK_001",
        timestamp=datetime.now(timezone.utc),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Mock raw narrative"
    )

    span = EntitySpan(
        text="Mock",
        canonical_form="mock_token",
        start_char=0,
        end_char=4,
        entity_type="ASSET"
    )
    norm = IncidentNormalizedRecord(
        log_id=raw.log_id,
        timestamp=raw.timestamp,
        asset_id=raw.asset_id,
        asset_type=raw.asset_type,
        raw_narrative=raw.raw_narrative,
        spans=[span]
    )

    inference = ModelInferenceResult(
        log_id=norm.log_id,
        raw_sif_p_score=0.85,
        calibrated_sif_p_score=0.82,
        deterministic_override=False,
        routing=RoutingBucket.CRITICAL_ESCALATION,
        matched_iogp_rules=[LifeSavingRule.LINE_OF_FIRE],
        triad=OperationalTriad(
            activity=None,
            asset_location=span,
            failed_barrier=None
        ),
        latency_ms=12.4
    )

    summary = AssetRiskSummary(
        asset_id=norm.asset_id,
        asset_type=norm.asset_type,
        total_logs=1,
        sif_precursor_count=1,
        spd_score=100.0,
        recurrent_failed_barriers=[]
    )

    assert inference.log_id == raw.log_id
    assert summary.asset_id == raw.asset_id
