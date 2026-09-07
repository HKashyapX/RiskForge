from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from riskforge.core.contracts import (
    AssetType,
    EntitySpan,
    IncidentNormalizedRecord,
    IncidentRawRecord,
    LifeSavingRule,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)

def test_incident_raw_record_valid():
    rec = IncidentRawRecord(
        log_id="LOG_001",
        timestamp=datetime.now(timezone.utc),
        asset_id="RIG_07",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Pressure surge detected."
    )
    assert rec.log_id == "LOG_001"
    assert rec.asset_type == AssetType.DRILLING_RIG

def test_incident_raw_record_invalid_enum():
    with pytest.raises(ValidationError):
        IncidentRawRecord(
            log_id="LOG_002",
            timestamp=datetime.now(timezone.utc),
            asset_id="RIG_07",
            asset_type="invalid_rig_type",
            raw_narrative="Test narrative"
        )

def test_model_inference_result_bounds():
    with pytest.raises(ValidationError):
        ModelInferenceResult(
            log_id="LOG_003",
            raw_sif_p_score=1.5,
            calibrated_sif_p_score=0.5,
            deterministic_override=False,
            routing=RoutingBucket.AUTO_DISMISS,
            matched_iogp_rules=[],
            triad=OperationalTriad(),
            latency_ms=10.0
        )
