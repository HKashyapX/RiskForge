from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class AssetType(str, Enum):
    DRILLING_RIG = "drilling_rig"
    WORKOVER_RIG = "workover_rig"
    GAS_GATHERING_STATION = "gas_gathering_station"
    OIL_COLLECTING_STATION = "oil_collecting_station"
    PIPELINE_NETWORK = "pipeline_network"

class LifeSavingRule(str, Enum):
    BYPASSING_SAFETY_CONTROLS = "bypassing_safety_controls"
    CONFINED_SPACE = "confined_space"
    DRIVING = "driving"
    ENERGY_ISOLATION = "energy_isolation"
    HOT_WORK = "hot_work"
    LINE_OF_FIRE = "line_of_fire"
    SAFE_MECHANICAL_LIFTING = "safe_mechanical_lifting"
    TOXIC_GAS = "toxic_gas"
    WORK_AT_HEIGHT = "work_at_height"

class RoutingBucket(str, Enum):
    AUTO_DISMISS = "auto_dismiss"
    HITL_REVIEW = "hitl_review"
    CRITICAL_ESCALATION = "critical_escalation"

class EntitySpan(BaseModel):
    text: str
    canonical_form: str
    start_char: int
    end_char: int
    entity_type: str

class IncidentRawRecord(BaseModel):
    log_id: str
    timestamp: datetime
    asset_id: str
    asset_type: AssetType
    raw_narrative: str
    reporter_severity_rank: Optional[str] = None

class IncidentNormalizedRecord(BaseModel):
    log_id: str
    timestamp: datetime
    asset_id: str
    asset_type: AssetType
    raw_narrative: str
    spans: List[EntitySpan]

class OperationalTriad(BaseModel):
    activity: Optional[EntitySpan] = None
    asset_location: Optional[EntitySpan] = None
    failed_barrier: Optional[EntitySpan] = None

class ModelInferenceResult(BaseModel):
    log_id: str
    raw_sif_p_score: float = Field(..., ge=0.0, le=1.0)
    calibrated_sif_p_score: float = Field(..., ge=0.0, le=1.0)
    deterministic_override: bool
    routing: RoutingBucket
    matched_iogp_rules: List[LifeSavingRule]
    triad: OperationalTriad
    latency_ms: float

class AssetRiskSummary(BaseModel):
    asset_id: str
    asset_type: AssetType
    total_logs: int
    sif_precursor_count: int
    spd_score: float
    recurrent_failed_barriers: List[str]
