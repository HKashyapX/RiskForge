from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


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


class ScoringMode(str, Enum):
    """What actually produced a score.  Heuristic and model numbers are never
    mixed: the scoring mode is recorded with every result so dashboards and
    audits can always tell rule-based scores from model probabilities."""

    ONNX_MODEL = "onnx-model"
    HEURISTIC = "heuristic"


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
    reporter_severity_rank: str | None = None

class IncidentNormalizedRecord(BaseModel):
    log_id: str
    timestamp: datetime
    asset_id: str
    asset_type: AssetType
    raw_narrative: str
    spans: list[EntitySpan]

class OperationalTriad(BaseModel):
    activity: EntitySpan | None = None
    asset_location: EntitySpan | None = None
    failed_barrier: EntitySpan | None = None

class ModelInferenceResult(BaseModel):
    log_id: str
    raw_sif_p_score: float = Field(..., ge=0.0, le=1.0)
    calibrated_sif_p_score: float = Field(..., ge=0.0, le=1.0)
    deterministic_override: bool
    routing: RoutingBucket
    matched_iogp_rules: list[LifeSavingRule]
    triad: OperationalTriad
    latency_ms: float
    scoring_mode: ScoringMode = Field(
        default=ScoringMode.HEURISTIC,
        description="Engine that produced the scores: onnx-model or heuristic.",
    )
    engine_name: str = Field(
        default="heuristic-rules-v1",
        min_length=1,
        max_length=128,
        description="Stable identifier of the scoring engine implementation.",
    )
    model_version: str | None = Field(
        default=None,
        max_length=128,
        description="Artifact manifest model version; required for onnx-model results.",
    )
    calibration_version: str | None = Field(
        default=None,
        max_length=128,
        description="Calibration metadata version applied to the raw score, if any.",
    )

    @model_validator(mode="after")
    def _model_scores_require_model_provenance(self) -> "ModelInferenceResult":
        if self.scoring_mode is ScoringMode.ONNX_MODEL:
            if not self.model_version:
                raise ValueError(
                    "model_version is required when scoring_mode is onnx-model"
                )
            if self.engine_name == "heuristic-rules-v1":
                raise ValueError(
                    "onnx-model results must not carry the heuristic engine name"
                )
        if self.scoring_mode is ScoringMode.HEURISTIC and self.model_version is not None:
            raise ValueError(
                "heuristic results must not claim a model version; heuristic "
                "scores are not model probabilities"
            )
        return self

class AssetRiskSummary(BaseModel):
    asset_id: str
    asset_type: AssetType
    total_logs: int
    sif_precursor_count: int
    spd_score: float
    recurrent_failed_barriers: list[str]
