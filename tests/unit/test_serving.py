from datetime import datetime, timezone

import numpy as np

from riskforge.core.contracts import (
    AssetType,
    EntitySpan,
    IncidentNormalizedRecord,
    RoutingBucket,
)
from riskforge.serving.engine import ONNXInferenceEngine
from riskforge.serving.postprocessor import InferencePostprocessor


def _record() -> IncidentNormalizedRecord:
    text = "Driller observed 3000 psi pressure with BOP leak."
    start = text.index("BOP")
    return IncidentNormalizedRecord(
        log_id="SERVE_001",
        timestamp=datetime.now(timezone.utc),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative=text,
        spans=[
            EntitySpan(
                text="BOP",
                canonical_form="blowout_preventer",
                start_char=start,
                end_char=start + 3,
                entity_type="BARRIER",
            )
        ],
    )


class _Node:
    def __init__(self, name: str) -> None:
        self.name = name


class _Session:
    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self):
        return [_Node("input_ids"), _Node("attention_mask")]

    def get_outputs(self):
        return [_Node("sif_logits"), _Node("iogp_logits")]

    def run(self, output_names, feed):
        self.calls += 1
        batch = len(feed["input_ids"])
        return [np.zeros((batch, 1)), np.zeros((batch, 9))]


def test_postprocessor_override_and_contract() -> None:
    result = InferencePostprocessor().process(
        _record(), np.array([[0.0]]), np.zeros((1, 9)), latency_ms=1.25
    )
    assert result.deterministic_override
    assert result.calibrated_sif_p_score == 0.95
    assert result.routing is RoutingBucket.CRITICAL_ESCALATION
    assert result.triad.failed_barrier is not None


def test_engine_chunks_dynamic_batches() -> None:
    session = _Session()
    engine = ONNXInferenceEngine("unused.onnx", session=session, max_batch_size=2)
    records = [_record(), _record(), _record()]
    results = engine.infer_batch(records, np.ones((3, 8)), np.ones((3, 8)))
    assert len(results) == 3
    assert session.calls == 2
