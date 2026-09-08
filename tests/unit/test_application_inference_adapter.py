from datetime import UTC, datetime

import numpy as np
import pytest

from riskforge.application.exceptions import InferenceApplicationError, ResultCorrelationError
from riskforge.application.inference_adapter import (
    EncodedIncident,
    IncidentInputEncoder,
    ServingInferenceAdapter,
    ServingInferenceEngine,
)
from riskforge.application.protocols import InferenceEngine
from riskforge.core.contracts import (
    AssetType,
    IncidentNormalizedRecord,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)

NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


def _record(log_id: str) -> IncidentNormalizedRecord:
    return IncidentNormalizedRecord(
        log_id=log_id,
        timestamp=NOW,
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Synthetic incident.",
        spans=[],
    )


def _result(log_id: str) -> ModelInferenceResult:
    return ModelInferenceResult(
        log_id=log_id,
        raw_sif_p_score=0.2,
        calibrated_sif_p_score=0.2,
        deterministic_override=False,
        routing=RoutingBucket.AUTO_DISMISS,
        matched_iogp_rules=[],
        triad=OperationalTriad(),
        latency_ms=1.0,
    )


class FakeEncoder:
    def __init__(self) -> None:
        self.single_calls: list[str] = []
        self.batch_calls: list[tuple[str, ...]] = []

    def encode(self, record: IncidentNormalizedRecord) -> EncodedIncident:
        self.single_calls.append(record.log_id)
        ids = np.array([1, 2, 3], dtype=np.int64)
        return EncodedIncident(ids, np.ones_like(ids))

    def encode_batch(
        self, records: list[IncidentNormalizedRecord]
    ) -> EncodedIncident:
        self.batch_calls.append(tuple(record.log_id for record in records))
        ids = np.ones((len(records), 3), dtype=np.int64)
        return EncodedIncident(ids, np.ones_like(ids))


class FakeEngine:
    def __init__(self, results=None, *, fail: bool = False) -> None:
        self.results = results
        self.fail = fail
        self.single_calls = []
        self.batch_calls = []

    def infer(self, record, input_ids, attention_mask, token_type_ids=None):
        if self.fail:
            raise RuntimeError("serving unavailable")
        self.single_calls.append(
            (record.log_id, input_ids.copy(), attention_mask.copy(), token_type_ids)
        )
        return self.results or _result(record.log_id)

    def infer_batch(self, records, input_ids, attention_mask, token_type_ids=None):
        if self.fail:
            raise RuntimeError("serving unavailable")
        self.batch_calls.append(
            (
                tuple(record.log_id for record in records),
                input_ids.copy(),
                attention_mask.copy(),
                token_type_ids,
            )
        )
        return self.results or [_result(record.log_id) for record in records]


def test_adapter_protocols_are_runtime_checkable() -> None:
    assert isinstance(FakeEncoder(), IncidentInputEncoder)
    assert isinstance(FakeEngine(), ServingInferenceEngine)
    assert isinstance(ServingInferenceAdapter(FakeEngine(), FakeEncoder()), InferenceEngine)


def test_single_inference_forwards_encoded_tensors() -> None:
    encoder = FakeEncoder()
    engine = FakeEngine()
    adapter = ServingInferenceAdapter(engine, encoder)
    result = adapter.infer(_record("LOG_1"))
    assert result == _result("LOG_1")
    assert encoder.single_calls == ["LOG_1"]
    assert len(engine.single_calls) == 1
    assert engine.single_calls[0][1].shape == (3,)


def test_batch_inference_preserves_log_id_order() -> None:
    encoder = FakeEncoder()
    records = [_record("LOG_1"), _record("LOG_2"), _record("LOG_3")]
    engine = FakeEngine(results=[_result("LOG_3"), _result("LOG_1"), _result("LOG_2")])
    results = ServingInferenceAdapter(engine, encoder).infer_batch(records)
    assert [result.log_id for result in results] == ["LOG_1", "LOG_2", "LOG_3"]
    assert encoder.batch_calls == [("LOG_1", "LOG_2", "LOG_3")]
    assert engine.batch_calls[0][1].shape == (3, 3)


def test_batch_duplicate_result_ids_are_rejected() -> None:
    records = [_record("LOG_1"), _record("LOG_2")]
    engine = FakeEngine(results=[_result("LOG_1"), _result("LOG_1")])
    with pytest.raises(ResultCorrelationError):
        ServingInferenceAdapter(engine, FakeEncoder()).infer_batch(records)


def test_batch_missing_result_ids_are_rejected() -> None:
    records = [_record("LOG_1"), _record("LOG_2")]
    engine = FakeEngine(results=[_result("LOG_1")])
    with pytest.raises(ResultCorrelationError):
        ServingInferenceAdapter(engine, FakeEncoder()).infer_batch(records)


def test_serving_failures_are_translated_without_narrative() -> None:
    record = _record("LOG_1")
    with pytest.raises(InferenceApplicationError, match="serving inference failed"):
        ServingInferenceAdapter(FakeEngine(fail=True), FakeEncoder()).infer(record)


def test_encoded_incident_validates_shapes_and_casts_dtype() -> None:
    encoded = EncodedIncident([1, 2], [1, 1])
    assert encoded.input_ids.dtype == np.dtype(np.int64)
    assert encoded.attention_mask.dtype == np.dtype(np.int64)
    with pytest.raises(ValueError):
        EncodedIncident([1, 2], [1])
    with pytest.raises(ValueError):
        EncodedIncident([[1, 2]], [[1, 2]], [[1], [2]])


def test_empty_batch_does_not_call_encoder_or_engine() -> None:
    encoder = FakeEncoder()
    engine = FakeEngine()
    assert ServingInferenceAdapter(engine, encoder).infer_batch([]) == []
    assert encoder.batch_calls == []
    assert engine.batch_calls == []
