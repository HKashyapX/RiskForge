"""Bridge application-level incidents to the tensor-based serving boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from riskforge.application.exceptions import (
    InferenceApplicationError,
    ResultCorrelationError,
)
from riskforge.core.contracts import IncidentNormalizedRecord, ModelInferenceResult


@dataclass(frozen=True)
class EncodedIncident:
    """Validated token tensors produced by an injected incident encoder."""

    input_ids: np.ndarray
    attention_mask: np.ndarray
    token_type_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        input_ids = np.asarray(self.input_ids, dtype=np.int64)
        attention_mask = np.asarray(self.attention_mask, dtype=np.int64)
        token_types = (
            None if self.token_type_ids is None else np.asarray(self.token_type_ids, dtype=np.int64)
        )
        if input_ids.ndim not in (1, 2):
            raise ValueError("input_ids must be rank 1 or rank 2")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")
        if token_types is not None and token_types.shape != input_ids.shape:
            raise ValueError("token_type_ids must match input_ids shape")
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "attention_mask", attention_mask)
        object.__setattr__(self, "token_type_ids", token_types)


@runtime_checkable
class IncidentInputEncoder(Protocol):
    """Application-facing tokenization/encoding dependency."""

    def encode(self, record: IncidentNormalizedRecord) -> EncodedIncident:
        """Encode one normalized incident."""

    def encode_batch(self, records: Sequence[IncidentNormalizedRecord]) -> EncodedIncident:
        """Encode a batch with one tensor row per input record, preserving order."""


@runtime_checkable
class ServingInferenceEngine(Protocol):
    """Application-facing protocol for tensor-based serving engines."""

    def infer(
        self,
        record: IncidentNormalizedRecord,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        token_type_ids: np.ndarray | None = None,
    ) -> ModelInferenceResult:
        """Run one normalized incident through a tensor-based engine."""

    def infer_batch(
        self,
        records: Sequence[IncidentNormalizedRecord],
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        token_type_ids: np.ndarray | None = None,
    ) -> Sequence[ModelInferenceResult]:
        """Run a batch through a tensor-based engine."""


class ServingInferenceAdapter:
    """Adapt tensor-based serving to the application inference protocol."""

    def __init__(self, engine: ServingInferenceEngine, encoder: IncidentInputEncoder) -> None:
        self._engine = engine
        self._encoder = encoder

    def infer(self, record: IncidentNormalizedRecord) -> ModelInferenceResult:
        try:
            encoded = self._encoder.encode(record)
            result = self._engine.infer(
                record,
                encoded.input_ids,
                encoded.attention_mask,
                encoded.token_type_ids,
            )
        except (ResultCorrelationError, ValueError):
            raise
        except Exception as error:
            raise InferenceApplicationError("serving inference failed") from error
        if result.log_id != record.log_id:
            raise ResultCorrelationError("serving result log_id does not match request")
        return result

    def infer_batch(
        self, records: Sequence[IncidentNormalizedRecord]
    ) -> list[ModelInferenceResult]:
        record_list = list(records)
        if not record_list:
            return []
        try:
            encoded = self._encoder.encode_batch(record_list)
            results = list(
                self._engine.infer_batch(
                    record_list,
                    encoded.input_ids,
                    encoded.attention_mask,
                    encoded.token_type_ids,
                )
            )
        except (ResultCorrelationError, ValueError):
            raise
        except Exception as error:
            raise InferenceApplicationError("serving batch inference failed") from error
        if len(results) != len(record_list):
            raise ResultCorrelationError("serving result count does not match request count")
        by_log_id: dict[str, ModelInferenceResult] = {}
        for result in results:
            if result.log_id in by_log_id:
                raise ResultCorrelationError("serving returned duplicate log_id values")
            by_log_id[result.log_id] = result
        expected = {record.log_id for record in record_list}
        if set(by_log_id) != expected:
            raise ResultCorrelationError("serving result log_id set does not match request")
        return [by_log_id[record.log_id] for record in record_list]
