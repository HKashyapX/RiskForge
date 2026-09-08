"""Deterministic application orchestration for normalized incidents."""

from __future__ import annotations

from collections.abc import Sequence

from riskforge.application.exceptions import (
    DuplicateLogIdError,
    InferenceApplicationError,
    MetricsApplicationError,
    ResultCorrelationError,
)
from riskforge.application.protocols import InferenceEngine, MetricsService
from riskforge.core.contracts import AssetRiskSummary, IncidentNormalizedRecord, ModelInferenceResult


class ApplicationService:
    """Coordinate inference and metrics through narrow injected dependencies."""

    def __init__(self, inference_engine: InferenceEngine, metrics_service: MetricsService) -> None:
        self._inference_engine = inference_engine
        self._metrics_service = metrics_service

    def process_incident(self, record: IncidentNormalizedRecord) -> ModelInferenceResult:
        """Process one normalized incident and preserve its correlation identifier."""
        try:
            result = self._inference_engine.infer(record)
        except Exception as error:
            raise InferenceApplicationError("incident inference failed") from error
        return self._validate_single_correlation(record, result)

    def process_batch(
        self, records: Sequence[IncidentNormalizedRecord]
    ) -> list[ModelInferenceResult]:
        """Process a batch, rejecting duplicate IDs and restoring caller order."""
        record_list = list(records)
        self._reject_duplicate_log_ids(record_list)
        if not record_list:
            return []
        try:
            results = list(self._inference_engine.infer_batch(record_list))
        except Exception as error:
            raise InferenceApplicationError("incident batch inference failed") from error
        return self._correlate_batch(record_list, results)

    def get_asset_summary(self, asset_id: str) -> AssetRiskSummary:
        """Read analytics through the injected application-facing interface."""
        try:
            return self._metrics_service.asset_summary(asset_id)
        except Exception as error:
            raise MetricsApplicationError("asset metrics lookup failed") from error

    @staticmethod
    def _reject_duplicate_log_ids(records: Sequence[IncidentNormalizedRecord]) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for record in records:
            if record.log_id in seen and record.log_id not in duplicates:
                duplicates.append(record.log_id)
            seen.add(record.log_id)
        if duplicates:
            raise DuplicateLogIdError(
                "duplicate log_id values in batch: " + ", ".join(sorted(duplicates))
            )

    @staticmethod
    def _validate_single_correlation(
        record: IncidentNormalizedRecord, result: ModelInferenceResult
    ) -> ModelInferenceResult:
        if result.log_id != record.log_id:
            raise ResultCorrelationError("inference result log_id does not match request")
        return result

    @staticmethod
    def _correlate_batch(
        records: Sequence[IncidentNormalizedRecord],
        results: Sequence[ModelInferenceResult],
    ) -> list[ModelInferenceResult]:
        if len(records) != len(results):
            raise ResultCorrelationError("inference result count does not match request count")
        by_log_id: dict[str, ModelInferenceResult] = {}
        for result in results:
            if result.log_id in by_log_id:
                raise ResultCorrelationError("inference returned duplicate log_id values")
            by_log_id[result.log_id] = result
        expected = {record.log_id for record in records}
        if set(by_log_id) != expected:
            raise ResultCorrelationError("inference result log_id set does not match request")
        return [by_log_id[record.log_id] for record in records]
