"""Narrow application-facing dependencies for orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from riskforge.core.contracts import AssetRiskSummary, IncidentNormalizedRecord, ModelInferenceResult


@runtime_checkable
class InferenceEngine(Protocol):
    """Application-facing model inference capability."""

    def infer(self, record: IncidentNormalizedRecord) -> ModelInferenceResult:
        """Process one normalized incident."""

    def infer_batch(
        self, records: Sequence[IncidentNormalizedRecord]
    ) -> Sequence[ModelInferenceResult]:
        """Process normalized incidents in the supplied order."""


@runtime_checkable
class MetricsService(Protocol):
    """Application-facing read interface for risk analytics."""

    def asset_summary(self, asset_id: str) -> AssetRiskSummary:
        """Return the current summary for one asset."""
