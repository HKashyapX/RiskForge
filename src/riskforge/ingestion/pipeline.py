"""Transport-neutral ingestion pipeline: parse → validate → normalize → score.

The pipeline only orchestrates: parsing lives in ``parsers``, semantic
validation in this module, normalization behind an injected port (the runtime
composer binds the gazetteer there, keeping the boundary that ``ingestion``
must not import ``normalization`` directly), and scoring behind the standard
application inference port.  The API layer stays free of any parsing or
normalization imports.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from riskforge.core.contracts import IncidentNormalizedRecord, ModelInferenceResult
from riskforge.ingestion.exceptions import ReportValidationError
from riskforge.ingestion.parsers import dedupe_records, parse_report


@dataclass(frozen=True)
class IngestionItemResult:
    """Outcome for one ingested report."""

    log_id: str
    status: str  # "normalized" | "failed"
    normalized: IncidentNormalizedRecord | None = None
    result: ModelInferenceResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class IngestionRunResult:
    """Aggregate outcome of an ingestion request."""

    received: int
    normalized: int
    failed: int
    items: tuple[IngestionItemResult, ...]

    @property
    def ok(self) -> bool:
        return self.failed == 0


class IngestionPipeline:
    """Parse, validate, normalize, and score submitted reports."""

    def __init__(
        self,
        normalize: Callable[[object], IncidentNormalizedRecord],
        inference_engine: object,
    ) -> None:
        self._normalize = normalize
        self._engine = inference_engine

    def run(self, data: bytes, fmt: str) -> IngestionRunResult:
        records = dedupe_records(parse_report(data, fmt))
        items: list[IngestionItemResult] = []
        normalized_count = 0
        failed_count = 0
        for record in records:
            try:
                normalized = self._normalize(record)
                if hasattr(self._engine, "infer_batch"):
                    results: Sequence[ModelInferenceResult] = list(
                        self._engine.infer_batch([normalized])
                    )
                else:
                    results = [self._engine.infer(normalized)]
            except Exception as error:  # noqa: BLE001 - per-report isolation
                failed_count += 1
                items.append(
                    IngestionItemResult(
                        log_id=getattr(record, "log_id", "unknown"),
                        status="failed",
                        error=str(error)[:300],
                        result=None,
                    )
                )
                continue
            normalized_count += 1
            items.append(
                IngestionItemResult(
                    log_id=normalized.log_id,
                    status="normalized",
                    normalized=normalized,
                    result=results[0] if results else None,
                    error=None,
                )
            )
        return IngestionRunResult(
            received=len(records),
            normalized=normalized_count,
            failed=failed_count,
            items=tuple(items),
        )

    def validate_records(self, records: Sequence[object]) -> None:
        """Semantic validation hook: narratives must carry usable text."""
        errors: list[dict[str, str]] = []
        for record in records:
            narrative = getattr(record, "raw_narrative", "") or ""
            if not narrative.strip():
                errors.append(
                    {"log_id": getattr(record, "log_id", "unknown"), "error": "empty narrative"}
                )
        if errors:
            raise ReportValidationError("reports failed validation", errors)
