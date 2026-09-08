"""Aggregate ModelInferenceResult records into per-asset AssetRiskSummary objects."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from riskforge.core.contracts import (
    AssetRiskSummary,
    AssetType,
    ModelInferenceResult,
    RoutingBucket,
)


def _is_sif_precursor(result: ModelInferenceResult) -> bool:
    """Determine whether an inference result represents a SIF Precursor.

    Uses the authoritative upstream classification: RoutingBucket.CRITICAL_ESCALATION.
    This is a read-only consumption of the serving layer's classification.
    """
    return result.routing == RoutingBucket.CRITICAL_ESCALATION


def _extract_failed_barrier_canonical(result: ModelInferenceResult) -> str | None:
    """Extract the canonical barrier name from the triad's failed_barrier, if present."""
    if result.triad.failed_barrier is not None:
        return result.triad.failed_barrier.canonical_form
    return None


class MetricsAggregator:
    """Aggregates ModelInferenceResult records into per-asset AssetRiskSummary objects.

    This is a read-only aggregation layer. It does not modify input records,
    perform ML inference, or create separate SIF-P classifications.

    Args:
        min_barrier_recurrence: Minimum occurrence count for a barrier to be
            classified as recurrent. Must be >= 1. Default is 2.
    """

    def __init__(self, min_barrier_recurrence: int = 2) -> None:
        if not isinstance(min_barrier_recurrence, int) or min_barrier_recurrence < 1:
            raise ValueError(
                f"min_barrier_recurrence must be a positive integer, "
                f"got {min_barrier_recurrence!r}"
            )
        self.min_barrier_recurrence = min_barrier_recurrence

    def compute(
        self,
        results: List[ModelInferenceResult],
        asset_mapping: Dict[str, Tuple[str, AssetType]],
    ) -> List[AssetRiskSummary]:
        """Aggregate inference results into per-asset risk summaries.

        Args:
            results: List of ModelInferenceResult records from the serving layer.
            asset_mapping: Explicit mapping from log_id to (asset_id, asset_type).
                The caller/orchestration layer owns this mapping.  Keyed by log_id;
                therefore a single log_id cannot map to multiple assets.

        Returns:
            List of AssetRiskSummary, one per unique asset_id present in the mapping
            for the given results. Sorted by asset_id for deterministic output.

        Raises:
            ValueError: If any result log_id is missing from asset_mapping.

        Duplicate-log_id semantics:
            ``ModelInferenceResult`` does not enforce log_id uniqueness.
            Duplicate log_ids in *results* are treated as separate records
            and all contribute to aggregation (total_logs, SIF-P count,
            barrier counts).  Each duplicate resolves through the same
            ``asset_mapping`` entry because the mapping is keyed by log_id.
            Callers must supply unique log_ids when different asset
            associations are intended.
        """
        # --- Validate: every result must have a mapping entry ---
        missing_ids = sorted(
            r.log_id for r in results if r.log_id not in asset_mapping
        )
        if missing_ids:
            raise ValueError(
                f"Missing asset mapping for log_ids: {missing_ids}"
            )

        # --- Group results by asset_id ---
        # Structure: asset_id -> (asset_type, list of results)
        asset_groups: Dict[str, Tuple[AssetType, List[ModelInferenceResult]]] = {}
        for result in results:
            asset_id, asset_type = asset_mapping[result.log_id]
            if asset_id not in asset_groups:
                asset_groups[asset_id] = (asset_type, [])
            asset_groups[asset_id][1].append(result)

        # --- Build summaries (sorted by asset_id for determinism) ---
        summaries: List[AssetRiskSummary] = []
        for asset_id in sorted(asset_groups):
            asset_type, group_results = asset_groups[asset_id]
            total_logs = len(group_results)

            # SIF-Precursor count: read-only check against upstream routing
            sif_precursor_count = sum(
                1 for r in group_results if _is_sif_precursor(r)
            )

            # SPD = (SIF-P count / Total Logs) * 100
            spd_score = (sif_precursor_count / total_logs) * 100 if total_logs > 0 else 0.0

            # Barrier degradation: count canonical barrier names
            barrier_counter: Counter[str] = Counter()
            for r in group_results:
                canonical = _extract_failed_barrier_canonical(r)
                if canonical is not None:
                    barrier_counter[canonical] += 1

            # Recurrent barriers: those meeting the threshold
            recurrent_failed_barriers = sorted(
                name
                for name, count in barrier_counter.items()
                if count >= self.min_barrier_recurrence
            )

            summaries.append(
                AssetRiskSummary(
                    asset_id=asset_id,
                    asset_type=asset_type,
                    total_logs=total_logs,
                    sif_precursor_count=sif_precursor_count,
                    spd_score=spd_score,
                    recurrent_failed_barriers=recurrent_failed_barriers,
                )
            )

        return summaries


def compute_asset_risk_summary(
    results: List[ModelInferenceResult],
    asset_mapping: Dict[str, Tuple[str, AssetType]],
    min_barrier_recurrence: int = 2,
) -> List[AssetRiskSummary]:
    """Compute per-asset risk summaries from inference results.

    Convenience wrapper around MetricsAggregator for stateless usage.

    Args:
        results: List of ModelInferenceResult records from the serving layer.
        asset_mapping: Explicit mapping from log_id to (asset_id, asset_type).
        min_barrier_recurrence: Minimum occurrences for a barrier to be recurrent.

    Returns:
        List of AssetRiskSummary, one per unique asset_id.
    """
    return MetricsAggregator(min_barrier_recurrence).compute(results, asset_mapping)
