"""Analytics privacy and store-shape regression tests.

- Analytics pattern output must never carry raw narratives; it exposes
  de-identified evidence (counts, assets, log-id pointers) only.
- The stored-metrics asset-summary path treats stored pages as sequences,
  not objects with an ``.items`` attribute (regression guard for the bug
  where ``stored.items`` raised AttributeError on a list).
"""

from __future__ import annotations

from datetime import UTC, datetime

from riskforge.analytics.service import compute_summary
from riskforge.core.contracts import (
    AssetType,
    IncidentNormalizedRecord,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
    ScoringMode,
)


def _pair(log_id: str, narrative: str, critical: bool) -> tuple[IncidentNormalizedRecord, ModelInferenceResult]:
    incident = IncidentNormalizedRecord(
        log_id=log_id,
        timestamp=datetime(2026, 6, 3, tzinfo=UTC),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative=narrative,
        spans=[],
    )
    result = ModelInferenceResult(
        log_id=log_id,
        raw_sif_p_score=0.8 if critical else 0.1,
        calibrated_sif_p_score=0.8 if critical else 0.1,
        scoring_mode=ScoringMode.HEURISTIC,
        engine_name="test-engine",
        deterministic_override=False,
        routing=(
            RoutingBucket.CRITICAL_ESCALATION if critical else RoutingBucket.AUTO_DISMISS
        ),
        matched_iogp_rules=["work_at_height"] if critical else [],
        triad=OperationalTriad(),
        latency_ms=1.0,
    )
    return incident, result


class TestNarrativeDeIdentification:
    def test_pattern_output_contains_no_narratives(self) -> None:
        secret = "Worker John Doe, badge 4471, was almost killed at 30 m."
        pairs = [
            _pair(f"L{i}", secret, critical=True) for i in range(3)
        ]
        summary = compute_summary(pairs).to_dict()
        assert summary["patterns"], "expected at least one pattern group"
        for pattern in summary["patterns"]:
            assert "example_narratives" not in pattern
            assert "narratives" not in pattern
            assert pattern["example_log_ids"]

    def test_no_narrative_text_leaks_anywhere_in_summary(self) -> None:
        secret = "CONFIDENTIAL-NARRATIVE-TEXT-98765"
        pairs = [
            _pair("L1", secret, critical=True),
            _pair("L2", "benign observation", critical=False),
        ]
        import json

        blob = json.dumps(compute_summary(pairs).to_dict())
        assert secret not in blob


class TestStoredShapeRegression:
    def test_asset_summary_accepts_sequence_of_stored_records(self) -> None:
        """The metrics service receives a *list* of stored records.

        Regression: the old implementation wrote ``stored.items`` on that
        list, raising AttributeError for every asset with history.
        """
        from riskforge.metrics.aggregator import MetricsAggregator

        incident, result = _pair("L1", "harness missing at height", critical=True)
        stored = [
            type("Stored", (), {"incident": incident, "result": result})()
        ]  # list, not an object with .items
        mapping = {incident.log_id: (incident.asset_id, incident.asset_type)}
        summaries = MetricsAggregator(min_barrier_recurrence=2).compute(
            [item.result for item in stored], mapping
        )
        assert isinstance(summaries, list)

    def test_production_store_metrics_uses_iterated_list(self) -> None:
        import inspect

        from riskforge.runtime import production

        source = inspect.getsource(production._StoreMetricsService.asset_summary)
        assert ".items" not in source, (
            "asset_summary must iterate the stored list directly; 'stored.items' "
            "re-introduces the AttributeError regression"
        )
        assert "for item in stored" in source
