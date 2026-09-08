"""Edge case and error handling tests for MetricsAggregator."""

from __future__ import annotations

import copy
import pytest
from riskforge.core.contracts import (
    AssetRiskSummary,
    AssetType,
    EntitySpan,
    LifeSavingRule,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)
from riskforge.metrics.aggregator import MetricsAggregator, compute_asset_risk_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    log_id: str,
    routing: RoutingBucket = RoutingBucket.AUTO_DISMISS,
    *,
    calibrated_score: float = 0.3,
    raw_score: float = 0.3,
    failed_barrier: EntitySpan | None = None,
    deterministic_override: bool = False,
) -> ModelInferenceResult:
    return ModelInferenceResult(
        log_id=log_id,
        raw_sif_p_score=raw_score,
        calibrated_sif_p_score=calibrated_score,
        deterministic_override=deterministic_override,
        routing=routing,
        matched_iogp_rules=[],
        triad=OperationalTriad(failed_barrier=failed_barrier),
        latency_ms=5.0,
    )


def _make_barrier(canonical: str, text: str | None = None) -> EntitySpan:
    """Build an EntitySpan representing a failed barrier."""
    display = text or canonical
    return EntitySpan(
        text=display,
        canonical_form=canonical,
        start_char=0,
        end_char=len(display),
        entity_type="BARRIER",
    )


# ---------------------------------------------------------------------------
# Empty Input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_results_returns_empty_list(self) -> None:
        mapping = {"L1": ("RIG_A", AssetType.DRILLING_RIG)}
        summaries = compute_asset_risk_summary([], mapping)
        assert summaries == []

    def test_empty_results_and_mapping_returns_empty(self) -> None:
        summaries = compute_asset_risk_summary([], {})
        assert summaries == []

    def test_empty_results_with_mapping_returns_empty(self) -> None:
        mapping = {"L1": ("RIG_A", AssetType.DRILLING_RIG)}
        summaries = compute_asset_risk_summary([], mapping)
        assert summaries == []


# ---------------------------------------------------------------------------
# Missing Mapping Errors
# ---------------------------------------------------------------------------

class TestMissingMapping:
    def test_one_missing_mapping_raises_value_error(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.AUTO_DISMISS),
            _make_result("L2", RoutingBucket.AUTO_DISMISS),
        ]
        mapping = {
            "L1": ("RIG_A", AssetType.DRILLING_RIG),
            # L2 missing
        }
        with pytest.raises(ValueError, match="Missing asset mapping"):
            compute_asset_risk_summary(results, mapping)

    def test_multiple_missing_mappings_raises_value_error(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.AUTO_DISMISS),
            _make_result("L2", RoutingBucket.AUTO_DISMISS),
            _make_result("L3", RoutingBucket.AUTO_DISMISS),
        ]
        mapping = {
            "L1": ("RIG_A", AssetType.DRILLING_RIG),
            # L2 and L3 missing
        }
        with pytest.raises(ValueError, match="Missing asset mapping"):
            compute_asset_risk_summary(results, mapping)

    def test_error_lists_missing_log_ids(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.AUTO_DISMISS),
            _make_result("MISSING_1", RoutingBucket.AUTO_DISMISS),
            _make_result("MISSING_2", RoutingBucket.AUTO_DISMISS),
        ]
        mapping = {"L1": ("RIG_A", AssetType.DRILLING_RIG)}
        with pytest.raises(ValueError, match="MISSING_1") as exc_info:
            compute_asset_risk_summary(results, mapping)
        error_msg = str(exc_info.value)
        assert "MISSING_1" in error_msg
        assert "MISSING_2" in error_msg

    def test_error_message_is_sorted(self) -> None:
        results = [
            _make_result("Z", RoutingBucket.AUTO_DISMISS),
            _make_result("A", RoutingBucket.AUTO_DISMISS),
        ]
        mapping = {"A": ("RIG_A", AssetType.DRILLING_RIG)}
        with pytest.raises(ValueError, match="Missing asset mapping") as exc_info:
            compute_asset_risk_summary(results, mapping)
        # Z is after A in the sorted list inside the error
        assert "['Z']" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Extra Mapping Entries (no corresponding results)
# ---------------------------------------------------------------------------

class TestExtraMappingEntries:
    def test_extra_mapping_entries_ignored(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.AUTO_DISMISS),
        ]
        mapping = {
            "L1": ("RIG_A", AssetType.DRILLING_RIG),
            "L_EXTRA": ("RIG_B", AssetType.WORKOVER_RIG),  # No results for this
        }
        summaries = compute_asset_risk_summary(results, mapping)
        assert len(summaries) == 1
        assert summaries[0].asset_id == "RIG_A"


# ---------------------------------------------------------------------------
# Duplicate Log IDs
# ---------------------------------------------------------------------------

class TestDuplicateLogIds:
    def test_duplicate_results_same_log_id(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION, calibrated_score=0.8),
            _make_result("L1", RoutingBucket.AUTO_DISMISS),  # duplicate log_id
        ]
        mapping = {"L1": ("RIG_A", AssetType.DRILLING_RIG)}
        summaries = compute_asset_risk_summary(results, mapping)
        s = summaries[0]
        # Both records are counted (the contract doesn't forbid duplicates)
        assert s.total_logs == 2
        assert s.sif_precursor_count == 1
        assert s.spd_score == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Immutability Checks
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_model_inference_result_not_mutated(self) -> None:
        original = _make_result("L1", RoutingBucket.CRITICAL_ESCALATION, calibrated_score=0.8)
        original_copy = original.model_copy(deep=True)
        results = [original]
        mapping = {"L1": ("RIG_A", AssetType.DRILLING_RIG)}
        compute_asset_risk_summary(results, mapping)
        # Original should be unchanged
        assert original.log_id == original_copy.log_id
        assert original.routing == original_copy.routing
        assert original.calibrated_sif_p_score == original_copy.calibrated_sif_p_score
        assert original.triad.failed_barrier == original_copy.triad.failed_barrier

    def test_asset_mapping_not_mutated(self) -> None:
        results = [_make_result("L1", RoutingBucket.AUTO_DISMISS)]
        mapping = {"L1": ("RIG_A", AssetType.DRILLING_RIG)}
        mapping_copy = copy.deepcopy(mapping)
        compute_asset_risk_summary(results, mapping)
        assert mapping == mapping_copy

    def test_input_list_not_mutated(self) -> None:
        results = [_make_result("L1", RoutingBucket.AUTO_DISMISS)]
        results_copy = [r.model_copy(deep=True) for r in results]
        mapping = {"L1": ("RIG_A", AssetType.DRILLING_RIG)}
        compute_asset_risk_summary(results, mapping)
        assert len(results) == len(results_copy)
        for r, rc in zip(results, results_copy):
            assert r.log_id == rc.log_id
            assert r.routing == rc.routing


# ---------------------------------------------------------------------------
# Constructor Validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_zero_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="min_barrier_recurrence"):
            MetricsAggregator(min_barrier_recurrence=0)

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="min_barrier_recurrence"):
            MetricsAggregator(min_barrier_recurrence=-1)

    def test_non_integer_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="min_barrier_recurrence"):
            MetricsAggregator(min_barrier_recurrence=1.5)  # type: ignore[arg-type]

    def test_string_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="min_barrier_recurrence"):
            MetricsAggregator(min_barrier_recurrence="two")  # type: ignore[arg-type]

    def test_threshold_one_is_valid(self) -> None:
        aggregator = MetricsAggregator(min_barrier_recurrence=1)
        assert aggregator.min_barrier_recurrence == 1


# ---------------------------------------------------------------------------
# SPD Edge Cases
# ---------------------------------------------------------------------------

class TestSPDEdgeCases:
    def test_single_log_zero_sif_p(self) -> None:
        results = [_make_result("L1", RoutingBucket.AUTO_DISMISS)]
        mapping = {"L1": ("RIG_A", AssetType.DRILLING_RIG)}
        summaries = compute_asset_risk_summary(results, mapping)
        assert summaries[0].spd_score == 0.0

    def test_single_log_one_sif_p(self) -> None:
        results = [_make_result("L1", RoutingBucket.CRITICAL_ESCALATION, calibrated_score=0.8)]
        mapping = {"L1": ("RIG_A", AssetType.DRILLING_RIG)}
        summaries = compute_asset_risk_summary(results, mapping)
        assert summaries[0].spd_score == pytest.approx(100.0)

    def test_large_batch(self) -> None:
        results = [
            _make_result(f"L{i}", RoutingBucket.CRITICAL_ESCALATION if i % 2 == 0 else RoutingBucket.AUTO_DISMISS)
            for i in range(100)
        ]
        mapping = {f"L{i}": ("RIG_A", AssetType.DRILLING_RIG) for i in range(100)}
        summaries = compute_asset_risk_summary(results, mapping)
        s = summaries[0]
        assert s.total_logs == 100
        assert s.sif_precursor_count == 50
        assert s.spd_score == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Barrier Extraction Edge Cases
# ---------------------------------------------------------------------------

class TestBarrierEdgeCases:
    def test_barrier_across_assets(self) -> None:
        b = _make_barrier("blowout_preventer")
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION, failed_barrier=b),
            _make_result("L2", RoutingBucket.HITL_REVIEW, failed_barrier=b),
            _make_result("L3", RoutingBucket.AUTO_DISMISS, failed_barrier=b),
            _make_result("L4", RoutingBucket.HITL_REVIEW, failed_barrier=b),
        ]
        mapping = {
            "L1": ("RIG_A", AssetType.DRILLING_RIG),
            "L2": ("RIG_A", AssetType.DRILLING_RIG),
            "L3": ("RIG_B", AssetType.PIPELINE_NETWORK),
            "L4": ("RIG_B", AssetType.PIPELINE_NETWORK),
        }
        summaries = compute_asset_risk_summary(results, mapping)
        # Both assets have barrier appearing 2 times
        for s in summaries:
            assert s.recurrent_failed_barriers == ["blowout_preventer"]

    def test_mixed_barriers_and_none(self) -> None:
        b1 = _make_barrier("blowout_preventer")
        b2 = _make_barrier("pressure_relief_valve")
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION, failed_barrier=b1),
            _make_result("L2", RoutingBucket.HITL_REVIEW, failed_barrier=None),
            _make_result("L3", RoutingBucket.AUTO_DISMISS, failed_barrier=b2),
            _make_result("L4", RoutingBucket.HITL_REVIEW, failed_barrier=b1),
            _make_result("L5", RoutingBucket.AUTO_DISMISS, failed_barrier=b2),
        ]
        mapping = {r.log_id: ("RIG_A", AssetType.DRILLING_RIG) for r in results}
        summaries = compute_asset_risk_summary(results, mapping)
        assert sorted(summaries[0].recurrent_failed_barriers) == [
            "blowout_preventer",
            "pressure_relief_valve",
        ]

    def test_different_barrier_canonical_names(self) -> None:
        b1 = _make_barrier("barrier_A", text="First barrier")
        b2 = _make_barrier("barrier_B", text="Second barrier")
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION, failed_barrier=b1),
            _make_result("L2", RoutingBucket.HITL_REVIEW, failed_barrier=b1),
            _make_result("L3", RoutingBucket.AUTO_DISMISS, failed_barrier=b2),
        ]
        mapping = {r.log_id: ("RIG_A", AssetType.DRILLING_RIG) for r in results}
        summaries = compute_asset_risk_summary(results, mapping)
        # barrier_A appears 2 times (recurrent), barrier_B appears 1 time (not recurrent)
        assert summaries[0].recurrent_failed_barriers == ["barrier_A"]


# ---------------------------------------------------------------------------
# Configurable Threshold Behavior
# ---------------------------------------------------------------------------

class TestConfigurableThreshold:
    def test_threshold_1_makes_every_single_barrier_recurrent(self) -> None:
        b = _make_barrier("blowout_preventer")
        results = [_make_result("L1", RoutingBucket.CRITICAL_ESCALATION, failed_barrier=b)]
        mapping = {"L1": ("RIG_A", AssetType.DRILLING_RIG)}
        aggregator = MetricsAggregator(min_barrier_recurrence=1)
        summaries = aggregator.compute(results, mapping)
        assert summaries[0].recurrent_failed_barriers == ["blowout_preventer"]

    def test_threshold_5_requires_five_occurrences(self) -> None:
        b = _make_barrier("blowout_preventer")
        results = [
            _make_result(f"L{i}", RoutingBucket.HITL_REVIEW, failed_barrier=b)
            for i in range(4)
        ]
        mapping = {f"L{i}": ("RIG_A", AssetType.DRILLING_RIG) for i in range(4)}
        aggregator = MetricsAggregator(min_barrier_recurrence=5)
        summaries = aggregator.compute(results, mapping)
        assert summaries[0].recurrent_failed_barriers == []
