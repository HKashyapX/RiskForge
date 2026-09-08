"""Core tests for MetricsAggregator and compute_asset_risk_summary."""

from __future__ import annotations

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
    matched_rules: list[LifeSavingRule] | None = None,
) -> ModelInferenceResult:
    """Build a minimal ModelInferenceResult for testing."""
    return ModelInferenceResult(
        log_id=log_id,
        raw_sif_p_score=raw_score,
        calibrated_sif_p_score=calibrated_score,
        deterministic_override=deterministic_override,
        routing=routing,
        matched_iogp_rules=matched_rules or [],
        triad=OperationalTriad(failed_barrier=failed_barrier),
        latency_ms=5.0,
    )


def _make_barrier(canonical: str, text: str | None = None) -> EntitySpan:
    """Build an EntitySpan representing a failed barrier."""
    return EntitySpan(
        text=text or canonical,
        canonical_form=canonical,
        start_char=0,
        end_char=len(text or canonical),
        entity_type="BARRIER",
    )


# ---------------------------------------------------------------------------
# Single Asset Tests
# ---------------------------------------------------------------------------

class TestSingleAsset:
    def test_single_asset_zero_sif_p(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.AUTO_DISMISS),
            _make_result("L2", RoutingBucket.HITL_REVIEW),
            _make_result("L3", RoutingBucket.AUTO_DISMISS),
        ]
        mapping = {"L1": ("RIG_A", AssetType.DRILLING_RIG),
                    "L2": ("RIG_A", AssetType.DRILLING_RIG),
                    "L3": ("RIG_A", AssetType.DRILLING_RIG)}
        summaries = compute_asset_risk_summary(results, mapping)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.asset_id == "RIG_A"
        assert s.asset_type == AssetType.DRILLING_RIG
        assert s.total_logs == 3
        assert s.sif_precursor_count == 0
        assert s.spd_score == 0.0
        assert s.recurrent_failed_barriers == []

    def test_single_asset_one_sif_p(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION, calibrated_score=0.8),
            _make_result("L2", RoutingBucket.AUTO_DISMISS),
        ]
        mapping = {"L1": ("RIG_A", AssetType.DRILLING_RIG),
                    "L2": ("RIG_A", AssetType.DRILLING_RIG)}
        summaries = compute_asset_risk_summary(results, mapping)
        s = summaries[0]
        assert s.total_logs == 2
        assert s.sif_precursor_count == 1
        assert s.spd_score == pytest.approx(50.0)

    def test_single_asset_multiple_sif_p(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION),
            _make_result("L2", RoutingBucket.CRITICAL_ESCALATION),
            _make_result("L3", RoutingBucket.HITL_REVIEW),
            _make_result("L4", RoutingBucket.CRITICAL_ESCALATION),
        ]
        mapping = {r.log_id: ("RIG_A", AssetType.DRILLING_RIG) for r in results}
        summaries = compute_asset_risk_summary(results, mapping)
        s = summaries[0]
        assert s.sif_precursor_count == 3
        assert s.spd_score == pytest.approx(75.0)


class TestSPD100Percent:
    def test_all_critical_escalation(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION),
            _make_result("L2", RoutingBucket.CRITICAL_ESCALATION),
        ]
        mapping = {r.log_id: ("RIG_A", AssetType.WORKOVER_RIG) for r in results}
        summaries = compute_asset_risk_summary(results, mapping)
        s = summaries[0]
        assert s.spd_score == pytest.approx(100.0)
        assert s.sif_precursor_count == 2
        assert s.total_logs == 2


class TestFractionalSPD:
    def test_one_of_three(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION),
            _make_result("L2", RoutingBucket.HITL_REVIEW),
            _make_result("L3", RoutingBucket.AUTO_DISMISS),
        ]
        mapping = {r.log_id: ("RIG_A", AssetType.DRILLING_RIG) for r in results}
        summaries = compute_asset_risk_summary(results, mapping)
        assert summaries[0].spd_score == pytest.approx(33.333333333333336)


# ---------------------------------------------------------------------------
# Multiple Assets
# ---------------------------------------------------------------------------

class TestMultipleAssets:
    def test_two_assets_independent(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION, calibrated_score=0.8),
            _make_result("L2", RoutingBucket.AUTO_DISMISS),
            _make_result("L3", RoutingBucket.HITL_REVIEW),
            _make_result("L4", RoutingBucket.CRITICAL_ESCALATION, calibrated_score=0.7),
            _make_result("L5", RoutingBucket.CRITICAL_ESCALATION, calibrated_score=0.9),
        ]
        mapping = {
            "L1": ("RIG_A", AssetType.DRILLING_RIG),
            "L2": ("RIG_A", AssetType.DRILLING_RIG),
            "L3": ("RIG_B", AssetType.PIPELINE_NETWORK),
            "L4": ("RIG_B", AssetType.PIPELINE_NETWORK),
            "L5": ("RIG_B", AssetType.PIPELINE_NETWORK),
        }
        summaries = compute_asset_risk_summary(results, mapping)
        assert len(summaries) == 2
        # Sorted by asset_id
        assert summaries[0].asset_id == "RIG_A"
        assert summaries[0].total_logs == 2
        assert summaries[0].sif_precursor_count == 1
        assert summaries[0].spd_score == pytest.approx(50.0)
        assert summaries[1].asset_id == "RIG_B"
        assert summaries[1].total_logs == 3
        assert summaries[1].sif_precursor_count == 2
        assert summaries[1].spd_score == pytest.approx(200.0 / 3.0)

    def test_output_deterministic_order(self) -> None:
        results = [
            _make_result("L3", RoutingBucket.AUTO_DISMISS),
            _make_result("L1", RoutingBucket.AUTO_DISMISS),
            _make_result("L2", RoutingBucket.AUTO_DISMISS),
        ]
        mapping = {
            "L3": ("ZIG", AssetType.PIPELINE_NETWORK),
            "L1": ("AIG", AssetType.DRILLING_RIG),
            "L2": ("MIG", AssetType.GAS_GATHERING_STATION),
        }
        summaries = compute_asset_risk_summary(results, mapping)
        ids = [s.asset_id for s in summaries]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# All Routing Buckets
# ---------------------------------------------------------------------------

class TestAllRoutingBuckets:
    def test_all_three_routing_buckets(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION),
            _make_result("L2", RoutingBucket.HITL_REVIEW),
            _make_result("L3", RoutingBucket.AUTO_DISMISS),
        ]
        mapping = {r.log_id: ("RIG_A", AssetType.DRILLING_RIG) for r in results}
        summaries = compute_asset_risk_summary(results, mapping)
        s = summaries[0]
        assert s.total_logs == 3
        assert s.sif_precursor_count == 1  # only CRITICAL_ESCALATION
        assert s.spd_score == pytest.approx(100.0 / 3.0)


# ---------------------------------------------------------------------------
# Barrier Tests
# ---------------------------------------------------------------------------

class TestFailedBarriers:
    def test_no_failed_barriers(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.AUTO_DISMISS, failed_barrier=None),
            _make_result("L2", RoutingBucket.AUTO_DISMISS, failed_barrier=None),
        ]
        mapping = {r.log_id: ("RIG_A", AssetType.DRILLING_RIG) for r in results}
        summaries = compute_asset_risk_summary(results, mapping)
        assert summaries[0].recurrent_failed_barriers == []

    def test_single_barrier_once(self) -> None:
        barrier = _make_barrier("blowout_preventer")
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION, failed_barrier=barrier),
            _make_result("L2", RoutingBucket.AUTO_DISMISS, failed_barrier=None),
        ]
        mapping = {r.log_id: ("RIG_A", AssetType.DRILLING_RIG) for r in results}
        summaries = compute_asset_risk_summary(results, mapping)
        # Default threshold is 2, so one occurrence is NOT recurrent
        assert summaries[0].recurrent_failed_barriers == []

    def test_single_barrier_twice_recurrent(self) -> None:
        barrier = _make_barrier("blowout_preventer")
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION, failed_barrier=barrier),
            _make_result("L2", RoutingBucket.HITL_REVIEW, failed_barrier=barrier),
        ]
        mapping = {r.log_id: ("RIG_A", AssetType.DRILLING_RIG) for r in results}
        summaries = compute_asset_risk_summary(results, mapping)
        assert summaries[0].recurrent_failed_barriers == ["blowout_preventer"]

    def test_multiple_barriers(self) -> None:
        b1 = _make_barrier("blowout_preventer")
        b2 = _make_barrier("pressure_relief_valve")
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION, failed_barrier=b1),
            _make_result("L2", RoutingBucket.HITL_REVIEW, failed_barrier=b1),
            _make_result("L3", RoutingBucket.AUTO_DISMISS, failed_barrier=b2),
            _make_result("L4", RoutingBucket.HITL_REVIEW, failed_barrier=b2),
        ]
        mapping = {r.log_id: ("RIG_A", AssetType.DRILLING_RIG) for r in results}
        summaries = compute_asset_risk_summary(results, mapping)
        # Both barriers appear twice, which meets default threshold of 2
        assert sorted(summaries[0].recurrent_failed_barriers) == [
            "blowout_preventer",
            "pressure_relief_valve",
        ]

    def test_recurrence_exactly_at_threshold(self) -> None:
        barrier = _make_barrier("energy_isolation_loto")
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION, failed_barrier=barrier),
            _make_result("L2", RoutingBucket.AUTO_DISMISS, failed_barrier=barrier),
        ]
        mapping = {r.log_id: ("RIG_A", AssetType.DRILLING_RIG) for r in results}
        aggregator = MetricsAggregator(min_barrier_recurrence=2)
        summaries = aggregator.compute(results, mapping)
        assert summaries[0].recurrent_failed_barriers == ["energy_isolation_loto"]

    def test_recurrence_below_threshold(self) -> None:
        barrier = _make_barrier("well_control_line")
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION, failed_barrier=barrier),
            _make_result("L2", RoutingBucket.AUTO_DISMISS, failed_barrier=barrier),
        ]
        mapping = {r.log_id: ("RIG_A", AssetType.DRILLING_RIG) for r in results}
        aggregator = MetricsAggregator(min_barrier_recurrence=3)
        summaries = aggregator.compute(results, mapping)
        assert summaries[0].recurrent_failed_barriers == []

    def test_recurrence_above_threshold(self) -> None:
        barrier = _make_barrier("well_control_line")
        results = [
            _make_result(f"L{i}", RoutingBucket.HITL_REVIEW, failed_barrier=barrier)
            for i in range(5)
        ]
        mapping = {r.log_id: ("RIG_A", AssetType.DRILLING_RIG) for r in results}
        aggregator = MetricsAggregator(min_barrier_recurrence=3)
        summaries = aggregator.compute(results, mapping)
        assert summaries[0].recurrent_failed_barriers == ["well_control_line"]

    def test_barrier_recurrence_default_threshold(self) -> None:
        aggregator = MetricsAggregator()
        assert aggregator.min_barrier_recurrence == 2


# ---------------------------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------------------------

class TestConvenienceFunction:
    def test_compute_asset_risk_summary(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION, calibrated_score=0.8),
            _make_result("L2", RoutingBucket.AUTO_DISMISS),
        ]
        mapping = {"L1": ("RIG_A", AssetType.DRILLING_RIG),
                    "L2": ("RIG_A", AssetType.DRILLING_RIG)}
        summaries = compute_asset_risk_summary(results, mapping)
        assert len(summaries) == 1
        assert isinstance(summaries[0], AssetRiskSummary)


# ---------------------------------------------------------------------------
# Contract Compatibility
# ---------------------------------------------------------------------------

class TestContractCompatibility:
    def test_asset_risk_summary_fields(self) -> None:
        results = [
            _make_result("L1", RoutingBucket.CRITICAL_ESCALATION, calibrated_score=0.8),
        ]
        mapping = {"L1": ("RIG_A", AssetType.DRILLING_RIG)}
        summaries = compute_asset_risk_summary(results, mapping)
        s = summaries[0]
        # Verify all fields exist and have correct types
        assert isinstance(s.asset_id, str)
        assert isinstance(s.asset_type, AssetType)
        assert isinstance(s.total_logs, int)
        assert isinstance(s.sif_precursor_count, int)
        assert isinstance(s.spd_score, float)
        assert isinstance(s.recurrent_failed_barriers, list)

    def test_asset_type_in_output_matches_mapping(self) -> None:
        results = [_make_result("L1", RoutingBucket.AUTO_DISMISS)]
        mapping = {"L1": ("RIG_A", AssetType.GAS_GATHERING_STATION)}
        summaries = compute_asset_risk_summary(results, mapping)
        assert summaries[0].asset_type == AssetType.GAS_GATHERING_STATION

    def test_all_asset_types(self) -> None:
        asset_types = list(AssetType)
        for i, at in enumerate(asset_types):
            lid = f"L{i}"
            results = [_make_result(lid, RoutingBucket.AUTO_DISMISS)]
            mapping = {lid: (f"ASSET_{i}", at)}
            summaries = compute_asset_risk_summary(results, mapping)
            assert summaries[0].asset_type == at


# ---------------------------------------------------------------------------
# Public Imports
# ---------------------------------------------------------------------------

class TestPublicImports:
    def test_imports_from_metrics(self) -> None:
        from riskforge.metrics import MetricsAggregator, compute_asset_risk_summary
        assert callable(compute_asset_risk_summary)
        assert callable(MetricsAggregator)

    def test_all_exports(self) -> None:
        import riskforge.metrics
        assert hasattr(riskforge.metrics, "MetricsAggregator")
        assert hasattr(riskforge.metrics, "compute_asset_risk_summary")
