"""Unit tests for analytics computations and recommendation mapping."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from riskforge.analytics import build_recommendations, compute_summary
from riskforge.core.contracts import (
    AssetType,
    EntitySpan,
    IncidentNormalizedRecord,
    LifeSavingRule,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)


def _pair(
    log_id: str,
    asset_id: str,
    timestamp: datetime,
    routing: RoutingBucket,
    rules: tuple[LifeSavingRule, ...] = (),
    failed_barrier: str | None = None,
):
    incident = IncidentNormalizedRecord(
        log_id=log_id,
        timestamp=timestamp,
        asset_id=asset_id,
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative=f"Narrative for {log_id}",
        spans=[],
    )
    triad = OperationalTriad()
    if failed_barrier:
        triad = OperationalTriad(
            failed_barrier=EntitySpan(
                text=failed_barrier,
                canonical_form=failed_barrier,
                start_char=0,
                end_char=len(failed_barrier),
                entity_type="BARRIER",
            )
        )
    result = ModelInferenceResult(
        log_id=log_id,
        raw_sif_p_score=0.9 if routing == RoutingBucket.CRITICAL_ESCALATION else 0.1,
        calibrated_sif_p_score=0.9 if routing == RoutingBucket.CRITICAL_ESCALATION else 0.1,
        deterministic_override=False,
        routing=routing,
        matched_iogp_rules=list(rules),
        triad=triad,
        latency_ms=1.0,
    )
    return incident, result


class TestComputeSummary:
    def test_density_and_routing_counts(self):
        pairs = [
            _pair("A", "R1", datetime(2026, 6, 1, tzinfo=UTC), RoutingBucket.CRITICAL_ESCALATION),
            _pair("B", "R1", datetime(2026, 6, 2, tzinfo=UTC), RoutingBucket.AUTO_DISMISS),
            _pair("C", "R2", datetime(2026, 6, 3, tzinfo=UTC), RoutingBucket.HITL_REVIEW),
        ]
        summary = compute_summary(pairs).to_dict()
        assert summary["total_reports"] == 3
        assert summary["sif_precursors"] == 1
        assert summary["sif_precursor_density"] == round(1 / 3, 4)
        assert summary["routing_counts"] == {
            "critical_escalation": 1,
            "auto_dismiss": 1,
            "hitl_review": 1,
        }

    def test_empty_store_is_safe(self):
        summary = compute_summary([]).to_dict()
        assert summary["total_reports"] == 0
        assert summary["weekly_trend"] == []
        assert summary["patterns"] == []

    def test_assets_sorted_by_spd_then_id(self):
        base = datetime(2026, 6, 1, tzinfo=UTC)
        pairs = [
            _pair("A", "R1", base, RoutingBucket.CRITICAL_ESCALATION),
            _pair("B", "R1", base, RoutingBucket.AUTO_DISMISS),
            _pair("C", "R2", base, RoutingBucket.CRITICAL_ESCALATION),
            _pair("D", "R3", base, RoutingBucket.AUTO_DISMISS),
        ]
        assets = compute_summary(pairs).to_dict()["assets"]
        # R2 (SPD 1.0) ranks first; R1 and R3 (0.5, 0.0) follow, ties broken by id.
        assert [a["asset_id"] for a in assets] == ["R2", "R1", "R3"]
        assert assets[0]["spd"] == 1.0
        assert assets[1]["spd"] == 0.5
        assert assets[2]["spd"] == 0.0

    def test_weekly_trend_groups_by_monday(self):
        base = datetime(2026, 6, 8, tzinfo=UTC)  # a Monday
        pairs = [
            _pair("A", "R1", base, RoutingBucket.CRITICAL_ESCALATION),
            _pair("B", "R1", base + timedelta(days=2), RoutingBucket.CRITICAL_ESCALATION),
            _pair("C", "R1", base + timedelta(days=8), RoutingBucket.AUTO_DISMISS),
        ]
        trend = compute_summary(pairs).to_dict()["weekly_trend"]
        assert len(trend) == 2
        assert trend[0]["reports"] == 2 and trend[0]["sif_precursors"] == 2
        assert trend[1]["reports"] == 1 and trend[1]["sif_precursors"] == 0

    def test_emerging_risk_flags_statistical_spike(self):
        pairs = []
        log_index = 0
        # Four baseline weeks with exactly one precursor per week.
        for week in range(4):
            monday = datetime(2026, 5, 4, tzinfo=UTC) + timedelta(weeks=week)
            pairs.append(
                _pair(
                    f"B{log_index}",
                    "R1",
                    monday,
                    RoutingBucket.CRITICAL_ESCALATION,
                )
            )
            log_index += 1
            pairs.append(
                _pair(f"N{log_index}", "R1", monday, RoutingBucket.AUTO_DISMISS)
            )
            log_index += 1
        # Latest week: a clear spike of five precursors.
        latest = datetime(2026, 6, 1, tzinfo=UTC)
        for _ in range(5):
            pairs.append(
                _pair(f"S{log_index}", "R1", latest, RoutingBucket.CRITICAL_ESCALATION)
            )
            log_index += 1
        summary = compute_summary(pairs).to_dict()
        assert len(summary["emerging_risks"]) == 1
        flag = summary["emerging_risks"][0]
        assert flag["sif_precursors"] == 5
        assert flag["sif_precursors"] > flag["threshold"]

    def test_patterns_group_rule_combinations_across_narratives(self):
        base = datetime(2026, 6, 1, tzinfo=UTC)
        pairs = [
            _pair(
                "A", "R1", base, RoutingBucket.CRITICAL_ESCALATION,
                rules=(LifeSavingRule.WORK_AT_HEIGHT,),
            ),
            _pair(
                "B", "R2", base, RoutingBucket.CRITICAL_ESCALATION,
                rules=(LifeSavingRule.WORK_AT_HEIGHT,),
            ),
            _pair(
                "C", "R3", base, RoutingBucket.HITL_REVIEW,
                rules=(LifeSavingRule.TOXIC_GAS,),
            ),
        ]
        patterns = compute_summary(pairs).to_dict()["patterns"]
        assert patterns[0]["count"] == 2
        assert patterns[0]["rule_combination"] == ["work_at_height"]
        assert sorted(patterns[0]["assets"]) == ["R1", "R2"]

    def test_barrier_recurrence(self):
        base = datetime(2026, 6, 1, tzinfo=UTC)
        pairs = [
            _pair("A", "R1", base, RoutingBucket.CRITICAL_ESCALATION, failed_barrier="blowout_preventer"),
            _pair("B", "R1", base, RoutingBucket.CRITICAL_ESCALATION, failed_barrier="blowout_preventer"),
            _pair("C", "R2", base, RoutingBucket.HITL_REVIEW, failed_barrier="energy_isolation_loto"),
        ]
        barriers = compute_summary(pairs).to_dict()["failed_barriers"]
        assert barriers[0] == {"barrier": "blowout_preventer", "failures": 2}


class TestRecommendations:
    def test_rules_map_to_recommendations(self):
        result = ModelInferenceResult(
            log_id="X",
            raw_sif_p_score=0.9,
            calibrated_sif_p_score=0.9,
            deterministic_override=False,
            routing=RoutingBucket.CRITICAL_ESCALATION,
            matched_iogp_rules=[LifeSavingRule.WORK_AT_HEIGHT],
            triad=OperationalTriad(),
            latency_ms=1.0,
        )
        recommendations = build_recommendations(result)
        assert recommendations
        assert any("fall protection" in rec.lower() for rec in recommendations)

    def test_failed_barrier_adds_barrier_specific_action(self):
        result = ModelInferenceResult(
            log_id="X",
            raw_sif_p_score=0.9,
            calibrated_sif_p_score=0.9,
            deterministic_override=False,
            routing=RoutingBucket.CRITICAL_ESCALATION,
            matched_iogp_rules=[],
            triad=OperationalTriad(
                failed_barrier=EntitySpan(
                    text="PRV",
                    canonical_form="pressure_relief_valve",
                    start_char=0,
                    end_char=3,
                    entity_type="BARRIER",
                )
            ),
            latency_ms=1.0,
        )
        recommendations = build_recommendations(result)
        assert any("relief" in rec.lower() for rec in recommendations)

    def test_benign_result_has_no_recommendations(self):
        result = ModelInferenceResult(
            log_id="X",
            raw_sif_p_score=0.05,
            calibrated_sif_p_score=0.05,
            deterministic_override=False,
            routing=RoutingBucket.AUTO_DISMISS,
            matched_iogp_rules=[],
            triad=OperationalTriad(),
            latency_ms=1.0,
        )
        assert build_recommendations(result) == ()
