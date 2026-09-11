"""Decision-table tests for the deterministic heuristic engine."""

from __future__ import annotations

from datetime import UTC, datetime

from riskforge.core.contracts import (
    AssetType,
    EntitySpan,
    IncidentNormalizedRecord,
    RoutingBucket,
)
from riskforge.serving.heuristic_engine import HeuristicRuleEngine


def _record(narrative: str, spans: list[EntitySpan] | None = None) -> IncidentNormalizedRecord:
    return IncidentNormalizedRecord(
        log_id="LOG_T1",
        timestamp=datetime(2026, 3, 1, tzinfo=UTC),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative=narrative,
        spans=spans or [],
    )


class TestScoring:
    def test_benign_narrative_scores_low_and_auto_dismiss(self):
        result = HeuristicRuleEngine().infer(_record("Housekeeping completed near the gate."))
        assert result.routing == RoutingBucket.AUTO_DISMISS
        assert result.raw_sif_p_score <= 0.40
        assert result.matched_iogp_rules == []
        assert result.deterministic_override is False

    def test_work_at_height_with_missing_protection_escalates(self):
        narrative = "Technician working at height on a scaffold; harness missing at 12 m."
        result = HeuristicRuleEngine().infer(_record(narrative))
        assert result.routing in {RoutingBucket.HITL_REVIEW, RoutingBucket.CRITICAL_ESCALATION}
        assert "work_at_height" in result.matched_iogp_rules

    def test_worker_exposure_raises_score(self):
        with_worker = HeuristicRuleEngine().infer(_record("Crane lifting near worker."))
        without_worker = HeuristicRuleEngine().infer(_record("Crane lifting near the office."))
        assert with_worker.raw_sif_p_score > without_worker.raw_sif_p_score

    def test_negated_worker_does_not_raise_score(self):
        plain = HeuristicRuleEngine().infer(_record("Toxic gas release h2s 20 ppm near a worker."))
        negated = HeuristicRuleEngine().infer(_record("Toxic gas release h2s 20 ppm with no worker nearby."))
        assert negated.raw_sif_p_score < plain.raw_sif_p_score

    def test_sub_threshold_rule_with_energy_and_failure_is_forced(self):
        narrative = (
            "Vehicle reversing during a site journey; the tanker was still "
            "pressurized at 200 psi while the hose was connected."
        )
        result = HeuristicRuleEngine().infer(_record(narrative))
        assert result.deterministic_override is True
        assert result.routing == RoutingBucket.CRITICAL_ESCALATION
        assert result.raw_sif_p_score >= 0.65

    def test_pure_high_energy_failure_without_rule_vocabulary_escalates(self):
        narrative = "The 250 psi line was still pressurized when the plug was removed."
        result = HeuristicRuleEngine().infer(_record(narrative))
        assert result.deterministic_override is True
        assert result.routing == RoutingBucket.CRITICAL_ESCALATION
        assert result.matched_iogp_rules == []

    def test_h2s_above_threshold_is_critical(self):
        result = HeuristicRuleEngine().infer(_record("H2s 25 ppm leak while technician present."))
        assert "toxic_gas" in result.matched_iogp_rules
        assert result.routing == RoutingBucket.CRITICAL_ESCALATION

    def test_height_below_threshold_does_not_force(self):
        result = HeuristicRuleEngine().infer(_record("Step ladder at 1 m; worker changed a bulb."))
        assert result.deterministic_override is False

    def test_every_lifesaving_rule_is_matchable(self):
        expected = {
            "energy_isolation": "Loto isolation was not applied before opening the line.",
            "confined_space": "Technician entered a confined space tank without a permit.",
            "work_at_height": "Worker fell from an unprotected edge on a scaffold.",
            "toxic_gas": "H2s detector alarmed during a gas release.",
            "line_of_fire": "Worker caught between suspended load and wall; dropped object.",
            "bypassing_safety_controls": "Operator bypassed the interlock and disabled the alarm.",
            "safe_mechanical_lifting": "Crane lifting with a damaged sling; rigging frayed.",
            "hot_work": "Hot work welding sparks near a flammable storage area.",
            "driving": "Vehicle reversing without a spotter during a site journey.",
        }
        engine = HeuristicRuleEngine()
        for rule, narrative in expected.items():
            result = engine.infer(_record(narrative))
            assert rule in result.matched_iogp_rules, (rule, narrative, result.matched_iogp_rules)

    def test_repeated_scoring_is_deterministic_except_latency(self):
        engine = HeuristicRuleEngine()
        first = engine.infer(_record("Confined space entry without gas test."))
        second = engine.infer(_record("Confined space entry without gas test."))
        assert first.model_copy(update={"latency_ms": 0.0}) == second.model_copy(
            update={"latency_ms": 0.0}
        )


class TestTriad:
    def test_failed_barrier_detected_from_failure_window(self):
        narrative = "Pressure relief valve leaking while the crew operated the swabbing unit on the monkey board."
        spans = [
            EntitySpan(
                text="swabbing unit",
                canonical_form="swab_unit",
                start_char=narrative.index("swabbing unit"),
                end_char=narrative.index("swabbing unit") + len("swabbing unit"),
                entity_type="ACTIVITY",
            ),
            EntitySpan(
                text="pressure relief valve",
                canonical_form="pressure_relief_valve",
                start_char=narrative.index("Pressure relief valve"),
                end_char=narrative.index("Pressure relief valve") + len("Pressure relief valve"),
                entity_type="BARRIER",
            ),
            EntitySpan(
                text="monkey board",
                canonical_form="monkey_board",
                start_char=narrative.index("monkey board"),
                end_char=narrative.index("monkey board") + len("monkey board"),
                entity_type="LOCATION",
            ),
        ]
        triad = HeuristicRuleEngine().infer(_record(narrative, spans)).triad
        assert triad.activity is not None and triad.activity.canonical_form == "swab_unit"
        assert triad.asset_location is not None and triad.asset_location.canonical_form == "monkey_board"
        assert triad.failed_barrier is not None
        assert triad.failed_barrier.canonical_form == "pressure_relief_valve"

    def test_intact_barrier_is_not_marked_failed(self):
        narrative = "Pressure relief valve functioned normally during swabbing."
        spans = [
            EntitySpan(
                text="Pressure relief valve",
                canonical_form="pressure_relief_valve",
                start_char=0,
                end_char=21,
                entity_type="BARRIER",
            ),
        ]
        triad = HeuristicRuleEngine().infer(_record(narrative, spans)).triad
        assert triad.failed_barrier is None


class TestBatch:
    def test_batch_preserves_order(self):
        engine = HeuristicRuleEngine()
        records = [
            _record("H2s 20 ppm leak with technician exposed."),
            _record("Housekeeping completed."),
            _record("Worker on an unprotected edge at height; harness missing."),
        ]
        results = list(engine.infer_batch(records))
        assert [r.log_id for r in results] == [r.log_id for r in records]
        assert results[0].routing == RoutingBucket.CRITICAL_ESCALATION
        assert results[1].routing == RoutingBucket.AUTO_DISMISS
        assert results[2].routing in {RoutingBucket.HITL_REVIEW, RoutingBucket.CRITICAL_ESCALATION}

    def test_engine_reports_honest_mode(self):
        engine = HeuristicRuleEngine()
        assert engine.mode == "deterministic-heuristics"
        assert engine.engine_name == "heuristic-rules-v1"
