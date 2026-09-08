from datetime import UTC, datetime
from math import log

import numpy as np
import pytest

from riskforge.core.contracts import (
    AssetType,
    EntitySpan,
    IncidentNormalizedRecord,
    RoutingBucket,
)
from riskforge.serving.postprocessor import InferencePostprocessor


def _record(narrative: str, *, barrier: bool = True) -> IncidentNormalizedRecord:
    spans = []
    if barrier:
        start = narrative.index("BOP")
        spans.append(
            EntitySpan(
                text="BOP",
                canonical_form="blowout_preventer",
                start_char=start,
                end_char=start + 3,
                entity_type="BARRIER",
            )
        )
    return IncidentNormalizedRecord(
        log_id="OVERRIDE_001",
        timestamp=datetime.now(UTC),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative=narrative,
        spans=spans,
    )


def _process(record: IncidentNormalizedRecord, *, score: float = 0.1):
    logit = log(score / (1.0 - score))
    return InferencePostprocessor().process(
        record, np.array([[logit]]), np.zeros((1, 9)), latency_ms=0.1
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.399999, RoutingBucket.AUTO_DISMISS),
        (0.4, RoutingBucket.HITL_REVIEW),
        (0.649999, RoutingBucket.HITL_REVIEW),
        (0.65, RoutingBucket.CRITICAL_ESCALATION),
    ],
)
def test_routing_boundaries_are_exact(score, expected) -> None:
    result = InferencePostprocessor(calibrator=lambda _: score).process(
        _record("BOP inspected with no failure and no personnel present."),
        np.array([[0.0]]),
        np.zeros((1, 9)),
        latency_ms=0.1,
    )
    assert result.routing is expected
    assert not result.deterministic_override


@pytest.mark.parametrize(
    ("hazard", "expected"),
    [
        ("149.999 psi pressure", False),
        ("150 psi pressure", True),
        ("1.799 m working height", False),
        ("1.8 m working height", True),
        ("H2S 9.999 ppm", False),
        ("H2S 10 ppm", True),
    ],
)
def test_critical_energy_thresholds_are_exact(hazard, expected) -> None:
    result = _process(_record(f"BOP failed while driller was exposed to {hazard}."))
    assert result.deterministic_override is expected
    if expected:
        assert result.calibrated_sif_p_score == 0.95
        assert result.routing is RoutingBucket.CRITICAL_ESCALATION


@pytest.mark.parametrize(
    "narrative",
    [
        "BOP did not fail while driller was exposed to 150 psi pressure.",
        "No BOP leak occurred; driller later observed 150 psi pressure.",
        "BOP was never bypassed, but driller worked around 150 psi pressure.",
    ],
)
def test_explicitly_negated_failure_does_not_override(narrative) -> None:
    assert not _process(_record(narrative)).deterministic_override


def test_positive_failure_in_another_clause_survives_unrelated_negation() -> None:
    narrative = (
        "No BOP leak occurred during inspection. However, the BOP failed later. "
        "A driller was exposed to 150 psi pressure."
    )
    assert _process(_record(narrative)).deterministic_override


def test_multiple_clauses_can_supply_the_required_triad() -> None:
    narrative = "The BOP failed. Pressure reached 150 psi. A driller was exposed."
    assert _process(_record(narrative)).deterministic_override


@pytest.mark.parametrize(
    ("narrative", "barrier"),
    [
        ("BOP failed at 150 psi pressure with no worker present.", True),
        ("BOP failed while the driller observed routine low-energy work.", True),
        ("Valve failed at 150 psi pressure while a driller was exposed.", False),
    ],
)
def test_missing_override_component_prevents_escalation(narrative, barrier) -> None:
    assert not _process(_record(narrative, barrier=barrier)).deterministic_override
