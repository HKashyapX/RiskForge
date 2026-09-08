from datetime import UTC, datetime

import pytest

from riskforge.core.contracts import AssetType, IncidentNormalizedRecord
from riskforge.supervision.engine import ABSTAIN, NON_SIF, SIF_P
from riskforge.supervision.heuristics import (
    lf_dropped_object_exposure,
    lf_h2s_confinement_breach,
    lf_height_fall_hazard,
    lf_isolation_loto_breach,
    lf_low_energy_noise,
    lf_well_control_breach,
)


def _record(text: str) -> IncidentNormalizedRecord:
    return IncidentNormalizedRecord(
        log_id="HAZARD",
        timestamp=datetime.now(UTC),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative=text,
        spans=[],
    )


@pytest.mark.parametrize(
    ("lf", "positive", "negated"),
    [
        (
            lf_well_control_breach,
            "Driller saw 3000 psi pressure when the seal ruptured.",
            "Driller saw 3000 psi pressure but no seal rupture occurred.",
        ),
        (
            lf_height_fall_hazard,
            "Worker on scaffold had harness unhooked.",
            "Worker on scaffold had no harness damage.",
        ),
        (
            lf_dropped_object_exposure,
            "Crane cable snapped above personnel underneath.",
            "Crane inspection found no cable snapped above personnel underneath.",
        ),
        (
            lf_h2s_confinement_breach,
            "H2S leak detected and SCBA failed near worker.",
            "H2S alarm tested with no SCBA failure near worker.",
        ),
        (
            lf_isolation_loto_breach,
            "LOTO was bypassed at the energized motor.",
            "LOTO was not bypassed at the energized motor.",
        ),
    ],
)
def test_hazard_positive_and_explicit_negation(lf, positive, negated) -> None:
    assert lf(_record(positive)) == SIF_P
    assert lf(_record(negated)) == NON_SIF


@pytest.mark.parametrize(
    ("lf", "text"),
    [
        (
            lf_well_control_breach,
            "No seal leak during inspection. However seal ruptured at 3000 psi near driller.",
        ),
        (
            lf_height_fall_hazard,
            "No harness damage was found, but the harness failed later on the scaffold.",
        ),
        (
            lf_dropped_object_exposure,
            "No load dropped initially; later the crane load slipped above personnel underneath.",
        ),
        (
            lf_h2s_confinement_breach,
            "No SCBA failure on test. However H2S leaked and the SCBA failed near the crew.",
        ),
        (
            lf_isolation_loto_breach,
            "LOTO was not bypassed on panel one; however LOTO was bypassed at the live motor.",
        ),
    ],
)
def test_contrast_boundary_preserves_later_positive_evidence(lf, text) -> None:
    assert lf(_record(text)) == SIF_P


def test_sentence_boundary_stops_negation_scope() -> None:
    text = "No leak was found. The seal ruptured during a 3000 psi surge near the driller."
    assert lf_well_control_breach(_record(text)) == SIF_P


def test_multiple_hazards_vote_independently_and_noise_abstains() -> None:
    text = (
        "Driller saw a seal rupture during a 3000 psi surge. "
        "A crane load slipped above personnel underneath. Housekeeping found dust in the office."
    )
    record = _record(text)
    assert lf_well_control_breach(record) == SIF_P
    assert lf_dropped_object_exposure(record) == SIF_P
    assert lf_low_energy_noise(record) == ABSTAIN


@pytest.mark.parametrize(
    "text",
    [
        "Housekeeping removed dust in the office.",
        "A small water puddle was cleaned in the canteen.",
        "First aid treated a minor bruise in the mess room.",
    ],
)
def test_low_energy_noise_is_non_sif(text) -> None:
    assert lf_low_energy_noise(_record(text)) == NON_SIF
