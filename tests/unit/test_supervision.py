from datetime import datetime, timezone
import numpy as np
import pytest

from riskforge.core.contracts import AssetType, IncidentNormalizedRecord, IncidentRawRecord
from riskforge.normalization.gazetteer import SpanPreservingGazetteer
from riskforge.supervision.engine import (
    ABSTAIN,
    NON_SIF,
    SIF_P,
    DawidSkeneLabelModel,
    LFAnalysis,
    LFApplier,
)
from riskforge.supervision.heuristics import DEFAULT_LFS, lf_low_energy_noise, lf_well_control_breach

@pytest.fixture
def normalizer():
    return SpanPreservingGazetteer()

def test_lf_well_control_breach_detection(normalizer):
    raw = IncidentRawRecord(
        log_id="S_01",
        timestamp=datetime.now(timezone.utc),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Observed 3000 psi surge. BOP seal leak while driller operating floor."
    )
    norm = normalizer.process(raw)
    assert lf_well_control_breach(norm) == SIF_P

def test_lf_well_control_breach_negation_scope(normalizer):
    raw = IncidentRawRecord(
        log_id="S_02",
        timestamp=datetime.now(timezone.utc),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Pressure test 5000 psi on BOP. Zero leak observed while driller operating floor."
    )
    norm = normalizer.process(raw)
    assert lf_well_control_breach(norm) == NON_SIF

def test_lf_low_energy_noise_detection(normalizer):
    raw = IncidentRawRecord(
        log_id="S_03",
        timestamp=datetime.now(timezone.utc),
        asset_id="GGS_01",
        asset_type=AssetType.GAS_GATHERING_STATION,
        raw_narrative="Housekeeping carried out in mess room, cleaned small water puddle."
    )
    norm = normalizer.process(raw)
    assert lf_low_energy_noise(norm) == NON_SIF

def test_dawid_skene_em_convergence():
    # Matrix of 4 samples across 3 LFs
    # Sample 0, 1: SIF-P consensus
    # Sample 2, 3: NON_SIF consensus
    L = np.array([
        [SIF_P, SIF_P, ABSTAIN],
        [SIF_P, SIF_P, SIF_P],
        [NON_SIF, ABSTAIN, NON_SIF],
        [NON_SIF, NON_SIF, NON_SIF],
    ], dtype=np.int32)

    model = DawidSkeneLabelModel(num_classes=2, max_iter=25)
    model.fit(L)
    probs = model.predict_proba(L)

    assert probs.shape == (4, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[0, SIF_P] > 0.80
    assert probs[1, SIF_P] > 0.90
    assert probs[2, NON_SIF] > 0.80
    assert probs[3, NON_SIF] > 0.90

def test_lf_analysis_summary(normalizer):
    raw1 = IncidentRawRecord(
        log_id="S_04",
        timestamp=datetime.now(timezone.utc),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="3000 psi kick. BOP leak with driller on floor."
    )
    raw2 = IncidentRawRecord(
        log_id="S_05",
        timestamp=datetime.now(timezone.utc),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Housekeeping in office yard."
    )
    records = [normalizer.process(raw1), normalizer.process(raw2)]

    applier = LFApplier(lfs=DEFAULT_LFS)
    L = applier.apply(records)
    analysis = LFAnalysis(L=L, lfs=DEFAULT_LFS)
    summary = analysis.summary()

    assert len(summary) == len(DEFAULT_LFS)
    for lf_name, stats in summary.items():
        assert "coverage" in stats
        assert "overlaps" in stats
        assert "conflicts" in stats
