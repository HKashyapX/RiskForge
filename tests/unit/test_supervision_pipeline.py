from datetime import UTC, datetime

import numpy as np
import pytest

from riskforge.core.contracts import AssetType, IncidentNormalizedRecord
from riskforge.supervision.engine import ABSTAIN, NON_SIF, SIF_P, labeling_function
from riskforge.supervision.pipeline import WeakSupervisionPipeline


def _record(log_id: str) -> IncidentNormalizedRecord:
    return IncidentNormalizedRecord(
        log_id=log_id,
        timestamp=datetime.now(UTC),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative=log_id,
        spans=[],
    )


@labeling_function(name="keyword_sif")
def _sif_lf(record: IncidentNormalizedRecord) -> int:
    return SIF_P if record.raw_narrative.startswith("sif") else NON_SIF


@labeling_function(name="selective")
def _selective_lf(record: IncidentNormalizedRecord) -> int:
    return ABSTAIN if "unknown" in record.raw_narrative else _sif_lf(record)


def test_pipeline_produces_transport_neutral_rows() -> None:
    records = [_record("sif-1"), _record("sif-2"), _record("non-sif"), _record("unknown")]
    pipeline = WeakSupervisionPipeline(lfs=[_sif_lf, _selective_lf])

    batch = pipeline.fit_transform(records)
    rows = batch.to_rows()

    assert batch.votes.shape == (4, 2)
    assert batch.probabilities.shape == (4, 2)
    assert np.allclose(batch.probabilities.sum(axis=1), 1.0)
    assert rows[0]["log_id"] == "sif-1"
    assert rows[0]["lf_votes"] == {"keyword_sif": SIF_P, "selective": SIF_P}
    assert rows[0]["lf_versions"] == {"keyword_sif": "1.0.0", "selective": "1.0.0"}
    assert rows[0]["p_sif_p"] > rows[0]["p_non_sif"]
    assert rows[-1]["lf_votes"]["selective"] == ABSTAIN


def test_batch_arrays_are_read_only() -> None:
    batch = WeakSupervisionPipeline(lfs=[_sif_lf]).fit_transform([_record("sif")])

    with pytest.raises(ValueError):
        batch.votes[0, 0] = NON_SIF
    with pytest.raises(ValueError):
        batch.probabilities[0, 0] = 0.0


def test_pipeline_rejects_duplicate_log_ids_and_empty_fit() -> None:
    pipeline = WeakSupervisionPipeline(lfs=[_sif_lf])

    with pytest.raises(ValueError, match="unique"):
        pipeline.fit_transform([_record("duplicate"), _record("duplicate")])
    with pytest.raises(ValueError, match="at least one"):
        pipeline.fit_transform([])


def test_transform_requires_a_fitted_label_model() -> None:
    pipeline = WeakSupervisionPipeline(lfs=[_sif_lf])

    with pytest.raises(RuntimeError):
        pipeline.transform([_record("sif")])
