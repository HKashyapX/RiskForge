import json

import pytest

from riskforge.supervision.benchmark import benchmark_supervision, synthetic_corpus
from riskforge.supervision.engine import LabelingFunction, labeling_function
from riskforge.supervision.pipeline import WeakSupervisionPipeline


def test_labeling_function_metadata_is_explicit_and_validated() -> None:
    @labeling_function(name="versioned", version="2.1.0")
    def versioned(record) -> int:
        return 0

    assert versioned.name == "versioned"
    assert versioned.version == "2.1.0"
    with pytest.raises(ValueError, match="version"):
        LabelingFunction("invalid", lambda _: 0, version=" ")


def test_pipeline_rows_include_reproducible_lf_versions() -> None:
    @labeling_function(name="stable", version="3.0.0")
    def stable(record) -> int:
        return 0

    batch = WeakSupervisionPipeline(lfs=[stable]).fit_transform(synthetic_corpus(2))
    rows = batch.to_rows()

    assert batch.lf_versions == ("3.0.0",)
    assert rows[0]["lf_versions"] == {"stable": "3.0.0"}
    assert not batch.votes.flags.writeable
    assert not batch.probabilities.flags.writeable


def test_corpus_benchmark_is_serialization_ready_and_deterministic() -> None:
    report = benchmark_supervision(record_count=120, iterations=2)

    assert report["record_count"] == 120
    assert report["lf_apply_latency"]["sample_count"] == 2
    assert report["dawid_skene_fit_latency"]["sample_count"] == 2
    assert [item["name"] for item in report["lf_metadata"]] == [
        "lf_well_control_breach",
        "lf_height_fall_hazard",
        "lf_dropped_object_exposure",
        "lf_h2s_confinement_breach",
        "lf_isolation_loto_breach",
        "lf_low_energy_noise",
    ]
    json.dumps(report)


def test_synthetic_corpus_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        synthetic_corpus(0)
