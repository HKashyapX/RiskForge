import numpy as np
import pytest

from riskforge.supervision.engine import (
    ABSTAIN,
    NON_SIF,
    SIF_P,
    DawidSkeneLabelModel,
    LFAnalysis,
    LFApplier,
    labeling_function,
)


def test_fit_returns_model_and_normalizes_parameters() -> None:
    labels = np.array(
        [
            [SIF_P, SIF_P, ABSTAIN],
            [SIF_P, SIF_P, SIF_P],
            [NON_SIF, ABSTAIN, NON_SIF],
            [NON_SIF, NON_SIF, NON_SIF],
        ],
        dtype=np.int32,
    )
    model = DawidSkeneLabelModel(max_iter=50)

    assert model.fit(labels) is model
    assert model.error_rates is not None
    assert np.allclose(model.class_priors.sum(), 1.0)
    assert np.allclose(model.error_rates.sum(axis=2), 1.0)
    assert np.all(np.isfinite(model.predict_proba(labels)))
    assert 1 <= model.n_iter_ <= model.max_iter


def test_all_abstain_prediction_falls_back_to_learned_prior() -> None:
    labels = np.array(
        [[SIF_P, SIF_P], [SIF_P, ABSTAIN], [NON_SIF, NON_SIF]], dtype=np.int32
    )
    model = DawidSkeneLabelModel().fit(labels)

    probability = model.predict_proba(np.array([[ABSTAIN, ABSTAIN]], dtype=np.int32))[0]

    assert np.allclose(probability, model.class_priors)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([SIF_P, NON_SIF]),
        np.array([[SIF_P, 2]], dtype=np.int32),
        np.array([[SIF_P, 0.5]], dtype=np.float64),
        np.empty((0, 2), dtype=np.int32),
        np.empty((2, 0), dtype=np.int32),
    ],
)
def test_fit_rejects_invalid_label_matrices(labels: np.ndarray) -> None:
    with pytest.raises(ValueError):
        DawidSkeneLabelModel().fit(labels)


def test_predict_validates_fitted_state_and_lf_count() -> None:
    model = DawidSkeneLabelModel()
    with pytest.raises(RuntimeError):
        model.predict_proba(np.array([[SIF_P]], dtype=np.int32))

    model.fit(np.array([[SIF_P, SIF_P], [NON_SIF, NON_SIF]], dtype=np.int32))
    with pytest.raises(ValueError):
        model.predict_proba(np.array([[SIF_P]], dtype=np.int32))


def test_lf_applier_rejects_invalid_votes() -> None:
    @labeling_function()
    def invalid_vote(record: object) -> int:
        return 3

    with pytest.raises(ValueError, match="invalid_vote"):
        LFApplier([invalid_vote]).apply([object()])


def test_lf_applier_rejects_non_integer_votes() -> None:
    @labeling_function()
    def invalid_type(record: object) -> int:
        return "sif"  # type: ignore[return-value]

    with pytest.raises(TypeError, match="invalid_type"):
        LFApplier([invalid_type]).apply([object()])


def test_lf_applier_preserves_empty_matrix_shape() -> None:
    @labeling_function()
    def abstains(record: object) -> int:
        return ABSTAIN

    assert LFApplier([abstains]).apply([]).shape == (0, 1)


def test_lf_analysis_rejects_duplicate_names() -> None:
    @labeling_function(name="duplicate")
    def first(record: object) -> int:
        return ABSTAIN

    @labeling_function(name="duplicate")
    def second(record: object) -> int:
        return ABSTAIN

    with pytest.raises(ValueError, match="unique"):
        LFAnalysis(np.array([[ABSTAIN, ABSTAIN]], dtype=np.int32), [first, second])


def test_dawid_skene_recovers_deterministic_imbalanced_synthetic_truth() -> None:
    rng = np.random.default_rng(26140)
    truth = np.zeros(2000, dtype=np.int32)
    truth[rng.choice(len(truth), size=200, replace=False)] = SIF_P
    votes = np.full((len(truth), 7), ABSTAIN, dtype=np.int32)

    for lf_index, accuracy in enumerate((0.97, 0.93, 0.89, 0.85, 0.81)):
        active = rng.random(len(truth)) < 0.8
        correct = rng.random(len(truth)) < accuracy
        emitted = np.where(correct, truth, 1 - truth)
        votes[active, lf_index] = emitted[active]
    votes[:, 5] = ABSTAIN
    adversarial_correct = rng.random(len(truth)) < 0.15
    votes[:, 6] = np.where(adversarial_correct, truth, 1 - truth)

    model = DawidSkeneLabelModel(max_iter=200, tol=1e-8).fit(votes)
    prediction = model.predict(votes)

    assert np.mean(prediction == truth) > 0.95
    assert model.converged_
    assert np.all(np.isfinite(model.predict_proba(votes)))
    assert model.error_rates is not None
    assert np.allclose(model.error_rates.sum(axis=2), 1.0)


def test_all_abstaining_and_adversarial_lfs_remain_numerically_stable() -> None:
    votes = np.array(
        [
            [SIF_P, SIF_P, ABSTAIN, NON_SIF],
            [SIF_P, SIF_P, ABSTAIN, NON_SIF],
            [NON_SIF, NON_SIF, ABSTAIN, SIF_P],
            [NON_SIF, NON_SIF, ABSTAIN, SIF_P],
        ],
        dtype=np.int32,
    )
    model = DawidSkeneLabelModel(max_iter=100).fit(votes)
    probabilities = model.predict_proba(votes)

    assert np.array_equal(model.predict(votes), np.array([SIF_P, SIF_P, NON_SIF, NON_SIF]))
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
