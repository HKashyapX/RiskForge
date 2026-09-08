"""Weak-label application, diagnostics, and Dawid-Skene aggregation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

ABSTAIN = -1
NON_SIF = 0
SIF_P = 1


class LabelingFunction:
    """Named callable that emits ``ABSTAIN``, ``NON_SIF``, or ``SIF_P``."""

    def __init__(self, name: str, f: Callable[[Any], int], version: str = "1.0.0") -> None:
        if not name:
            raise ValueError("labeling-function name must not be empty")
        if not version or not version.strip():
            raise ValueError("labeling-function version must not be empty")
        if not callable(f):
            raise TypeError("labeling-function implementation must be callable")
        self.name = name
        self.version = version
        self.f = f

    def __call__(self, record: Any) -> int:
        return self.f(record)


def labeling_function(
    name: str | None = None, *, version: str = "1.0.0"
) -> Callable[[Callable[[Any], int]], LabelingFunction]:
    def decorator(function: Callable[[Any], int]) -> LabelingFunction:
        return LabelingFunction(name=name or function.__name__, f=function, version=version)

    return decorator


def _validate_vote_matrix(
    labels: np.ndarray, *, num_classes: int, require_nonempty: bool
) -> np.ndarray:
    matrix = np.asarray(labels)
    if matrix.ndim != 2:
        raise ValueError("label matrix must be two-dimensional")
    if require_nonempty and (matrix.shape[0] == 0 or matrix.shape[1] == 0):
        raise ValueError("label matrix must contain samples and labeling functions")
    if not np.issubdtype(matrix.dtype, np.integer) and (
        not np.all(np.isfinite(matrix)) or not np.all(matrix == np.floor(matrix))
    ):
        raise ValueError("label matrix must contain integer class identifiers")
    matrix = matrix.astype(np.int32, copy=False)
    invalid = (matrix != ABSTAIN) & ((matrix < 0) | (matrix >= num_classes))
    if np.any(invalid):
        raise ValueError("label matrix contains an invalid class identifier")
    return matrix


class LFApplier:
    def __init__(self, lfs: Sequence[LabelingFunction]) -> None:
        if not lfs:
            raise ValueError("at least one labeling function is required")
        self.lfs = tuple(lfs)

    def apply(self, records: Iterable[Any]) -> np.ndarray:
        rows: list[list[int]] = []
        for record_index, record in enumerate(records):
            row: list[int] = []
            for lf in self.lfs:
                vote = lf(record)
                if isinstance(vote, bool) or not isinstance(vote, (int, np.integer)):
                    raise TypeError(
                        f"labeling function {lf.name!r} returned a non-integer vote "
                        f"for record {record_index}"
                    )
                vote = int(vote)
                if vote not in {ABSTAIN, NON_SIF, SIF_P}:
                    raise ValueError(
                        f"labeling function {lf.name!r} returned invalid vote {vote} "
                        f"for record {record_index}"
                    )
                row.append(vote)
            rows.append(row)
        if not rows:
            return np.empty((0, len(self.lfs)), dtype=np.int32)
        return np.asarray(rows, dtype=np.int32)


class LFAnalysis:
    def __init__(self, L: np.ndarray, lfs: Sequence[LabelingFunction]) -> None:
        self.lfs = tuple(lfs)
        self.L = _validate_vote_matrix(L, num_classes=2, require_nonempty=False)
        if self.L.shape[1] != len(self.lfs):
            raise ValueError("label matrix must contain one column per labeling function")
        names = [lf.name for lf in self.lfs]
        if len(names) != len(set(names)):
            raise ValueError("labeling-function names must be unique")
        self.num_records, self.num_lfs = self.L.shape

    def summary(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        denominator = float(self.num_records) if self.num_records else 1.0
        active_counts = np.sum(self.L != ABSTAIN, axis=1)
        for index, lf in enumerate(self.lfs):
            column = self.L[:, index]
            active = column != ABSTAIN
            overlap = active & (active_counts > 1)
            conflict = np.zeros(self.num_records, dtype=bool)
            for row_index in np.flatnonzero(active):
                other_votes = np.delete(self.L[row_index], index)
                conflict[row_index] = np.any(
                    (other_votes != ABSTAIN) & (other_votes != column[row_index])
                )
            stats[lf.name] = {
                "coverage": round(float(np.sum(active)) / denominator, 4),
                "overlaps": round(float(np.sum(overlap)) / denominator, 4),
                "conflicts": round(float(np.sum(active & conflict)) / denominator, 4),
            }
        return stats


class DawidSkeneLabelModel:
    """Estimate latent labels and LF confusion matrices with EM.

    Abstentions are treated as missing observations. Additive smoothing prevents
    perfect or unused labeling functions from producing zero likelihoods.
    """

    def __init__(
        self,
        num_classes: int = 2,
        max_iter: int = 100,
        tol: float = 1e-5,
        smoothing: float = 1e-2,
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        if tol <= 0.0:
            raise ValueError("tol must be positive")
        if smoothing <= 0.0:
            raise ValueError("smoothing must be positive")
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.tol = tol
        self.smoothing = smoothing
        self.class_priors = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
        self.error_rates: np.ndarray | None = None
        self.n_iter_ = 0
        self.converged_ = False

    def _initial_posteriors(self, labels: np.ndarray) -> np.ndarray:
        counts = np.full(
            (labels.shape[0], self.num_classes), self.smoothing, dtype=np.float64
        )
        for label in range(self.num_classes):
            counts[:, label] += np.sum(labels == label, axis=1)
        posterior = counts / counts.sum(axis=1, keepdims=True)
        return np.asarray(posterior, dtype=np.float64)

    def _m_step(self, labels: np.ndarray, posterior: np.ndarray) -> None:
        priors = posterior.sum(axis=0) + self.smoothing
        self.class_priors = priors / priors.sum()
        rates = np.empty(
            (labels.shape[1], self.num_classes, self.num_classes), dtype=np.float64
        )
        for lf_index in range(labels.shape[1]):
            observed = labels[:, lf_index]
            active = observed != ABSTAIN
            for truth in range(self.num_classes):
                counts = np.full(self.num_classes, self.smoothing, dtype=np.float64)
                if np.any(active):
                    counts += np.bincount(
                        observed[active],
                        weights=posterior[active, truth],
                        minlength=self.num_classes,
                    )
                rates[lf_index, truth] = counts / counts.sum()
        self.error_rates = rates

    def _e_step(self, labels: np.ndarray) -> np.ndarray:
        if self.error_rates is None:
            raise RuntimeError("label model has not been fitted")
        log_probability = np.broadcast_to(
            np.log(self.class_priors), (labels.shape[0], self.num_classes)
        ).copy()
        for lf_index in range(labels.shape[1]):
            observed = labels[:, lf_index]
            active_rows = np.flatnonzero(observed != ABSTAIN)
            if active_rows.size:
                log_probability[active_rows] += np.log(
                    self.error_rates[lf_index, :, observed[active_rows]]
                )
        log_probability -= log_probability.max(axis=1, keepdims=True)
        posterior = np.exp(log_probability)
        posterior /= posterior.sum(axis=1, keepdims=True)
        return np.asarray(posterior, dtype=np.float64)

    def fit(self, L: np.ndarray) -> DawidSkeneLabelModel:
        labels = _validate_vote_matrix(
            L, num_classes=self.num_classes, require_nonempty=True
        )
        posterior = self._initial_posteriors(labels)
        self.converged_ = False
        for iteration in range(1, self.max_iter + 1):
            self._m_step(labels, posterior)
            updated = self._e_step(labels)
            self.n_iter_ = iteration
            if np.max(np.abs(updated - posterior)) < self.tol:
                posterior = updated
                self.converged_ = True
                break
            posterior = updated
        self._m_step(labels, posterior)
        return self

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        if self.error_rates is None:
            raise RuntimeError("LabelModel must be fitted before calling predict_proba.")
        labels = _validate_vote_matrix(
            L, num_classes=self.num_classes, require_nonempty=False
        )
        if labels.shape[1] != self.error_rates.shape[0]:
            raise ValueError("labeling-function count differs from the fitted matrix")
        return self._e_step(labels)

    def predict(self, L: np.ndarray) -> np.ndarray:
        return np.asarray(np.argmax(self.predict_proba(L), axis=1), dtype=np.int64)
