"""Transport-neutral orchestration for weak-label generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from riskforge.core.contracts import IncidentNormalizedRecord
from riskforge.supervision.engine import (
    NON_SIF,
    SIF_P,
    DawidSkeneLabelModel,
    LabelingFunction,
    LFApplier,
)
from riskforge.supervision.heuristics import DEFAULT_LFS


@dataclass(frozen=True)
class WeakLabelBatch:
    log_ids: tuple[str, ...]
    lf_names: tuple[str, ...]
    votes: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        sample_count = len(self.log_ids)
        if self.votes.shape != (sample_count, len(self.lf_names)):
            raise ValueError("vote matrix shape does not match batch metadata")
        if self.probabilities.shape != (sample_count, 2):
            raise ValueError("probability matrix must have shape [samples, 2]")
        if not np.all(np.isfinite(self.probabilities)):
            raise ValueError("probabilities must be finite")
        if not np.allclose(self.probabilities.sum(axis=1), 1.0):
            raise ValueError("each probability row must sum to one")
        self.votes.flags.writeable = False
        self.probabilities.flags.writeable = False

    def to_rows(self) -> list[dict[str, Any]]:
        """Return serialization-ready rows without choosing a storage format."""
        rows: list[dict[str, Any]] = []
        for index, log_id in enumerate(self.log_ids):
            rows.append(
                {
                    "log_id": log_id,
                    "lf_votes": {
                        name: int(vote)
                        for name, vote in zip(self.lf_names, self.votes[index], strict=True)
                    },
                    "p_non_sif": float(self.probabilities[index, NON_SIF]),
                    "p_sif_p": float(self.probabilities[index, SIF_P]),
                }
            )
        return rows


class WeakSupervisionPipeline:
    def __init__(
        self,
        lfs: Sequence[LabelingFunction] = DEFAULT_LFS,
        label_model: DawidSkeneLabelModel | None = None,
    ) -> None:
        self.lfs = tuple(lfs)
        self.applier = LFApplier(self.lfs)
        self.label_model = label_model or DawidSkeneLabelModel(num_classes=2)

    def fit_transform(
        self, records: Sequence[IncidentNormalizedRecord]
    ) -> WeakLabelBatch:
        self._validate_records(records, require_nonempty=True)
        votes = self.applier.apply(records)
        self.label_model.fit(votes)
        return self._batch(records, votes)

    def transform(self, records: Sequence[IncidentNormalizedRecord]) -> WeakLabelBatch:
        self._validate_records(records, require_nonempty=False)
        votes = self.applier.apply(records)
        return self._batch(records, votes)

    def _batch(
        self, records: Sequence[IncidentNormalizedRecord], votes: np.ndarray
    ) -> WeakLabelBatch:
        probabilities = self.label_model.predict_proba(votes)
        return WeakLabelBatch(
            log_ids=tuple(record.log_id for record in records),
            lf_names=tuple(lf.name for lf in self.lfs),
            votes=votes.copy(),
            probabilities=probabilities.copy(),
        )

    @staticmethod
    def _validate_records(
        records: Sequence[IncidentNormalizedRecord], *, require_nonempty: bool
    ) -> None:
        if require_nonempty and not records:
            raise ValueError("at least one normalized record is required for fitting")
        log_ids = [record.log_id for record in records]
        if len(log_ids) != len(set(log_ids)):
            raise ValueError("normalized record log_id values must be unique")
