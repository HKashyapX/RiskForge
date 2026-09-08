"""Convert model outputs into stable, validated serving contracts."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

from riskforge.core.contracts import (
    EntitySpan,
    IncidentNormalizedRecord,
    LifeSavingRule,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)

_FAILURE_RE = re.compile(
    r"\b(?:bypass(?:ed)?|defeat(?:ed)?|disable[ds]?|fail(?:ed|ure)?|"
    r"leak(?:ed|ing)?|breach(?:ed)?|missing|removed|not\s+isolated)\b",
    re.IGNORECASE,
)
_NEGATED_FAILURE_RE = re.compile(
    r"\b(?:no|not|never|without|zero|nil)\b(?:\W+\w+){0,3}\W+"
    r"(?:leak|failure|breach|bypass)\b",
    re.IGNORECASE,
)
_HIGH_ENERGY_RE = re.compile(
    r"\b(?:high[- ]pressure|energized|live\s+(?:line|wire)|toxic\s+gas|"
    r"suspended\s+load|work(?:ing)?\s+at\s+height)\b",
    re.IGNORECASE,
)
_PRESSURE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*psi\b", re.IGNORECASE)
_HEIGHT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:m|metre|meter)s?\b", re.IGNORECASE)
_H2S_RE = re.compile(r"\b(?:h2s|hydrogen\s+sulfide)\D{0,12}(\d+(?:\.\d+)?)\s*ppm\b", re.IGNORECASE)
_WORKER_RE = re.compile(
    r"\b(?:worker|personnel|person|crew|driller|derrickman|operator|roustabout|"
    r"roughneck|technician|contractor|employee)\b",
    re.IGNORECASE,
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _sif_probabilities(output: np.ndarray) -> np.ndarray:
    logits = np.asarray(output, dtype=np.float64)
    if logits.ndim == 0:
        logits = logits.reshape(1)
    if logits.ndim == 2 and logits.shape[1] == 2:
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities[:, 1] / probabilities.sum(axis=1)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 1:
        raise ValueError("SIF output must have shape [batch], [batch, 1], or [batch, 2]")
    return _sigmoid(logits)


def _first_span(record: IncidentNormalizedRecord, types: set[str]) -> EntitySpan | None:
    return next((span for span in record.spans if span.entity_type.upper() in types), None)


def _triad(record: IncidentNormalizedRecord) -> OperationalTriad:
    return OperationalTriad(
        activity=_first_span(record, {"ACTIVITY"}),
        asset_location=_first_span(record, {"ASSET", "LOCATION", "ASSET_LOCATION"}),
        failed_barrier=_first_span(record, {"BARRIER", "BARRIER_STATE", "FAILED_BARRIER"}),
    )


def _barrier_override(record: IncidentNormalizedRecord, triad: OperationalTriad) -> bool:
    if triad.failed_barrier is None:
        return False
    narrative = record.raw_narrative
    failure = _FAILURE_RE.search(narrative)
    if failure is None or _NEGATED_FAILURE_RE.search(narrative):
        return False
    pressure = any(float(match.group(1)) >= 150.0 for match in _PRESSURE_RE.finditer(narrative))
    height = any(float(match.group(1)) >= 1.8 for match in _HEIGHT_RE.finditer(narrative))
    h2s = any(float(match.group(1)) >= 10.0 for match in _H2S_RE.finditer(narrative))
    high_energy = bool(_HIGH_ENERGY_RE.search(narrative) or pressure or height or h2s)
    return high_energy and _WORKER_RE.search(narrative) is not None


class InferencePostprocessor:
    def __init__(
        self,
        *,
        rule_threshold: float = 0.5,
        auto_dismiss_threshold: float = 0.4,
        critical_threshold: float = 0.65,
        calibrator: Callable[[float], float] | None = None,
    ) -> None:
        if not 0.0 <= rule_threshold <= 1.0:
            raise ValueError("rule_threshold must be between zero and one")
        if not 0.0 <= auto_dismiss_threshold < critical_threshold <= 1.0:
            raise ValueError("routing thresholds must satisfy 0 <= dismiss < critical <= 1")
        self.rule_threshold = rule_threshold
        self.auto_dismiss_threshold = auto_dismiss_threshold
        self.critical_threshold = critical_threshold
        self.calibrator = calibrator
        self.rules = tuple(LifeSavingRule)

    def process_batch(
        self,
        records: Sequence[IncidentNormalizedRecord],
        sif_logits: np.ndarray,
        iogp_logits: np.ndarray,
        *,
        latency_ms: float | Sequence[float],
    ) -> list[ModelInferenceResult]:
        raw_scores = _sif_probabilities(sif_logits)
        rule_logits = np.asarray(iogp_logits, dtype=np.float64)
        if rule_logits.ndim == 1:
            rule_logits = rule_logits.reshape(1, -1)
        if rule_logits.ndim != 2 or rule_logits.shape[1] != len(self.rules):
            raise ValueError(f"IOGP output must have shape [batch, {len(self.rules)}]")
        if len(records) != len(raw_scores) or len(records) != rule_logits.shape[0]:
            raise ValueError("record and output batch sizes must match")
        rule_scores = _sigmoid(rule_logits)
        if np.isscalar(latency_ms):
            latencies = np.full(len(records), float(latency_ms), dtype=np.float64)
        else:
            latencies = np.asarray(latency_ms, dtype=np.float64)
            if latencies.shape != (len(records),):
                raise ValueError("latency_ms must be scalar or have one value per record")

        results: list[ModelInferenceResult] = []
        for index, record in enumerate(records):
            raw_score = float(np.clip(raw_scores[index], 0.0, 1.0))
            calibrated = raw_score if self.calibrator is None else float(self.calibrator(raw_score))
            calibrated = float(np.clip(calibrated, 0.0, 1.0))
            triad = _triad(record)
            override = _barrier_override(record, triad)
            if override:
                calibrated = max(calibrated, 0.95)
            if override or calibrated >= self.critical_threshold:
                routing = RoutingBucket.CRITICAL_ESCALATION
            elif calibrated < self.auto_dismiss_threshold:
                routing = RoutingBucket.AUTO_DISMISS
            else:
                routing = RoutingBucket.HITL_REVIEW
            matched = [
                rule for rule, score in zip(self.rules, rule_scores[index], strict=True)
                if score >= self.rule_threshold
            ]
            results.append(
                ModelInferenceResult(
                    log_id=record.log_id,
                    raw_sif_p_score=raw_score,
                    calibrated_sif_p_score=calibrated,
                    deterministic_override=override,
                    routing=routing,
                    matched_iogp_rules=matched,
                    triad=triad,
                    latency_ms=max(0.0, float(latencies[index])),
                )
            )
        return results

    def process(
        self,
        record: IncidentNormalizedRecord,
        sif_logits: np.ndarray,
        iogp_logits: np.ndarray,
        *,
        latency_ms: float,
    ) -> ModelInferenceResult:
        return self.process_batch(
            [record], sif_logits, iogp_logits, latency_ms=latency_ms
        )[0]


Postprocessor = InferencePostprocessor


def postprocess_outputs(
    record: IncidentNormalizedRecord,
    sif_logits: np.ndarray,
    iogp_logits: np.ndarray,
    *,
    latency_ms: float,
    postprocessor: InferencePostprocessor | None = None,
) -> ModelInferenceResult:
    return (postprocessor or InferencePostprocessor()).process(
        record, sif_logits, iogp_logits, latency_ms=latency_ms
    )
