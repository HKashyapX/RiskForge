"""Deterministic, explainable inference engine for model-free deployment.

Implements the application-facing ``InferenceEngine`` protocol with the
IOGP Life-Saving-Rule vocabulary and a documented risk policy instead of a
trained statistical model.  This keeps the platform fully operational and
honest before a trained ONNX artifact exists: every score is derived from
explicit, auditable lexical rules (hazard category, failed barrier, worker
exposure, measured high-energy quantities) rather than fabricated ML.

This is NOT a neural network and must never be presented as one; the
composer reports this engine's mode through readiness and the README's
capability statement.  When ``RISKFORGE_MODEL_PATH``/``RISKFORGE_MANIFEST_PATH``
point at a real artifact, ``ONNXInferenceEngine`` is used instead.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from riskforge.core.contracts import (
    EntitySpan,
    IncidentNormalizedRecord,
    LifeSavingRule,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)

# Deterministic thresholds mirrored from config/threshold_policy.yaml.
_AUTO_DISMISS_MAX = 0.40
_CRITICAL_MIN = 0.65
_PRESSURE_PSI = 150.0
_HEIGHT_METERS = 1.8
_H2S_PPM = 10.0
_SCORE_CAP = 0.98
_FORCED_ESCALATION_SCORE = 0.90

_FAILURE_RE = re.compile(
    r"\b(?:bypass(?:ed)?|defeat(?:ed)?|disabl(?:ed|e[sd]?)|fail(?:ed|ure[sd]?)?|"
    r"leak(?:ed|ing)?|breach(?:ed)?|missing|removed|not\s+isolated|"
    r"no\s+(?:loto|isolation|guard|barricade)|"
    r"still\s+(?:pressuri[sz]ed|live|energi[sz]ed))\b",
    re.IGNORECASE,
)
_WORKER_RE = re.compile(
    r"\b(?:worker|personnel|crew|technician|operator|contractor|driller|"
    r"derrickman|roustabout|roughneck|fitter|helper)\b",
    re.IGNORECASE,
)
_NEGATED_WORKER_RE = re.compile(
    r"\b(?:no|not|never|without|zero|nil)\b(?:\W+\w+){0,3}\W+"
    r"\b(?:worker|personnel|crew|technician|operator|contractor|driller|"
    r"derrickman|roustabout|roughneck|fitter|helper)\b",
    re.IGNORECASE,
)
_PSI_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*psi\b", re.IGNORECASE)
_HEIGHT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:m|metre|meter)s?\b", re.IGNORECASE)
_HEIGHT_CONTEXT_RE = re.compile(
    r"\b(?:height|elevat(?:ed|ion)|scaffold|ladder|platform|monkey\s*board|"
    r"derrick|above\s+ground|roof)\b",
    re.IGNORECASE,
)
_H2S_RE = re.compile(
    r"\b(?:h2s|hydrogen\s+sulfide)\D{0,12}(\d+(?:\.\d+)?)\s*ppm\b", re.IGNORECASE
)

_BARRIER_WINDOW = 40


@dataclass(frozen=True)
class _Rule:
    rule: LifeSavingRule
    hazard: re.Pattern[str]
    barrier: re.Pattern[str]
    base: float


_RULES: tuple[_Rule, ...] = (
    _Rule(
        LifeSavingRule.ENERGY_ISOLATION,
        re.compile(
            r"\b(?:isolation|isolat(?:e|ed|ing)|loto|lockout|tagout|"
            r"de-?energi[sz]|live\s+(?:line|wire|equipment)|stored\s+pressure|"
            r"bleed(?:ing)?)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:isolation|loto|lockout|tagout|blind|spade|bleed)\b", re.IGNORECASE),
        0.60,
    ),
    _Rule(
        LifeSavingRule.CONFINED_SPACE,
        re.compile(
            r"\b(?:confined\s+space|tank|vessel|cellar|sump|manhole|"
            r"entry\s+permit|pit)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:gas\s+test|permit|ventilation|standby(?:\s*man)?|blower)\b", re.IGNORECASE),
        0.60,
    ),
    _Rule(
        LifeSavingRule.WORK_AT_HEIGHT,
        re.compile(
            r"\b(?:fall(?:ing)?|slip(?:ped)?|unprotected\s+edge|scaffold|ladder|"
            r"work(?:ing)?\s+at\s+height|heights?)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:fall\s+arrest|fall\s+protection|harness|guardrail|lifeline)\b", re.IGNORECASE),
        0.55,
    ),
    _Rule(
        LifeSavingRule.TOXIC_GAS,
        re.compile(
            r"\b(?:h2s|hydrogen\s+sulfide|toxic\s+gas|gas\s+leak|gas\s+release|"
            r"lpg|methane|carbon\s+monoxide)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:detector|monitor|breathing\s+apparatus|scba|mask|alarm)\b", re.IGNORECASE),
        0.65,
    ),
    _Rule(
        LifeSavingRule.LINE_OF_FIRE,
        re.compile(
            r"\b(?:line\s+of\s+fire|struck\s+by|pinch\s+point|crush(?:ed)?|"
            r"caught\s+(?:between|in)|entangl\w*|dropped|falling\s+(?:object|material)|"
            r"suspended\s+load)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:barricad\w*|exclusion\s+zone|spotter|tag\s+line|netting)\b", re.IGNORECASE),
        0.50,
    ),
    _Rule(
        LifeSavingRule.BYPASSING_SAFETY_CONTROLS,
        re.compile(
            r"\b(?:bypass\w*|defeat\w*|disabl\w*|removed\s+guard|interlock|"
            r"alarm\s+(?:disabled|ignored))\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:guard|interlock|alarm|relief\s+valve|psv|prv)\b", re.IGNORECASE),
        0.55,
    ),
    _Rule(
        LifeSavingRule.SAFE_MECHANICAL_LIFTING,
        re.compile(
            r"\b(?:lift(?:ing|ed)?|crane|hoist|winch|rigging|sling|derrick)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:tag\s*line|load\s+test|swl|outrigger|lift\s+plan)\b", re.IGNORECASE),
        0.50,
    ),
    _Rule(
        LifeSavingRule.HOT_WORK,
        re.compile(
            r"\b(?:hot\s+work|welding|weld\w*|grinding|grinder|flame|spark\w*|"
            r"ignition\s+source)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:fire\s+watch|fire\s+blanket|permit|extinguisher)\b", re.IGNORECASE),
        0.50,
    ),
    _Rule(
        LifeSavingRule.DRIVING,
        re.compile(
            r"\b(?:driving|vehicle|journey|rtv|pickup|reversing|convoy)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:seat\s*belt|speed\s+limit|spotter|journey\s+plan)\b", re.IGNORECASE),
        0.40,
    ),
)


class HeuristicRuleEngine:
    """Deterministic SIF-precursor scorer implementing ``InferenceEngine``.

    Stateless and thread-safe: the same narrative always yields the same
    result.  ``engine_name`` and ``mode`` make the engine's nature explicit
    wherever it is surfaced (readiness, logs, dashboards).
    """

    engine_name = "heuristic-rules-v1"

    @property
    def mode(self) -> str:
        return "deterministic-heuristics"

    def infer(self, record: IncidentNormalizedRecord) -> ModelInferenceResult:
        started = time.perf_counter()
        narrative = record.raw_narrative or ""

        matched: list[LifeSavingRule] = []
        best = 0.0
        worker_exposed = bool(_WORKER_RE.search(narrative)) and not _NEGATED_WORKER_RE.search(narrative)
        any_failure = bool(_FAILURE_RE.search(narrative))
        psi = _PSI_RE.search(narrative)
        height = _HEIGHT_RE.search(narrative) if _HEIGHT_CONTEXT_RE.search(narrative) else None
        h2s = _H2S_RE.search(narrative)
        energy_exceeded = bool(
            (psi and float(psi.group(1)) >= _PRESSURE_PSI)
            or (height and float(height.group(1)) >= _HEIGHT_METERS)
            or (h2s and float(h2s.group(1)) >= _H2S_PPM)
        )

        for entry in _RULES:
            if not entry.hazard.search(narrative):
                continue
            matched.append(entry.rule)
            score = entry.base
            if _FAILURE_RE.search(narrative) and entry.barrier.search(narrative):
                score += 0.15
            if worker_exposed:
                score += 0.10
            if energy_exceeded:
                score += 0.20
            best = max(best, score)

        if not matched:
            probability = 0.05
            routing = RoutingBucket.AUTO_DISMISS
            deterministic_override = False
        else:
            probability = min(best, _SCORE_CAP)
            routing = self._route(probability)
            if energy_exceeded and any_failure and probability < _FORCED_ESCALATION_SCORE:
                probability = _FORCED_ESCALATION_SCORE
                escalated = RoutingBucket.CRITICAL_ESCALATION
                deterministic_override = escalated is not routing
                routing = escalated
            else:
                deterministic_override = False

        # High energy with a failed/absent control is a SIF precursor even
        # when no Life-Saving-Rule vocabulary is present in the narrative.
        if (
            not matched
            and energy_exceeded
            and any_failure
        ):
            probability = _FORCED_ESCALATION_SCORE
            routing = RoutingBucket.CRITICAL_ESCALATION
            deterministic_override = True

        result = ModelInferenceResult(
            log_id=record.log_id,
            raw_sif_p_score=round(probability, 4),
            calibrated_sif_p_score=round(probability, 4),
            deterministic_override=deterministic_override,
            routing=routing,
            matched_iogp_rules=sorted(matched, key=lambda rule: rule.value),
            triad=self._triad(record),
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
        return result

    def infer_batch(
        self, records: Sequence[IncidentNormalizedRecord]
    ) -> Sequence[ModelInferenceResult]:
        return [self.infer(record) for record in records]

    @staticmethod
    def _route(probability: float) -> RoutingBucket:
        if probability >= _CRITICAL_MIN:
            return RoutingBucket.CRITICAL_ESCALATION
        if probability >= _AUTO_DISMISS_MAX:
            return RoutingBucket.HITL_REVIEW
        return RoutingBucket.AUTO_DISMISS

    @staticmethod
    def _triad(record: IncidentNormalizedRecord) -> OperationalTriad:
        activity: EntitySpan | None = None
        location: EntitySpan | None = None
        failed_barrier: EntitySpan | None = None
        narrative = record.raw_narrative or ""
        for span in record.spans:
            if activity is None and span.entity_type == "ACTIVITY":
                activity = span
            if location is None and span.entity_type == "LOCATION":
                location = span
            if failed_barrier is None and span.entity_type == "BARRIER":
                window = narrative[max(0, span.start_char - _BARRIER_WINDOW) : span.end_char + _BARRIER_WINDOW]
                if _FAILURE_RE.search(window):
                    failed_barrier = span
        return OperationalTriad(
            activity=activity,
            asset_location=location,
            failed_barrier=failed_barrier,
        )
