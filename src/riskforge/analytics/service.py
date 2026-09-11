"""Operational analytics over stored incident results: trends, patterns, barriers.

Pure functions over ``(IncidentNormalizedRecord, ModelInferenceResult)`` pairs;
no I/O and no persistence imports, so any composer can bind them to a store.
All outputs are deterministic for a given input set.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from riskforge.analytics.recommendations import DISCLAIMER
from riskforge.core.contracts import AssetType, RoutingBucket


def _week_start(value: datetime) -> datetime:
    """Return the Monday 00:00 UTC week start for a timestamp."""
    utc = value if value.tzinfo is not None else value.replace(tzinfo=None)
    days_since_monday = utc.weekday()
    monday = (utc - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return monday


class AnalyticsSummary:
    """Computed operational summary for dashboard consumption."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    @property
    def payload(self) -> dict[str, Any]:
        return self._payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


def compute_summary(
    stored: Sequence[tuple[Any, Any]],
    *,
    trend_weeks: int = 12,
    emerging_z_threshold: float = 1.5,
) -> AnalyticsSummary:
    """Compute the operational analytics summary.

    Parameters
    ----------
    stored:
        Sequence of ``(incident, result)`` pairs (core contract models).
    trend_weeks:
        Length of the rolling weekly trend window.
    emerging_z_threshold:
        Mean + z * stddev of prior weeks above which the latest week's
        precursor count is flagged as an emerging risk.
    """
    incidents = [pair[0] for pair in stored]
    results = [pair[1] for pair in stored]
    total = len(results)
    precursors = [r for r in results if r.routing == RoutingBucket.CRITICAL_ESCALATION]
    sif_density = (len(precursors) / total) if total else 0.0

    routing_counts = Counter(r.routing.value for r in results)
    rule_counts = Counter(rule.value for r in results for rule in r.matched_iogp_rules)

    # ── Asset summaries (SPD per asset) ────────────────────────────────
    asset_totals: Counter[str] = Counter()
    asset_sif: Counter[str] = Counter()
    asset_types: dict[str, AssetType] = {}
    for incident, result in zip(incidents, results, strict=True):
        asset_totals[incident.asset_id] += 1
        asset_types[incident.asset_id] = incident.asset_type
        if result.routing == RoutingBucket.CRITICAL_ESCALATION:
            asset_sif[incident.asset_id] += 1
    assets = [
        {
            "asset_id": asset_id,
            "asset_type": asset_types[asset_id].value,
            "total_reports": asset_totals[asset_id],
            "sif_precursors": asset_sif[asset_id],
            "spd": round(asset_sif[asset_id] / asset_totals[asset_id], 4)
            if asset_totals[asset_id]
            else 0.0,
        }
        for asset_id in sorted(asset_totals)
    ]
    assets.sort(key=lambda item: (-item["spd"], item["asset_id"]))

    # ── Weekly trend + emerging-risk detection ─────────────────────────
    by_week: dict[datetime, list[bool]] = defaultdict(list)
    for incident, result in zip(incidents, results, strict=True):
        by_week[_week_start(incident.timestamp)].append(
            result.routing == RoutingBucket.CRITICAL_ESCALATION
        )
    weeks = sorted(by_week)
    trend = [
        {
            "week_start": week.isoformat(),
            "reports": len(by_week[week]),
            "sif_precursors": sum(by_week[week]),
            "density": round(sum(by_week[week]) / len(by_week[week]), 4) if by_week[week] else 0.0,
        }
        for week in weeks
    ]

    emerging: list[dict[str, Any]] = []
    if len(weeks) >= 3:
        window = trend[-(trend_weeks + 1) : -1]
        prior = window[-trend_weeks:] if window else trend[:-1]
        counts = [point["sif_precursors"] for point in prior]
        if len(counts) >= 2:
            mean = sum(counts) / len(counts)
            variance = sum((c - mean) ** 2 for c in counts) / len(counts)
            stddev = variance ** 0.5
            latest = trend[-1]
            # A spike over a perfectly constant baseline (stddev == 0) is the
            # clearest emerging-risk signal, so it must NOT be suppressed;
            # there is no division by stddev here, only multiplication.
            if latest["sif_precursors"] > mean + emerging_z_threshold * stddev:
                emerging.append(
                    {
                        "week_start": latest["week_start"],
                        "sif_precursors": latest["sif_precursors"],
                        "prior_mean": round(mean, 2),
                        "threshold": round(mean + emerging_z_threshold * stddev, 2),
                        "note": "Precursor count is rising above the recent weekly baseline",
                    }
                )

    # ── Recurring patterns across assets (rule + barrier co-occurrence) ─
    pattern_groups: dict[tuple[str, ...], dict[str, Any]] = {}
    for incident, result in zip(incidents, results, strict=True):
        if not result.matched_iogp_rules:
            continue
        key = tuple(sorted(rule.value for rule in result.matched_iogp_rules))
        group = pattern_groups.setdefault(
            key, {"rule_combination": list(key), "count": 0, "assets": set(), "narratives": []}
        )
        group["count"] += 1
        group["assets"].add(incident.asset_id)
        if len(group["narratives"]) < 3:
            group["narratives"].append(incident.raw_narrative[:160])
    patterns = sorted(
        (
            {
                "rule_combination": group["rule_combination"],
                "count": group["count"],
                "assets": sorted(group["assets"]),
                "example_narratives": group["narratives"],
            }
            for group in pattern_groups.values()
        ),
        key=lambda item: (-item["count"], item["rule_combination"]),
    )

    # ── Barrier recurrence across reports ──────────────────────────────
    barrier_counts = Counter(
        result.triad.failed_barrier.canonical_form
        for result in results
        if result.triad.failed_barrier is not None
    )
    barriers = [
        {"barrier": barrier, "failures": count}
        for barrier, count in barrier_counts.most_common()
    ]

    return AnalyticsSummary(
        {
            "disclaimer": DISCLAIMER,
            "total_reports": total,
            "sif_precursors": len(precursors),
            "sif_precursor_density": round(sif_density, 4),
            "routing_counts": dict(routing_counts),
            "matched_rule_counts": dict(rule_counts.most_common()),
            "assets": assets,
            "weekly_trend": trend,
            "emerging_risks": emerging,
            "patterns": patterns[:20],
            "failed_barriers": barriers,
        }
    )
