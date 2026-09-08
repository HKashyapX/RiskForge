"""Corpus-scale runtime and memory benchmarks for weak supervision."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from riskforge.core.contracts import AssetType, IncidentNormalizedRecord
from riskforge.supervision.engine import DawidSkeneLabelModel, LFAnalysis, LFApplier
from riskforge.supervision.heuristics import DEFAULT_LFS

_TEMPLATES = (
    "Driller saw a seal rupture during a 3000 psi surge.",
    "Worker on scaffold had harness unhooked.",
    "Crane load slipped above personnel underneath.",
    "H2S leaked and SCBA failed near the crew.",
    "LOTO was bypassed at the energized motor.",
    "Housekeeping removed dust in the office.",
)


def current_rss_bytes() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
    except (IndexError, OSError, ValueError):
        return 0
    return pages * os.sysconf("SC_PAGE_SIZE")


def synthetic_corpus(record_count: int) -> list[IncidentNormalizedRecord]:
    if record_count < 1:
        raise ValueError("record_count must be positive")
    timestamp = datetime(2000, 1, 1, tzinfo=UTC)
    return [
        IncidentNormalizedRecord(
            log_id=f"SUPERVISION_BENCHMARK_{index:08d}",
            timestamp=timestamp,
            asset_id="BENCHMARK_ASSET",
            asset_type=AssetType.DRILLING_RIG,
            raw_narrative=_TEMPLATES[index % len(_TEMPLATES)],
            spans=[],
        )
        for index in range(record_count)
    ]


def _latency(samples: list[float]) -> dict[str, float | int]:
    return {
        "sample_count": len(samples),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def benchmark_supervision(record_count: int = 10_000, iterations: int = 5) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    records = synthetic_corpus(record_count)
    applier = LFApplier(DEFAULT_LFS)
    apply_samples: list[float] = []
    fit_samples: list[float] = []
    rss_before = current_rss_bytes()
    votes = applier.apply(records)

    for _ in range(iterations):
        started = perf_counter()
        votes = applier.apply(records)
        apply_samples.append((perf_counter() - started) * 1000.0)

        started = perf_counter()
        DawidSkeneLabelModel().fit(votes)
        fit_samples.append((perf_counter() - started) * 1000.0)

    rss_after = current_rss_bytes()
    return {
        "record_count": record_count,
        "iterations": iterations,
        "lf_metadata": [
            {"name": lf.name, "version": lf.version} for lf in DEFAULT_LFS
        ],
        "lf_apply_latency": _latency(apply_samples),
        "dawid_skene_fit_latency": _latency(fit_samples),
        "rss_growth_bytes": max(0, rss_after - rss_before),
        "lf_analysis": LFAnalysis(votes, DEFAULT_LFS).summary(),
    }
