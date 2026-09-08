"""Air-gapped latency and memory benchmarks for the serving engine."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from riskforge.core.contracts import AssetType, IncidentNormalizedRecord
from riskforge.serving.batching import ConcurrentRequestBatcher
from riskforge.serving.engine import ONNXInferenceEngine

DEFAULT_BATCH_SIZES = (1, 8, 16, 32)


@dataclass(frozen=True)
class LatencySummary:
    sample_count: int
    p50_ms: float
    p95_ms: float
    minimum_ms: float
    maximum_ms: float


@dataclass(frozen=True)
class BatchBenchmark:
    batch_size: int
    iterations: int
    latency: LatencySummary
    rss_growth_bytes: int


def percentile(samples: list[float], percentile_value: float) -> float:
    """Calculate an interpolated percentile without an optional statistics dependency."""
    if not samples:
        raise ValueError("at least one latency sample is required")
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be between zero and 100")
    ordered = sorted(float(sample) for sample in samples)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_summary(samples: list[float]) -> LatencySummary:
    return LatencySummary(
        sample_count=len(samples),
        p50_ms=percentile(samples, 50.0),
        p95_ms=percentile(samples, 95.0),
        minimum_ms=min(samples),
        maximum_ms=max(samples),
    )


def current_rss_bytes() -> int:
    """Read current resident memory without network or third-party monitoring agents."""
    statm = Path("/proc/self/statm")
    try:
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
    except (IndexError, OSError, ValueError):
        return 0
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def synthetic_record(index: int) -> IncidentNormalizedRecord:
    """Create non-production input used only to exercise serving performance."""
    return IncidentNormalizedRecord(
        log_id=f"BENCHMARK_{index:06d}",
        timestamp=datetime(2000, 1, 1, tzinfo=UTC),
        asset_id="BENCHMARK_ASSET",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="",
        spans=[],
    )


def _inputs(batch_size: int, sequence_length: int) -> tuple[list[Any], np.ndarray, np.ndarray]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    records = [synthetic_record(index) for index in range(batch_size)]
    input_ids = np.zeros((batch_size, sequence_length), dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    return records, input_ids, attention_mask


def benchmark_warm_batches(
    engine: ONNXInferenceEngine,
    *,
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
    iterations: int = 100,
    sequence_length: int = 256,
) -> list[BatchBenchmark]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    results: list[BatchBenchmark] = []
    for batch_size in batch_sizes:
        if batch_size > engine.max_batch_size:
            raise ValueError("benchmark batch size exceeds engine maximum")
        records, input_ids, attention_mask = _inputs(batch_size, sequence_length)
        engine.infer_batch(records, input_ids, attention_mask)
        rss_before = current_rss_bytes()
        samples: list[float] = []
        for _ in range(iterations):
            started = perf_counter()
            engine.infer_batch(records, input_ids, attention_mask)
            samples.append((perf_counter() - started) * 1000.0)
        rss_after = current_rss_bytes()
        results.append(
            BatchBenchmark(
                batch_size=batch_size,
                iterations=iterations,
                latency=latency_summary(samples),
                rss_growth_bytes=max(0, rss_after - rss_before),
            )
        )
    return results


def benchmark_concurrent_requests(
    engine: ONNXInferenceEngine,
    *,
    concurrency: int = 32,
    iterations: int = 20,
    sequence_length: int = 256,
    max_queue_delay_ms: float = 5.0,
) -> dict[str, Any]:
    if concurrency < 1 or iterations < 1:
        raise ValueError("concurrency and iterations must be positive")
    input_ids = np.zeros(sequence_length, dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    samples: list[float] = []
    rss_before = current_rss_bytes()

    def invoke(index: int, batcher: ConcurrentRequestBatcher) -> float:
        started = perf_counter()
        future = batcher.submit(synthetic_record(index), input_ids, attention_mask)
        future.result()
        return (perf_counter() - started) * 1000.0

    with ConcurrentRequestBatcher(
        engine,
        max_queue_size=max(concurrency * 2, engine.max_batch_size),
        max_queue_delay_ms=max_queue_delay_ms,
    ) as batcher, ThreadPoolExecutor(max_workers=concurrency) as callers:
        for iteration in range(iterations):
            submissions = [
                callers.submit(invoke, iteration * concurrency + index, batcher)
                for index in range(concurrency)
            ]
            samples.extend(submission.result() for submission in submissions)
    rss_after = current_rss_bytes()
    return {
        "concurrency": concurrency,
        "iterations": iterations,
        "request_count": concurrency * iterations,
        "latency": asdict(latency_summary(samples)),
        "rss_growth_bytes": max(0, rss_after - rss_before),
    }


def benchmark_artifact(
    model_path: str | Path,
    manifest_path: str | Path,
    *,
    iterations: int = 100,
    concurrent_iterations: int = 20,
    concurrency: int = 32,
    sequence_length: int = 256,
    max_queue_delay_ms: float = 5.0,
) -> dict[str, Any]:
    rss_before = current_rss_bytes()
    started = perf_counter()
    engine = ONNXInferenceEngine.from_artifact(
        model_path, manifest_path, max_batch_size=max(DEFAULT_BATCH_SIZES), warmup=False
    )
    record, input_ids, attention_mask = _inputs(1, sequence_length)
    engine.infer_batch(record, input_ids, attention_mask)
    cold_start_ms = (perf_counter() - started) * 1000.0
    rss_after_cold = current_rss_bytes()
    engine.warmup(sequence_length)

    warm_batches = benchmark_warm_batches(
        engine, iterations=iterations, sequence_length=sequence_length
    )
    concurrent = benchmark_concurrent_requests(
        engine,
        concurrency=concurrency,
        iterations=concurrent_iterations,
        sequence_length=sequence_length,
        max_queue_delay_ms=max_queue_delay_ms,
    )
    return {
        "cold": {
            "load_and_first_inference_ms": cold_start_ms,
            "rss_growth_bytes": max(0, rss_after_cold - rss_before),
        },
        "warm_batches": [asdict(result) for result in warm_batches],
        "concurrent": concurrent,
        "target_latency_ms": 35.0,
    }
