"""Air-gapped latency and memory benchmarks for the serving engine."""

from __future__ import annotations

import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any, Self

import numpy as np
import onnxruntime as ort
import psutil

from riskforge.core.contracts import AssetType, IncidentNormalizedRecord
from riskforge.serving.artifact import ModelArtifactManifest
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
    baseline_rss_bytes: int
    peak_rss_bytes: int
    final_rss_bytes: int


@dataclass(frozen=True)
class MemorySummary:
    baseline_rss_bytes: int
    peak_rss_bytes: int
    final_rss_bytes: int
    rss_growth_bytes: int


class PeakRssMonitor:
    """Sample total process RSS on every supported development platform."""

    def __init__(self, poll_interval_seconds: float = 0.001) -> None:
        if poll_interval_seconds <= 0.0:
            raise ValueError("poll interval must be positive")
        self._poll_interval_seconds = poll_interval_seconds
        self._stop = Event()
        self._thread: Thread | None = None
        self._baseline = 0
        self._peak = 0
        self._final = 0

    def __enter__(self) -> Self:
        self._baseline = self._read()
        self._peak = self._baseline
        self._stop.clear()
        self._thread = Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._final = self._read()
        self._peak = max(self._peak, self._final)

    def summary(self) -> MemorySummary:
        if self._thread is None or not self._stop.is_set():
            raise RuntimeError("memory summary is available only after monitoring completes")
        return MemorySummary(
            baseline_rss_bytes=self._baseline,
            peak_rss_bytes=self._peak,
            final_rss_bytes=self._final,
            rss_growth_bytes=max(0, self._final - self._baseline),
        )

    def _read(self) -> int:
        return current_rss_bytes()

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self._poll_interval_seconds):
            self._peak = max(self._peak, self._read())


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
    """Return total resident process memory using a cross-platform implementation."""
    try:
        return int(psutil.Process().memory_info().rss)
    except psutil.Error as error:
        statm = Path("/proc/self/statm")
        try:
            resident_pages = int(statm.read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, IndexError, OSError, ValueError):
            raise RuntimeError("process RSS measurement is unavailable") from error


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
        samples: list[float] = []
        with PeakRssMonitor() as memory:
            for _ in range(iterations):
                started = perf_counter()
                engine.infer_batch(records, input_ids, attention_mask)
                samples.append((perf_counter() - started) * 1000.0)
        memory_summary = memory.summary()
        results.append(
            BatchBenchmark(
                batch_size=batch_size,
                iterations=iterations,
                latency=latency_summary(samples),
                rss_growth_bytes=memory_summary.rss_growth_bytes,
                baseline_rss_bytes=memory_summary.baseline_rss_bytes,
                peak_rss_bytes=memory_summary.peak_rss_bytes,
                final_rss_bytes=memory_summary.final_rss_bytes,
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

    def invoke(index: int, batcher: ConcurrentRequestBatcher) -> float:
        started = perf_counter()
        future = batcher.submit(synthetic_record(index), input_ids, attention_mask)
        future.result()
        return (perf_counter() - started) * 1000.0

    with PeakRssMonitor() as memory, ConcurrentRequestBatcher(
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
    memory_summary = memory.summary()
    return {
        "concurrency": concurrency,
        "iterations": iterations,
        "request_count": concurrency * iterations,
        "latency": asdict(latency_summary(samples)),
        "memory": asdict(memory_summary),
        "rss_growth_bytes": memory_summary.rss_growth_bytes,
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
    target_latency_ms: float = 35.0,
    target_peak_rss_bytes: int = 1_200_000_000,
) -> dict[str, Any]:
    if target_latency_ms <= 0.0 or target_peak_rss_bytes < 1:
        raise ValueError("acceptance targets must be positive")
    model = Path(model_path)
    manifest = ModelArtifactManifest.load(manifest_path)
    if sequence_length > manifest.max_sequence_length:
        raise ValueError("benchmark sequence length exceeds artifact maximum")
    with PeakRssMonitor() as overall_memory:
        with PeakRssMonitor() as cold_memory:
            started = perf_counter()
            engine = ONNXInferenceEngine.from_artifact(
                model, manifest_path, max_batch_size=max(DEFAULT_BATCH_SIZES), warmup=False
            )
            record, input_ids, attention_mask = _inputs(1, sequence_length)
            engine.infer_batch(record, input_ids, attention_mask)
            cold_start_ms = (perf_counter() - started) * 1000.0
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
    cold_summary = cold_memory.summary()
    overall_summary = overall_memory.summary()
    batch_one = next(result for result in warm_batches if result.batch_size == 1)
    latency_passed = batch_one.latency.p95_ms <= target_latency_ms
    memory_passed = overall_summary.peak_rss_bytes <= target_peak_rss_bytes
    quantization_passed = manifest.quantization.upper() == "INT8"
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact": {
            "backbone": manifest.backbone,
            "model_sha256": manifest.model_sha256,
            "model_size_bytes": model.stat().st_size,
            "quantization": manifest.quantization,
            "maximum_sequence_length": manifest.max_sequence_length,
        },
        "environment": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "byte_order": sys.byteorder,
            "onnxruntime_version": ort.__version__,
            "onnx_intra_op_threads": 4,
            "onnx_inter_op_threads": 1,
        },
        "benchmark_parameters": {
            "batch_sizes": list(DEFAULT_BATCH_SIZES),
            "iterations": iterations,
            "concurrent_iterations": concurrent_iterations,
            "concurrency": concurrency,
            "sequence_length": sequence_length,
            "max_queue_delay_ms": max_queue_delay_ms,
        },
        "measurement_scope": "pretokenized_onnx_execution_and_postprocessing",
        "cold": {
            "load_and_first_inference_ms": cold_start_ms,
            "memory": asdict(cold_summary),
            "rss_growth_bytes": cold_summary.rss_growth_bytes,
        },
        "warm_batches": [asdict(result) for result in warm_batches],
        "concurrent": concurrent,
        "process_memory": asdict(overall_summary),
        "targets": {
            "warm_batch_1_p95_ms": target_latency_ms,
            "peak_rss_bytes": target_peak_rss_bytes,
            "quantization": "INT8",
        },
        "acceptance": {
            "warm_batch_1_latency_passed": latency_passed,
            "peak_memory_passed": memory_passed,
            "int8_artifact_passed": quantization_passed,
            "passed": latency_passed and memory_passed and quantization_passed,
        },
        "target_latency_ms": target_latency_ms,
    }
