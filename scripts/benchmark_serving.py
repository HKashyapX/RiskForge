"""Run RiskForge serving benchmarks against a validated ONNX artifact bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskforge.serving.benchmark import benchmark_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="path to the ONNX model")
    parser.add_argument("manifest", type=Path, help="path to the artifact manifest")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--concurrent-iterations", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--max-queue-delay-ms", type=float, default=5.0)
    parser.add_argument("--target-latency-ms", type=float, default=35.0)
    parser.add_argument("--target-peak-rss-bytes", type=int, default=1_200_000_000)
    parser.add_argument(
        "--enforce-targets",
        action="store_true",
        help="exit unsuccessfully when the production acceptance targets are missed",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = benchmark_artifact(
        args.model,
        args.manifest,
        iterations=args.iterations,
        concurrent_iterations=args.concurrent_iterations,
        concurrency=args.concurrency,
        sequence_length=args.sequence_length,
        max_queue_delay_ms=args.max_queue_delay_ms,
        target_latency_ms=args.target_latency_ms,
        target_peak_rss_bytes=args.target_peak_rss_bytes,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if not args.enforce_targets or report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
