"""Run the air-gapped RiskForge weak-supervision corpus benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskforge.supervision.benchmark import benchmark_supervision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(
        benchmark_supervision(args.records, args.iterations), indent=2, sort_keys=True
    )
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
