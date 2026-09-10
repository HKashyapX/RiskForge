"""Privacy-safe, streaming inspection for local modeling datasets."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetInspection:
    """Dataset metadata that intentionally excludes record values."""

    path: str
    format: str
    records: int
    fields: tuple[str, ...]
    field_types: Mapping[str, str]
    size_bytes: int


def _jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"record on line {line_number} is not an object")
            yield value


def _csv_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def inspect_dataset(path: Path) -> DatasetInspection:
    """Inspect JSONL or CSV metadata without retaining or returning values."""

    if path.is_symlink():
        raise ValueError("refusing to inspect a symbolic link")
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = _jsonl_records(path)
        format_name = "jsonl"
    elif suffix == ".csv":
        records = _csv_records(path)
        format_name = "csv"
    else:
        raise ValueError("supported formats are .jsonl and .csv")

    count = 0
    first: dict[str, Any] | None = None
    for record in records:
        if first is None:
            first = record
        count += 1

    if first is None:
        fields: tuple[str, ...] = ()
        field_types: dict[str, str] = {}
    else:
        fields = tuple(sorted(str(field) for field in first))
        field_types = {field: type(first[field]).__name__ for field in fields}

    return DatasetInspection(
        path=str(path),
        format=format_name,
        records=count,
        fields=fields,
        field_types=field_types,
        size_bytes=path.stat().st_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report dataset shape and field types without record values."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = [inspect_dataset(path) for path in args.paths]
    if args.as_json:
        print(json.dumps([asdict(report) for report in reports], indent=2, sort_keys=True))
    else:
        for report in reports:
            print(f"{report.path}: {report.records} records, {report.size_bytes} bytes")
            for field in report.fields:
                print(f"  {field}: {report.field_types[field]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
