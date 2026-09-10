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
    nested_schema: Mapping[str, str]
    size_bytes: int


def _record_type(value: Any) -> str:
    if value is None:
        return "null"
    return type(value).__name__


def _merge_type(schema: dict[str, str], path: str, value: Any) -> None:
    observed = _record_type(value)
    previous = schema.get(path)
    if previous is None:
        schema[path] = observed
        return
    if observed not in previous.split("|"):
        schema[path] = "|".join(sorted({*previous.split("|"), observed}))


def _collect_nested_schema(
    value: Any,
    schema: dict[str, str],
    *,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 4,
) -> None:
    if depth >= max_depth:
        return
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            _merge_type(schema, path, child)
            _collect_nested_schema(
                child,
                schema,
                prefix=path,
                depth=depth + 1,
                max_depth=max_depth,
            )
    elif isinstance(value, list) and value:
        item_path = f"{prefix}[]"
        for child in value:
            _merge_type(schema, item_path, child)
            _collect_nested_schema(
                child,
                schema,
                prefix=item_path,
                depth=depth + 1,
                max_depth=max_depth,
            )


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
                raise TypeError(f"record on line {line_number} is not an object")
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
    nested_schema: dict[str, str] = {}
    for record in records:
        if first is None:
            first = record
        _collect_nested_schema(record, nested_schema)
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
        nested_schema=dict(sorted(nested_schema.items())),
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
            for field, field_type in report.nested_schema.items():
                print(f"  {field}: {field_type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
