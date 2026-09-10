"""Streaming structural validation for prepared RiskForge SIF datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetValidationReport:
    """Aggregate validation results that contain no incident values."""

    path: str
    records: int
    valid_records: int
    invalid_records: int
    duplicate_ids: int
    issue_counts: Mapping[str, int]

    @property
    def valid(self) -> bool:
        return self.records > 0 and self.invalid_records == 0


_REQUIRED_TYPES: dict[str, type[Any]] = {
    "id": str,
    "input": dict,
    "input.narrative": str,
    "labels": dict,
    "labels.sif_potential": str,
    "labels.sif_confidence": str,
    "labels.sif_label_source": str,
    "provenance": dict,
    "provenance.pipeline_version": str,
    "provenance.source_datasets": list,
}

_NON_EMPTY_STRINGS = {
    "id",
    "input.narrative",
    "labels.sif_potential",
    "labels.sif_confidence",
    "labels.sif_label_source",
    "provenance.pipeline_version",
}


def _nested_value(record: Mapping[str, Any], dotted_path: str) -> tuple[bool, Any]:
    value: Any = record
    for component in dotted_path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return False, None
        value = value[component]
    return True, value


def _validate_record(record: Mapping[str, Any], issues: Counter[str]) -> bool:
    valid = True
    for path, expected_type in _REQUIRED_TYPES.items():
        present, value = _nested_value(record, path)
        if not present:
            issues[f"missing:{path}"] += 1
            valid = False
            continue
        if not isinstance(value, expected_type):
            issues[f"type:{path}"] += 1
            valid = False
            continue
        if path in _NON_EMPTY_STRINGS and not value.strip():
            issues[f"empty:{path}"] += 1
            valid = False

    present, sources = _nested_value(record, "provenance.source_datasets")
    if present and isinstance(sources, list):
        if not sources or not all(isinstance(source, str) and source.strip() for source in sources):
            issues["invalid:provenance.source_datasets"] += 1
            valid = False
    return valid


def validate_sif_dataset(path: Path) -> DatasetValidationReport:
    """Validate a JSONL dataset in constant record memory.

    Duplicate detection retains only identifier hashes/strings, not complete
    records. Reports expose aggregate issue codes and never incident values.
    """

    if path.is_symlink():
        raise ValueError("refusing to validate a symbolic link")
    if path.suffix.lower() != ".jsonl":
        raise ValueError("SIF dataset contract requires a .jsonl file")
    if not path.is_file():
        raise FileNotFoundError(path)

    issues: Counter[str] = Counter()
    seen_ids: set[str] = set()
    records = 0
    valid_records = 0
    duplicate_ids = 0

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                issues["invalid:json"] += 1
                continue
            if not isinstance(record, dict):
                issues["type:record"] += 1
                continue

            record_valid = _validate_record(record, issues)
            present, incident_id = _nested_value(record, "id")
            if present and isinstance(incident_id, str) and incident_id.strip():
                if incident_id in seen_ids:
                    issues["duplicate:id"] += 1
                    duplicate_ids += 1
                    record_valid = False
                else:
                    seen_ids.add(incident_id)
            if record_valid:
                valid_records += 1

    return DatasetValidationReport(
        path=str(path),
        records=records,
        valid_records=valid_records,
        invalid_records=records - valid_records,
        duplicate_ids=duplicate_ids,
        issue_counts=dict(sorted(issues.items())),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate prepared SIF JSONL structure without printing record values."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = [validate_sif_dataset(path) for path in args.paths]
    if args.as_json:
        payload = [asdict(report) | {"valid": report.valid} for report in reports]
        print(json.dumps(payload, indent=2))
    else:
        for report in reports:
            status = "PASS" if report.valid else "FAIL"
            print(
                f"{status} {report.path}: records={report.records}, "
                f"valid={report.valid_records}, invalid={report.invalid_records}, "
                f"duplicates={report.duplicate_ids}"
            )
            for issue, count in report.issue_counts.items():
                print(f"  {issue}: {count}")
    return 0 if all(report.valid for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
