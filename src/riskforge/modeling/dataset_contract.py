"""Streaming structural validation for prepared RiskForge SIF datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    cross_split_duplicates: int
    issue_counts: Mapping[str, int]
    label_counts: Mapping[str, Mapping[str, int]]

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

_PROFILED_LABELS = (
    "labels.sif_potential",
    "labels.sif_confidence",
    "labels.actual_severity",
    "labels.supervision_tier",
)

_SAFE_CATEGORY = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


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


def _identifier_digest(identifier: str) -> bytes:
    return hashlib.blake2b(identifier.encode("utf-8"), digest_size=16).digest()


def _safe_category(value: Any) -> str:
    if isinstance(value, str) and _SAFE_CATEGORY.fullmatch(value):
        return value
    return "<invalid-or-redacted>"


def _validate_sif_dataset(
    path: Path,
    prior_id_digests: set[bytes],
) -> tuple[DatasetValidationReport, set[bytes]]:
    """Validate a JSONL dataset without retaining complete records.

    Duplicate detection retains only fixed-size identifier digests. Reports
    expose aggregate issue codes and never incident values.
    """

    if path.is_symlink():
        raise ValueError("refusing to validate a symbolic link")
    if path.suffix.lower() != ".jsonl":
        raise ValueError("SIF dataset contract requires a .jsonl file")
    if not path.is_file():
        raise FileNotFoundError(path)

    issues: Counter[str] = Counter()
    seen_ids: set[bytes] = set()
    label_counts = {path: Counter[str]() for path in _PROFILED_LABELS}
    records = 0
    valid_records = 0
    duplicate_ids = 0
    cross_split_duplicates = 0

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
            for label_path, counts in label_counts.items():
                present, value = _nested_value(record, label_path)
                counts[_safe_category(value) if present else "<missing>"] += 1

            present, incident_id = _nested_value(record, "id")
            if present and isinstance(incident_id, str) and incident_id.strip():
                identifier_digest = _identifier_digest(incident_id)
                if identifier_digest in seen_ids:
                    issues["duplicate:id"] += 1
                    duplicate_ids += 1
                    record_valid = False
                elif identifier_digest in prior_id_digests:
                    issues["leakage:id"] += 1
                    cross_split_duplicates += 1
                    record_valid = False
                else:
                    seen_ids.add(identifier_digest)
            if record_valid:
                valid_records += 1

    report = DatasetValidationReport(
        path=str(path),
        records=records,
        valid_records=valid_records,
        invalid_records=records - valid_records,
        duplicate_ids=duplicate_ids,
        cross_split_duplicates=cross_split_duplicates,
        issue_counts=dict(sorted(issues.items())),
        label_counts={
            label_path: dict(sorted(counts.items()))
            for label_path, counts in label_counts.items()
        },
    )
    return report, seen_ids


def validate_sif_dataset(path: Path) -> DatasetValidationReport:
    """Validate one prepared SIF dataset without exposing record values."""

    report, _ = _validate_sif_dataset(path, set())
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate prepared SIF JSONL structure without printing record values."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports: list[DatasetValidationReport] = []
    prior_id_digests: set[bytes] = set()
    for path in args.paths:
        report, current_id_digests = _validate_sif_dataset(path, prior_id_digests)
        reports.append(report)
        prior_id_digests.update(current_id_digests)
    if args.as_json:
        payload = [asdict(report) | {"valid": report.valid} for report in reports]
        print(json.dumps(payload, indent=2))
    else:
        for report in reports:
            status = "PASS" if report.valid else "FAIL"
            print(
                f"{status} {report.path}: records={report.records}, "
                f"valid={report.valid_records}, invalid={report.invalid_records}, "
                f"duplicates={report.duplicate_ids}, "
                f"cross_split_duplicates={report.cross_split_duplicates}"
            )
            for issue, count in report.issue_counts.items():
                print(f"  {issue}: {count}")
            for label_path, counts in report.label_counts.items():
                summary = ", ".join(
                    f"{label}={count}" for label, count in counts.items()
                )
                print(f"  {label_path}: {summary}")
    return 0 if all(report.valid for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
