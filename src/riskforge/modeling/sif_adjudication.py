"""Create a local, spreadsheet-safe queue for adjudicating ambiguous SIF labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any


_HEADER = (
    "record_digest",
    "source_split",
    "narrative_for_review",
    "current_label",
    "reviewer_label",
    "reviewer_id",
    "reviewed_at_utc",
    "rationale",
)


def _nested_value(record: dict[str, Any], *path: str) -> Any:
    value: Any = record
    for component in path:
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def _digest(identifier: str) -> str:
    return hashlib.blake2b(identifier.encode("utf-8"), digest_size=16).hexdigest()


def _spreadsheet_safe(value: str) -> str:
    """Prevent imported text from being interpreted as a spreadsheet formula."""

    if value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _open_private_output(path: Path) -> Any:
    if path.suffix.lower() != ".csv":
        raise ValueError("adjudication output must be a .csv file")
    if path.is_symlink():
        raise ValueError("refusing to replace a symbolic link")
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        os.chmod(path, 0o600)
        return os.fdopen(descriptor, "w", encoding="utf-8", newline="")
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def create_adjudication_queue(paths: Sequence[Path], output: Path) -> int:
    """Write ambiguous gold records to a new private CSV and return row count."""

    seen_digests: set[str] = set()
    rows = 0
    try:
        with _open_private_output(output) as stream:
            writer = csv.writer(stream)
            writer.writerow(_HEADER)

            for path in paths:
                if path.is_symlink():
                    raise ValueError("refusing to read a symbolic link")
                if path.suffix.lower() != ".jsonl":
                    raise ValueError("adjudication input must be a .jsonl file")
                if not path.is_file():
                    raise FileNotFoundError(path)

                with path.open(encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"{path}: invalid JSON at record {line_number}"
                            ) from exc
                        if not isinstance(record, dict):
                            raise TypeError(f"{path}: record {line_number} is not an object")

                        label = _nested_value(record, "labels", "sif_potential")
                        if label != "possible":
                            continue

                        identifier = record.get("id")
                        narrative = _nested_value(record, "input", "narrative")
                        if not isinstance(identifier, str) or not identifier.strip():
                            raise ValueError(f"{path}: record {line_number} has no valid id")
                        if not isinstance(narrative, str) or not narrative.strip():
                            raise ValueError(
                                f"{path}: record {line_number} has no review narrative"
                            )

                        digest = _digest(identifier)
                        if digest in seen_digests:
                            raise ValueError("duplicate ambiguous record across input splits")
                        seen_digests.add(digest)

                        writer.writerow(
                            (
                                digest,
                                _spreadsheet_safe(path.stem),
                                _spreadsheet_safe(narrative),
                                "possible",
                                "",
                                "",
                                "",
                                "",
                            )
                        )
                        rows += 1
    except BaseException:
        output.unlink(missing_ok=True)
        raise

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a private CSV for human yes/no adjudication of gold records "
            "currently labeled possible."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = create_adjudication_queue(args.paths, args.output)
    print(f"Created private adjudication queue: rows={rows}, output={args.output}")
    print("Allowed reviewer_label values: yes, no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
