"""Report parsers: CSV, JSON, JSONL, XLSX, and PDF sources.

Each parser converts source bytes into ``IncidentRawRecord`` candidates.
Parsers are deliberately dumb: they map columns/fields onto the raw-record
schema and perform no semantic validation — that lives in the ingestion
pipeline.  PDF extraction is best-effort text reconstruction because
line-of-business safety forms rarely have machine-readable structure; its
limits are documented rather than hidden.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from riskforge.core.contracts import AssetType, IncidentRawRecord
from riskforge.ingestion.exceptions import ReportParseError

_REQUIRED_COLUMNS = ("log_id", "timestamp", "asset_id", "asset_type", "raw_narrative")

# Column synonyms accepted for tabular sources, mirroring the XLSX parser so
# every flat-file format handles the same real-world header variations.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "log_id": ("log_id", "report_id", "id"),
    "timestamp": ("timestamp", "date", "datetime", "reported_at"),
    "asset_id": ("asset_id", "location", "asset"),
    "asset_type": ("asset_type", "asset_category"),
    "raw_narrative": ("raw_narrative", "narrative", "description", "details"),
    "reporter_severity_rank": ("reporter_severity_rank", "severity", "severity_rank"),
}


def _canonical_header(name: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower()) if name is not None else ""


def _resolve_columns(headers: list[str]) -> dict[str, str] | None:
    """Map canonical column names onto actual headers, or None if a required one is missing."""
    mapping: dict[str, str] = {}
    for canonical_name, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in headers:
                mapping[canonical_name] = alias
                break
    missing = [column for column in _REQUIRED_COLUMNS if column not in mapping]
    if missing:
        raise ReportParseError(
            f"missing required column(s): {', '.join(sorted(missing))} "
            f"(accepted aliases: {', '.join(sorted({a for v in _COLUMN_ALIASES.values() for a in v}))})"
        )
    return mapping


def _coerce_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        raise ReportParseError("empty timestamp")
    normalized = text.replace("Z", "+00:00") if text.endswith("Z") else text
    parsed = datetime.fromisoformat(normalized)
    return parsed


def _coerce_asset_type(value: Any) -> AssetType:
    if isinstance(value, AssetType):
        return value
    return AssetType(str(value).strip().lower())


def _build_record(row: dict[str, Any], source: str) -> IncidentRawRecord:
    missing = [column for column in _REQUIRED_COLUMNS if row.get(column) in (None, "")]
    if missing:
        raise ReportParseError(
            f"{source}: missing required field(s): {', '.join(sorted(missing))}"
        )
    try:
        return IncidentRawRecord(
            log_id=str(row["log_id"]).strip(),
            timestamp=_coerce_timestamp(row["timestamp"]),
            asset_id=str(row["asset_id"]).strip(),
            asset_type=_coerce_asset_type(row["asset_type"]),
            raw_narrative=str(row["raw_narrative"]),
            reporter_severity_rank=(
                str(row["reporter_severity_rank"]).strip()
                if row.get("reporter_severity_rank") not in (None, "")
                else None
            ),
        )
    except ReportParseError:
        raise
    except Exception as error:
        raise ReportParseError(f"{source}: {error}") from error


def _apply_column_aliases(row: dict[str, Any]) -> dict[str, Any]:
    """Rename aliased keys (e.g. ``narrative``) to canonical contract names."""
    resolved: dict[str, Any] = dict(row)
    for canonical_name, aliases in _COLUMN_ALIASES.items():
        if canonical_name in resolved:
            continue
        for alias in aliases:
            if alias in resolved:
                resolved[canonical_name] = resolved[alias]
                break
    return resolved


def _iter_csv(text: str, delimiter: str, source: str) -> list[IncidentRawRecord]:
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        raise ReportParseError(f"{source}: no header row found")
    header_map = {_canonical_header(name): name for name in reader.fieldnames}
    mapping = _resolve_columns(list(header_map))
    records: list[IncidentRawRecord] = []
    for line_number, row in enumerate(reader, start=2):
        if all((value or "").strip() == "" for value in row.values()):
            continue
        normalized = {
            canonical: row.get(header_map.get(alias, "")) for canonical, alias in mapping.items()
        }
        try:
            records.append(_build_record(normalized, f"{source} line {line_number}"))
        except ReportParseError:
            raise
        except Exception as error:
            raise ReportParseError(f"{source} line {line_number}: {error}") from error
    if not records:
        raise ReportParseError(f"{source}: no data rows found")
    return records


def parse_csv(data: bytes) -> list[IncidentRawRecord]:
    """Parse comma-separated safety reports."""
    return _iter_csv(data.decode("utf-8-sig"), ",", "csv")


def parse_tsv(data: bytes) -> list[IncidentRawRecord]:
    """Parse tab-separated safety reports (Excel 'CSV (tab delimited)' export)."""
    return _iter_csv(data.decode("utf-8-sig"), "\t", "tsv")


def parse_json(data: bytes) -> list[IncidentRawRecord]:
    """Parse a JSON array of report objects."""
    try:
        payload = json.loads(data.decode("utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ReportParseError(f"json: {error}") from error
    if not isinstance(payload, list):
        raise ReportParseError("json: top-level value must be an array")
    records: list[IncidentRawRecord] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ReportParseError(f"json: entry {index} is not an object")
        records.append(_build_record(_apply_column_aliases(item), f"json entry {index}"))
    return records


def parse_jsonl(data: bytes) -> list[IncidentRawRecord]:
    """Parse newline-delimited JSON report objects."""
    records: list[IncidentRawRecord] = []
    for line_number, line in enumerate(data.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReportParseError(f"jsonl line {line_number}: {error}") from error
        if not isinstance(item, dict):
            raise ReportParseError(f"jsonl line {line_number}: entry is not an object")
        records.append(_build_record(_apply_column_aliases(item), f"jsonl line {line_number}"))
    if not records:
        raise ReportParseError("jsonl: no entries found")
    return records


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def parse_xlsx(data: bytes) -> list[IncidentRawRecord]:
    """Parse the first worksheet of an Excel workbook.

    Column headers are matched case-insensitively and a handful of common
    synonyms are accepted (``narrative``/``description`` for the report text,
    ``date``/``datetime`` for the timestamp).  Only the stdlib-openpyxl pair
    is used; no network or native code.
    """
    try:
        import openpyxl
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ReportParseError("xlsx support requires the openpyxl package") from error
    try:
        workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as error:
        raise ReportParseError(f"xlsx: {error}") from error
    sheet = workbook.worksheets[0]
    rows = sheet.iter_rows(values_only=True)
    header_row = next(rows, None)
    if header_row is None:
        raise ReportParseError("xlsx: worksheet is empty")

    def canonical(name: Any) -> str:
        return re.sub(r"[\s\-]+", "_", str(name).strip().lower()) if name is not None else ""

    headers = [canonical(cell) for cell in header_row]

    def resolve(*names: str) -> str | None:
        for name in names:
            if canonical(name) in headers:
                return canonical(name)
        return None

    column_log_id = resolve("log_id", "report_id", "id")
    column_timestamp = resolve("timestamp", "date", "datetime", "reported_at")
    column_asset_id = resolve("asset_id", "location", "asset")
    column_asset_type = resolve("asset_type", "asset_category")
    column_narrative = resolve("raw_narrative", "narrative", "description", "details")
    missing = [
        label
        for label, column in (
            ("log_id", column_log_id),
            ("timestamp", column_timestamp),
            ("asset_id", column_asset_id),
            ("asset_type", column_asset_type),
            ("raw_narrative", column_narrative),
        )
        if column is None
    ]
    if missing:
        raise ReportParseError(f"xlsx: missing required column(s): {', '.join(missing)}")

    records: list[IncidentRawRecord] = []
    for row_number, row in enumerate(rows, start=2):
        values = dict(zip(headers, row, strict=False))
        mapped = {
            "log_id": values.get(column_log_id),
            "timestamp": values.get(column_timestamp),
            "asset_id": values.get(column_asset_id),
            "asset_type": values.get(column_asset_type),
            "raw_narrative": values.get(column_narrative),
        }
        severity = _first_present(values, "reporter_severity_rank", "severity")
        if severity is not None:
            mapped["reporter_severity_rank"] = severity
        try:
            records.append(_build_record(mapped, f"xlsx row {row_number}"))
        except ReportParseError:
            raise
        except Exception as error:
            raise ReportParseError(f"xlsx row {row_number}: {error}") from error
    if not records:
        raise ReportParseError("xlsx: no data rows found")
    return records


def parse_pdf(data: bytes) -> list[IncidentRawRecord]:
    """Best-effort PDF extraction: one report per page.

    Safety-form PDFs are usually printed forms, so each page is treated as a
    single report.  Required fields are recovered from common label patterns
    (``Log ID:``, ``Date:``, ``Location:``, ``Asset Type:``); any remaining
    body text becomes the narrative.  Pages missing required fields raise a
    parse error naming the page — callers can re-export the form rather than
    silently losing the report.
    """
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ReportParseError("pdf support requires the pypdf package") from error
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as error:
        raise ReportParseError(f"pdf: {error}") from error

    import re

    label_patterns = {
        "log_id": re.compile(r"log\s*id\s*[:#-]?\s*([A-Za-z0-9._-]+)", re.IGNORECASE),
        "timestamp": re.compile(
            r"(?:date(?:\s*/?\s*time)?|reported(?:\s*on)?)\s*[:#-]?\s*"
            r"(\d{4}-\d{2}-\d{2}(?:[T ][0-9:.+-]+(?:Z|[+-]\d{2}:?\d{2})?)?)",
            re.IGNORECASE,
        ),
        "asset_id": re.compile(
            r"(?:location|asset(?:\s*id)?)\s*[:#-]?\s*([A-Za-z0-9._ -]{1,64})", re.IGNORECASE
        ),
        "asset_type": re.compile(
            r"asset\s*type\s*[:#-]?\s*([A-Za-z_]{1,40})", re.IGNORECASE
        ),
    }
    records: list[IncidentRawRecord] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:
            raise ReportParseError(f"pdf page {page_number}: extraction failed") from error
        fields: dict[str, str] = {}
        for key, pattern in label_patterns.items():
            match = pattern.search(text)
            if match:
                fields[key] = match.group(1).strip()
        narrative = text
        if fields:
            first_used = min(text.find(fields[key]) for key in fields if key in text)
            narrative = text[first_used:].strip()
        fields["raw_narrative"] = narrative
        missing = [key for key in _REQUIRED_COLUMNS if not fields.get(key)]
        if missing:
            raise ReportParseError(
                f"pdf page {page_number}: could not recover {', '.join(missing)}; "
                "re-export the form or provide a structured format"
            )
        records.append(_build_record(fields, f"pdf page {page_number}"))
    if not records:
        raise ReportParseError("pdf: no pages found")
    return records


PARSERS: dict[str, Callable[[bytes], list[IncidentRawRecord]]] = {
    "csv": parse_csv,
    "tsv": parse_tsv,
    "json": parse_json,
    "jsonl": parse_jsonl,
    "xlsx": parse_xlsx,
    "pdf": parse_pdf,
}

SUPPORTED_FORMATS: tuple[str, ...] = tuple(sorted(PARSERS))


def parse_report(data: bytes, fmt: str) -> list[IncidentRawRecord]:
    """Parse report bytes using the parser registered for *fmt*."""
    parser = PARSERS.get(fmt.strip().lower())
    if parser is None:
        raise ReportParseError(
            f"unsupported format {fmt!r}; supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    return parser(data)


def dedupe_records(records: Sequence[IncidentRawRecord]) -> list[IncidentRawRecord]:
    """Drop duplicate log_ids within one source, keeping the first."""
    seen: set[str] = set()
    unique: list[IncidentRawRecord] = []
    for record in records:
        if record.log_id in seen:
            continue
        seen.add(record.log_id)
        unique.append(record)
    return unique
