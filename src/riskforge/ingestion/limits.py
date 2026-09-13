"""Format-aware resource limits for report ingestion.

Every parser declares how dangerous its format can be, and
:func:`check_source` refuses inputs that would expand into unbounded work
*before* parsing: row bombs (CSV/JSONL), spreadsheet expansion bombs (XLSX),
decompression bombs (any container format), and page bombs (PDF).  JSON
nesting depth and single-record size are bounded too.

The default limits are calibrated for realistic OIL safety-report uploads
(a few thousand rows at most); operators can tighten them through
``RISKFORGE_INGEST_LIMIT_*`` environment variables but never disable them.
"""

from __future__ import annotations

import os

from riskforge.ingestion.exceptions import ResourceLimitError

# ── Default limits ────────────────────────────────────────────────────
DEFAULT_MAX_RECORDS = 5_000
DEFAULT_MAX_BYTES = 20 * 1024 * 1024  # transport cap; runtime may be tighter
DEFAULT_MAX_JSON_DEPTH = 8
DEFAULT_MAX_LINE_BYTES = 1_048_576  # 1 MiB per JSONL line / per CSV row
DEFAULT_MAX_PDF_PAGES = 200
DEFAULT_MAX_XLSX_EXPANSION_RATIO = 100  # uncompressed XML vs compressed bytes
DEFAULT_MAX_XLSX_CELLS = 5_000_000


class _Limits:
    """Resolved, positive limit values; never zero or negative."""

    def __init__(self) -> None:
        self.max_records = _env_int("RISKFORGE_INGEST_LIMIT_MAX_RECORDS", DEFAULT_MAX_RECORDS)
        self.max_bytes = _env_int("RISKFORGE_INGEST_LIMIT_MAX_BYTES", DEFAULT_MAX_BYTES)
        self.max_json_depth = _env_int(
            "RISKFORGE_INGEST_LIMIT_MAX_JSON_DEPTH", DEFAULT_MAX_JSON_DEPTH
        )
        self.max_line_bytes = _env_int(
            "RISKFORGE_INGEST_LIMIT_MAX_LINE_BYTES", DEFAULT_MAX_LINE_BYTES
        )
        self.max_pdf_pages = _env_int(
            "RISKFORGE_INGEST_LIMIT_MAX_PDF_PAGES", DEFAULT_MAX_PDF_PAGES
        )
        self.max_xlsx_expansion_ratio = _env_int(
            "RISKFORGE_INGEST_LIMIT_MAX_XLSX_EXPANSION_RATIO",
            DEFAULT_MAX_XLSX_EXPANSION_RATIO,
        )
        self.max_xlsx_cells = _env_int(
            "RISKFORGE_INGEST_LIMIT_MAX_XLSX_CELLS", DEFAULT_MAX_XLSX_CELLS
        )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ResourceLimitError(f"{name} must be an integer") from error
    if value < 1:
        raise ResourceLimitError(f"{name} must be positive")
    return value


def limits() -> _Limits:
    """Resolve limits from the environment (fresh per call; cheap dict reads)."""
    return _Limits()


def check_source(data: bytes, *, fmt: str) -> None:
    """Refuse source bytes that violate the format's resource limits."""
    cap = limits().max_bytes
    if len(data) > cap:
        raise ResourceLimitError(
            f"{fmt}: document exceeds the {cap // (1024 * 1024)} MiB ingest limit"
        )


def check_record_count(count: int, *, fmt: str) -> None:
    """Refuse parsed record volumes above the per-document cap."""
    max_records = limits().max_records
    if count > max_records:
        raise ResourceLimitError(
            f"{fmt}: {count} reports exceed the {max_records} per-document limit"
        )


def check_line_size(line: bytes | str, *, fmt: str, line_number: int) -> None:
    cap = limits().max_line_bytes
    if len(line) > cap:
        raise ResourceLimitError(
            f"{fmt} line {line_number}: exceeds the {cap}-byte line limit"
        )


def check_json_depth(node: object, *, fmt: str) -> None:
    def depth(value: object) -> int:
        if isinstance(value, dict):
            return 1 + max((depth(v) for v in value.values()), default=0)
        if isinstance(value, list):
            return 1 + max((depth(v) for v in value), default=0)
        return 0

    max_depth = limits().max_json_depth
    if depth(node) > max_depth:
        raise ResourceLimitError(
            f"{fmt}: nesting exceeds the {max_depth}-level depth limit"
        )


def check_pdf_pages(page_count: int, *, fmt: str = "pdf") -> None:
    max_pages = limits().max_pdf_pages
    if page_count > max_pages:
        raise ResourceLimitError(
            f"{fmt}: {page_count} pages exceed the {max_pages}-page limit"
        )


def check_xlsx_expansion(data: bytes, *, sheet_cells: int) -> None:
    """Refuse spreadsheet-expansion bombs before materializing rows.

    XLSX is a ZIP of XML; a small file can expand enormously.  Compare the
    declared cell volume against the compressed size and enforce an absolute
    cell ceiling.
    """
    cap = limits()
    if sheet_cells > cap.max_xlsx_cells:
        raise ResourceLimitError(
            f"xlsx: {sheet_cells} cells exceed the {cap.max_xlsx_cells}-cell limit"
        )
    compressed = max(len(data), 1)
    estimated = sheet_cells * 32  # conservative per-cell XML overhead
    if estimated // compressed > cap.max_xlsx_expansion_ratio and estimated > cap.max_xlsx_cells:
        raise ResourceLimitError(
            "xlsx: worksheet expansion exceeds the safety ratio; refusing to parse"
        )
