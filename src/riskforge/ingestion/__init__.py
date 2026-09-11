"""Report ingestion subsystem (CSV/TSV/JSON/JSONL/XLSX/PDF → normalized records).

Boundary notes: this subsystem must not import normalization directly —
normalization is injected as a callable by the runtime composer.  The API
layer consumes only this package's public surface.
"""

from riskforge.ingestion.exceptions import (
    IngestionError,
    ReportParseError,
    ReportValidationError,
    UnsupportedFormatError,
)
from riskforge.ingestion.parsers import (
    PARSERS,
    SUPPORTED_FORMATS,
    dedupe_records,
    parse_csv,
    parse_json,
    parse_jsonl,
    parse_pdf,
    parse_report,
    parse_tsv,
    parse_xlsx,
)
from riskforge.ingestion.pipeline import IngestionPipeline

__all__ = [
    "PARSERS",
    "SUPPORTED_FORMATS",
    "IngestionError",
    "IngestionPipeline",
    "ReportParseError",
    "ReportValidationError",
    "UnsupportedFormatError",
    "dedupe_records",
    "parse_csv",
    "parse_json",
    "parse_jsonl",
    "parse_pdf",
    "parse_report",
    "parse_tsv",
    "parse_xlsx",
]
