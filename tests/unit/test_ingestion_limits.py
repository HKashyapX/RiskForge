"""Tests for format-aware ingestion resource limits.

Proves that row bombs, spreadsheet expansion bombs, oversized sources,
deep-nested JSON, page bombs, and giant lines are refused with
``ResourceLimitError`` before parsing does unbounded work.
"""

from __future__ import annotations

import json

import pytest

from riskforge.ingestion.exceptions import ResourceLimitError
from riskforge.ingestion.limits import (
    check_json_depth,
    check_line_size,
    check_pdf_pages,
    check_record_count,
    check_source,
    check_xlsx_expansion,
)
from riskforge.ingestion.parsers import parse_json, parse_jsonl, parse_report


class TestCheckFunctions:
    def test_oversized_source_rejected(self) -> None:
        with pytest.raises(ResourceLimitError, match="ingest limit"):
            check_source(b"x" * (21 * 1024 * 1024), fmt="csv")

    def test_row_bomb_rejected(self) -> None:
        with pytest.raises(ResourceLimitError, match="per-document limit"):
            check_record_count(50_001, fmt="csv")

    def test_giant_line_rejected(self) -> None:
        with pytest.raises(ResourceLimitError, match="line limit"):
            check_line_size("x" * 2_000_000, fmt="jsonl", line_number=7)

    def test_deep_json_rejected(self) -> None:
        deep: object = []
        for _ in range(20):
            deep = [deep]
        with pytest.raises(ResourceLimitError, match="depth"):
            check_json_depth(deep, fmt="json")

    def test_page_bomb_rejected(self) -> None:
        with pytest.raises(ResourceLimitError, match="page limit"):
            check_pdf_pages(500)

    def test_xlsx_cell_bomb_rejected(self) -> None:
        with pytest.raises(ResourceLimitError, match="cell"):
            check_xlsx_expansion(b"PK" + b"0" * 64, sheet_cells=10_000_000)

    def test_env_limits_are_positive(self, monkeypatch) -> None:
        monkeypatch.setenv("RISKFORGE_INGEST_LIMIT_MAX_RECORDS", "0")
        from riskforge.ingestion.limits import _Limits

        with pytest.raises(ResourceLimitError, match="positive"):
            _Limits()

    def test_env_limits_apply(self, monkeypatch) -> None:
        monkeypatch.setenv("RISKFORGE_INGEST_LIMIT_MAX_RECORDS", "2")
        from riskforge.ingestion.limits import _Limits

        assert _Limits().max_records == 2


def _record_line(log_id: str) -> str:
    return json.dumps(
        {
            "log_id": log_id,
            "timestamp": "2026-03-01T08:00:00Z",
            "asset_id": "RIG_01",
            "asset_type": "drilling_rig",
            "raw_narrative": "Routine inspection completed.",
        }
    )


class TestParserEnforcement:
    def test_jsonl_row_bomb_refused(self, monkeypatch) -> None:
        monkeypatch.setenv("RISKFORGE_INGEST_LIMIT_MAX_RECORDS", "3")
        data = ("\n".join(_record_line(f"L{i}") for i in range(5)) + "\n").encode()
        with pytest.raises(ResourceLimitError, match="per-document limit"):
            parse_jsonl(data)

    def test_json_depth_bomb_refused(self) -> None:
        payload: object = {"log_id": "L1"}
        for _ in range(20):
            payload = {"nested": payload}
        data = json.dumps([payload]).encode()
        with pytest.raises(ResourceLimitError, match="depth"):
            parse_json(data)

    def test_normal_documents_parse_within_limits(self) -> None:
        data = ("\n".join(_record_line(f"L{i}") for i in range(3)) + "\n").encode()
        assert len(parse_jsonl(data)) == 3

    def test_oversized_document_refused_before_parsing(self, monkeypatch) -> None:
        monkeypatch.setenv("RISKFORGE_INGEST_LIMIT_MAX_BYTES", "1024")
        with pytest.raises(ResourceLimitError, match="ingest limit"):
            parse_report(b"x" * 2048, fmt="csv")
        # Even an unsupported format is refused at the size gate first.
        with pytest.raises(ResourceLimitError):
            parse_report(b"x" * 2048, fmt="docx")
