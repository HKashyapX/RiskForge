"""Parser and pipeline tests for the ingestion subsystem."""

from __future__ import annotations

import io

import openpyxl
import pytest

from riskforge.ingestion.exceptions import ReportParseError
from riskforge.ingestion.parsers import (
    dedupe_records,
    parse_csv,
    parse_json,
    parse_jsonl,
    parse_tsv,
    parse_xlsx,
)


class TestCsvParser:
    def test_parses_rows_with_all_columns(self):
        data = (
            b"log_id,timestamp,asset_id,asset_type,raw_narrative,reporter_severity_rank\n"
            b"A-1,2026-03-01T08:00:00Z,RIG_01,drilling_rig,Worker fell from ladder,high\n"
        )
        records = parse_csv(data)
        assert len(records) == 1
        record = records[0]
        assert record.log_id == "A-1"
        assert record.asset_id == "RIG_01"
        assert record.reporter_severity_rank == "high"
        assert record.timestamp.year == 2026

    def test_missing_required_column_raises_with_column_names(self):
        data = b"log_id,timestamp\nA-1,2026-03-01T08:00:00Z\n"
        with pytest.raises(ReportParseError, match="asset_id"):
            parse_csv(data)

    def test_malformed_timestamp_raises(self):
        data = (
            b"log_id,timestamp,asset_id,asset_type,raw_narrative\n"
            b"A-1,not-a-date,RIG_01,drilling_rig,text\n"
        )
        with pytest.raises(ReportParseError, match="csv line 2"):
            parse_csv(data)

    def test_blank_rows_are_skipped(self):
        data = (
            b"log_id,timestamp,asset_id,asset_type,raw_narrative\n"
            b"A-1,2026-03-01T08:00:00Z,RIG_01,drilling_rig,text\n"
            b",,,,\n"
            b"A-2,2026-03-02T08:00:00Z,RIG_01,drilling_rig,text two\n"
        )
        assert [r.log_id for r in parse_csv(data)] == ["A-1", "A-2"]

    def test_unknown_asset_type_raises(self):
        data = (
            b"log_id,timestamp,asset_id,asset_type,raw_narrative\n"
            b"A-1,2026-03-01T08:00:00Z,RIG_01,hovercraft,text\n"
        )
        with pytest.raises(ReportParseError):
            parse_csv(data)


class TestTsvParser:
    def test_parses_tab_separated_rows(self):
        data = (
            b"log_id\ttimestamp\tasset_id\tasset_type\traw_narrative\n"
            b"T-1\t2026-03-01T08:00:00Z\tRIG_01\tdrilling_rig\ttext\n"
        )
        assert parse_tsv(data)[0].log_id == "T-1"


class TestJsonParsers:
    def test_json_array(self):
        data = b'[{"log_id":"J-1","timestamp":"2026-03-01T08:00:00Z","asset_id":"RIG_01","asset_type":"drilling_rig","raw_narrative":"text"}]'
        assert parse_json(data)[0].log_id == "J-1"

    def test_json_rejects_non_array(self):
        with pytest.raises(ReportParseError, match="array"):
            parse_json(b'{"log_id": "J-1"}')

    def test_jsonl_multiple_lines(self):
        data = (
            b'{"log_id":"L-1","timestamp":"2026-03-01T08:00:00Z","asset_id":"RIG_01","asset_type":"drilling_rig","raw_narrative":"a"}\n'
            b'{"log_id":"L-2","timestamp":"2026-03-02T08:00:00Z","asset_id":"RIG_01","asset_type":"drilling_rig","raw_narrative":"b"}\n'
        )
        assert [r.log_id for r in parse_jsonl(data)] == ["L-1", "L-2"]

    def test_jsonl_bad_line_raises_with_line_number(self):
        data = (
            b'{"log_id":"L-1","timestamp":"2026-03-01T08:00:00Z","asset_id":"RIG_01","asset_type":"drilling_rig","raw_narrative":"a"}\n'
            b"not json\n"
        )
        with pytest.raises(ReportParseError, match="line 2"):
            parse_jsonl(data)


class TestXlsxParser:
    def _workbook_bytes(self, headers: list[str], rows: list[list]) -> bytes:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(headers)
        for row in rows:
            sheet.append(row)
        buffer = io.BytesIO()
        workbook.save(buffer)
        return buffer.getvalue()

    def test_parses_canonical_headers(self):
        data = self._workbook_bytes(
            ["log_id", "timestamp", "asset_id", "asset_type", "raw_narrative"],
            [["X-1", "2026-03-01", "RIG_01", "drilling_rig", "text"]],
        )
        assert parse_xlsx(data)[0].log_id == "X-1"

    def test_parses_synonym_headers_with_spaces(self):
        data = self._workbook_bytes(
            ["Report ID", "Date", "Location", "Asset Type", "Description"],
            [["X-1", "2026-03-01", "RIG_01", "drilling_rig", "narrative text"]],
        )
        record = parse_xlsx(data)[0]
        assert record.log_id == "X-1"
        assert record.raw_narrative == "narrative text"

    def test_missing_columns_raise_with_names(self):
        data = self._workbook_bytes(["Report ID", "Date"], [["X-1", "2026-03-01"]])
        with pytest.raises(ReportParseError, match="asset_id"):
            parse_xlsx(data)


class TestDedupe:
    def test_keeps_first_of_duplicate_log_ids(self):
        from datetime import UTC, datetime

        from riskforge.core.contracts import AssetType, IncidentRawRecord

        timestamp = datetime(2026, 3, 1, tzinfo=UTC)
        first = IncidentRawRecord(
            log_id="D-1", timestamp=timestamp, asset_id="A", asset_type=AssetType.DRILLING_RIG,
            raw_narrative="first",
        )
        second = IncidentRawRecord(
            log_id="D-1", timestamp=timestamp, asset_id="A", asset_type=AssetType.DRILLING_RIG,
            raw_narrative="second",
        )
        unique = dedupe_records([first, second])
        assert len(unique) == 1
        assert unique[0].raw_narrative == "first"
