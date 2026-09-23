"""The writers."""

from __future__ import annotations

import csv
import json

import pytest

from research_harvest.export import COLUMNS, write_csv, write_jsonl, write_xlsx


@pytest.fixture
def articles(article_factory):
    from research_harvest.pipeline.enrich import enrich_article

    return [
        enrich_article(article_factory(source_id="1", year=2022, title="First paper")),
        enrich_article(article_factory(source_id="2", year=2023, title="Second paper", doi="10.1/b")),
    ]


class TestJsonl:
    def test_writes_one_object_per_line(self, articles, tmp_path):
        path = tmp_path / "out.jsonl"
        assert write_jsonl(articles, path) == 2

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["title"] == "First paper"

    def test_keeps_lists_as_lists(self, articles, tmp_path):
        path = tmp_path / "out.jsonl"
        write_jsonl(articles, path)
        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert isinstance(record["authors"], list)
        assert isinstance(record["models_mentioned"], list)

    def test_creates_missing_directories(self, articles, tmp_path):
        path = tmp_path / "nested" / "deep" / "out.jsonl"
        write_jsonl(articles, path)
        assert path.exists()

    def test_an_empty_run_writes_an_empty_file(self, tmp_path):
        path = tmp_path / "out.jsonl"
        assert write_jsonl([], path) == 0
        assert path.read_text(encoding="utf-8") == ""


class TestCsv:
    def test_writes_a_header_and_a_row_per_article(self, articles, tmp_path):
        path = tmp_path / "out.csv"
        assert write_csv(articles, path) == 2

        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 2
        assert list(rows[0].keys()) == list(COLUMNS)

    def test_flattens_lists_into_text(self, articles, tmp_path):
        path = tmp_path / "out.csv"
        write_csv(articles, path)
        with path.open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        assert row["authors"] == "Jha V; Sharma R"

    def test_survives_a_comma_and_a_newline_in_a_title(self, article_factory, tmp_path):
        path = tmp_path / "out.csv"
        write_csv([article_factory(title="Commas, and\nnewlines")], path)
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["title"] == "Commas, and newlines"


class TestXlsx:
    def test_writes_one_sheet_per_year(self, articles, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "out.xlsx"
        assert write_xlsx(articles, path) == 2

        book = openpyxl.load_workbook(path)
        assert sorted(book.sheetnames) == ["2022", "2023"]
        assert [c.value for c in book["2022"][1]] == list(COLUMNS)

    def test_can_write_a_single_sheet_instead(self, articles, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "out.xlsx"
        write_xlsx(articles, path, sheet_per_year=False)
        assert openpyxl.load_workbook(path).sheetnames == ["results"]

    def test_undated_records_get_their_own_sheet(self, article_factory, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "out.xlsx"
        write_xlsx([article_factory(year=None)], path)
        assert openpyxl.load_workbook(path).sheetnames == ["undated"]

    def test_an_empty_run_still_produces_a_readable_file(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "out.xlsx"
        assert write_xlsx([], path) == 0
        assert openpyxl.load_workbook(path).sheetnames == ["results"]
