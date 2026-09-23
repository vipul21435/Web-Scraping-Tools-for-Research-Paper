"""The command line, with the network stubbed out at the source boundary."""

from __future__ import annotations

import json

import pytest

from research_harvest import cli
from research_harvest.harvest import HarvestResult
from research_harvest.pipeline.enrich import enrich_article


@pytest.fixture
def stub_harvest(monkeypatch, article_factory):
    """Replaces the harvest call so the CLI is tested without any HTTP."""
    calls: dict = {}

    def fake_harvest(query, **kwargs):
        calls["query"] = query
        calls.update(kwargs)
        result = HarvestResult()
        result.articles = [
            enrich_article(article_factory(source_id="1", title="First paper", year=2022)),
            enrich_article(article_factory(source_id="2", title="Second paper", year=2023)),
        ]
        result.fetched = 2
        return result

    monkeypatch.setattr(cli, "harvest", fake_harvest)
    return calls


class TestArguments:
    def test_query_is_required(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])

    def test_defaults_are_sensible(self):
        args = cli.build_parser().parse_args(["a query"])
        assert args.source == ["pubmed"]
        assert args.limit == 100
        assert args.format == "jsonl"

    def test_rejects_an_unknown_source(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["q", "--source", "google-scholar"])

    def test_rejects_an_unknown_method_group(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["q", "--method-group", "telepathy"])


class TestRun:
    def test_writes_jsonl_to_stdout_by_default(self, stub_harvest, capsys):
        assert cli.main(["a query", "--quiet"]) == 0

        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["title"] == "First paper"

    def test_writes_a_file_when_asked(self, stub_harvest, tmp_path, capsys):
        out = tmp_path / "results.csv"
        assert cli.main(["a query", "--format", "csv", "-o", str(out), "--quiet"]) == 0
        assert out.exists()
        assert "First paper" in out.read_text(encoding="utf-8")

    def test_writes_an_excel_workbook(self, stub_harvest, tmp_path):
        pytest.importorskip("openpyxl")
        out = tmp_path / "results.xlsx"
        assert cli.main(["a query", "--format", "xlsx", "-o", str(out), "--quiet"]) == 0
        assert out.stat().st_size > 0

    def test_passes_filters_through(self, stub_harvest):
        cli.main(["a query", "--year-from", "2019", "--require-abstract",
                  "--must-mention", "SVM", "CNN", "--quiet"])

        assert stub_harvest["year_from"] == 2019
        spec = stub_harvest["filters"]
        assert spec.require_abstract is True
        assert spec.must_mention == ["SVM", "CNN"]

    def test_passes_the_source_list_through(self, stub_harvest):
        cli.main(["a query", "--source", "pubmed", "openalex", "--quiet"])
        assert stub_harvest["sources"] == ["pubmed", "openalex"]

    def test_reports_a_summary_on_stderr_not_stdout(self, stub_harvest, capsys):
        cli.main(["a query"])
        captured = capsys.readouterr()
        assert "record(s) kept" in captured.err
        assert "record(s) kept" not in captured.out, "stdout must stay pipeable"

    def test_rejects_a_limit_below_one(self, capsys):
        assert cli.main(["q", "--limit", "0"]) == 2
        assert "--limit must be at least 1" in capsys.readouterr().err

    def test_rejects_a_reversed_year_range(self, capsys):
        assert cli.main(["q", "--year-from", "2020", "--year-to", "2010"]) == 2
        assert "cannot be later than" in capsys.readouterr().err

    def test_reports_failure_when_a_source_errored_and_nothing_came_back(self, monkeypatch):
        def failing(query, **kwargs):
            result = HarvestResult()
            result.errors = ["pubmed: unreachable"]
            return result

        monkeypatch.setattr(cli, "harvest", failing)
        assert cli.main(["q", "--quiet"]) == 1

    def test_finding_nothing_is_a_success(self, monkeypatch):
        monkeypatch.setattr(cli, "harvest", lambda query, **kwargs: HarvestResult())
        assert cli.main(["q", "--quiet"]) == 0
