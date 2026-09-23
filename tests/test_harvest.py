"""The orchestration layer."""

from __future__ import annotations

import pytest

from research_harvest.harvest import harvest
from research_harvest.models import Article
from research_harvest.pipeline import FilterSpec
from research_harvest.sources import SOURCES, SourceError


class FakeSource:
    """A source that returns whatever it was told to, without any HTTP."""

    name = "fake"
    payload: list[Article] = []
    error: str = ""

    def search(self, query, *, limit, year_from, year_to):
        if self.error:
            raise SourceError(self.error)
        return list(self.payload)[:limit]


@pytest.fixture
def fake_source(monkeypatch):
    monkeypatch.setitem(SOURCES, "fake", FakeSource)
    return FakeSource


class TestHarvest:
    def test_runs_every_stage(self, fake_source, article_factory):
        fake_source.payload = [article_factory(source_id="1")]
        fake_source.error = ""

        result = harvest("q", sources=["fake"])

        assert result.fetched == 1
        assert len(result.articles) == 1
        assert result.articles[0].models_mentioned == ["Random Forest"], "enrichment ran"

    def test_removes_duplicates_across_sources(self, fake_source, article_factory):
        fake_source.payload = [
            article_factory(source_id="1", doi="10.1/x"),
            article_factory(source_id="2", doi="10.1/x", title="Same paper, other wording"),
        ]
        fake_source.error = ""

        result = harvest("q", sources=["fake"])

        assert result.fetched == 2
        assert len(result.articles) == 1
        assert result.dedupe.removed == 1

    def test_applies_filters_and_counts_what_they_dropped(self, fake_source, article_factory):
        fake_source.payload = [
            article_factory(source_id="1", doi="10.1/a"),
            article_factory(source_id="2", doi="10.1/b", abstract="", title="No abstract"),
        ]
        fake_source.error = ""

        result = harvest("q", sources=["fake"], filters=FilterSpec(require_abstract=True))

        assert len(result.articles) == 1
        assert result.filtered_out == 1

    def test_a_failing_source_is_recorded_not_raised(self, fake_source):
        fake_source.payload = []
        fake_source.error = "unreachable"

        result = harvest("q", sources=["fake"])

        assert result.articles == []
        assert result.errors and "unreachable" in result.errors[0]

    def test_an_unknown_source_is_reported(self):
        result = harvest("q", sources=["nowhere"])
        assert result.errors and "Unknown source" in result.errors[0]

    def test_the_summary_describes_the_run(self, fake_source, article_factory):
        fake_source.payload = [article_factory(source_id="1")]
        fake_source.error = ""

        summary = harvest("q", sources=["fake"]).summary()
        assert "Fetched 1 record(s)" in summary
        assert "1 record(s) kept" in summary
