"""Ties the sources and the stages together into one run."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from research_harvest.models import Article
from research_harvest.pipeline import (
    DedupeReport,
    FilterSpec,
    apply_filters,
    clean_articles,
    deduplicate,
    enrich_articles,
)
from research_harvest.sources import SOURCES, SourceError

logger = logging.getLogger(__name__)


@dataclass
class HarvestResult:
    """The articles, plus enough detail to explain how many were dropped and why."""

    articles: list[Article] = field(default_factory=list)
    fetched: int = 0
    dedupe: DedupeReport | None = None
    filtered_out: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Fetched {self.fetched} record(s)."]
        if self.dedupe:
            lines.append(self.dedupe.summary() + ".")
        if self.filtered_out:
            lines.append(f"{self.filtered_out} record(s) removed by filters.")
        lines.append(f"{len(self.articles)} record(s) kept.")
        for error in self.errors:
            lines.append(f"Warning: {error}")
        return " ".join(lines)


def harvest(
    query: str,
    *,
    sources: Sequence[str] = ("pubmed",),
    limit_per_source: int = 100,
    year_from: int | None = None,
    year_to: int | None = None,
    filters: FilterSpec | None = None,
) -> HarvestResult:
    """
    Run the whole pipeline: fetch, clean, deduplicate, enrich, filter.

    One source failing does not abandon the run - the others still return, and
    the failure is recorded on the result.
    """
    result = HarvestResult()
    collected: list[Article] = []

    for name in sources:
        source_class = SOURCES.get(name)
        if source_class is None:
            result.errors.append(f"Unknown source {name!r}; known sources are {', '.join(sorted(SOURCES))}.")
            continue

        try:
            found = list(
                source_class().search(
                    query, limit=limit_per_source, year_from=year_from, year_to=year_to
                )
            )
        except SourceError as exc:
            logger.error("Source %s failed: %s", name, exc)
            result.errors.append(f"{name}: {exc}")
            continue

        logger.info("%s returned %d record(s)", name, len(found))
        collected.extend(found)

    result.fetched = len(collected)

    cleaned = clean_articles(collected)
    deduped, report = deduplicate(cleaned)
    result.dedupe = report

    enriched = list(enrich_articles(deduped))

    kept = apply_filters(enriched, filters)
    result.filtered_out = len(enriched) - len(kept)
    result.articles = kept

    return result
