"""
Duplicate removal.

Searching two catalogues for one topic returns the same paper twice, so records
are collapsed on the strongest identifier they share: DOI first, then the
source's own id, then a normalised title. Where two copies describe the same
work, the more complete one is kept - a PubMed record with a structured abstract
beats an OpenAlex stub of the same paper.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from research_harvest.models import Article


@dataclass
class DedupeReport:
    """What the stage did, so a run can be explained afterwards."""

    kept: int = 0
    removed: int = 0
    by_doi: int = 0
    by_source_id: int = 0
    by_title: int = 0
    duplicate_titles: list[str] = field(default_factory=list)

    @property
    def seen(self) -> int:
        return self.kept + self.removed

    def summary(self) -> str:
        return (
            f"{self.seen} record(s) in, {self.kept} kept, {self.removed} duplicate(s) removed "
            f"(doi: {self.by_doi}, source id: {self.by_source_id}, title: {self.by_title})"
        )


def _completeness(article: Article) -> tuple[int, int, int, int]:
    """How much a record actually carries; bigger wins when two describe one work."""
    return (
        len(article.abstract),
        len(article.authors),
        1 if article.doi else 0,
        1 if article.year else 0,
    )


def deduplicate(articles: Iterable[Article]) -> tuple[list[Article], DedupeReport]:
    report = DedupeReport()
    kept: list[Article] = []
    index: dict[tuple[str, str], int] = {}

    for article in articles:
        keys: list[tuple[str, str]] = []
        if article.doi:
            keys.append(("doi", article.doi))
        keys.append(("source", f"{article.source}:{article.source_id}"))
        if article.normalised_title:
            keys.append(("title", article.normalised_title))

        position = next((index[key] for key in keys if key in index), None)

        if position is None:
            index.update({key: len(kept) for key in keys})
            kept.append(article)
            report.kept += 1
            continue

        matched_kind = next(kind for kind, value in keys if (kind, value) in index)
        if matched_kind == "doi":
            report.by_doi += 1
        elif matched_kind == "source":
            report.by_source_id += 1
        else:
            report.by_title += 1
        report.removed += 1
        report.duplicate_titles.append(article.title)

        # Prefer whichever copy carries more, then re-index under both records' keys.
        if _completeness(article) > _completeness(kept[position]):
            kept[position] = article
        index.update(dict.fromkeys(keys, position))

    return kept, report
