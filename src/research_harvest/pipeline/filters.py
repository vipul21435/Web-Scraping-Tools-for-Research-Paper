"""Query-driven filtering, applied after enrichment so it can use what was found."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from research_harvest.models import Article
from research_harvest.pipeline.enrich import method_groups


@dataclass
class FilterSpec:
    """Everything a run can narrow results by. Unset fields do not filter."""

    year_from: int | None = None
    year_to: int | None = None
    require_abstract: bool = False
    require_doi: bool = False
    must_mention: list[str] = field(default_factory=list)
    method_group: str | None = None
    min_dataset_size: int | None = None
    title_matches: str | None = None

    def __post_init__(self) -> None:
        self._title_pattern = re.compile(self.title_matches, re.IGNORECASE) if self.title_matches else None
        self._required = [term.casefold() for term in self.must_mention]

    def matches(self, article: Article) -> bool:
        if self.year_from and (article.year is None or article.year < self.year_from):
            return False
        if self.year_to and (article.year is None or article.year > self.year_to):
            return False
        if self.require_abstract and not article.has_abstract:
            return False
        if self.require_doi and not article.doi:
            return False
        if self.min_dataset_size and (article.dataset_size or 0) < self.min_dataset_size:
            return False
        if self.method_group and self.method_group not in method_groups(article):
            return False
        if self._title_pattern and not self._title_pattern.search(article.title):
            return False

        if self._required:
            haystack = f"{article.title} {article.abstract}".casefold()
            mentioned = {term.casefold() for term in article.models_mentioned}
            for term in self._required:
                if term not in mentioned and term not in haystack:
                    return False

        return True


def apply_filters(articles: Iterable[Article], spec: FilterSpec | None) -> list[Article]:
    if spec is None:
        return list(articles)
    return [article for article in articles if spec.matches(article)]
