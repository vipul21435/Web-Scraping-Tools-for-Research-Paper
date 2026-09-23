"""The single record shape every source produces and every stage passes on."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Punctuation and casing differ between sources for the same paper, so titles are
# reduced to comparable letters and digits before they are matched.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise_title(title: str) -> str:
    """A title reduced to the form used for duplicate detection."""
    return _NON_ALNUM.sub(" ", title.casefold()).strip()


def normalise_doi(doi: str | None) -> str | None:
    """A bare lowercase DOI, with any resolver prefix removed."""
    if not doi:
        return None
    cleaned = doi.strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned.strip() or None


@dataclass
class Article:
    """One scholarly work, however it was found."""

    title: str
    source: str
    source_id: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str = ""
    doi: str | None = None
    url: str = ""
    keywords: list[str] = field(default_factory=list)

    # Filled in by the enrichment stage rather than by a source.
    models_mentioned: list[str] = field(default_factory=list)
    dataset_size: int | None = None
    headline_metric: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.title = " ".join(self.title.split())
        self.abstract = " ".join(self.abstract.split())
        self.doi = normalise_doi(self.doi)

    @property
    def normalised_title(self) -> str:
        return normalise_title(self.title)

    @property
    def has_abstract(self) -> bool:
        return bool(self.abstract.strip())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_row(self) -> dict[str, Any]:
        """A flat form suitable for CSV and Excel, where lists cannot be stored."""
        row = self.to_dict()
        row["authors"] = "; ".join(self.authors)
        row["keywords"] = "; ".join(self.keywords)
        row["models_mentioned"] = "; ".join(self.models_mentioned)
        metric = self.headline_metric
        row["headline_metric"] = f"{metric['name']} {metric['value']}" if metric else ""
        return row
