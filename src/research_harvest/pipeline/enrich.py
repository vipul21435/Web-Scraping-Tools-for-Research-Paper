"""
Pulls structured facts out of an abstract.

The original scanned for roughly 200 keywords, took "the biggest number in any
sentence" as the dataset size, and glued together every sentence containing the
word "best" as the top model. All three are replaced here with rules that are
narrow enough to be right and small enough to test.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Any

from research_harvest.keywords import GROUP_OF, find_terms
from research_harvest.models import Article

# "1,234 patients", "n = 500", "a dataset of 2000 compounds".
_COHORT = re.compile(
    r"""(?:
          \b(?:n\s*=\s*)(?P<n>\d[\d,]*)
        | \b(?P<count>\d[\d,]*)\s+(?:patients|subjects|participants|samples|compounds|molecules|images|records|cases|sequences|drugs)\b
        | \b(?:dataset|cohort|corpus|database)\s+of\s+(?P<of>\d[\d,]*)\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# "accuracy of 94.2%", "AUC = 0.91", "F1-score of 0.88".
_METRIC = re.compile(
    r"\b(?P<name>accuracy|AUROC|AUC-ROC|AUC|F1[- ]?score|F1|precision|recall|sensitivity|specificity|R2|R\^2|RMSE|MAE)\b"
    r"\s*(?:of|was|is|:|=|reached|achieved|reaching)?\s*"
    r"(?P<value>\d{1,3}(?:\.\d+)?)\s*(?P<pct>%)?",
    re.IGNORECASE,
)

_METRIC_CANON = {
    "auroc": "AUC",
    "auc-roc": "AUC",
    "auc": "AUC",
    "f1-score": "F1",
    "f1 score": "F1",
    "f1score": "F1",
    "f1": "F1",
    "r^2": "R2",
    "r2": "R2",
}


def extract_dataset_size(text: str) -> int | None:
    """The largest plausible cohort size stated in the text, or None."""
    if not text:
        return None

    sizes: list[int] = []
    for match in _COHORT.finditer(text):
        raw = match.group("n") or match.group("count") or match.group("of")
        if not raw:
            continue
        try:
            value = int(raw.replace(",", ""))
        except ValueError:
            continue
        # A "dataset" of two is a typo or a coincidence; ten million is a year.
        if 2 <= value <= 50_000_000:
            sizes.append(value)

    return max(sizes) if sizes else None


def extract_headline_metric(text: str) -> dict[str, Any] | None:
    """The best reported score, normalised to a 0-1 fraction where it is a rate."""
    if not text:
        return None

    best: dict[str, Any] | None = None
    for match in _METRIC.finditer(text):
        raw_name = match.group("name").lower().replace("_", " ")
        name = _METRIC_CANON.get(raw_name, raw_name.upper())

        try:
            value = float(match.group("value"))
        except ValueError:
            continue

        is_rate = name in {"ACCURACY", "AUC", "F1", "PRECISION", "RECALL", "SENSITIVITY", "SPECIFICITY", "R2"}
        if match.group("pct") or (is_rate and value > 1):
            value = value / 100.0

        # Error measures have no fixed range, so they are recorded but not ranked.
        if is_rate and not 0.0 <= value <= 1.0:
            continue

        candidate = {"name": name, "value": round(value, 4), "rate": is_rate}
        if best is None or (candidate["rate"] and (not best["rate"] or candidate["value"] > best["value"])):
            best = candidate

    return best


def enrich_article(article: Article) -> Article:
    """Attach the method mentions, cohort size and headline score to a record."""
    searchable = f"{article.title}. {article.abstract}"
    article.models_mentioned = find_terms(searchable)
    article.dataset_size = extract_dataset_size(article.abstract)
    article.headline_metric = extract_headline_metric(article.abstract)
    return article


def enrich_articles(articles: Iterable[Article]) -> Iterator[Article]:
    for article in articles:
        yield enrich_article(article)


def method_groups(article: Article) -> set[str]:
    """Which families of method a record mentions, e.g. {"classical_ml"}."""
    return {GROUP_OF[term.casefold()] for term in article.models_mentioned if term.casefold() in GROUP_OF}
