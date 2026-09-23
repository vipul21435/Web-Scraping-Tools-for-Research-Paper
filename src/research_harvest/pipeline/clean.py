"""Text tidying, applied before anything tries to read meaning out of a record."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Iterator

from research_harvest.models import Article

_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

# PubMed titles routinely arrive wrapped in brackets (translated titles) and
# with a trailing full stop that no other source adds.
_BRACKETED = re.compile(r"^\[(.+?)\]\.?$")

# Copyright and funding boilerplate carries no signal and skews keyword counts.
_BOILERPLATE = re.compile(
    r"(©|\bCopyright\b|\bAll rights reserved\b|\bThis article is protected by copyright\b).*$",
    re.IGNORECASE | re.DOTALL,
)


def clean_text(value: str) -> str:
    """Unescape entities, drop markup, and collapse whitespace."""
    if not value:
        return ""
    text = html.unescape(value)
    text = _TAG.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def clean_title(title: str) -> str:
    cleaned = clean_text(title)
    match = _BRACKETED.match(cleaned)
    if match:
        cleaned = match.group(1).strip()
    return cleaned.rstrip(".").strip() if cleaned.endswith(".") else cleaned


def clean_abstract(abstract: str) -> str:
    cleaned = clean_text(abstract)
    return _BOILERPLATE.sub("", cleaned).strip()


def clean_article(article: Article) -> Article:
    article.title = clean_title(article.title)
    article.abstract = clean_abstract(article.abstract)
    article.authors = [clean_text(a) for a in article.authors if clean_text(a)]
    article.keywords = [clean_text(k) for k in article.keywords if clean_text(k)]
    article.journal = clean_text(article.journal)
    return article


def clean_articles(articles: Iterable[Article]) -> Iterator[Article]:
    for article in articles:
        yield clean_article(article)
