"""Shared fixtures. Nothing in the suite touches the network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_harvest.models import Article

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def pubmed_esearch() -> str:
    return (FIXTURES / "pubmed_esearch.json").read_text(encoding="utf-8")


@pytest.fixture
def pubmed_efetch() -> str:
    return (FIXTURES / "pubmed_efetch.xml").read_text(encoding="utf-8")


@pytest.fixture
def openalex_works() -> dict:
    return json.loads((FIXTURES / "openalex_works.json").read_text(encoding="utf-8"))


def make_article(**overrides) -> Article:
    """An article with sensible defaults, so a test only states what it cares about."""
    defaults = {
        "title": "Machine learning for blood brain barrier permeability",
        "source": "pubmed",
        "source_id": "12345",
        "abstract": "We trained a Random Forest on 1,200 compounds and reached an accuracy of 91.5%.",
        "authors": ["Jha V", "Sharma R"],
        "year": 2023,
        "journal": "Journal of Testing",
        "doi": "10.1234/test.2023",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
    }
    defaults.update(overrides)
    return Article(**defaults)


@pytest.fixture
def article_factory():
    return make_article
