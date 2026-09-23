"""Pipeline stages. Each takes articles and returns articles."""

from research_harvest.pipeline.clean import clean_articles
from research_harvest.pipeline.dedupe import DedupeReport, deduplicate
from research_harvest.pipeline.enrich import enrich_articles
from research_harvest.pipeline.filters import FilterSpec, apply_filters

__all__ = [
    "clean_articles",
    "deduplicate",
    "DedupeReport",
    "enrich_articles",
    "apply_filters",
    "FilterSpec",
]
