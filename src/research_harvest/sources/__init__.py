"""Literature sources. Each one turns a query into `Article` records."""

from research_harvest.sources.base import Source, SourceError
from research_harvest.sources.openalex import OpenAlexSource
from research_harvest.sources.pubmed import PubMedSource

SOURCES: dict[str, type[Source]] = {
    "pubmed": PubMedSource,
    "openalex": OpenAlexSource,
}

__all__ = ["Source", "SourceError", "PubMedSource", "OpenAlexSource", "SOURCES"]
