"""
PubMed through the official E-utilities REST API.

Two calls do the work. `esearch` returns the PMIDs for a query, `efetch` returns
the full records for up to 200 of them at a time, so a year of results costs a
couple of requests.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from xml.etree import ElementTree

from research_harvest.models import Article
from research_harvest.sources.base import HttpClient, SourceError

logger = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI allows 3 requests/second anonymously and 10 with an API key.
RATE_WITHOUT_KEY = 3.0
RATE_WITH_KEY = 9.0

FETCH_BATCH = 200


def _text(node: ElementTree.Element | None) -> str:
    """All text under a node, including the tags PubMed uses for italics."""
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


class PubMedSource:
    """Searches PubMed and returns parsed articles."""

    name = "pubmed"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: HttpClient | None = None,
        email: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("NCBI_API_KEY") or None
        self.email = email or os.environ.get("NCBI_EMAIL") or None
        self.client = client or HttpClient(
            requests_per_second=RATE_WITH_KEY if self.api_key else RATE_WITHOUT_KEY
        )

    def _common_params(self) -> dict[str, str]:
        params = {"db": "pubmed", "tool": "research-harvest"}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return params

    def search(
        self,
        query: str,
        *,
        limit: int = 100,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> Iterator[Article]:
        pmids = self._search_ids(query, limit=limit, year_from=year_from, year_to=year_to)
        logger.info("PubMed matched %d record(s) for %r", len(pmids), query)

        for start in range(0, len(pmids), FETCH_BATCH):
            yield from self._fetch(pmids[start : start + FETCH_BATCH])

    def _search_ids(
        self,
        query: str,
        *,
        limit: int,
        year_from: int | None,
        year_to: int | None,
    ) -> list[str]:
        params = self._common_params()
        params.update({"term": query, "retmode": "json", "retmax": str(min(limit, 10000)), "sort": "relevance"})
        if year_from or year_to:
            params["datetype"] = "pdat"
            params["mindate"] = str(year_from or 1800)
            params["maxdate"] = str(year_to or 3000)

        response = self.client.get(f"{EUTILS}/esearch.fcgi", params=params)
        try:
            payload = response.json()
        except ValueError as exc:
            raise SourceError(f"PubMed returned a non-JSON search response: {exc}") from exc

        return list(payload.get("esearchresult", {}).get("idlist", []))[:limit]

    def _fetch(self, pmids: list[str]) -> Iterator[Article]:
        if not pmids:
            return

        params = self._common_params()
        params.update({"id": ",".join(pmids), "retmode": "xml"})
        response = self.client.get(f"{EUTILS}/efetch.fcgi", params=params)

        try:
            root = ElementTree.fromstring(response.content)
        except ElementTree.ParseError as exc:
            raise SourceError(f"PubMed returned XML that could not be parsed: {exc}") from exc

        for citation in root.iter("PubmedArticle"):
            article = self._parse(citation)
            if article is not None:
                yield article

    def _parse(self, citation: ElementTree.Element) -> Article | None:
        """
        Turn one PubmedArticle element into an Article.

        Every field is optional. Thousands of PubMed records have no abstract,
        plenty have no DOI, and some carry a collective author instead of a
        list of names, so nothing here assumes an element exists.
        """
        pmid = _text(citation.find(".//PMID"))
        title = _text(citation.find(".//ArticleTitle"))
        if not pmid or not title:
            logger.debug("Skipping a record with no PMID or no title")
            return None

        # Structured abstracts arrive as several labelled sections.
        parts: list[str] = []
        for chunk in citation.findall(".//Abstract/AbstractText"):
            label = chunk.get("Label")
            body = _text(chunk)
            if body:
                parts.append(f"{label}: {body}" if label else body)

        authors: list[str] = []
        for author in citation.findall(".//AuthorList/Author"):
            last = _text(author.find("LastName"))
            initials = _text(author.find("Initials"))
            collective = _text(author.find("CollectiveName"))
            if last:
                authors.append(f"{last} {initials}".strip())
            elif collective:
                authors.append(collective)

        doi = None
        for ident in citation.findall(".//ArticleId"):
            if ident.get("IdType") == "doi":
                doi = _text(ident)
                break

        return Article(
            title=title,
            source=self.name,
            source_id=pmid,
            abstract=" ".join(parts),
            authors=authors,
            year=self._year(citation),
            journal=_text(citation.find(".//Journal/Title")),
            doi=doi,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            keywords=[k for k in (_text(n) for n in citation.findall(".//KeywordList/Keyword")) if k],
        )

    @staticmethod
    def _year(citation: ElementTree.Element) -> int | None:
        """The publication year, from whichever of PubMed's date fields exists."""
        for path in (".//Journal/JournalIssue/PubDate/Year", ".//PubMedPubDate/Year", ".//ArticleDate/Year"):
            value = _text(citation.find(path))
            if value.isdigit():
                return int(value)

        # Some records only carry a MedlineDate such as "2019 Nov-Dec".
        medline = _text(citation.find(".//Journal/JournalIssue/PubDate/MedlineDate"))
        for token in medline.split():
            if len(token) == 4 and token.isdigit():
                return int(token)
        return None
