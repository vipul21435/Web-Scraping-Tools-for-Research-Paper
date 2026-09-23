"""
OpenAlex through its official REST API.

OpenAlex is an open catalogue of over 250 million works with a documented API
and no key. It is the second source here because Google Scholar, which covers
similar ground, publishes no API and blocks automated access.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

from research_harvest.models import Article
from research_harvest.sources.base import HttpClient, SourceError

logger = logging.getLogger(__name__)

API = "https://api.openalex.org/works"
PER_PAGE = 200


class OpenAlexSource:
    """Searches OpenAlex and returns parsed articles."""

    name = "openalex"

    def __init__(
        self,
        *,
        mailto: str | None = None,
        client: HttpClient | None = None,
    ) -> None:
        # Supplying a contact address puts you in OpenAlex's faster pool.
        self.mailto = mailto or os.environ.get("OPENALEX_MAILTO") or None
        self.client = client or HttpClient(requests_per_second=5.0)

    def search(
        self,
        query: str,
        *,
        limit: int = 100,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> Iterator[Article]:
        filters = []
        if year_from:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if year_to:
            filters.append(f"to_publication_date:{year_to}-12-31")

        cursor = "*"
        returned = 0

        # Cursor paging, because OpenAlex refuses offset paging past 10,000 results.
        while returned < limit and cursor:
            params: dict[str, Any] = {
                "search": query,
                "per-page": str(min(PER_PAGE, limit - returned)),
                "cursor": cursor,
            }
            if filters:
                params["filter"] = ",".join(filters)
            if self.mailto:
                params["mailto"] = self.mailto

            response = self.client.get(API, params=params)
            try:
                payload = response.json()
            except ValueError as exc:
                raise SourceError(f"OpenAlex returned a non-JSON response: {exc}") from exc

            results = payload.get("results", [])
            if not results:
                break

            for record in results:
                article = self._parse(record)
                if article is not None:
                    yield article
                    returned += 1
                    if returned >= limit:
                        return

            cursor = payload.get("meta", {}).get("next_cursor")

    def _parse(self, record: dict[str, Any]) -> Article | None:
        title = (record.get("display_name") or "").strip()
        work_id = (record.get("id") or "").rsplit("/", 1)[-1]
        if not title or not work_id:
            return None

        authorships = record.get("authorships") or []
        authors = [
            name
            for name in ((a.get("author") or {}).get("display_name") for a in authorships)
            if name
        ]

        location = record.get("primary_location") or {}
        venue = (location.get("source") or {}).get("display_name") or ""

        return Article(
            title=title,
            source=self.name,
            source_id=work_id,
            abstract=self._abstract(record.get("abstract_inverted_index")),
            authors=authors,
            year=record.get("publication_year"),
            journal=venue,
            doi=record.get("doi"),
            url=record.get("doi") or record.get("id") or "",
            keywords=[
                name
                for name in ((c.get("display_name")) for c in (record.get("concepts") or [])[:8])
                if name
            ],
        )

    @staticmethod
    def _abstract(inverted: dict[str, list[int]] | None) -> str:
        """
        Rebuild an abstract from OpenAlex's inverted index.

        It stores `{"word": [positions]}` rather than the text, for licensing
        reasons, so the words have to be put back in position order.
        """
        if not inverted:
            return ""

        positions: list[tuple[int, str]] = [
            (position, word) for word, spots in inverted.items() for position in spots
        ]
        if not positions:
            return ""

        positions.sort()
        return " ".join(word for _, word in positions)
