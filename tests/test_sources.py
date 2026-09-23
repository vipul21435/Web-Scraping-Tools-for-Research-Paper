"""The API clients, driven entirely from recorded responses."""

from __future__ import annotations

import pytest
import responses

from research_harvest.sources.base import HttpClient, RateLimiter, SourceError
from research_harvest.sources.openalex import OpenAlexSource
from research_harvest.sources.pubmed import EUTILS, PubMedSource

OPENALEX_API = "https://api.openalex.org/works"


@pytest.fixture
def fast_client() -> HttpClient:
    """No pacing and no backoff, so the suite does not sit waiting."""
    return HttpClient(requests_per_second=0, max_retries=2, timeout=1)


class TestRateLimiter:
    def test_a_rate_of_zero_disables_waiting(self):
        assert RateLimiter(0).min_interval == 0

    def test_interval_is_the_inverse_of_the_rate(self):
        assert RateLimiter(4).min_interval == pytest.approx(0.25)


class TestHttpClient:
    @responses.activate
    def test_retries_a_server_error_then_succeeds(self, fast_client):
        responses.add(responses.GET, "https://example.test/x", status=503)
        responses.add(responses.GET, "https://example.test/x", json={"ok": True}, status=200)

        assert fast_client.get("https://example.test/x").json() == {"ok": True}
        assert len(responses.calls) == 2

    @responses.activate
    def test_gives_up_after_the_retry_budget(self, fast_client):
        responses.add(responses.GET, "https://example.test/x", status=500)
        responses.add(responses.GET, "https://example.test/x", status=500)

        with pytest.raises(SourceError, match="Giving up"):
            fast_client.get("https://example.test/x")

    @responses.activate
    def test_does_not_retry_a_client_error(self, fast_client):
        responses.add(responses.GET, "https://example.test/x", status=400)

        with pytest.raises(SourceError, match="HTTP 400"):
            fast_client.get("https://example.test/x")
        assert len(responses.calls) == 1, "a 400 will fail identically forever"

    @responses.activate
    def test_sends_an_identifying_user_agent(self, fast_client):
        responses.add(responses.GET, "https://example.test/x", json={}, status=200)
        fast_client.get("https://example.test/x")
        assert "research-harvest" in responses.calls[0].request.headers["User-Agent"]


class TestPubMedSource:
    @responses.activate
    def test_parses_recorded_records(self, fast_client, pubmed_esearch, pubmed_efetch):
        responses.add(responses.GET, f"{EUTILS}/esearch.fcgi", body=pubmed_esearch, content_type="application/json")
        responses.add(responses.GET, f"{EUTILS}/efetch.fcgi", body=pubmed_efetch, content_type="application/xml")

        articles = list(PubMedSource(client=fast_client).search("blood brain barrier", limit=10))

        assert len(articles) == 3
        for article in articles:
            assert article.source == "pubmed"
            assert article.title
            assert article.source_id.isdigit()
            assert article.url.endswith(f"/{article.source_id}/")
            assert isinstance(article.authors, list)

    @responses.activate
    def test_reads_years_dois_and_abstracts(self, fast_client, pubmed_esearch, pubmed_efetch):
        responses.add(responses.GET, f"{EUTILS}/esearch.fcgi", body=pubmed_esearch, content_type="application/json")
        responses.add(responses.GET, f"{EUTILS}/efetch.fcgi", body=pubmed_efetch, content_type="application/xml")

        articles = list(PubMedSource(client=fast_client).search("q", limit=10))

        assert any(a.year and 1990 < a.year < 2100 for a in articles)
        assert any(a.doi for a in articles)
        assert any(a.has_abstract for a in articles)

    @responses.activate
    def test_honours_the_limit(self, fast_client, pubmed_esearch, pubmed_efetch):
        responses.add(responses.GET, f"{EUTILS}/esearch.fcgi", body=pubmed_esearch, content_type="application/json")
        responses.add(responses.GET, f"{EUTILS}/efetch.fcgi", body=pubmed_efetch, content_type="application/xml")

        list(PubMedSource(client=fast_client).search("q", limit=1))
        fetched = responses.calls[1].request.url
        assert fetched.count(",") == 0, "only one id should be fetched"

    @responses.activate
    def test_sends_a_date_range_when_years_are_given(self, fast_client, pubmed_esearch, pubmed_efetch):
        responses.add(responses.GET, f"{EUTILS}/esearch.fcgi", body=pubmed_esearch, content_type="application/json")
        responses.add(responses.GET, f"{EUTILS}/efetch.fcgi", body=pubmed_efetch, content_type="application/xml")

        list(PubMedSource(client=fast_client).search("q", limit=5, year_from=2019, year_to=2021))

        url = responses.calls[0].request.url
        assert "mindate=2019" in url and "maxdate=2021" in url and "datetype=pdat" in url

    @responses.activate
    def test_no_results_is_not_an_error(self, fast_client):
        responses.add(
            responses.GET, f"{EUTILS}/esearch.fcgi",
            json={"esearchresult": {"idlist": []}}, status=200,
        )
        assert list(PubMedSource(client=fast_client).search("nothing", limit=10)) == []

    @responses.activate
    def test_a_record_without_a_title_is_skipped_not_fatal(self, fast_client):
        responses.add(
            responses.GET, f"{EUTILS}/esearch.fcgi",
            json={"esearchresult": {"idlist": ["1", "2"]}}, status=200,
        )
        responses.add(
            responses.GET, f"{EUTILS}/efetch.fcgi",
            body=(
                "<PubmedArticleSet>"
                "<PubmedArticle><MedlineCitation><PMID>1</PMID></MedlineCitation></PubmedArticle>"
                "<PubmedArticle><MedlineCitation><PMID>2</PMID>"
                "<Article><ArticleTitle>A real title</ArticleTitle></Article>"
                "</MedlineCitation></PubmedArticle>"
                "</PubmedArticleSet>"
            ),
            content_type="application/xml",
        )

        articles = list(PubMedSource(client=fast_client).search("q", limit=10))
        assert [a.source_id for a in articles] == ["2"]

    @responses.activate
    def test_a_record_with_no_abstract_still_comes_back(self, fast_client):
        # The version this replaced raised AttributeError here and lost the run.
        responses.add(responses.GET, f"{EUTILS}/esearch.fcgi", json={"esearchresult": {"idlist": ["7"]}})
        responses.add(
            responses.GET, f"{EUTILS}/efetch.fcgi",
            body=(
                "<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>7</PMID>"
                "<Article><ArticleTitle>No abstract here</ArticleTitle></Article>"
                "</MedlineCitation></PubmedArticle></PubmedArticleSet>"
            ),
            content_type="application/xml",
        )

        article = next(iter(PubMedSource(client=fast_client).search("q", limit=1)))
        assert article.title == "No abstract here"
        assert article.abstract == ""

    @responses.activate
    def test_keeps_the_labels_of_a_structured_abstract(self, fast_client):
        responses.add(responses.GET, f"{EUTILS}/esearch.fcgi", json={"esearchresult": {"idlist": ["8"]}})
        responses.add(
            responses.GET, f"{EUTILS}/efetch.fcgi",
            body=(
                "<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>8</PMID><Article>"
                "<ArticleTitle>Structured</ArticleTitle><Abstract>"
                '<AbstractText Label="METHODS">We used an SVM.</AbstractText>'
                '<AbstractText Label="RESULTS">It worked.</AbstractText>'
                "</Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
            ),
            content_type="application/xml",
        )

        article = next(iter(PubMedSource(client=fast_client).search("q", limit=1)))
        assert article.abstract == "METHODS: We used an SVM. RESULTS: It worked."

    @responses.activate
    def test_falls_back_to_a_medline_date_for_the_year(self, fast_client):
        responses.add(responses.GET, f"{EUTILS}/esearch.fcgi", json={"esearchresult": {"idlist": ["9"]}})
        responses.add(
            responses.GET, f"{EUTILS}/efetch.fcgi",
            body=(
                "<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>9</PMID><Article>"
                "<ArticleTitle>Dated oddly</ArticleTitle>"
                "<Journal><JournalIssue><PubDate><MedlineDate>2019 Nov-Dec</MedlineDate></PubDate></JournalIssue></Journal>"
                "</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
            ),
            content_type="application/xml",
        )

        assert next(iter(PubMedSource(client=fast_client).search("q", limit=1))).year == 2019

    @responses.activate
    def test_unparseable_xml_is_reported_clearly(self, fast_client):
        responses.add(responses.GET, f"{EUTILS}/esearch.fcgi", json={"esearchresult": {"idlist": ["1"]}})
        responses.add(responses.GET, f"{EUTILS}/efetch.fcgi", body="<not xml", content_type="application/xml")

        with pytest.raises(SourceError, match="could not be parsed"):
            list(PubMedSource(client=fast_client).search("q", limit=1))

    def test_uses_the_faster_rate_when_an_api_key_is_present(self):
        with_key = PubMedSource(api_key="abc")
        without = PubMedSource(api_key=None)
        assert with_key.client.limiter.min_interval < without.client.limiter.min_interval


class TestOpenAlexSource:
    @responses.activate
    def test_parses_recorded_works(self, fast_client, openalex_works):
        responses.add(responses.GET, OPENALEX_API, json=openalex_works, status=200)

        articles = list(OpenAlexSource(client=fast_client).search("bbb", limit=10))

        assert len(articles) == 3
        for article in articles:
            assert article.source == "openalex"
            assert article.title
            assert article.source_id.startswith("W")

    @responses.activate
    def test_rebuilds_abstracts_from_the_inverted_index(self, fast_client, openalex_works):
        responses.add(responses.GET, OPENALEX_API, json=openalex_works, status=200)

        articles = list(OpenAlexSource(client=fast_client).search("bbb", limit=10))
        assert any(a.has_abstract for a in articles)
        assert all("[" not in a.abstract for a in articles)

    @responses.activate
    def test_stops_once_the_limit_is_reached(self, fast_client, openalex_works):
        responses.add(responses.GET, OPENALEX_API, json=openalex_works, status=200)

        assert len(list(OpenAlexSource(client=fast_client).search("bbb", limit=2))) == 2

    @responses.activate
    def test_sends_publication_date_filters(self, fast_client, openalex_works):
        responses.add(responses.GET, OPENALEX_API, json=openalex_works, status=200)

        list(OpenAlexSource(client=fast_client).search("bbb", limit=1, year_from=2015, year_to=2020))

        url = responses.calls[0].request.url
        assert "from_publication_date" in url and "to_publication_date" in url

    @responses.activate
    def test_an_empty_page_ends_the_run(self, fast_client):
        responses.add(responses.GET, OPENALEX_API, json={"results": [], "meta": {}}, status=200)
        assert list(OpenAlexSource(client=fast_client).search("nothing", limit=50)) == []

    @responses.activate
    def test_a_work_without_a_title_is_skipped(self, fast_client):
        responses.add(
            responses.GET, OPENALEX_API,
            json={"results": [{"id": "https://openalex.org/W1"}, {"id": "https://openalex.org/W2", "display_name": "Real"}],
                  "meta": {"next_cursor": None}},
            status=200,
        )
        articles = list(OpenAlexSource(client=fast_client).search("q", limit=10))
        assert [a.source_id for a in articles] == ["W2"]

    def test_rebuilding_an_abstract_orders_by_position(self):
        inverted = {"learning": [1], "Machine": [0], "works": [2]}
        assert OpenAlexSource._abstract(inverted) == "Machine learning works"

    def test_rebuilding_handles_a_repeated_word(self):
        assert OpenAlexSource._abstract({"the": [0, 2], "cat": [1], "sat": [3]}) == "the cat the sat"

    def test_rebuilding_an_absent_index_gives_an_empty_string(self):
        assert OpenAlexSource._abstract(None) == ""
        assert OpenAlexSource._abstract({}) == ""
