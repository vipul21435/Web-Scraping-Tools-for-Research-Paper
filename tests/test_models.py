from research_harvest.models import Article, normalise_doi, normalise_title


class TestNormaliseDoi:
    def test_strips_resolver_prefixes_and_case(self):
        assert normalise_doi("https://doi.org/10.1234/ABC") == "10.1234/abc"
        assert normalise_doi("http://doi.org/10.1234/abc") == "10.1234/abc"
        assert normalise_doi("doi:10.1234/ABC") == "10.1234/abc"
        assert normalise_doi("  10.1234/abc  ") == "10.1234/abc"

    def test_empty_values_become_none(self):
        assert normalise_doi(None) is None
        assert normalise_doi("") is None
        assert normalise_doi("   ") is None


class TestNormaliseTitle:
    def test_ignores_punctuation_and_case(self):
        assert normalise_title("Deep Learning: A Review!") == normalise_title("deep learning a review")

    def test_collapses_runs_of_separators(self):
        assert normalise_title("A --- B") == "a b"


class TestArticle:
    def test_collapses_whitespace_on_construction(self, article_factory):
        article = article_factory(title="  Spaced   out\n title ", abstract="a\n\nb")
        assert article.title == "Spaced out title"
        assert article.abstract == "a b"

    def test_normalises_its_doi(self, article_factory):
        assert article_factory(doi="https://doi.org/10.1/X").doi == "10.1/x"

    def test_reports_whether_an_abstract_is_present(self, article_factory):
        assert article_factory().has_abstract
        assert not article_factory(abstract="").has_abstract
        assert not article_factory(abstract="   ").has_abstract

    def test_row_form_flattens_every_list(self, article_factory):
        article = article_factory()
        article.models_mentioned = ["SVM", "CNN"]
        article.headline_metric = {"name": "AUC", "value": 0.91, "rate": True}

        row = article.to_row()
        assert row["authors"] == "Jha V; Sharma R"
        assert row["models_mentioned"] == "SVM; CNN"
        assert row["headline_metric"] == "AUC 0.91"
        assert all(not isinstance(value, (list, dict)) for value in row.values())

    def test_row_form_copes_with_a_missing_metric(self, article_factory):
        assert article_factory().to_row()["headline_metric"] == ""

    def test_defaults_are_not_shared_between_instances(self):
        first = Article(title="A", source="s", source_id="1")
        second = Article(title="B", source="s", source_id="2")
        first.authors.append("Someone")
        assert second.authors == []
