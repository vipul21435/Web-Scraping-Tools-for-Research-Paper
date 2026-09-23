"""Cleaning, deduplication, enrichment and filtering."""

from __future__ import annotations

import pytest

from research_harvest.keywords import find_terms
from research_harvest.pipeline.clean import clean_abstract, clean_article, clean_text, clean_title
from research_harvest.pipeline.dedupe import deduplicate
from research_harvest.pipeline.enrich import (
    enrich_article,
    extract_dataset_size,
    extract_headline_metric,
    method_groups,
)
from research_harvest.pipeline.filters import FilterSpec, apply_filters


class TestClean:
    def test_unescapes_entities_and_strips_markup(self):
        assert clean_text("A &amp; B <i>italic</i>") == "A & B italic"

    def test_collapses_whitespace(self):
        assert clean_text("a\n\n  b\tc") == "a b c"

    def test_unwraps_a_bracketed_translated_title(self):
        assert clean_title("[Efeito da barreira].") == "Efeito da barreira"

    def test_leaves_an_ordinary_title_alone(self):
        assert clean_title("Deep learning in medicine") == "Deep learning in medicine"

    def test_removes_copyright_boilerplate_from_an_abstract(self):
        text = "Real findings here. © 2023 Elsevier Ltd. All rights reserved."
        assert clean_abstract(text) == "Real findings here."

    def test_keeps_an_abstract_that_has_no_boilerplate(self):
        assert clean_abstract("Just the science.") == "Just the science."

    def test_empty_input_stays_empty(self):
        assert clean_text("") == "" and clean_abstract("") == "" and clean_title("") == ""

    def test_cleaning_an_article_drops_blank_authors(self, article_factory):
        article = clean_article(article_factory(authors=["Jha V", "  ", ""]))
        assert article.authors == ["Jha V"]


class TestDedupe:
    def test_collapses_two_records_sharing_a_doi(self, article_factory):
        kept, report = deduplicate([
            article_factory(source="pubmed", source_id="1", doi="10.1/x"),
            article_factory(source="openalex", source_id="W1", doi="https://doi.org/10.1/X", title="Different wording"),
        ])
        assert len(kept) == 1
        assert report.by_doi == 1 and report.removed == 1

    def test_collapses_two_records_sharing_a_title(self, article_factory):
        kept, report = deduplicate([
            article_factory(source="pubmed", source_id="1", doi=None, title="Deep Learning: A Review"),
            article_factory(source="openalex", source_id="W1", doi=None, title="deep learning a review"),
        ])
        assert len(kept) == 1 and report.by_title == 1

    def test_keeps_genuinely_different_records(self, article_factory):
        kept, report = deduplicate([
            article_factory(source_id="1", doi="10.1/a", title="First paper"),
            article_factory(source_id="2", doi="10.1/b", title="Second paper"),
        ])
        assert len(kept) == 2 and report.removed == 0

    def test_prefers_the_copy_that_carries_more(self, article_factory):
        thin = article_factory(source="openalex", source_id="W1", doi="10.1/x", abstract="", authors=[])
        full = article_factory(source="pubmed", source_id="1", doi="10.1/x",
                               abstract="A full abstract with detail.", authors=["Jha V"])

        kept, _ = deduplicate([thin, full])
        assert len(kept) == 1
        assert kept[0].abstract == "A full abstract with detail."
        assert kept[0].source == "pubmed"

    def test_the_first_copy_wins_when_they_are_equally_complete(self, article_factory):
        first = article_factory(source="pubmed", source_id="1", doi="10.1/x")
        second = article_factory(source="openalex", source_id="W1", doi="10.1/x")
        kept, _ = deduplicate([first, second])
        assert kept[0].source == "pubmed"

    def test_the_same_record_twice_from_one_source_is_collapsed(self, article_factory):
        kept, report = deduplicate([
            article_factory(source="pubmed", source_id="1", doi=None),
            article_factory(source="pubmed", source_id="1", doi=None),
        ])
        assert len(kept) == 1 and report.removed == 1

    def test_an_empty_run_reports_nothing(self):
        kept, report = deduplicate([])
        assert kept == [] and report.seen == 0
        assert "0 record(s) in" in report.summary()

    def test_records_with_no_doi_are_not_treated_as_equal(self, article_factory):
        kept, _ = deduplicate([
            article_factory(source_id="1", doi=None, title="Paper one"),
            article_factory(source_id="2", doi=None, title="Paper two"),
        ])
        assert len(kept) == 2, "a missing DOI must not act as a shared key"


class TestKeywordMatching:
    def test_prefers_the_longest_name(self):
        assert "Convolutional Neural Network" in find_terms("We used a Convolutional Neural Network.")

    def test_does_not_match_inside_a_longer_word(self):
        assert find_terms("The organ was scanned.") == []
        assert find_terms("A cannon fired.") == []

    def test_matching_ignores_case(self):
        assert find_terms("we used an svm") == ["SVM"]

    def test_reports_each_method_once(self):
        assert find_terms("SVM and SVM and SVM") == ["SVM"]

    def test_finds_several_methods_in_order(self):
        assert find_terms("First XGBoost, then BERT.") == ["XGBoost", "BERT"]

    def test_empty_text_finds_nothing(self):
        assert find_terms("") == []


class TestDatasetSize:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("We studied 1,204 patients.", 1204),
            ("The cohort (n = 512) was split.", 512),
            ("A dataset of 8153 compounds was used.", 8153),
            ("We used 340 images and 90 records.", 340),
        ],
    )
    def test_reads_a_stated_cohort_size(self, text, expected):
        assert extract_dataset_size(text) == expected

    def test_ignores_text_with_no_cohort(self):
        assert extract_dataset_size("The barrier is selective.") is None
        assert extract_dataset_size("") is None

    def test_ignores_implausible_values(self):
        # The version this replaced took the largest number in any sentence,
        # so a p-value or a year became the "dataset size".
        assert extract_dataset_size("Published in 2021 with p < 0.001.") is None


class TestHeadlineMetric:
    @pytest.mark.parametrize(
        "text,name,value",
        [
            ("an accuracy of 94.2%", "ACCURACY", 0.942),
            ("AUC = 0.91", "AUC", 0.91),
            ("F1-score of 0.88", "F1", 0.88),
            ("accuracy was 88", "ACCURACY", 0.88),
        ],
    )
    def test_reads_and_normalises_a_score(self, text, name, value):
        metric = extract_headline_metric(text)
        assert metric["name"] == name
        assert metric["value"] == pytest.approx(value)

    def test_picks_the_best_of_several(self):
        metric = extract_headline_metric("accuracy of 80% but AUC = 0.95")
        assert metric["value"] == pytest.approx(0.95)

    def test_returns_nothing_when_no_score_is_reported(self):
        assert extract_headline_metric("We discuss the mechanism.") is None
        assert extract_headline_metric("") is None

    def test_discards_a_rate_outside_zero_to_one(self):
        assert extract_headline_metric("accuracy of 4000") is None


class TestEnrich:
    def test_fills_in_every_derived_field(self, article_factory):
        article = enrich_article(article_factory())
        assert article.models_mentioned == ["Random Forest"]
        assert article.dataset_size == 1200
        assert article.headline_metric["name"] == "ACCURACY"

    def test_searches_the_title_as_well_as_the_abstract(self, article_factory):
        article = enrich_article(article_factory(title="A CNN approach", abstract="No methods named."))
        assert "CNN" in article.models_mentioned

    def test_reports_the_families_a_record_belongs_to(self, article_factory):
        article = enrich_article(article_factory(abstract="We used XGBoost and a CNN."))
        assert method_groups(article) == {"classical_ml", "architecture"}


class TestFilters:
    def test_no_spec_keeps_everything(self, article_factory):
        articles = [article_factory(source_id="1"), article_factory(source_id="2")]
        assert apply_filters(articles, None) == articles

    def test_requires_an_abstract(self, article_factory):
        articles = [article_factory(source_id="1"), article_factory(source_id="2", abstract="")]
        assert len(apply_filters(articles, FilterSpec(require_abstract=True))) == 1

    def test_requires_a_doi(self, article_factory):
        articles = [article_factory(source_id="1"), article_factory(source_id="2", doi=None)]
        assert len(apply_filters(articles, FilterSpec(require_doi=True))) == 1

    def test_filters_by_year_range(self, article_factory):
        articles = [article_factory(source_id=str(y), year=y) for y in (2015, 2020, 2024)]
        kept = apply_filters(articles, FilterSpec(year_from=2018, year_to=2022))
        assert [a.year for a in kept] == [2020]

    def test_a_record_with_no_year_fails_a_year_filter(self, article_factory):
        assert apply_filters([article_factory(year=None)], FilterSpec(year_from=2000)) == []

    def test_filters_by_required_term(self, article_factory):
        articles = [
            enrich_article(article_factory(source_id="1", abstract="We used an SVM.")),
            enrich_article(article_factory(source_id="2", abstract="We used a CNN.")),
        ]
        kept = apply_filters(articles, FilterSpec(must_mention=["SVM"]))
        assert [a.source_id for a in kept] == ["1"]

    def test_filters_by_method_family(self, article_factory):
        articles = [
            enrich_article(article_factory(source_id="1", abstract="A Random Forest model.")),
            enrich_article(article_factory(source_id="2", abstract="A Transformer model.")),
        ]
        kept = apply_filters(articles, FilterSpec(method_group="classical_ml"))
        assert [a.source_id for a in kept] == ["1"]

    def test_filters_by_minimum_cohort_size(self, article_factory):
        small = enrich_article(article_factory(source_id="1", abstract="We studied 50 patients."))
        large = enrich_article(article_factory(source_id="2", abstract="We studied 5000 patients."))
        kept = apply_filters([small, large], FilterSpec(min_dataset_size=1000))
        assert [a.source_id for a in kept] == ["2"]

    def test_filters_by_a_title_pattern(self, article_factory):
        articles = [
            article_factory(source_id="1", title="Deep learning for BBB"),
            article_factory(source_id="2", title="A clinical trial"),
        ]
        kept = apply_filters(articles, FilterSpec(title_matches=r"deep\s+learning"))
        assert [a.source_id for a in kept] == ["1"]

    def test_several_conditions_all_have_to_hold(self, article_factory):
        article = enrich_article(article_factory(year=2023, abstract="An SVM on 5000 samples."))
        assert apply_filters([article], FilterSpec(year_from=2020, must_mention=["SVM"], min_dataset_size=1000))
        assert not apply_filters([article], FilterSpec(year_from=2020, must_mention=["CNN"]))
