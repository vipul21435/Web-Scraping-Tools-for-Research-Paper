# research-harvest

**Literature harvesting for systematic reviews - search, deduplicate, and pull the numbers out of the abstracts.**

Give it a topic. It searches PubMed and OpenAlex, merges the results, removes the
papers that appear in both, and extracts what each abstract actually reports:
which methods were used, how large the cohort was, and the headline score.

[![CI](https://github.com/vipul21435/Web-Scraping-Tools-for-Research-Paper/actions/workflows/ci.yml/badge.svg)](https://github.com/vipul21435/Web-Scraping-Tools-for-Research-Paper/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.13-blue)
![Tests](https://img.shields.io/badge/tests-115-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

```bash
research-harvest "blood brain barrier machine learning" \
    --source pubmed openalex --limit 200 --require-abstract \
    --format xlsx -o out/bbb.xlsx
```

```
INFO PubMed matched 200 record(s) for 'blood brain barrier machine learning'
INFO openalex returned 200 record(s)
Fetched 400 record(s). 400 record(s) in, 361 kept, 39 duplicate(s) removed
(doi: 37, source id: 0, title: 2). 12 record(s) removed by filters. 349 record(s) kept.
Wrote 349 record(s) to out/bbb.xlsx
```

---

## What it produces

One row per paper, with the derived columns filled in from the abstract:

| title | year | models_mentioned | dataset_size | headline_metric |
|---|---|---|---|---|
| Machine learning based dynamic consensus model for predicting blood-brain barrier permeability | 2023 | XGBoost; Random Forest | 8153 | ACCURACY 0.978 |
| MATH: A Deep Learning Approach in QSAR for Estrogen Receptor Alpha Inhibitors | 2023 | QSAR; Transformer | - | AUC 0.977 |
| Exploring blood-brain barrier passage using atomic weighted vector and machine learning | 2024 | Gradient Boosting; SVM | - | ACCURACY 0.98 |

Those are real rows from a real run, not illustrations.

## Why it was rewritten

The first version drove a headless Chrome browser over the PubMed and Google
Scholar web pages, opened every article in turn, and slept five seconds between
each one. It worked, slowly, until it did not:

| Then | Now |
|---|---|
| Selenium + a 17 MB `chromedriver.exe` committed twice to the repo | Two official REST APIs, no browser, no binaries |
| `sleep(5)` between every article, hundreds of page loads per year of results | Batched `efetch` - a year of results in a couple of requests |
| Crashed with `AttributeError` on the first paper without an abstract | Every field optional; a thin record still comes back |
| Google Scholar, which blocks automated access - the committed `scholar.xlsx` was empty | OpenAlex, an open catalogue of 250M+ works with a documented free API |
| "Biggest number in any sentence" as the dataset size | Cohort patterns (`n = 512`, `1,204 patients`) with a plausibility range |
| Every sentence containing "best" glued together as the top model | A metric parser that normalises `94.2%`, `AUC = 0.91` and `F1 of 0.88` |
| ~200 hand-listed keywords with repeats, frameworks mixed in with architectures | 114 grouped terms, deduplicated, longest-match-first, word-boundary safe |
| No deduplication, though two sources return the same paper | DOI -> source id -> normalised title, keeping the fuller copy |
| Hardcoded years, hardcoded page counts, bare `input()` | A proper CLI with filters |
| No tests, no packaging, no CI | 115 tests, 95% coverage, packaged, linted, CI on 3.9-3.13 |

## Install

```bash
git clone https://github.com/vipul21435/Web-Scraping-Tools-for-Research-Paper.git
cd Web-Scraping-Tools-for-Research-Paper

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # or: pip install -e ".[excel]" for xlsx only
```

Python 3.9 or newer. No API key needed for either source, and no browser.

## Use

```bash
# The basics - JSONL to stdout, so it pipes into jq
research-harvest "drug discovery deep learning" --limit 50 | jq -r .title

# Both catalogues, recent work only, straight to a spreadsheet
research-harvest "blood brain barrier permeability" \
    --source pubmed openalex --year-from 2018 --require-abstract \
    --format xlsx -o out/bbb.xlsx

# Only papers that used a classical ML method on a decent-sized cohort
research-harvest "QSAR" --method-group classical_ml --min-dataset-size 500

# Only papers that name both of these
research-harvest "molecular property prediction" --must-mention Transformer "Graph Neural Network"
```

### Options

| Flag | What it does |
|---|---|
| `--source` | `pubmed`, `openalex`, or both (default: `pubmed`) |
| `--limit` | maximum records **per source** (default: 100) |
| `--year-from` / `--year-to` | restrict publication years at the API, not after the fact |
| `--require-abstract` / `--require-doi` | drop records missing either |
| `--must-mention` | keep only records naming every listed term |
| `--method-group` | `architecture`, `language_model`, `classical_ml`, `unsupervised`, `reinforcement`, `domain` |
| `--min-dataset-size` | keep only records reporting at least this cohort size |
| `--title-matches` | keep only titles matching a regular expression |
| `--format` | `jsonl` (default), `csv`, `xlsx` |
| `-o` | output path; without it, JSONL goes to stdout |

Progress and summaries go to stderr, results to stdout, so `research-harvest ... | jq`
works as you would expect.

### As a library

```python
from research_harvest.harvest import harvest
from research_harvest.pipeline import FilterSpec

result = harvest(
    "blood brain barrier machine learning",
    sources=["pubmed", "openalex"],
    limit_per_source=200,
    filters=FilterSpec(require_abstract=True, min_dataset_size=100),
)

print(result.summary())
for article in result.articles:
    print(article.year, article.title, article.models_mentioned)
```

## How it works

```
  query
    |
    +-- sources/pubmed.py      esearch -> efetch, batched 200 at a time
    +-- sources/openalex.py    cursor paging, abstracts rebuilt from the inverted index
    |
    v
  pipeline/clean.py     entities, markup, bracketed titles, copyright boilerplate
    v
  pipeline/dedupe.py    DOI -> source id -> normalised title; fuller copy wins
    v
  pipeline/enrich.py    method mentions, cohort size, headline metric
    v
  pipeline/filters.py   year, abstract, DOI, terms, method family, cohort size
    v
  export.py             jsonl | csv | xlsx (one sheet per year)
```

Each source is just an object with a `search()` method returning `Article`
records, so adding Crossref or Semantic Scholar means writing one file and adding
one line to `sources/__init__.py`.

### Being a good citizen

Both APIs are free and neither asks for a key, which is worth not abusing. Every
request goes through a rate limiter - 3/second for PubMed, 9/second if you set
`NCBI_API_KEY`, 5/second for OpenAlex - and retries with exponential backoff on
429s and 5xx, but never on a 4xx that would fail identically forever. Set
`NCBI_EMAIL` or `OPENALEX_MAILTO` to identify yourself and land in the faster pool.

## Tests

```bash
pytest                                   # 115 tests, ~4 seconds
pytest --cov=research_harvest            # 95% coverage
ruff check .
```

Nothing in the suite touches the network. The API clients are driven from
recorded PubMed XML and OpenAlex JSON in `tests/fixtures/`, so the tests are fast
and stay honest when NCBI is down.

## Licence

[MIT](LICENSE).
