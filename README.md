# research-harvest

Searches PubMed and OpenAlex for a topic, merges the results, throws away the papers that turn up in both, and reads what each abstract actually reports: which methods were used, how big the cohort was, and the best score claimed.

[![CI](https://github.com/vipul21435/Web-Scraping-Tools-for-Research-Paper/actions/workflows/ci.yml/badge.svg)](https://github.com/vipul21435/Web-Scraping-Tools-for-Research-Paper/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%20to%203.13-blue)
![Tests](https://img.shields.io/badge/tests-115-brightgreen)
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

## What you get

One row per paper. The last three columns are read out of the abstract rather than copied from a field:

| title | year | models_mentioned | dataset_size | headline_metric |
|---|---|---|---|---|
| Machine learning based dynamic consensus model for predicting blood-brain barrier permeability | 2023 | XGBoost; Random Forest | 8153 | ACCURACY 0.978 |
| MATH: A Deep Learning Approach in QSAR for Estrogen Receptor Alpha Inhibitors | 2023 | QSAR; Transformer | | AUC 0.977 |
| Exploring blood-brain barrier passage using atomic weighted vector and machine learning | 2024 | Gradient Boosting; SVM | | ACCURACY 0.98 |

Those rows came out of an actual run. The blanks are papers that never state a cohort size, which is common.

## Install

```bash
git clone https://github.com/vipul21435/Web-Scraping-Tools-for-Research-Paper.git
cd Web-Scraping-Tools-for-Research-Paper

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.9 or newer. Neither API needs a key.

## Usage

```bash
# JSONL on stdout, so it pipes
research-harvest "drug discovery deep learning" --limit 50 | jq -r .title

# both sources, recent work, into a spreadsheet
research-harvest "blood brain barrier permeability" \
    --source pubmed openalex --year-from 2018 --require-abstract \
    --format xlsx -o out/bbb.xlsx

# only classical ML papers with a decent cohort
research-harvest "QSAR" --method-group classical_ml --min-dataset-size 500

# only papers that name both of these
research-harvest "molecular property prediction" --must-mention Transformer "Graph Neural Network"
```

| Flag | Does |
|---|---|
| `--source` | `pubmed`, `openalex`, or both. Default is pubmed |
| `--limit` | cap per source, default 100 |
| `--year-from` / `--year-to` | filtered at the API, not afterwards |
| `--require-abstract`, `--require-doi` | drop records missing either |
| `--must-mention` | keep records naming every term listed |
| `--method-group` | architecture, language_model, classical_ml, unsupervised, reinforcement, domain |
| `--min-dataset-size` | keep records reporting at least this cohort |
| `--title-matches` | regex against the title |
| `--format` | jsonl, csv, xlsx |
| `-o` | output path; without it JSONL goes to stdout |

Logs go to stderr and results to stdout, so piping into jq works.

From Python:

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

## How it fits together

A query goes to the sources, which turn API responses into `Article` records. Then it runs through four stages in `pipeline/`: clean strips entities and markup and the copyright boilerplate at the end of abstracts, dedupe collapses records, enrich reads the methods and numbers, filter narrows. Whatever survives goes to `export.py`.

Deduplication matches on DOI first, then the source's own id, then a normalised title. When two records describe the same paper it keeps the fuller one, so a PubMed record with a structured abstract beats an OpenAlex stub of the same work.

A source is just an object with a `search()` method returning `Article` records, so adding Crossref or Semantic Scholar is one file and one line in `sources/__init__.py`.

### Rate limits

Both APIs are free and neither asks for a key, so it is worth not hammering them. Every request goes through a limiter: 3 a second for PubMed, 9 if you set `NCBI_API_KEY`, 5 for OpenAlex. It retries 429s and 5xx with backoff and does not retry a 4xx, since that will fail the same way forever. Set `NCBI_EMAIL` or `OPENALEX_MAILTO` to identify yourself and land in the faster pool.

## Tests

```bash
pytest                          # 115 tests, about 4 seconds
pytest --cov=research_harvest   # 95%
ruff check .
```

Nothing in the suite touches the network. The API clients run against recorded PubMed XML and OpenAlex JSON in `tests/fixtures/`, so the tests still pass when NCBI is down.

## History

This started as two scripts that drove a headless Chrome browser over the PubMed and Google Scholar web pages, opening every article in turn with a five second sleep between each. The repo also carried 35 MB of committed `chromedriver.exe`, twice.

PubMed publishes E-utilities for exactly this, free and without a key, so the browser went. Google Scholar has no API and blocks automated access, which is why the `scholar.xlsx` committed alongside it was empty; OpenAlex covers similar ground with a documented API, so that replaced it.

The old extraction was guesswork. Dataset size was the largest number in any sentence, so a year or a p-value often won. The top model was every sentence containing the word "best", concatenated. Those are now cohort patterns with a plausibility range, and a metric parser that normalises `94.2%`, `AUC = 0.91` and `F1 of 0.88`.

## Licence

[MIT](LICENSE).
