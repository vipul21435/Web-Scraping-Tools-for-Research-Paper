"""Writers. JSONL and CSV always work; Excel needs the optional `excel` extra."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from research_harvest.models import Article

COLUMNS: Sequence[str] = (
    "title",
    "authors",
    "year",
    "journal",
    "doi",
    "url",
    "source",
    "source_id",
    "models_mentioned",
    "dataset_size",
    "headline_metric",
    "keywords",
    "abstract",
)


def write_jsonl(articles: Iterable[Article], path: Path) -> int:
    """One JSON object per line, which keeps every field intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for article in articles:
            handle.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_csv(articles: Iterable[Article], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for article in articles:
            writer.writerow(article.to_row())
            count += 1
    return count


def write_xlsx(articles: Iterable[Article], path: Path, *, sheet_per_year: bool = True) -> int:
    """
    Excel output, optionally split into one sheet per publication year.

    Written with openpyxl directly, which keeps pandas out of the dependency
    list for a job that only needs a workbook and some rows.
    """
    try:
        from openpyxl import Workbook
    except ImportError as exc:  # pragma: no cover (depends on the install extra)
        raise RuntimeError(
            "Excel output needs the optional dependency. Install it with:\n"
            '    pip install "research-harvest[excel]"'
        ) from exc

    records = list(articles)
    path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    workbook.remove(workbook.active)

    if sheet_per_year:
        by_year: dict[str, list[Article]] = {}
        for article in records:
            by_year.setdefault(str(article.year) if article.year else "undated", []).append(article)
        groups = sorted(by_year.items())
    else:
        groups = [("results", records)]

    if not groups:
        groups = [("results", [])]

    for name, group in groups:
        sheet = workbook.create_sheet(title=name[:31])
        sheet.append(list(COLUMNS))
        for article in group:
            row = article.to_row()
            sheet.append([row.get(column, "") for column in COLUMNS])

    workbook.save(path)
    return len(records)


WRITERS = {"jsonl": write_jsonl, "csv": write_csv, "xlsx": write_xlsx}
