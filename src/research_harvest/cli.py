"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from research_harvest import __version__
from research_harvest.export import WRITERS
from research_harvest.harvest import harvest
from research_harvest.keywords import GROUPS
from research_harvest.pipeline import FilterSpec
from research_harvest.sources import SOURCES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-harvest",
        description=(
            "Search PubMed and OpenAlex, remove duplicates, pull out the methods, "
            "cohort sizes and headline scores reported in each abstract, and write "
            "the result to JSONL, CSV or Excel."
        ),
        epilog=(
            "Examples:\n"
            "  research-harvest 'blood brain barrier machine learning' --limit 50\n"
            "  research-harvest 'drug discovery deep learning' --source pubmed openalex \\\n"
            "      --year-from 2018 --require-abstract --format xlsx -o out/results.xlsx\n"
            "  research-harvest 'QSAR' --method-group classical_ml --min-dataset-size 500\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("query", help="the search phrase, e.g. 'blood brain barrier machine learning'")
    parser.add_argument("--version", action="version", version=f"research-harvest {__version__}")

    fetch = parser.add_argument_group("fetching")
    fetch.add_argument(
        "--source", nargs="+", default=["pubmed"], choices=sorted(SOURCES),
        help="which catalogues to search (default: pubmed)",
    )
    fetch.add_argument("--limit", type=int, default=100, help="maximum records per source (default: 100)")
    fetch.add_argument("--year-from", type=int, help="earliest publication year")
    fetch.add_argument("--year-to", type=int, help="latest publication year")

    narrow = parser.add_argument_group("filtering")
    narrow.add_argument("--require-abstract", action="store_true", help="drop records with no abstract")
    narrow.add_argument("--require-doi", action="store_true", help="drop records with no DOI")
    narrow.add_argument("--must-mention", nargs="+", default=[], metavar="TERM",
                        help="keep only records mentioning every one of these terms")
    narrow.add_argument("--method-group", choices=sorted(GROUPS),
                        help="keep only records mentioning a method from this family")
    narrow.add_argument("--min-dataset-size", type=int, help="keep only records reporting at least this cohort size")
    narrow.add_argument("--title-matches", metavar="REGEX", help="keep only records whose title matches this pattern")

    output = parser.add_argument_group("output")
    output.add_argument("-o", "--output", type=Path, help="where to write (default: stdout as JSONL)")
    output.add_argument("--format", choices=sorted(WRITERS), default="jsonl", help="output format (default: jsonl)")
    output.add_argument("--single-sheet", action="store_true",
                        help="for xlsx, write one sheet instead of one per year")
    output.add_argument("-q", "--quiet", action="store_true", help="only report errors")
    output.add_argument("-v", "--verbose", action="store_true", help="report each request")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if args.limit < 1:
        print("--limit must be at least 1.", file=sys.stderr)
        return 2
    if args.year_from and args.year_to and args.year_from > args.year_to:
        print("--year-from cannot be later than --year-to.", file=sys.stderr)
        return 2

    filters = FilterSpec(
        require_abstract=args.require_abstract,
        require_doi=args.require_doi,
        must_mention=args.must_mention,
        method_group=args.method_group,
        min_dataset_size=args.min_dataset_size,
        title_matches=args.title_matches,
    )

    result = harvest(
        args.query,
        sources=args.source,
        limit_per_source=args.limit,
        year_from=args.year_from,
        year_to=args.year_to,
        filters=filters,
    )

    if not args.quiet:
        print(result.summary(), file=sys.stderr)

    if args.output:
        writer = WRITERS[args.format]
        if args.format == "xlsx":
            written = writer(result.articles, args.output, sheet_per_year=not args.single_sheet)
        else:
            written = writer(result.articles, args.output)
        if not args.quiet:
            print(f"Wrote {written} record(s) to {args.output}", file=sys.stderr)
    else:
        import json

        for article in result.articles:
            print(json.dumps(article.to_dict(), ensure_ascii=False))

    # Nothing found is not a crash, but a source that failed outright is.
    if result.errors and not result.articles:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
