#!/usr/bin/env python3
"""Audit fund-specific documentary coverage and emit ingestion backlog."""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "src")

from financial_pipeline.config import settings
from financial_pipeline.storage.document_repo import DocumentRepository

DEFAULT_TYPES = (
    "fund_prospectus",
    "fund_fact_sheet",
    "fund_strategy_document",
    "regulatory_filing",
    "annual_report",
    "portfolio_disclosure",
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fund_names", nargs="+")
    parser.add_argument("--types", nargs="*", default=list(DEFAULT_TYPES))
    args = parser.parse_args()

    if not settings.postgres_url:
        raise RuntimeError("POSTGRES_URL is required")

    repo = DocumentRepository(settings.postgres_url)
    coverage = repo.documentary_coverage(
        fund_names=args.fund_names,
        required_document_types=args.types,
    )
    backlog = repo.documentary_ingestion_backlog(
        fund_names=args.fund_names,
        required_document_types=args.types,
    )

    ratios = [float(item["coverage_ratio"]) for item in coverage.values()]
    print(json.dumps({
        "fund_count": len(coverage),
        "required_document_types": args.types,
        "coverage_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
        "fully_covered_funds": sum(1 for item in coverage.values() if item["covered"]),
        "missing_combinations": len(backlog),
        "funds": coverage,
        "ingestion_backlog": backlog,
    }, indent=2))

if __name__ == "__main__":
    main()
