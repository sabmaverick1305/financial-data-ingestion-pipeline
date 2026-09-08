#!/usr/bin/env python3
"""Audit whether fund-specific documentary evidence is indexed for beta."""
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
    args = parser.parse_args()

    if not settings.postgres_url:
        raise RuntimeError("POSTGRES_URL is required")

    repo = DocumentRepository(settings.postgres_url)
    coverage = repo.documentary_coverage(
        fund_names=args.fund_names,
        required_document_types=DEFAULT_TYPES,
    )
    covered = sum(1 for item in coverage.values() if item["covered"])
    total = len(coverage)
    print(json.dumps({
        "covered": covered,
        "total": total,
        "coverage_ratio": covered / total if total else 0.0,
        "funds": coverage,
    }, indent=2))

if __name__ == "__main__":
    main()
