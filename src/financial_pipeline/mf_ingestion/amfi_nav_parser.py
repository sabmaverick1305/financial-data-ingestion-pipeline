"""Parser for AMFI's own NAVAll.txt bulk file format.

Moved out of scripts/fetch_amfi_nav.py (a dev CLI script, not part of the
installed src/financial_pipeline package — pyproject.toml's packages.find
only covers `src`) so this logic is a normal, importable dependency for
production code, not something reached via sys.path hacks. Used by both
scripts/fetch_amfi_nav.py (fetch + upload CLI) and
services/amfi_category_source.py (the live per-scheme category lookup used
by mf_ingestion/sync.py's entity resolution — see
domain/resolution/entity_resolution_rules.yaml's cross_source_scheme_identity
rule for why scheme_code is a valid AMFI<->mfapi join key here).
"""
from __future__ import annotations

import io
import re

import pandas as pd
import structlog

log = structlog.get_logger()

_CATEGORY_HEADER_RE = re.compile(r"^.*\bSchemes\((.+)\)\s*$")


def parse_nav_text(raw: bytes) -> pd.DataFrame:
    """Parse the AMFI NAV text format into a tidy DataFrame.

    The file format interleaves two different kinds of no-semicolon header
    lines with data rows that look like Scheme Code;ISIN1;ISIN2;Scheme
    Name;NAV;Date:
      - real category headers, e.g. "Open Ended Schemes(Debt Scheme -
        Banking and PSU Fund)" / "Close Ended Schemes(ELSS)" / "Interval
        Fund Schemes(Income)" — always end in "Schemes(...)".
      - AMC/fund-house sub-header lines nested under each category, e.g.
        "Aditya Birla Sun Life Mutual Fund" — no "Schemes(...)" suffix.
    Only the first kind should update `category`; treating every
    non-data line as a category header (the original bug here) makes
    `category` silently become the AMC name for every scheme after the
    first one in each category block, since the AMC header line always
    comes after the real category header and before the data rows.
    """
    category: str = ""
    rows: list[dict[str, str]] = []

    for line in io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue

        parts = line.split(";")
        if len(parts) < 6:
            if line.startswith("Scheme Code"):
                continue
            m = _CATEGORY_HEADER_RE.match(line)
            if m:
                category = m.group(1)
            # else: AMC/fund-house sub-header line — not a category change, ignore.
            continue

        scheme_code, isin_growth, isin_div_reinvest, scheme_name, nav, nav_date = (
            parts[0].strip(),
            parts[1].strip(),
            parts[2].strip(),
            parts[3].strip(),
            parts[4].strip(),
            parts[5].strip(),
        )

        # Skip the column-header row itself
        if scheme_code == "Scheme Code":
            continue

        rows.append(
            {
                "category": category,
                "scheme_code": scheme_code,
                "isin_growth": isin_growth,
                "isin_div_reinvestment": isin_div_reinvest,
                "scheme_name": scheme_name,
                "nav": nav,
                "nav_date": nav_date,
            }
        )

    df = pd.DataFrame(rows)
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["nav_date"] = pd.to_datetime(df["nav_date"], format="%d-%b-%Y", errors="coerce")
    log.info("amfi.parse.completed", rows=len(df))
    return df
