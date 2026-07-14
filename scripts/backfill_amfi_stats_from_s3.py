#!/usr/bin/env python3
"""Backfill amfi_fund_stats for months that have no structured data.

For each (period_year, period_month) in document_metadata that has zero rows
in amfi_fund_stats, this script:
  1. Downloads the raw PDF from S3 using s3_raw_key
  2. Extracts fund-category data with PyMuPDF (same approach as repair script)
  3. Upserts into amfi_fund_stats

Idempotent: ON CONFLICT DO UPDATE, safe to re-run.

Usage:
    cd /Users/sabyasachi/Documents/financial-data-ingestion-pipeline
    .venv/bin/python3 scripts/backfill_amfi_stats_from_s3.py [--dry-run] [--year YYYY]
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import boto3
import fitz  # PyMuPDF
import structlog
from sqlalchemy import create_engine, text as sa_text

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from financial_pipeline.config import settings
from financial_pipeline.logging import configure_logging
from financial_pipeline.text_to_sql.schema import (
    ALL_FUND_CATEGORIES,
    create_table,
    parse_int,
    parse_number,
)

log = structlog.get_logger()

_ALL_FUND_SET = set(ALL_FUND_CATEGORIES)

# Newer AMFI monthly reports (observed starting ~2023) prefix each category
# row with a lowercase roman-numeral list marker ("ii Large Cap Fund" instead
# of "Large Cap Fund"), which broke the old exact-line-match check silently —
# no error, just a much shorter results list (8-10/39 categories instead of
# the full 39). Restricted to i/v/x characters only (true roman-numeral
# prefixes, "i".."xvi") rather than a general prefix/suffix match, since a
# looser check (e.g. line.endswith(fund_name)) would also match unrelated
# scheme-name lines elsewhere in the PDF that happen to end with a category
# string, like "JM Large & Mid Cap Fund" or "Bajaj Finserv Small Cap Fund" —
# those aren't roman-numeral-prefixed, so this stays precise.
_ROMAN_PREFIX_RE = re.compile(r"^[ivx]{1,6}\s+")

UPSERT_SQL = """
INSERT INTO amfi_fund_stats
    (period_year, period_month, fund_category,
     no_of_schemes, no_of_folios,
     funds_mobilized, redemption, net_inflow, aum, avg_aum,
     source_document_id)
VALUES
    (:yr, :mo, :cat,
     :schemes, :folios,
     :mobilized, :redemption, :net_inflow, :aum, :avg_aum,
     :doc_id)
ON CONFLICT (period_year, period_month, fund_category)
DO UPDATE SET
    no_of_schemes    = EXCLUDED.no_of_schemes,
    no_of_folios     = EXCLUDED.no_of_folios,
    funds_mobilized  = EXCLUDED.funds_mobilized,
    redemption       = EXCLUDED.redemption,
    net_inflow       = EXCLUDED.net_inflow,
    aum              = EXCLUDED.aum,
    avg_aum          = EXCLUDED.avg_aum,
    source_document_id = EXCLUDED.source_document_id
"""


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_fund_stats_from_pdf(
    raw: bytes, year: int, month: int
) -> list[dict]:
    """Parse monthly fund stats from raw PDF bytes using PyMuPDF.

    Scans the first 3 pages for known fund-category names. For each match,
    collects the 7 numeric values that follow (schemes, folios, mobilized,
    redemption, net_inflow, aum, avg_aum).

    Returns list of dicts keyed by schema.py column names.
    """
    doc = fitz.open(stream=raw, filetype="pdf")

    # Collect text from first 3 pages — older reports sometimes span 2 pages
    all_text = "\n".join(doc[p].get_text() for p in range(min(3, len(doc))))
    lines = [l.strip() for l in all_text.splitlines() if l.strip()]

    results: list[dict] = []
    seen_funds: set[str] = set()

    for fund_name in ALL_FUND_CATEGORIES:
        if fund_name in seen_funds:
            continue
        for i, line in enumerate(lines):
            if _ROMAN_PREFIX_RE.sub("", line, count=1) != fund_name:
                continue
            # Collect numeric tokens after the fund name until the next fund /
            # subtotal / end of reasonable window
            vals: list[str] = []
            for j in range(i + 1, min(i + 20, len(lines))):
                v = lines[j]
                if v in _ALL_FUND_SET:
                    break
                if v.startswith(("Sub Total", "Grand Total", "Total")):
                    break
                vals.append(v)

            # Expect 7 columns; pad with "-" if fewer found
            padded = (vals + ["-"] * 7)[:7]

            row = {
                "fund_category":   fund_name,
                "no_of_schemes":   parse_int(padded[0]),
                "no_of_folios":    parse_int(padded[1]),
                "funds_mobilized": parse_number(padded[2]),
                "redemption":      parse_number(padded[3]),
                "net_inflow":      parse_number(padded[4]),
                "aum":             parse_number(padded[5]),
                "avg_aum":         parse_number(padded[6]),
            }
            # Accept row only if at least AUM is present
            if row["aum"] is not None:
                results.append(row)
                seen_funds.add(fund_name)
            break  # found this fund name — move to next category

    log.info("backfill.parsed_pdf", year=year, month=month, fund_rows=len(results))
    return results


# ── DB helpers ────────────────────────────────────────────────────────────────

def fetch_gap_months(
    engine, year_filter: int | None, from_year: int | None
) -> list[dict]:
    """Return one document per (year, month) that has zero amfi_fund_stats rows."""
    clauses: list[str] = []
    params: dict = {}
    if year_filter:
        clauses.append("dm.period_year = :year")
        params["year"] = year_filter
    if from_year:
        clauses.append("dm.period_year >= :from_year")
        params["from_year"] = from_year
    where = ("AND " + " AND ".join(clauses)) if clauses else ""

    sql = f"""
        SELECT DISTINCT ON (dm.period_year, dm.period_month)
               dm.document_id,
               dm.period_year,
               dm.period_month,
               dm.s3_raw_key,
               dm.file_name
        FROM document_metadata dm
        WHERE dm.document_type    = 'monthly_report'
          AND dm.processing_status = 'embedded'
          AND dm.file_type        = 'pdf'
          AND dm.s3_raw_key       IS NOT NULL
          {where}
          AND (
              SELECT COUNT(*) FROM amfi_fund_stats afs
              WHERE afs.period_year  = dm.period_year
                AND afs.period_month = dm.period_month
          ) < 20
        ORDER BY dm.period_year, dm.period_month, dm.document_id
    """
    with engine.connect() as conn:
        rows = conn.execute(sa_text(sql), params).mappings().all()
    return [dict(r) for r in rows]


def upsert_rows(
    engine,
    year: int,
    month: int,
    doc_id: str,
    rows: list[dict],
    dry_run: bool,
) -> None:
    if dry_run:
        for r in rows:
            log.info("backfill.dry_run",
                     year=year, month=month, fund=r["fund_category"],
                     aum=r["aum"])
        return
    with engine.begin() as conn:
        for r in rows:
            conn.execute(sa_text(UPSERT_SQL), {
                "yr":         year,
                "mo":         month,
                "cat":        r["fund_category"],
                "schemes":    r["no_of_schemes"],
                "folios":     r["no_of_folios"],
                "mobilized":  r["funds_mobilized"],
                "redemption": r["redemption"],
                "net_inflow": r["net_inflow"],
                "aum":        r["aum"],
                "avg_aum":    r["avg_aum"],
                "doc_id":     doc_id,
            })


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool, year_filter: int | None, from_year: int | None) -> None:
    configure_logging(level="INFO", fmt="console")

    engine = create_engine(settings.postgres_url)
    log.info("backfill.creating_table")
    create_table(engine)

    gaps = fetch_gap_months(engine, year_filter, from_year)
    log.info("backfill.gaps_found", count=len(gaps),
             year_filter=year_filter or "all",
             from_year=from_year or "all")

    if not gaps:
        log.info("backfill.nothing_to_do")
        return

    s3 = boto3.client(
        "s3",
        region_name           = settings.aws_region,
        aws_access_key_id     = settings.aws_access_key_id,
        aws_secret_access_key = settings.aws_secret_access_key,
    )

    ok = skipped = errors = 0

    for gap in gaps:
        yr       = gap["period_year"]
        mo       = gap["period_month"]
        s3_key   = gap["s3_raw_key"]
        doc_id   = str(gap["document_id"])
        file_name = gap["file_name"]

        logger = log.bind(year=yr, month=mo, file=file_name)

        # Download PDF from S3
        try:
            logger.info("backfill.downloading", key=s3_key)
            obj = s3.get_object(Bucket=settings.s3_bucket, Key=s3_key)
            raw = obj["Body"].read()
        except Exception as exc:
            logger.warning("backfill.s3_download_failed", error=str(exc))
            errors += 1
            continue

        # Extract fund stats
        rows = extract_fund_stats_from_pdf(raw, yr, mo)
        if not rows:
            logger.warning("backfill.no_rows_parsed")
            skipped += 1
            continue

        # Upsert
        upsert_rows(engine, yr, mo, doc_id, rows, dry_run)
        logger.info("backfill.upserted", rows=len(rows), dry_run=dry_run)
        ok += 1

    log.info("backfill.done",
             processed=ok,
             skipped=skipped,
             errors=errors,
             dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill amfi_fund_stats from S3 PDFs for months with no data"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and log without writing to DB")
    parser.add_argument("--year", type=int, default=None,
                        help="Limit to a single year (e.g. --year 2020)")
    parser.add_argument("--from-year", type=int, default=2020,
                        help="Skip years before this (default 2020; pre-2020 uses old AMC format)")
    args = parser.parse_args()
    main(dry_run=args.dry_run, year_filter=args.year, from_year=args.from_year)
