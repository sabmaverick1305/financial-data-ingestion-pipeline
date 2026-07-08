#!/usr/bin/env python3
"""Backfill amfi_amc_stats for pre-2020 monthly reports.

For each monthly report document in document_metadata that has zero rows in
amfi_amc_stats, this script:
  1. Downloads the raw PDF from S3 (s3_raw_key)
  2. Extracts AUM by scheme type (Table 4) and GRAND TOTAL mobilization (Table 1)
  3. Upserts into amfi_amc_stats

Idempotent (ON CONFLICT DO UPDATE). Safe to re-run.

Usage:
    cd /Users/sabyasachi/Documents/financial-data-ingestion-pipeline
    .venv/bin/python3 scripts/backfill_amfi_amc_stats_from_s3.py [--dry-run] [--year YYYY]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3
import fitz  # PyMuPDF
import structlog
from sqlalchemy import create_engine, text as sa_text

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from financial_pipeline.config import settings
from financial_pipeline.logging import configure_logging
from financial_pipeline.text_to_sql.schema_amc import (
    create_table,
    extract_rows_from_pdf_text,
)

log = structlog.get_logger()

UPSERT_SQL = """
INSERT INTO amfi_amc_stats
    (period_year, period_month, scheme_type,
     total_mobilized, redemption, net_inflow,
     aum, aum_pct, source_document_id)
VALUES
    (:yr, :mo, :scheme_type,
     :total_mobilized, :redemption, :net_inflow,
     :aum, :aum_pct, :doc_id)
ON CONFLICT (period_year, period_month, scheme_type)
DO UPDATE SET
    total_mobilized    = EXCLUDED.total_mobilized,
    redemption         = EXCLUDED.redemption,
    net_inflow         = EXCLUDED.net_inflow,
    aum                = EXCLUDED.aum,
    aum_pct            = EXCLUDED.aum_pct,
    source_document_id = EXCLUDED.source_document_id
"""


def fetch_gap_months(engine, year_filter: int | None) -> list[dict]:
    """Return one document per (year, month) with < 5 rows in amfi_amc_stats."""
    clauses = ["dm.document_type = 'monthly_report'",
                "dm.processing_status = 'embedded'",
                "dm.file_type = 'pdf'",
                "dm.s3_raw_key IS NOT NULL",
                "dm.period_year < 2020"]  # pre-2020 legacy format only
    params: dict = {}
    if year_filter:
        clauses.append("dm.period_year = :year")
        params["year"] = year_filter

    where = " AND ".join(clauses)
    sql = f"""
        SELECT DISTINCT ON (dm.period_year, dm.period_month)
               dm.document_id,
               dm.period_year,
               dm.period_month,
               dm.s3_raw_key,
               dm.file_name
        FROM document_metadata dm
        WHERE {where}
          AND (
              SELECT COUNT(*) FROM amfi_amc_stats aas
              WHERE aas.period_year  = dm.period_year
                AND aas.period_month = dm.period_month
          ) < 5
        ORDER BY dm.period_year, dm.period_month, dm.document_id
    """
    with engine.connect() as conn:
        rows = conn.execute(sa_text(sql), params).mappings().all()
    return [dict(r) for r in rows]


def main(dry_run: bool, year_filter: int | None) -> None:
    configure_logging(level="INFO", fmt="console")

    engine = create_engine(settings.postgres_url)
    log.info("backfill_amc.creating_table")
    create_table(engine)

    gaps = fetch_gap_months(engine, year_filter)
    log.info("backfill_amc.gaps_found", count=len(gaps),
             year_filter=year_filter or "all")

    if not gaps:
        log.info("backfill_amc.nothing_to_do")
        return

    s3 = boto3.client(
        "s3",
        region_name           = settings.aws_region,
        aws_access_key_id     = settings.aws_access_key_id,
        aws_secret_access_key = settings.aws_secret_access_key,
    )

    ok = skipped = errors = 0

    for gap in gaps:
        yr        = gap["period_year"]
        mo        = gap["period_month"]
        s3_key    = gap["s3_raw_key"]
        doc_id    = str(gap["document_id"])
        file_name = gap["file_name"]

        logger = log.bind(year=yr, month=mo, file=file_name)

        # Download PDF
        try:
            logger.info("backfill_amc.downloading", key=s3_key)
            obj = s3.get_object(Bucket=settings.s3_bucket, Key=s3_key)
            raw = obj["Body"].read()
        except Exception as exc:
            logger.warning("backfill_amc.s3_download_failed", error=str(exc))
            errors += 1
            continue

        # Extract text from all pages
        doc = fitz.open(stream=raw, filetype="pdf")
        pages_text = [doc[p].get_text() for p in range(len(doc))]

        # Parse fund stats
        rows = extract_rows_from_pdf_text(pages_text, yr, mo)
        valid_rows = [r for r in rows if r.get("aum") is not None
                      or r.get("total_mobilized") is not None]

        if not valid_rows:
            logger.warning("backfill_amc.no_rows_parsed")
            skipped += 1
            continue

        logger.info("backfill_amc.parsed", rows=len(valid_rows))

        if dry_run:
            for r in valid_rows:
                logger.info("backfill_amc.dry_run",
                             scheme_type=r["scheme_type"],
                             aum=r["aum"],
                             mobilized=r["total_mobilized"])
            ok += 1
            continue

        with engine.begin() as conn:
            for r in valid_rows:
                conn.execute(sa_text(UPSERT_SQL), {
                    "yr":               yr,
                    "mo":               mo,
                    "scheme_type":      r["scheme_type"],
                    "total_mobilized":  r["total_mobilized"],
                    "redemption":       r["redemption"],
                    "net_inflow":       r["net_inflow"],
                    "aum":              r["aum"],
                    "aum_pct":          r["aum_pct"],
                    "doc_id":           doc_id,
                })

        logger.info("backfill_amc.upserted", rows=len(valid_rows))
        ok += 1

    log.info("backfill_amc.done",
             processed=ok, skipped=skipped, errors=errors, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill amfi_amc_stats from pre-2020 S3 PDFs"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--year", type=int, default=None,
                        help="Limit to a single year")
    args = parser.parse_args()
    main(dry_run=args.dry_run, year_filter=args.year)
