#!/usr/bin/env python3
"""Populate amfi_fund_stats from existing document_chunks.

Scans all embedded monthly-report chunks in the DB, parses pipe-separated
fund-category rows using schema.extract_rows_from_chunk(), and upserts into
the amfi_fund_stats table.

Idempotent: uses INSERT ... ON CONFLICT DO UPDATE so re-running is safe.

Usage:
    cd /Users/sabyasachi/Documents/financial-data-ingestion-pipeline
    .venv/bin/python3 scripts/populate_amfi_stats.py [--dry-run] [--year YYYY]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from financial_pipeline.config import settings
from financial_pipeline.logging import configure_logging
from financial_pipeline.text_to_sql.schema import create_table, extract_rows_from_chunk

log = structlog.get_logger()

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
    no_of_schemes   = EXCLUDED.no_of_schemes,
    no_of_folios    = EXCLUDED.no_of_folios,
    funds_mobilized = EXCLUDED.funds_mobilized,
    redemption      = EXCLUDED.redemption,
    net_inflow      = EXCLUDED.net_inflow,
    aum             = EXCLUDED.aum,
    avg_aum         = EXCLUDED.avg_aum,
    source_document_id = EXCLUDED.source_document_id
"""


def fetch_chunks(engine, year: int | None) -> list[dict]:
    where = "WHERE dm.processing_status = 'embedded' AND dm.document_type = 'monthly_report'"
    params: dict = {}
    if year:
        where += " AND dc.period_year = :year"
        params["year"] = year
    sql = f"""
        SELECT DISTINCT ON (dc.period_year, dc.period_month, dc.document_id)
               dc.document_id, dc.period_year, dc.period_month, dc.text
        FROM document_chunks dc
        JOIN document_metadata dm ON dc.document_id = dm.document_id
        {where}
          AND dc.text ILIKE '%|%'
        ORDER BY dc.period_year, dc.period_month, dc.document_id, dc.chunk_index
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]


def main(dry_run: bool, year: int | None) -> None:
    configure_logging(level="INFO", fmt="console")
    engine = create_engine(settings.postgres_url)

    log.info("populate.creating_table")
    create_table(engine)

    chunks = fetch_chunks(engine, year)
    log.info("populate.chunks_fetched", count=len(chunks), year=year or "all")

    total_rows = 0
    skipped    = 0

    for chunk in chunks:
        yr    = chunk["period_year"]
        mo    = chunk["period_month"]
        doc_id = str(chunk["document_id"])

        fund_rows = extract_rows_from_chunk(chunk["text"])
        if not fund_rows:
            skipped += 1
            continue

        if dry_run:
            for r in fund_rows:
                log.info("populate.dry_run",
                         year=yr, month=mo, fund=r["fund_category"],
                         mobilized=r["funds_mobilized"])
            total_rows += len(fund_rows)
            continue

        with engine.begin() as conn:
            for r in fund_rows:
                conn.execute(text(UPSERT_SQL), {
                    "yr": yr, "mo": mo, "cat": r["fund_category"],
                    "schemes":    r["no_of_schemes"],
                    "folios":     r["no_of_folios"],
                    "mobilized":  r["funds_mobilized"],
                    "redemption": r["redemption"],
                    "net_inflow": r["net_inflow"],
                    "aum":        r["aum"],
                    "avg_aum":    r["avg_aum"],
                    "doc_id":     doc_id,
                })
        total_rows += len(fund_rows)
        log.info("populate.upserted",
                 year=yr, month=mo, rows=len(fund_rows))

    log.info("populate.done",
             total_rows=total_rows, skipped=skipped, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--year",    type=int, default=None)
    args = parser.parse_args()
    main(dry_run=args.dry_run, year=args.year)
