#!/usr/bin/env python3
"""Repair Jan/Feb 2025 chunks — Docling missed the main stats table.

Root cause: Docling's table extractor failed on the wide (10-column) fund
statistics table in page 1 of amjan2025repo.pdf and amfeb2025repo.pdf,
producing a tiny markdown that only contains the NFO appendix (page 2).
PyMuPDF extracts page 1 correctly.

This script:
  1. Downloads the raw PDFs from S3 for Jan and Feb 2025
  2. Extracts text with PyMuPDF, parses the fund-category table rows
  3. Formats them as pipe-separated markdown (matching the March 2025 format)
  4. Deletes the bad existing chunks from document_chunks
  5. Embeds the new chunk text and inserts into document_chunks

Usage:
    cd /Users/sabyasachi/Documents/financial-data-ingestion-pipeline
    .venv/bin/python3 scripts/repair_2025_q1_chunks.py [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import boto3
import fitz  # PyMuPDF
import structlog
from sentence_transformers import SentenceTransformer
from sqlalchemy import text as sa_text

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from financial_pipeline.config import settings

log = structlog.get_logger()

MODEL_NAME   = "all-MiniLM-L6-v2"
EMBED_DIM    = 384
CATEGORY_TAG = "monthly_report"

# Months to repair: (year, month, raw S3 key)
REPAIR_TARGETS = [
    (2025, 1, "amfi/research/2026-06-27/monthly/amjan2025repo.pdf"),
    (2025, 2, "amfi/research/2026-06-27/monthly/amfeb2025repo.pdf"),
]


# ── Text parsing ──────────────────────────────────────────────────────────────

# All open-ended equity fund categories in the AMFI stats table (in order)
_EQUITY_FUNDS = [
    "Multi Cap Fund",
    "Large Cap Fund",
    "Large & Mid Cap Fund",
    "Mid Cap Fund",
    "Small Cap Fund",
    "Dividend Yield Fund",
    "Value Fund/Contra Fund",
    "Focused Fund",
    "Sectoral/Thematic Funds",
    "ELSS",
    "Flexi Cap Fund",
]

# Debt + Hybrid + Other categories (same table, page 1)
_OTHER_FUNDS = [
    "Overnight Fund", "Liquid Fund", "Ultra Short Duration Fund",
    "Low Duration Fund", "Money Market Fund", "Short Duration Fund",
    "Medium Duration Fund", "Medium to Long Duration Fund",
    "Long Duration Fund", "Dynamic Bond Fund", "Corporate Bond Fund",
    "Credit Risk Fund", "Banking and PSU Fund", "Gilt Fund",
    "Gilt Fund with 10 year constant duration", "Floater Fund",
    "Conservative Hybrid Fund", "Balanced Hybrid Fund/Aggressive Hybrid Fund",
    "Dynamic Asset Allocation/Balanced Advantage Fund",
    "Multi Asset Allocation Fund", "Arbitrage Fund", "Equity Savings Fund",
    "Retirement Fund", "Childrens Fund",
    "Index Funds", "GOLD ETF", "Other ETFs",
    "Fund of funds investing overseas",
]

_ALL_FUND_NAMES = set(_EQUITY_FUNDS + _OTHER_FUNDS)

_TABLE_HEADER = (
    "| Row | Scheme Name | No. Schemes | No. Folios"
    " | Funds Mobilized (INR cr) | Redemption (INR cr)"
    " | Net Inflow (INR cr) | AUM (INR cr) | Avg AUM (INR cr) |"
)
_TABLE_SEP = "|---|---|---|---|---|---|---|---|---|"


def _parse_pdf_table(raw: bytes, year: int, month: int) -> str:
    """Extract main stats table from PDF using PyMuPDF, return as pipe-markdown.

    Strategy: scan stripped lines for known fund-category names. The line
    immediately before is the roman-numeral row index; the lines after (up to
    the next fund name or 'Sub Total') are the numeric column values.
    """
    doc = fitz.open(stream=raw, filetype="pdf")
    full_text = doc[0].get_text()
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]

    table_rows: list[str] = []

    for fund_name in _EQUITY_FUNDS + _OTHER_FUNDS:
        for i, line in enumerate(lines):
            if line != fund_name:
                continue
            # Row index is on the line immediately before the fund name
            row_idx = lines[i - 1] if i > 0 else ""
            # Values are on the lines after the fund name, until next known name
            # or a 'Sub Total' / 'Total' marker
            vals: list[str] = []
            for j in range(i + 1, min(i + 14, len(lines))):
                v = lines[j]
                if v in _ALL_FUND_NAMES:
                    break
                if v.startswith(("Sub Total", "Grand Total", "Total A", "Total B", "Total C")):
                    break
                vals.append(v)
            padded = (vals + ["-"] * 7)[:7]
            pipe_row = f"| {row_idx} {fund_name} | " + " | ".join(padded) + " |"
            table_rows.append(pipe_row)
            break  # found this fund — move to next

    if not table_rows:
        log.warning("repair.parse_table.no_rows", year=year, month=month)
        return full_text  # fallback: raw text

    md_lines = [
        f"## AMFI Monthly Statistics — {month}/{year}",
        "",
        _TABLE_HEADER,
        _TABLE_SEP,
    ] + table_rows

    return "\n".join(md_lines)


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_document_ids(engine, year: int, month: int) -> list[str]:
    """Return document_ids for PDF documents with the given year/month."""
    with engine.connect() as conn:
        rows = conn.execute(sa_text("""
            SELECT document_id, file_name
            FROM document_metadata
            WHERE period_year  = :y
              AND period_month = :m
              AND file_name ILIKE '%.pdf'
        """), {"y": year, "m": month}).mappings().all()
    return [(str(r["document_id"]), r["file_name"]) for r in rows]


def delete_old_chunks(engine, doc_id: str, dry_run: bool) -> int:
    with engine.connect() as conn:
        result = conn.execute(sa_text(
            "SELECT COUNT(*) FROM document_chunks WHERE document_id = :id"
        ), {"id": doc_id}).scalar()
        count = int(result or 0)
        if not dry_run:
            conn.execute(sa_text(
                "DELETE FROM document_chunks WHERE document_id = :id"
            ), {"id": doc_id})
            conn.commit()
    return count


def insert_chunk(engine, doc_id: str, chunk_index: int, text: str,
                 embedding: list[float], year: int, month: int,
                 file_name: str, dry_run: bool) -> None:
    if dry_run:
        return
    with engine.connect() as conn:
        conn.execute(sa_text("""
            INSERT INTO document_chunks
                (chunk_id, document_id, chunk_index, text,
                 char_start, char_end, token_count,
                 embedding, embedding_model,
                 period_year, period_month, category, created_at)
            VALUES
                (:cid, :did, :cidx, :txt,
                 0, :cend, :tok,
                 :emb, :emodel,
                 :yr, :mo, :cat, :now)
        """), {
            "cid":    str(uuid.uuid4()),
            "did":    doc_id,
            "cidx":   chunk_index,
            "txt":    text,
            "cend":   len(text),
            "tok":    len(text.split()),
            "emb":    str(embedding),
            "emodel": MODEL_NAME,
            "yr":     year,
            "mo":     month,
            "cat":    CATEGORY_TAG,
            "now":    datetime.now(tz=UTC),
        })
        conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool) -> None:
    from financial_pipeline.storage.document_repo import DocumentRepository
    from sqlalchemy import create_engine

    engine = create_engine(settings.postgres_url)
    s3 = boto3.client(
        "s3",
        region_name            = settings.aws_region,
        aws_access_key_id      = settings.aws_access_key_id,
        aws_secret_access_key  = settings.aws_secret_access_key,
    )

    log.info("repair.loading_model", model=MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    for (year, month, s3_key) in REPAIR_TARGETS:
        logger = log.bind(year=year, month=month)
        doc_pairs = get_document_ids(engine, year, month)
        if not doc_pairs:
            logger.warning("repair.no_documents_found")
            continue

        # Download raw PDF
        logger.info("repair.downloading", key=s3_key)
        obj = s3.get_object(Bucket=settings.s3_bucket, Key=s3_key)
        raw = obj["Body"].read()

        # Parse main stats table into pipe-separated markdown
        table_md = _parse_pdf_table(raw, year, month)
        line_count = table_md.count("\n") + 1
        mid_cap_present = "mid cap" in table_md.lower()
        logger.info("repair.parsed_table",
                    chars=len(table_md), lines=line_count,
                    mid_cap=mid_cap_present)

        # Embed the new chunk
        embedding = model.encode(table_md, normalize_embeddings=True).tolist()

        for doc_id, file_name in doc_pairs:
            logger.info("repair.processing_doc", doc_id=doc_id, file=file_name)

            # Delete stale chunks
            deleted = delete_old_chunks(engine, doc_id, dry_run)
            logger.info("repair.deleted_chunks", count=deleted, dry_run=dry_run)

            # Insert corrected chunk
            insert_chunk(
                engine, doc_id, chunk_index=0,
                text=table_md, embedding=embedding,
                year=year, month=month,
                file_name=file_name, dry_run=dry_run,
            )
            logger.info("repair.inserted_chunk", dry_run=dry_run)

    log.info("repair.done", dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and log without writing to DB")
    args = parser.parse_args()

    from financial_pipeline.logging import configure_logging
    configure_logging(level="INFO", fmt="console")

    main(dry_run=args.dry_run)
