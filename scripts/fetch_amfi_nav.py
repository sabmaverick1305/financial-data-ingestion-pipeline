#!/usr/bin/env python3
"""Fetch the AMFI India NAV file and upload raw + parsed CSV to S3.

Writes raw text to bronze/ (as-fetched) and the parsed CSV to silver/
(cleaned/typed) — see the bronze/silver/gold medallion convention.

Usage:
    python scripts/fetch_amfi_nav.py

Environment variables (via .env or shell):
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    S3_BUCKET
    S3_PREFIX        (default: bronze/amfi/nav)
    AWS_REGION       (default: ap-south-1)
"""

from __future__ import annotations

import sys
from datetime import date

import httpx
import structlog

sys.path.insert(0, "src")

from financial_pipeline.config import settings
from financial_pipeline.logging import configure_logging
from financial_pipeline.mf_ingestion.amfi_nav_parser import parse_nav_text
from financial_pipeline.services import entity_store, lineage
from financial_pipeline.storage.s3 import S3Storage

log = structlog.get_logger()

AMFI_NAV_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"


def fetch_nav_text() -> bytes:
    log.info("amfi.fetch.started", url=AMFI_NAV_URL)
    with httpx.Client(timeout=settings.request_timeout, follow_redirects=True) as client:
        for attempt in range(1, settings.max_retries + 1):
            try:
                response = client.get(AMFI_NAV_URL)
                response.raise_for_status()
                log.info("amfi.fetch.completed", bytes=len(response.content), attempt=attempt)
                return response.content
            except httpx.HTTPError as exc:
                log.warning("amfi.fetch.retry", attempt=attempt, error=str(exc))
                if attempt == settings.max_retries:
                    raise
    raise RuntimeError("unreachable")


def main() -> None:
    configure_logging(level=settings.log_level, fmt=settings.log_format)

    today = date.today().isoformat()  # e.g. 2026-06-27

    # No shared prefix on the storage object — bronze/ and silver/ are
    # siblings, not one nested under the other.
    storage = S3Storage(
        bucket=settings.s3_bucket,
        region=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )

    conn = entity_store.connect()
    cur = conn.cursor()
    run_id = lineage.start_run(cur, "amfi_nav_fetch")
    conn.commit()

    try:
        raw = fetch_nav_text()

        raw_key = storage.put_raw(
            key_suffix=f"{settings.s3_prefix}/{today}/NAVAll.txt",
            data=raw,
            content_type="text/plain",
        )
        log.info("upload.raw.done", s3_key=raw_key)
        lineage.record_transform(
            cur,
            run_id=run_id,
            stage="bronze_ingest",
            source_type="http",
            source_ref=AMFI_NAV_URL,
            target_type="s3",
            target_ref=raw_key,
            record_count=len(raw),
        )
        conn.commit()

        df = parse_nav_text(raw)

        csv_key = storage.put_csv(
            key_suffix=f"silver/amfi/nav_csv/{today}/NAVAll.csv",
            df=df,
        )
        log.info("upload.csv.done", s3_key=csv_key, rows=len(df))
        lineage.record_transform(
            cur,
            run_id=run_id,
            stage="bronze_to_silver",
            source_type="s3",
            source_ref=raw_key,
            target_type="s3",
            target_ref=csv_key,
            record_count=len(df),
        )
        conn.commit()

        lineage.complete_run(cur, run_id, "completed", rows=len(df))
        conn.commit()
    except Exception as exc:
        conn.rollback()
        lineage.complete_run(cur, run_id, "failed", error=str(exc))
        conn.commit()
        raise
    finally:
        cur.close()
        conn.close()

    print(f"Raw  -> s3://{settings.s3_bucket}/{raw_key}")
    print(f"CSV  -> s3://{settings.s3_bucket}/{csv_key}")
    print(f"Rows parsed: {len(df)}")


if __name__ == "__main__":
    main()
