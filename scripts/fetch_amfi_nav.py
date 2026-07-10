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

import io
import sys
from datetime import date

import httpx
import pandas as pd
import structlog

sys.path.insert(0, "src")

from financial_pipeline.config import settings
from financial_pipeline.logging import configure_logging
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


def parse_nav_text(raw: bytes) -> pd.DataFrame:
    """Parse the AMFI NAV text format into a tidy DataFrame.

    The file format interleaves category headers (no semicolons) with
    data rows that look like:
        Scheme Code;ISIN1;ISIN2;Scheme Name;NAV;Date
    """
    category: str = ""
    rows: list[dict[str, str]] = []

    for line in io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue

        parts = line.split(";")
        if len(parts) < 6:
            # Category / fund-house header line
            if not line.startswith("Scheme Code"):
                category = line
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

    raw = fetch_nav_text()

    raw_key = storage.put_raw(
        key_suffix=f"{settings.s3_prefix}/{today}/NAVAll.txt",
        data=raw,
        content_type="text/plain",
    )
    log.info("upload.raw.done", s3_key=raw_key)

    df = parse_nav_text(raw)

    csv_key = storage.put_csv(
        key_suffix=f"silver/amfi/nav_csv/{today}/NAVAll.csv",
        df=df,
    )
    log.info("upload.csv.done", s3_key=csv_key, rows=len(df))

    print(f"Raw  -> s3://{settings.s3_bucket}/{raw_key}")
    print(f"CSV  -> s3://{settings.s3_bucket}/{csv_key}")
    print(f"Rows parsed: {len(df)}")


if __name__ == "__main__":
    main()
