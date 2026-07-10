#!/usr/bin/env python3
"""Scrape the AMFI research page, download all XLS/PDF files, upload each to S3.

Usage:
    python scripts/fetch_amfi_research_files.py

S3 layout (bronze/ medallion layer — raw, as-downloaded files):
    s3://<S3_BUCKET>/bronze/amfi/monthly_aum/<YYYY-MM-DD>/<filename>
    s3://<S3_BUCKET>/bronze/amfi/quarterly_aum/<YYYY-MM-DD>/<filename>
    s3://<S3_BUCKET>/bronze/amfi/other/<YYYY-MM-DD>/<filename>
"""

from __future__ import annotations

import hashlib
import sys
from datetime import UTC, date, datetime

import structlog

sys.path.insert(0, "src")

from financial_pipeline.config import settings
from financial_pipeline.ingestion.page_scraper import (
    PageScraper,
    classify_filename,
    parse_filename_metadata,
)
from financial_pipeline.logging import configure_logging
from financial_pipeline.storage.document_repo import DocumentRepository
from financial_pipeline.storage.s3 import S3Storage

log = structlog.get_logger()

AMFI_RESEARCH_URL = "https://www.amfiindia.com/research-information/amfi-data"
S3_PREFIX = "bronze/amfi"

_DOC_TYPE = {
    "monthly": "monthly_report",
    "quarterly": "quarterly_report",
    "unknown": "research_document",
}

# classification (from classify_filename) -> bronze/ folder name
_BRONZE_FOLDER = {
    "monthly": "monthly_aum",
    "quarterly": "quarterly_aum",
    "unknown": "other",
}


def main() -> None:
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    today = date.today().isoformat()

    # One S3Storage per classification folder — bronze/amfi/{monthly_aum,
    # quarterly_aum,other}/{today}/, not a single shared date-prefix, since
    # classification is now the top-level bronze/ folder, not a subfolder.
    def _storage(classification: str) -> S3Storage:
        return S3Storage(
            bucket=settings.s3_bucket,
            prefix=f"{S3_PREFIX}/{_BRONZE_FOLDER[classification]}/{today}",
            region=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )

    storages = {
        "monthly": _storage("monthly"),
        "quarterly": _storage("quarterly"),
        "unknown": _storage("unknown"),
    }

    # Optional Postgres metadata store — skipped if POSTGRES_URL is not configured
    repo: DocumentRepository | None = None
    if settings.postgres_url:
        repo = DocumentRepository(settings.postgres_url)
        repo.create_tables()
    else:
        log.warning("db.postgres_url_not_set", msg="Metadata will not be recorded in PostgreSQL")

    scraper = PageScraper()

    # ── 1. Discover links ────────────────────────────────────────────────────
    links = scraper.find_links(AMFI_RESEARCH_URL)
    if not links:
        log.warning("scraper.no_links_found", url=AMFI_RESEARCH_URL)
        print("No XLS/PDF links found on the page.")
        return

    print(f"Found {len(links)} file(s) on {AMFI_RESEARCH_URL}")

    # ── 2. Download, upload, record metadata ────────────────────────────────
    uploaded: list[dict[str, str | int]] = []
    skipped: list[str] = []
    failed: list[str] = []

    for url in links:
        filename = PageScraper.filename_from_url(url)
        classification = classify_filename(filename)
        store = storages[classification]
        try:
            if store.exists(filename):
                skipped.append(filename)
                print(f"  –  [{classification:10s}]  {filename:50s} already in S3, skipped")
                continue

            started_at = datetime.now(tz=UTC)
            raw = scraper.download(url)
            key = store.put_raw(
                key_suffix=filename,
                data=raw,
                content_type=PageScraper.content_type(filename),
            )
            completed_at = datetime.now(tz=UTC)

            uploaded.append({"filename": filename, "s3_key": key, "bytes": len(raw), "class": classification})
            print(f"  ✓  [{classification:10s}]  {filename:50s} {len(raw):>10,} bytes  →  s3://{settings.s3_bucket}/{key}")

            # ── Postgres metadata + log ──────────────────────────────────────
            if repo:
                period_meta = parse_filename_metadata(filename)
                file_ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else None
                title = filename.rsplit(".", 1)[0].replace("-", " ").replace("_", " ")

                doc_id, action = repo.upsert_metadata(
                    source="amfi",
                    provider="amfiindia.com",
                    document_type=_DOC_TYPE[classification],
                    s3_raw_key=key,
                    original_url=url,
                    file_name=filename,
                    file_size_bytes=len(raw),
                    file_hash=hashlib.md5(raw).hexdigest(),
                    category=classification,
                    title=title,
                    file_type=file_ext,
                    **period_meta,
                )
                repo.log_stage(
                    document_id=doc_id,
                    stage="upload",
                    status=action,  # 'inserted' on first run, 'updated' on re-runs
                    message=f"Uploaded {len(raw):,} bytes to s3://{settings.s3_bucket}/{key}",
                    started_at=started_at,
                    completed_at=completed_at,
                )

        except Exception as exc:
            log.error("download.failed", url=url, error=str(exc))
            failed.append(url)
            print(f"  ✗  {filename} — {exc}")

    # ── 3. Summary ───────────────────────────────────────────────────────────
    counts = {c: sum(1 for u in uploaded if u["class"] == c) for c in storages}
    print(f"\n{'─' * 70}")
    print(
        f"Uploaded : {len(uploaded)}  (monthly={counts['monthly']}, quarterly={counts['quarterly']}, unknown={counts['unknown']})"
    )
    print(f"Skipped  : {len(skipped)}  (already in S3)")
    print(f"Failed   : {len(failed)}")
    print(f"S3 prefix: s3://{settings.s3_bucket}/{S3_PREFIX}/{{monthly_aum,quarterly_aum,other}}/{today}/")
    if repo:
        print(f"DB       : document_metadata rows inserted = {len(uploaded)}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
