"""Scrapes each registered AMC's SID page and uploads new PDFs to
bronze/sebi/sid/{amc_entity_id}/ (bronze/ medallion layer — raw, as-downloaded).

Self-contained: reuses ingestion/page_scraper.py's PageScraper (the same
static-HTML/Next.js-RSC extraction used for AMFI research files) and
storage/s3.py's S3Storage, but is otherwise independent of the AMFI PDF
pipeline — different source, different bronze/ subtree.

Scaffold-level: no dedicated Postgres metadata table yet (unlike
document_metadata for the AMFI pipeline) — idempotency is via S3Storage's
exists() check only. Add a sebi_sid_documents table (mirroring
document_metadata's shape) if/when this becomes a tracked production source.
"""

from __future__ import annotations

import structlog

from financial_pipeline.ingestion.page_scraper import PageScraper
from financial_pipeline.sebi_ingestion.amc_sources import AMC_SID_SOURCES
from financial_pipeline.storage.s3 import S3Storage

log = structlog.get_logger()


def sync_sid_documents(
    *,
    s3_bucket: str,
    aws_region: str = "ap-south-1",
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
) -> dict:
    scraper = PageScraper()
    total_found = 0
    total_uploaded = 0
    total_skipped = 0
    total_failed = 0

    for source in AMC_SID_SOURCES:
        src_log = log.bind(job="sebi_sid_sync", amc=source.amc_entity_id, page_url=source.page_url)
        store = S3Storage(
            bucket=s3_bucket,
            prefix=f"bronze/sebi/sid/{source.amc_entity_id}",
            region=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        try:
            links = scraper.find_links(source.page_url, extensions=(".pdf",))
        except Exception:
            src_log.exception("sebi_sid_sync.page_fetch_failed")
            continue

        total_found += len(links)
        src_log.info("sebi_sid_sync.links_found", count=len(links))

        for url in links:
            filename = PageScraper.filename_from_url(url)
            try:
                if store.exists(filename):
                    total_skipped += 1
                    continue
                raw = scraper.download(url)
                key = store.put_raw(key_suffix=filename, data=raw, content_type="application/pdf")
                total_uploaded += 1
                src_log.info("sebi_sid_sync.uploaded", filename=filename, s3_key=key, bytes=len(raw))
            except Exception:
                total_failed += 1
                src_log.exception("sebi_sid_sync.download_failed", url=url)

    result = {
        "sources": len(AMC_SID_SOURCES),
        "links_found": total_found,
        "uploaded": total_uploaded,
        "skipped": total_skipped,
        "failed": total_failed,
    }
    log.info("sebi_sid_sync.completed", **result)
    return result
