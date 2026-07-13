"""Scrapes each registered AMC's SID page and uploads new PDFs to
bronze/sebi/sid/{amc_entity_id}/ (bronze/ medallion layer — raw, as-downloaded).

Self-contained: reuses ingestion/page_scraper.py's PageScraper (the same
static-HTML/Next.js-RSC extraction used for AMFI research files) and
storage/s3.py's S3Storage, but is otherwise independent of the AMFI PDF
pipeline — different source, different bronze/ subtree.

Every uploaded SID gets a document_metadata row (reusing the existing table
rather than adding a new sebi_sid_documents one) and, per AMC source, a
mandatory entity-resolution call through services/lineage.py's
resolve_and_link — the resulting amc_entity_id is stored on each SID's
document_metadata row so a document is traceable back to its canonical AMC
entity, and every download is logged as an ingestion_lineage row. Idempotency
is still primarily via S3Storage's exists() check (unchanged).
"""

from __future__ import annotations

import hashlib

import structlog

from financial_pipeline.ingestion.page_scraper import PageScraper
from financial_pipeline.sebi_ingestion.amc_sources import AMC_SID_SOURCES
from financial_pipeline.services import entity_store, lineage
from financial_pipeline.services.entity_ingestion import ingest_amc
from financial_pipeline.storage.s3 import S3Storage

log = structlog.get_logger()


def _upsert_sid_metadata(cur, *, amc_slug: str, amc_entity_id, url: str, key: str, file_name: str, raw: bytes) -> None:
    """Insert/update a document_metadata row for one downloaded SID PDF,
    keyed on file_name like document_repo.py's upsert_metadata (AMFI
    pipeline) does — reusing that table's existing conflict convention
    rather than a new one. amc_slug (e.g. 'sbi_mf', the taxonomy.yaml id)
    is the human-readable provider; amc_entity_id is the resolved
    financial_entity_master UUID (nullable — entity resolution is
    best-effort, see resolve_and_link)."""
    cur.execute(
        """
        INSERT INTO document_metadata
            (source, provider, document_type, s3_raw_key, original_url,
             file_name, file_size_bytes, file_hash, amc_entity_id)
        VALUES ('sebi_sid', %s, 'sid', %s, %s, %s, %s, %s, %s)
        ON CONFLICT (file_name) DO UPDATE SET
            s3_raw_key = EXCLUDED.s3_raw_key,
            file_size_bytes = EXCLUDED.file_size_bytes,
            file_hash = EXCLUDED.file_hash,
            amc_entity_id = EXCLUDED.amc_entity_id,
            updated_at = NOW()
        """,
        (amc_slug, key, url, file_name, len(raw), hashlib.md5(raw).hexdigest(), amc_entity_id),
    )


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

    conn = entity_store.connect()
    cur = conn.cursor()
    run_id = lineage.start_run(cur, "sebi_sid_scrape")
    conn.commit()

    try:
        for source in AMC_SID_SOURCES:
            src_log = log.bind(job="sebi_sid_sync", amc=source.amc_entity_id, page_url=source.page_url)
            store = S3Storage(
                bucket=s3_bucket,
                prefix=f"bronze/sebi/sid/{source.amc_entity_id}",
                region=aws_region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )

            # Mandatory entity-resolution gateway: resolve this source's AMC
            # to a canonical financial_entity_master row once per source
            # (not once per document — it's the same organization for every
            # SID this source yields), so every SID download below can be
            # linked to it without a repeated resolve per PDF.
            amc_entity_id = lineage.resolve_and_link(
                conn,
                cur,
                run_id=run_id,
                ingest_fn=ingest_amc,
                source_ref=f"sebi_sid_source:{source.amc_entity_id}",
                raw_amc_name=source.amc_name,
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

                    _upsert_sid_metadata(
                        cur,
                        amc_slug=source.amc_entity_id,
                        amc_entity_id=amc_entity_id,
                        url=url,
                        key=key,
                        file_name=filename,
                        raw=raw,
                    )
                    lineage.record_transform(
                        cur,
                        run_id=run_id,
                        stage="bronze_ingest",
                        source_type="http",
                        source_ref=url,
                        target_type="s3",
                        target_ref=key,
                        record_count=1,
                    )
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    total_failed += 1
                    src_log.exception("sebi_sid_sync.download_failed", url=url)
                    lineage.record_transform(
                        cur,
                        run_id=run_id,
                        stage="bronze_ingest",
                        source_type="http",
                        source_ref=url,
                        target_type="s3",
                        target_ref=None,
                        status="failed",
                        error=str(exc),
                    )
                    conn.commit()

        result = {
            "sources": len(AMC_SID_SOURCES),
            "links_found": total_found,
            "uploaded": total_uploaded,
            "skipped": total_skipped,
            "failed": total_failed,
        }
        lineage.complete_run(cur, run_id, "completed" if total_failed == 0 else "completed_with_errors", **result)
        conn.commit()
        log.info("sebi_sid_sync.completed", **result)
        return result
    finally:
        cur.close()
        conn.close()
