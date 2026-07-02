#!/usr/bin/env python3
"""text-worker — low memory (PyMuPDF + pandas only, no ML models loaded).

Picks documents with status IN (uploaded, classified), claims them atomically
via SELECT ... FOR UPDATE SKIP LOCKED (so multiple text-workers can run
concurrently safely), extracts text via PyMuPDF (PDF) or pandas (XLS/XLSX),
uploads text.json + markdown.md (+ tables for spreadsheets) to S3, and
advances processing_status:

    PDF      -> text_extracted   (has_text_layer set; table/ocr-worker next)
    XLS/XLSX -> tables_extracted (no Docling needed; chunk-worker next)

Failures: record_failure() increments attempt_count. After MAX_ATTEMPTS the
document moves to 'failed' (terminal) instead of re-queueing.

Usage:
    python scripts/process_text_worker.py [--limit N] [--loop]
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, "src")

from _worker_common import (  # noqa: E402
    SCHEMA_VERSION,
    df_to_parquet_bytes,
    init,
    processed_prefix,
)
from financial_pipeline.config import settings  # noqa: E402
from financial_pipeline.processing.extractor import TextExtractor  # noqa: E402
from financial_pipeline.storage.document_repo import Status  # noqa: E402
import structlog  # noqa: E402

log = structlog.get_logger()


def process_one(doc: dict, s3, repo, extractor: TextExtractor) -> bool:
    doc_id = str(doc["document_id"])
    filename = doc.get("file_name", doc_id)
    file_type = doc.get("file_type") or filename.rsplit(".", 1)[-1]
    raw_key = doc["s3_raw_key"]
    prefix = processed_prefix(doc)
    logger = log.bind(document_id=doc_id, file_name=filename, prefix=prefix)

    started_at = datetime.now(tz=timezone.utc)
    try:
        logger.info("text_worker.downloading", key=raw_key)
        obj = s3.get_object(Bucket=settings.s3_bucket, Key=raw_key)
        raw = obj["Body"].read()

        result = extractor.extract(raw, file_type)

        text_payload = json.dumps({
            "document_id": doc_id,
            "file_name": filename,
            "has_text_layer": result.has_text_layer,
            "extraction_engine": result.extraction_engine,
            "schema_version": SCHEMA_VERSION,
            "metadata": result.metadata,
            "pages": result.pages,
            "figures": result.figures,
            "full_text": result.full_text,
        }, ensure_ascii=False).encode()
        s3.put_object(Bucket=settings.s3_bucket, Key=f"{prefix}/text.json",
                      Body=text_payload, ContentType="application/json")
        s3.put_object(Bucket=settings.s3_bucket, Key=f"{prefix}/markdown.md",
                      Body=result.markdown.encode(), ContentType="text/markdown")

        table_keys: list[str] = []
        for i, df in enumerate(result.tables, start=1):
            tbl_key = f"{prefix}/tables/table_{i:03d}.parquet"
            s3.put_object(Bucket=settings.s3_bucket, Key=tbl_key,
                          Body=df_to_parquet_bytes(df),
                          ContentType="application/octet-stream")
            table_keys.append(tbl_key)

        # Register spreadsheet sheets as table assets (page_number=None, sheet name as table_name)
        if result.tables:
            from financial_pipeline.processing.extractor import TableMeta
            sheets = (result.metadata or {}).get("sheets", [])
            sheet_meta = [
                TableMeta(table_name=s.get("sheet_name") if i < len(sheets) else None)
                for i, _ in enumerate(result.tables)
            ]
            repo.register_table_assets(
                document_id=doc_id,
                tables=result.tables,
                table_meta=sheet_meta,
                prefix=prefix,
            )

        completed_at = datetime.now(tz=timezone.utc)
        repo.log_stage(doc_id, "text_extraction", "success",
                       message=f"engine={result.extraction_engine} pages={len(result.pages)} "
                               f"chars={len(result.full_text)} tables={len(table_keys)}",
                       started_at=started_at, completed_at=completed_at)

        next_status = (
            Status.TABLES_EXTRACTED if file_type.lower() in ("xls", "xlsx")
            else Status.TEXT_EXTRACTED
        )
        repo.update_status(doc_id, next_status,
                           s3_processed_key=prefix,
                           has_text_layer=result.has_text_layer,
                           schema_version=SCHEMA_VERSION)

        logger.info("text_worker.done", next_status=next_status,
                    elapsed_s=round((completed_at - started_at).total_seconds(), 1))
        return True

    except Exception as exc:
        new_status = repo.record_failure(
            doc_id,
            error=str(exc),
            revert_status=Status.UPLOADED,  # text-worker is the first stage
        )
        repo.log_stage(doc_id, "text_extraction", "failed", message=str(exc))
        logger.error("text_worker.failed", error=str(exc), new_status=new_status)
        return False


def run_batch(repo, s3, extractor: TextExtractor, limit: int | None) -> int:
    docs = repo.claim_documents(
        Status.TEXT_PENDING,
        claim_status=Status.TEXT_PROCESSING,
        limit=limit,
    )
    if not docs:
        return 0

    print(f"text-worker: processing {len(docs)} document(s)…")
    succeeded = failed = 0
    for doc in docs:
        ok = process_one(doc, s3, repo, extractor)
        succeeded += ok
        failed += not ok
        print(f"  {'✓' if ok else '✗'}  {doc.get('file_name')}")
    print(f"  batch done — succeeded={succeeded} failed={failed}")
    return len(docs)


def main(limit: int | None = None, loop: bool = False) -> None:
    repo, s3 = init("text-worker")
    extractor = TextExtractor()

    if loop:
        print("text-worker: starting in --loop mode (self-drains until queue empty)")
        total = 0
        while True:
            count = run_batch(repo, s3, extractor, limit)
            total += count
            if count == 0:
                print(f"text-worker: queue empty after {total} documents. exiting.")
                break
            time.sleep(1)
    else:
        count = run_batch(repo, s3, extractor, limit)
        if count == 0:
            print("No documents pending text extraction.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--loop", action="store_true",
                        help="Keep processing until queue is empty, then exit")
    args = parser.parse_args()
    main(limit=args.limit, loop=args.loop)
