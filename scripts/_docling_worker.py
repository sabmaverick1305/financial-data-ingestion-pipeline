"""Shared run loop for table-worker and ocr-worker (high memory, Docling).

Both pick documents at status=text_extracted that the text-worker already
processed, re-download the raw PDF, run Docling (table-structure only, or
+OCR for scanned PDFs), and advance to tables_extracted.

Documents are claimed atomically (claim_documents(), SELECT ... FOR UPDATE
SKIP LOCKED) into a transient table_processing/ocr_processing status before
work starts so multiple tasks can run concurrently without grabbing the same
row. On failure, record_failure() increments the attempt counter and either
reverts to text_extracted (retry) or marks the document 'failed' if it has
exceeded MAX_ATTEMPTS.

In --loop mode each task self-drains its queue until empty, then exits. This
eliminates the need for a wave-based orchestrator re-launching tasks — a fixed
pool of N concurrent tasks each run until there is nothing left to do.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime

sys.path.insert(0, "src")

import structlog  # noqa: E402
from _worker_common import (  # noqa: E402
    SCHEMA_VERSION,
    df_to_parquet_bytes,
    init,
    processed_prefix,
)

from financial_pipeline.config import settings  # noqa: E402
from financial_pipeline.storage.document_repo import Status  # noqa: E402

log = structlog.get_logger()

LOOP_IDLE_SLEEP_S = 5  # seconds to wait when queue is empty in --loop mode


def process_one(doc: dict, s3, repo, extractor, stage_name: str, claim_status: str) -> bool:
    doc_id = str(doc["document_id"])
    filename = doc.get("file_name", doc_id)
    raw_key = doc["s3_raw_key"]
    prefix = doc.get("s3_processed_key") or processed_prefix(doc)
    logger = log.bind(document_id=doc_id, file_name=filename, prefix=prefix)

    started_at = datetime.now(tz=UTC)
    try:
        logger.info(f"{stage_name}.downloading", key=raw_key)
        obj = s3.get_object(Bucket=settings.s3_bucket, Key=raw_key)
        raw = obj["Body"].read()

        result = extractor.extract(raw)

        text_payload = json.dumps(
            {
                "document_id": doc_id,
                "file_name": filename,
                "has_text_layer": result.has_text_layer,
                "extraction_engine": result.extraction_engine,
                "schema_version": SCHEMA_VERSION,
                "metadata": result.metadata,
                "pages": result.pages,
                "figures": result.figures,
                "full_text": result.full_text,
            },
            ensure_ascii=False,
        ).encode()
        s3.put_object(
            Bucket=settings.s3_bucket,
            Key=f"{prefix}/text.json",
            Body=text_payload,
            ContentType="application/json",
        )
        s3.put_object(
            Bucket=settings.s3_bucket,
            Key=f"{prefix}/markdown.md",
            Body=result.markdown.encode(),
            ContentType="text/markdown",
        )

        table_keys: list[str] = []
        for i, df in enumerate(result.tables, start=1):
            tbl_key = f"{prefix}/tables/table_{i:03d}.parquet"
            s3.put_object(
                Bucket=settings.s3_bucket,
                Key=tbl_key,
                Body=df_to_parquet_bytes(df),
                ContentType="application/octet-stream",
            )
            table_keys.append(tbl_key)

        # Register each table in document_table_assets (page, shape, schema, S3 key)
        repo.register_table_assets(
            document_id=doc_id,
            tables=result.tables,
            table_meta=result.table_meta,
            prefix=prefix,
        )

        completed_at = datetime.now(tz=UTC)
        repo.log_stage(
            doc_id,
            stage_name,
            "success",
            message=f"engine={result.extraction_engine} tables={len(table_keys)} failed_tables={result.failed_tables}",
            started_at=started_at,
            completed_at=completed_at,
        )
        repo.update_status(doc_id, Status.TABLES_EXTRACTED, s3_processed_key=prefix, schema_version=SCHEMA_VERSION)

        logger.info(
            f"{stage_name}.done",
            tables=len(table_keys),
            elapsed_s=round((completed_at - started_at).total_seconds(), 1),
        )
        return True

    except Exception as exc:
        new_status = repo.record_failure(
            doc_id,
            error=str(exc),
            revert_status=Status.TEXT_EXTRACTED,
        )
        repo.log_stage(doc_id, stage_name, "failed", message=str(exc))
        logger.error(f"{stage_name}.failed", error=str(exc), new_status=new_status)
        return False


def run(
    worker_name: str,
    stage_name: str,
    extractor_cls,
    claim_status: str,
    has_text_layer: bool,
    limit: int | None,
    loop: bool = False,
) -> None:
    repo, s3 = init(worker_name)
    extractor = extractor_cls()

    def process_batch() -> int:
        docs = repo.claim_documents(
            Status.TABLE_PENDING,
            claim_status=claim_status,
            has_text_layer=has_text_layer,
            limit=limit,
        )
        if not docs:
            return 0
        print(f"{worker_name}: processing {len(docs)} document(s)…")
        succeeded = failed = 0
        for doc in docs:
            ok = process_one(doc, s3, repo, extractor, stage_name, claim_status)
            succeeded += ok
            failed += not ok
            print(f"  {'✓' if ok else '✗'}  {doc.get('file_name')}")
        print(f"  batch done — succeeded={succeeded} failed={failed}")
        return len(docs)

    if loop:
        print(f"{worker_name}: starting in --loop mode (self-drains until queue empty)")
        total = 0
        while True:
            processed = process_batch()
            total += processed
            if processed == 0:
                print(f"{worker_name}: queue empty after {total} documents. exiting.")
                break
            time.sleep(1)  # brief pause between batches to avoid thundering herd
    else:
        count = process_batch()
        if count == 0:
            print(f"{worker_name}: no documents pending.")
            sys.exit(0)
