#!/usr/bin/env python3
"""chunk-worker — lightweight (pure Python, no PDF/ML libraries loaded).

Picks documents at status=tables_extracted, claims them atomically via
SELECT ... FOR UPDATE SKIP LOCKED, reads back the text.json produced by the
earlier stage, chunks the full_text, uploads chunks.json, and marks the
document processed.

Failures: record_failure() increments attempt_count. After MAX_ATTEMPTS the
document moves to 'failed' (terminal).

Usage:
    python scripts/process_chunk_worker.py [--limit N] [--loop]
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, "src")

from _worker_common import SCHEMA_VERSION, init  # noqa: E402
from financial_pipeline.config import settings  # noqa: E402
from financial_pipeline.processing.chunker import chunk_text  # noqa: E402
from financial_pipeline.storage.document_repo import Status  # noqa: E402
import structlog  # noqa: E402

log = structlog.get_logger()


def process_one(doc: dict, s3, repo) -> bool:
    doc_id = str(doc["document_id"])
    filename = doc.get("file_name", doc_id)
    prefix = doc["s3_processed_key"]
    logger = log.bind(document_id=doc_id, file_name=filename, prefix=prefix)

    started_at = datetime.now(tz=timezone.utc)
    try:
        text_key = f"{prefix}/text.json"
        obj = s3.get_object(Bucket=settings.s3_bucket, Key=text_key)
        text_doc = json.loads(obj["Body"].read())
        full_text = text_doc.get("full_text", "")

        chunks = chunk_text(full_text)
        chunks_payload = json.dumps({
            "document_id": doc_id,
            "schema_version": SCHEMA_VERSION,
            "total_chunks": len(chunks),
            "chunks": chunks,
        }, ensure_ascii=False).encode()
        chunks_key = f"{prefix}/chunks.json"
        s3.put_object(Bucket=settings.s3_bucket, Key=chunks_key,
                      Body=chunks_payload, ContentType="application/json")

        completed_at = datetime.now(tz=timezone.utc)
        repo.log_stage(doc_id, "chunking", "success",
                       message=f"chunks={len(chunks)} key={chunks_key}",
                       started_at=started_at, completed_at=completed_at)
        repo.update_status(doc_id, Status.PROCESSED,
                           s3_processed_key=prefix,
                           schema_version=SCHEMA_VERSION)

        logger.info("chunk_worker.done", chunks=len(chunks),
                    elapsed_s=round((completed_at - started_at).total_seconds(), 1))
        return True

    except Exception as exc:
        new_status = repo.record_failure(
            doc_id,
            error=str(exc),
            revert_status=Status.TABLES_EXTRACTED,
        )
        repo.log_stage(doc_id, "chunking", "failed", message=str(exc))
        logger.error("chunk_worker.failed", error=str(exc), new_status=new_status)
        return False


def run_batch(repo, s3, limit: int | None) -> int:
    docs = repo.claim_documents(
        Status.CHUNK_PENDING,
        claim_status=Status.CHUNK_PROCESSING,
        limit=limit,
    )
    if not docs:
        return 0

    print(f"chunk-worker: processing {len(docs)} document(s)…")
    succeeded = failed = 0
    for doc in docs:
        ok = process_one(doc, s3, repo)
        succeeded += ok
        failed += not ok
        print(f"  {'✓' if ok else '✗'}  {doc.get('file_name')}")
    print(f"  batch done — succeeded={succeeded} failed={failed}")
    return len(docs)


def main(limit: int | None = None, loop: bool = False) -> None:
    repo, s3 = init("chunk-worker")

    if loop:
        print("chunk-worker: starting in --loop mode (self-drains until queue empty)")
        total = 0
        while True:
            count = run_batch(repo, s3, limit)
            total += count
            if count == 0:
                print(f"chunk-worker: queue empty after {total} documents. exiting.")
                break
            time.sleep(1)
    else:
        count = run_batch(repo, s3, limit)
        if count == 0:
            print("No documents pending chunking.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--loop", action="store_true",
                        help="Keep processing until queue is empty, then exit")
    args = parser.parse_args()
    main(limit=args.limit, loop=args.loop)
