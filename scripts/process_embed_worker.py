#!/usr/bin/env python3
"""embed-worker — lightweight (sentence-transformers CPU, no Docling/torch-GPU loaded).

Stage 5 of the pipeline: processed → embedded

Reads chunks.json from S3 (written by chunk-worker), generates dense vector
embeddings using sentence-transformers all-MiniLM-L6-v2 (384 dims, CPU-only),
and upserts them into the document_chunks pgvector table in RDS.

The model is cached after the first download (~90 MB) in the Docker image layer.
Pre-download at build time by adding to Dockerfile:
    RUN python -c "from sentence_transformers import SentenceTransformer; \
                   SentenceTransformer('all-MiniLM-L6-v2')"

Memory profile: ~500 MB (model in RAM) — suitable for a 2 GB Fargate task.
Batch size 64 is the sweet spot for CPU inference on MiniLM.

Usage:
    python scripts/process_embed_worker.py [--limit N] [--loop] [--batch-size N]
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, "src")

from _worker_common import init  # noqa: E402
from financial_pipeline.config import settings  # noqa: E402
from financial_pipeline.storage.document_repo import Status  # noqa: E402
import structlog  # noqa: E402

log = structlog.get_logger()

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM  = 384
ENCODE_BATCH_SIZE = 64   # chunks per encoding call


def load_model():
    """Load (or retrieve from cache) the sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    log.info("embed_worker.model_loading", model=MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    log.info("embed_worker.model_ready", model=MODEL_NAME, dims=EMBED_DIM)
    return model


def process_one(doc: dict, s3, repo, model) -> bool:
    doc_id    = str(doc["document_id"])
    filename  = doc.get("file_name", doc_id)
    prefix    = doc.get("s3_processed_key", "")
    logger    = log.bind(document_id=doc_id, file_name=filename)

    started_at = datetime.now(tz=timezone.utc)
    try:
        # ── 1. Read chunks.json from S3 ──────────────────────────────────────
        chunks_key = f"{prefix}/chunks.json"
        logger.info("embed_worker.reading_chunks", key=chunks_key)
        obj  = s3.get_object(Bucket=settings.s3_bucket, Key=chunks_key)
        data = json.loads(obj["Body"].read())
        chunks: list[dict] = data.get("chunks", [])

        if not chunks:
            logger.warning("embed_worker.no_chunks", key=chunks_key)
            repo.update_status(doc_id, Status.EMBEDDED)
            return True

        # ── 2. Generate embeddings in batches ────────────────────────────────
        texts = [c["text"] for c in chunks]
        logger.info("embed_worker.encoding", chunks=len(texts))

        embeddings = model.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,   # cosine sim = dot product after L2 norm
            convert_to_numpy=True,
        )

        # ── 3. Store in pgvector ─────────────────────────────────────────────
        stored = repo.store_chunk_embeddings(
            document_id  = doc_id,
            chunks       = chunks,
            embeddings   = embeddings.tolist(),
            model_name   = MODEL_NAME,
            period_year  = doc.get("period_year"),
            period_month = doc.get("period_month"),
            category     = doc.get("category"),
        )

        # ── 4. Advance status ────────────────────────────────────────────────
        completed_at = datetime.now(tz=timezone.utc)
        repo.log_stage(
            doc_id, "embedding", "success",
            message=f"model={MODEL_NAME} chunks={stored} dims={EMBED_DIM}",
            started_at=started_at, completed_at=completed_at,
        )
        repo.update_status(doc_id, Status.EMBEDDED)

        logger.info(
            "embed_worker.done",
            chunks=stored,
            elapsed_s=round((completed_at - started_at).total_seconds(), 1),
        )
        return True

    except Exception as exc:
        new_status = repo.record_failure(
            doc_id,
            error=str(exc),
            revert_status=Status.PROCESSED,
        )
        repo.log_stage(doc_id, "embedding", "failed", message=str(exc))
        logger.error("embed_worker.failed", error=str(exc), new_status=new_status)
        return False


def run_batch(repo, s3, model, limit: int | None) -> int:
    docs = repo.claim_documents(
        Status.EMBED_PENDING,
        claim_status=Status.EMBED_PROCESSING,
        limit=limit,
    )
    if not docs:
        return 0

    print(f"embed-worker: processing {len(docs)} document(s)…")
    succeeded = failed = 0
    for doc in docs:
        ok = process_one(doc, s3, repo, model)
        succeeded += ok
        failed    += not ok
        print(f"  {'✓' if ok else '✗'}  {doc.get('file_name')}")

    print(f"  batch done — succeeded={succeeded} failed={failed}")
    return len(docs)


def main(limit: int | None = None, loop: bool = False, batch_size: int = ENCODE_BATCH_SIZE) -> None:
    global ENCODE_BATCH_SIZE
    ENCODE_BATCH_SIZE = batch_size

    repo, s3 = init("embed-worker")
    model    = load_model()

    if loop:
        print("embed-worker: starting in --loop mode (self-drains until queue empty)")
        total = 0
        while True:
            count  = run_batch(repo, s3, model, limit)
            total += count
            if count == 0:
                print(f"embed-worker: queue empty after {total} documents. exiting.")
                break
            time.sleep(1)
    else:
        count = run_batch(repo, s3, model, limit)
        if count == 0:
            print("No documents pending embedding.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit",      type=int, default=None,
                        help="Max documents to claim per batch")
    parser.add_argument("--loop",       action="store_true",
                        help="Keep processing until queue is empty, then exit")
    parser.add_argument("--batch-size", type=int, default=ENCODE_BATCH_SIZE,
                        help=f"Encoding batch size (default {ENCODE_BATCH_SIZE})")
    args = parser.parse_args()
    main(limit=args.limit, loop=args.loop, batch_size=args.batch_size)
