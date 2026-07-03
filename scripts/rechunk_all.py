#!/usr/bin/env python3
"""Re-chunk all embedded documents using the table-aware chunker.

Replaces all existing document_chunks rows with chunks produced by the
updated chunker (which preserves markdown table headers in every chunk).

Usage:
    # Dry-run — shows what would happen, no DB writes
    python scripts/rechunk_all.py --dry-run

    # Re-chunk all 440 documents
    python scripts/rechunk_all.py

    # Re-chunk a single document (for testing)
    python scripts/rechunk_all.py --doc-id f2b6facc-407a-491e-a6ce-c3b7fec98ee0

    # Limit to N documents (for smoke-testing)
    python scripts/rechunk_all.py --limit 5

Progress is printed per document. If a document fails (missing markdown.md,
S3 error, etc.) it is skipped and logged — the script does not abort.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import boto3
import structlog
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from financial_pipeline.config import settings
from financial_pipeline.processing.chunker import chunk_text
from financial_pipeline.storage.document_repo import DocumentRepository

log = structlog.get_logger()

EMBED_BATCH = 32   # chunks per encode() call


def rechunk_document(
    doc_id: str,
    s3_key: str,
    period_year: int | None,
    period_month: int | None,
    category: str,
    s3_client,
    model: SentenceTransformer,
    conn,
    dry_run: bool = False,
) -> dict:
    """Re-chunk one document. Returns a stats dict."""
    markdown_key = f"{s3_key}/markdown.md"

    try:
        obj = s3_client.get_object(Bucket=settings.s3_bucket, Key=markdown_key)
        markdown_text = obj["Body"].read().decode("utf-8")
    except Exception as exc:
        return {"status": "skip", "reason": f"S3 missing: {exc}"}

    new_chunks = chunk_text(markdown_text)
    if not new_chunks:
        return {"status": "skip", "reason": "chunker returned 0 chunks"}

    # Embed all chunks for this document in one batch
    texts = [c["text"] for c in new_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=EMBED_BATCH)

    if dry_run:
        return {"status": "dry_run", "new_chunks": len(new_chunks)}

    # Atomic replace: delete old, insert new
    old_count = conn.execute(
        text("SELECT COUNT(*) FROM document_chunks WHERE document_id = :d"),
        {"d": doc_id},
    ).scalar()

    conn.execute(
        text("DELETE FROM document_chunks WHERE document_id = :d"),
        {"d": doc_id},
    )

    for i, (chunk, emb) in enumerate(zip(new_chunks, embeddings)):
        conn.execute(
            text("""
                INSERT INTO document_chunks
                    (chunk_id, document_id, chunk_index, text,
                     char_start, char_end, token_count,
                     embedding, embedding_model,
                     period_year, period_month, category, created_at)
                VALUES
                    (:cid, :doc_id, :idx, :text,
                     :start, :end, :tokens,
                     :emb, :model,
                     :year, :month, :cat, NOW())
            """),
            {
                "cid":    str(uuid.uuid4()),
                "doc_id": doc_id,
                "idx":    i,
                "text":   chunk["text"],
                "start":  chunk["start"],
                "end":    chunk["end"],
                "tokens": len(chunk["text"].split()),
                "emb":    str(emb.tolist()),
                "model":  settings.embed_model,
                "year":   period_year,
                "month":  period_month,
                "cat":    category,
            },
        )

    return {
        "status": "ok",
        "old_chunks": old_count,
        "new_chunks": len(new_chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run",  action="store_true", help="Print what would happen, no DB writes")
    parser.add_argument("--limit",    type=int, default=None, help="Process only N documents")
    parser.add_argument("--doc-id",   default=None, help="Re-chunk one specific document")
    args = parser.parse_args()

    repo = DocumentRepository(settings.postgres_url)
    s3 = boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )

    print(f"Loading embedding model {settings.embed_model}...")
    model = SentenceTransformer(settings.embed_model)
    print("Model ready.\n")

    # Fetch documents to process
    with repo._engine.connect() as conn:
        if args.doc_id:
            query = text("""
                SELECT document_id, s3_processed_key, period_year, period_month, category, file_name
                FROM document_metadata
                WHERE document_id = :d AND processing_status = 'embedded'
            """)
            docs = conn.execute(query, {"d": args.doc_id}).mappings().all()
        else:
            query = text("""
                SELECT document_id, s3_processed_key, period_year, period_month, category, file_name
                FROM document_metadata
                WHERE processing_status = 'embedded'
                ORDER BY period_year DESC NULLS LAST, period_month DESC NULLS LAST
                LIMIT :lim
            """)
            docs = conn.execute(query, {"lim": args.limit or 100_000}).mappings().all()

    total = len(docs)
    print(f"Documents to process: {total}  (dry_run={args.dry_run})\n")

    ok = skip = 0
    old_total = new_total = 0
    t0 = time.perf_counter()

    for i, doc in enumerate(docs, 1):
        doc_t0 = time.perf_counter()

        with repo._engine.begin() as conn:
            result = rechunk_document(
                doc_id      = str(doc["document_id"]),
                s3_key      = doc["s3_processed_key"],
                period_year = doc["period_year"],
                period_month= doc["period_month"],
                category    = doc["category"],
                s3_client   = s3,
                model       = model,
                conn        = conn,
                dry_run     = args.dry_run,
            )

        elapsed = int((time.perf_counter() - doc_t0) * 1000)

        if result["status"] in ("ok", "dry_run"):
            ok += 1
            old_total += result.get("old_chunks", 0)
            new_total += result.get("new_chunks", 0)
            print(
                f"[{i:>3}/{total}] {doc['file_name']:<35} "
                f"{result.get('old_chunks', '?'):>3} → {result['new_chunks']:>3} chunks  "
                f"({elapsed} ms)"
            )
        else:
            skip += 1
            print(f"[{i:>3}/{total}] SKIP {doc['file_name']}: {result['reason']}")

    total_ms = int((time.perf_counter() - t0) * 1000)
    print(f"""
{'─'*60}
Done in {total_ms/1000:.1f}s
  Processed : {ok}/{total}
  Skipped   : {skip}
  Old chunks: {old_total:,}
  New chunks: {new_total:,}
  Reduction : {100*(1 - new_total/max(old_total,1)):.0f}% fewer chunks, higher quality
{'─'*60}
""")


if __name__ == "__main__":
    main()
