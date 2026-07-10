"""Shared helpers for the document-processing worker scripts.

Pipeline:
    text-worker  (low memory)  -> table-worker (high memory) -\
                                                                 -> chunk-worker (low memory)
                               -> ocr-worker   (high memory) -/
"""

from __future__ import annotations

import io
import sys

import boto3
import structlog

sys.path.insert(0, "src")

from financial_pipeline.config import settings
from financial_pipeline.logging import configure_logging
from financial_pipeline.storage.document_repo import DocumentRepository

log = structlog.get_logger()

# Artifact schema version — still tracked per-document in Postgres
# (document_metadata.schema_version, see document_repo.py's update_status
# calls), not baked into the S3 path anymore now that path itself denotes
# the medallion layer (silver/ = cleaned/extracted, not raw).
SCHEMA_VERSION = "v1"


def processed_prefix(doc: dict) -> str:
    """silver/ medallion layer — cleaned/extracted text/chunks/tables,
    derived from the bronze/amfi/{monthly_aum,quarterly_aum,other}/ raw PDFs."""
    cat = doc.get("category", "unknown")
    if cat == "monthly" and doc.get("period_year") and doc.get("period_month"):
        return f"silver/amfi/monthly/{doc['period_year']}/{doc['period_month']:02d}"
    if cat == "quarterly" and doc.get("volume") and doc.get("issue"):
        return f"silver/amfi/quarterly/{doc.get('period_year', 'unknown')}/vol{doc['volume']}/issue{doc['issue']}"
    stem = (doc.get("file_name") or "unknown").rsplit(".", 1)[0]
    return f"silver/amfi/other/{stem}"


def make_repo() -> DocumentRepository:
    if not settings.postgres_url:
        print("POSTGRES_URL not set.")
        sys.exit(1)
    return DocumentRepository(settings.postgres_url)


def make_s3():
    return boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )


def init(worker_name: str):
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    repo = make_repo()
    # Reset any stale claims left by previously killed workers before
    # this instance starts claiming new work.
    reset = repo.reset_stale_claims()
    if reset:
        log.warning(
            "worker.stale_claims_recovered",
            worker=worker_name,
            count=len(reset),
            files=[r["file_name"] for r in reset[:5]],
        )
    return repo, make_s3()


def df_to_parquet_bytes(df) -> bytes:
    """Serialize a DataFrame to Parquet, with a fallback for pyarrow's
    "Expected bytes, got a 'int' object" failure on object columns that mix
    types — stringify and retry once rather than dropping the whole table."""
    buf = io.BytesIO()
    try:
        df.to_parquet(buf, index=False, engine="pyarrow")
    except Exception as exc:
        log.warning("worker.parquet_write_retrying_as_strings", error=str(exc))
        buf = io.BytesIO()
        df = df.astype(str)
        df.to_parquet(buf, index=False, engine="pyarrow")
    return buf.getvalue()
