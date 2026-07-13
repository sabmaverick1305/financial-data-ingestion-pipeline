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


class LineageContext:
    """Wraps one pipeline_run (services/lineage.py) for a single worker
    invocation's lifetime — opened once in main(), not once per document,
    same rationale as mf_ingestion/sync.py's per-thread connection reuse.
    Each process_one() call records its own bronze/silver/gold hop via
    record(); document_processing_log (repo.log_stage) is untouched and
    still the detailed per-document audit trail — this sits one level up,
    making cross-pipeline "what produced this artifact" queries possible.
    """

    def __init__(self, pipeline_name: str):
        from financial_pipeline.services import entity_store, lineage

        self._lineage = lineage
        self._conn = entity_store.connect()
        self._cur = self._conn.cursor()
        self.run_id = lineage.start_run(self._cur, pipeline_name)
        self._conn.commit()
        self._counts = {"success": 0, "failed": 0}

    def record(
        self,
        *,
        stage: str,
        source_type: str,
        source_ref: str,
        target_type: str,
        target_ref: str | None,
        status: str = "success",
        error: str | None = None,
        record_count: int | None = None,
    ) -> None:
        self._lineage.record_transform(
            self._cur,
            run_id=self.run_id,
            stage=stage,
            source_type=source_type,
            source_ref=source_ref,
            target_type=target_type,
            target_ref=target_ref,
            record_count=record_count,
            status=status,
            error=error,
        )
        self._conn.commit()
        self._counts["success" if status == "success" else "failed"] += 1

    def close(self) -> None:
        overall = "completed" if self._counts["failed"] == 0 else "completed_with_errors"
        self._lineage.complete_run(self._cur, self.run_id, overall, **self._counts)
        self._conn.commit()
        self._cur.close()
        self._conn.close()


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
