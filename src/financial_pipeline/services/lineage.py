"""Ingestion lineage — Bronze -> Silver -> Gold traceability, plus the
mandatory entity-resolution gateway.

Every ingestion pipeline (AMFI NAV fetch, mf_ingestion sync, SEBI SID scrape,
mf_performance, the document worker chain) writes bronze/silver/gold
artifacts independently, with no shared parent class (ingestion/base.py's
BaseIngester is not extended by any of them) and no unified way to answer
"what job run produced this row/file, and what did it derive from". This
module is that shared layer — a `pipeline_run` row per job execution and one
`ingestion_lineage` row per artifact-to-artifact transformation, generalizing
the pattern mf_ingestion/repository.py's start_ingestion_log/
complete_ingestion_log already established for mf_ingestion_log specifically.

resolve_and_link() additionally makes entity resolution structurally
unskippable for pipelines that mint scheme/AMC identity: any pipeline that
calls it gets both the entity-resolution write AND a lineage row for free,
so there's no way to add a new bronze/silver write path for entity-bearing
data without going through entity resolution too. It preserves the
best-effort-and-log semantics mf_ingestion/sync.py's _sync_entity already
established (see that function's docstring) — a bad match on one record
must never abort an otherwise-healthy ingestion run.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable

import structlog
from psycopg2.extensions import connection as Psycopg2Connection
from psycopg2.extensions import cursor as Psycopg2Cursor

log = structlog.get_logger()


def start_run(cur: Psycopg2Cursor, pipeline_name: str) -> str:
    """Opens a pipeline_run row, returns its run_id. Mirrors
    MFIngestionRepository.start_ingestion_log (mf_ingestion/repository.py)
    generalized to any pipeline_name rather than just mf_ingestion sync."""
    cur.execute(
        """
        INSERT INTO pipeline_run (pipeline_name, status)
        VALUES (%s, 'running')
        RETURNING run_id
        """,
        (pipeline_name,),
    )
    return str(cur.fetchone()[0])


def complete_run(cur: Psycopg2Cursor, run_id: str, status: str, **summary: Any) -> None:
    """Closes out a pipeline_run row. status should be one of
    'completed' | 'completed_with_errors' | 'failed'; summary is stored
    as-is in the jsonb summary column (counts, error text, whatever the
    caller finds useful — no fixed shape, unlike mf_ingestion_log's fixed
    columns, since pipelines vary widely in what they'd want to record)."""
    cur.execute(
        """
        UPDATE pipeline_run
        SET status = %s, completed_at = NOW(), summary = %s::jsonb
        WHERE run_id = %s
        """,
        (status, json.dumps(summary), run_id),
    )


def record_transform(
    cur: Psycopg2Cursor,
    *,
    run_id: str,
    stage: str,
    source_type: str | None,
    source_ref: str | None,
    target_type: str | None,
    target_ref: str | None,
    record_count: int | None = None,
    status: str = "success",
    error: str | None = None,
) -> None:
    """One row in ingestion_lineage for one artifact-to-artifact
    transformation. stage is one of bronze_ingest | bronze_to_silver |
    silver_to_gold | entity_resolution. Never raises — a lineage-write
    failure must not take down the ingestion it's describing, matching the
    best-effort convention every other logging call in these pipelines
    already follows (e.g. api/main.py's query_log write, mf_ingestion's
    entity sync)."""
    try:
        cur.execute(
            """
            INSERT INTO ingestion_lineage
                (run_id, stage, source_type, source_ref, target_type, target_ref,
                 record_count, status, error_message)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (run_id, stage, source_type, source_ref, target_type, target_ref, record_count, status, error),
        )
    except Exception as exc:  # noqa: BLE001 — best-effort, never break the caller's ingestion
        log.warning("lineage.record_transform_failed", run_id=run_id, stage=stage, error=str(exc))


def resolve_and_link(
    conn: Psycopg2Connection,
    cur: Psycopg2Cursor,
    *,
    run_id: str,
    ingest_fn: Callable[..., uuid.UUID | Any | None],
    source_ref: str,
    **ingest_kwargs: Any,
) -> Any | None:
    """The mandatory entity-resolution gateway: calls ingest_fn(cur,
    **ingest_kwargs) — e.g. services/entity_ingestion.py's
    ingest_scheme_plan or ingest_amc — and always records an
    entity_resolution lineage row alongside it, success or failure.

    Never raises: mirrors mf_ingestion/sync.py's original _sync_entity
    semantics exactly (a bad category/AMC match on one record must not
    abort an otherwise-healthy ingestion run) — this function is what that
    logic now runs through, so every pipeline that mints scheme/AMC
    identity gets the same guarantee instead of reimplementing its own
    try/except around ingest_scheme_plan/ingest_amc.

    Any pipeline that writes a record needing entity resolution should call
    this rather than ingest_fn directly — see
    tests/test_ingestion_gateway_coverage.py, which asserts
    entity_ingestion's ingest_* functions are never called anywhere except
    from here.
    """
    try:
        result = ingest_fn(cur, **ingest_kwargs)
        conn.commit()
        # ingest_scheme_plan returns a SchemePlanIngestResult dataclass (its
        # scheme_entity_id is the meaningful target); ingest_amc/ingest_category
        # return a bare entity_id UUID directly.
        if result is None:
            target_ref = None
        elif hasattr(result, "scheme_entity_id"):
            target_ref = str(result.scheme_entity_id)
        else:
            target_ref = str(result)
        record_transform(
            cur,
            run_id=run_id,
            stage="entity_resolution",
            source_type="postgres",
            source_ref=source_ref,
            target_type="postgres",
            target_ref=target_ref,
            status="success",
        )
        conn.commit()
        return result
    except Exception as exc:  # noqa: BLE001 — best-effort, keep the ingestion going
        conn.rollback()
        log.warning("lineage.entity_resolution_failed", run_id=run_id, source_ref=source_ref, error=str(exc))
        record_transform(
            cur,
            run_id=run_id,
            stage="entity_resolution",
            source_type="postgres",
            source_ref=source_ref,
            target_type="postgres",
            target_ref=None,
            status="failed",
            error=str(exc),
        )
        conn.commit()
        return None
