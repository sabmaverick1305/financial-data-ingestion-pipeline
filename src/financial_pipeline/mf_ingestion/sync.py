"""Orchestrates the mfapi.in scheme-master + NAV-history sync job.

Flow: S3 scheme list -> validate -> per-scheme mfapi fetch (retry x3 on
rate-limit/network errors, then give up on that scheme and move on) ->
upsert mf_scheme_master / mf_nav_history / mf_scheme_sync_status ->
mark schemes missing from this run inactive -> write a run-level summary
to mf_ingestion_log.

Schemes are processed concurrently via a thread pool — each scheme's work
(one HTTP fetch + a few DB round trips) is I/O-bound, so threads give a
near-linear speedup up to the DB connection pool size, without the
complexity of an async rewrite.

Fully independent of the AMFI PDF ingestion pipeline (ingestion/, processing/) —
different source, different tables, no shared code paths.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime

import httpx
import structlog
from sqlalchemy import create_engine
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from financial_pipeline.mf_ingestion.mfapi_client import MFApiClient, RateLimitedError
from financial_pipeline.mf_ingestion.models import NavRecord, SchemeRecord
from financial_pipeline.mf_ingestion.repository import MFIngestionRepository
from financial_pipeline.mf_ingestion.s3_source import read_scheme_master_json, resolve_latest_scheme_master_key
from financial_pipeline.mf_ingestion.validation import InvalidSchemeRecord, parse_scheme_record

log = structlog.get_logger()

# Retried up to 3 attempts total (2 retries) with exponential backoff; anything
# still failing after that is recorded against mf_scheme_sync_status and the
# job moves on to the next scheme rather than aborting the whole run.
_RETRYABLE = (RateLimitedError, httpx.TransportError, httpx.TimeoutException)
_STATUS_COUNT_KEYS = ("success", "failed", "rate_limited", "no_data")


def _parse_mfapi_date(raw: str) -> date:
    return datetime.strptime(raw, "%d-%m-%Y").date()  # mfapi returns DD-MM-YYYY


@dataclass
class _SchemeResult:
    scheme_code: str
    status: str
    inserted: int = 0
    updated: int = 0
    records_fetched: int = 0


def _process_scheme(
    repo: MFIngestionRepository,
    scheme: SchemeRecord,
    *,
    s3_key: str,
    start_date: str,
    end_date: str,
    request_delay_seconds: float,
) -> _SchemeResult:
    """Fetch one scheme's NAV history and upsert it. Runs inside a worker thread —
    each call gets its own DB connection checked out from the shared pool and its
    own short-lived httpx.Client (see MFApiClient.fetch_nav_history), so this is
    safe to run concurrently across threads.
    """
    # Per-scheme traceability: the master list has one S3 key shared by every
    # scheme in the batch, so suffix it with scheme_code to make each row's
    # raw_s3_key distinct rather than identical across the whole run.
    scheme_raw_s3_key = f"{s3_key}#{scheme.scheme_code}"
    client = MFApiClient()

    status = "failed"
    error_message: str | None = None
    http_status: int | None = None
    retry_count = 0
    inserted = updated = records_fetched = 0
    latest_nav_date: date | None = None
    latest_nav_value: float | None = None
    # Fetch first, then a single merged upsert_scheme call below — avoids a
    # second DB round trip just to backfill amc_name/category/scheme_type
    # from mfapi's meta (each round trip costs ~250ms+ to a cross-region RDS).
    resolved_scheme = scheme
    data: list = []

    try:
        for attempt in Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=20),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        ):
            with attempt:
                retry_count = attempt.retry_state.attempt_number - 1
                payload = client.fetch_nav_history(scheme.scheme_code, start_date, end_date)

        meta = payload.get("meta") or {}
        data = payload.get("data") or []
        if meta:
            resolved_scheme = SchemeRecord(
                scheme_code=scheme.scheme_code,
                scheme_name=scheme.scheme_name,
                amc_name=meta.get("fund_house"),
                category=meta.get("scheme_category"),
                scheme_type=meta.get("scheme_type"),
            )
        status = "no_data" if not data else "success"
    except RateLimitedError as exc:
        status = "rate_limited"
        error_message = str(exc)
        http_status = 429
    except httpx.HTTPStatusError as exc:
        status = "failed"
        error_message = str(exc)
        http_status = exc.response.status_code
    except Exception as exc:  # noqa: BLE001 — record any failure, keep the batch going
        status = "failed"
        error_message = str(exc)

    # Single upsert regardless of outcome — satisfies the FK that
    # mf_nav_history/mf_scheme_sync_status both require, and carries mfapi's
    # meta enrichment (amc_name/category/scheme_type) when available.
    repo.upsert_scheme(resolved_scheme, raw_s3_key=scheme_raw_s3_key)

    if status == "success":
        nav_records = [
            NavRecord(scheme_code=scheme.scheme_code, nav_date=_parse_mfapi_date(d["date"]), nav=float(d["nav"]))
            for d in data
        ]
        records_fetched = len(nav_records)
        inserted, updated = repo.upsert_nav_history(nav_records, raw_s3_key=scheme_raw_s3_key)
        latest = max(nav_records, key=lambda r: r.nav_date)
        latest_nav_date, latest_nav_value = latest.nav_date, latest.nav

    repo.upsert_sync_status(
        scheme_code=scheme.scheme_code,
        sync_status=status,
        requested_start_date=date.fromisoformat(start_date),
        requested_end_date=date.fromisoformat(end_date),
        latest_nav_date=latest_nav_date,
        latest_nav_value=latest_nav_value,
        records_fetched=records_fetched,
        records_inserted=inserted,
        records_updated=updated,
        retry_count=retry_count,
        http_status=http_status,
        error_message=error_message,
        raw_s3_key=scheme_raw_s3_key,
    )

    log.info("mf_ingestion.scheme_done", scheme_code=scheme.scheme_code, status=status, records=records_fetched)
    if request_delay_seconds:
        time.sleep(request_delay_seconds)

    return _SchemeResult(
        scheme_code=scheme.scheme_code, status=status, inserted=inserted, updated=updated, records_fetched=records_fetched
    )


def run_sync(
    *,
    postgres_url: str,
    s3_bucket: str,
    s3_key: str | None = None,
    s3_prefix: str | None = None,
    aws_region: str = "ap-south-1",
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    request_delay_seconds: float = 0.15,
    max_workers: int = 10,
) -> dict:
    end_date = end_date or date.today().isoformat()
    # pool_size >= max_workers so every worker thread gets its own connection
    # without blocking on the pool; max_overflow gives headroom for bursts.
    engine = create_engine(postgres_url, pool_pre_ping=True, pool_size=max_workers, max_overflow=max_workers)
    repo = MFIngestionRepository(engine)
    repo.create_tables()

    if s3_key is None:
        if not s3_prefix:
            raise ValueError("run_sync requires either s3_key or s3_prefix")
        s3_key = resolve_latest_scheme_master_key(
            bucket=s3_bucket,
            prefix=s3_prefix,
            region=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    run_log = log.bind(job="mf_ingestion_sync", s3_key=s3_key)
    run_log.info("mf_ingestion.started", start_date=start_date, end_date=end_date, max_workers=max_workers)

    raw_records = read_scheme_master_json(
        bucket=s3_bucket,
        key=s3_key,
        region=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    run_id = repo.start_ingestion_log(
        source_s3_key=s3_key,
        requested_start_date=date.fromisoformat(start_date),
        requested_end_date=date.fromisoformat(end_date),
    )

    valid: list[SchemeRecord] = []
    invalid_count = 0
    for raw in raw_records:
        try:
            valid.append(parse_scheme_record(raw))
        except InvalidSchemeRecord as exc:
            invalid_count += 1
            run_log.warning("mf_ingestion.invalid_record", reason=exc.reason, raw=exc.raw)

    counts = {f"{k}_count": 0 for k in _STATUS_COUNT_KEYS}
    counts["nav_rows_inserted"] = 0
    counts["nav_rows_updated"] = 0

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _process_scheme,
                repo,
                scheme,
                s3_key=s3_key,
                start_date=start_date,
                end_date=end_date,
                request_delay_seconds=request_delay_seconds,
            )
            for scheme in valid
        ]
        for future in as_completed(futures):
            result = future.result()
            counts[f"{result.status}_count"] += 1
            counts["nav_rows_inserted"] += result.inserted
            counts["nav_rows_updated"] += result.updated
            done += 1
            if done % 500 == 0:
                run_log.info("mf_ingestion.progress", done=done, total=len(valid), **counts)

    active_codes = {s.scheme_code for s in valid}
    inactive_marked = repo.mark_missing_inactive(active_codes)

    overall_status = (
        "completed" if counts["failed_count"] == 0 and counts["rate_limited_count"] == 0 else "completed_with_errors"
    )
    repo.complete_ingestion_log(
        run_id,
        total_schemes=len(raw_records),
        valid_schemes=len(valid),
        invalid_schemes=invalid_count,
        inactive_marked=inactive_marked,
        status=overall_status,
        error_summary=(
            None
            if overall_status == "completed"
            else f"{counts['failed_count']} failed, {counts['rate_limited_count']} rate-limited"
        ),
        **counts,
    )

    run_log.info(
        "mf_ingestion.completed",
        run_id=run_id,
        invalid_schemes=invalid_count,
        inactive_marked=inactive_marked,
        **counts,
    )
    return {"run_id": run_id, "invalid_schemes": invalid_count, "inactive_marked": inactive_marked, **counts}
