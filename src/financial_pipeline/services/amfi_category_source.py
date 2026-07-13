"""AMFI Category Source — the "AMFI" box in the entity resolution pipeline:

    AMFI -> Normalize -> Canonical Entity -> MFAPI Mapping -> Entity Relationships

Provides the real, authoritative scheme_code -> category mapping from
AMFI's own NAVAll.txt section headers (parsed by
mf_ingestion.amfi_nav_parser.parse_nav_text), so it can feed
services/entity_resolver.py's resolve_category() as its highest-priority
signal (see that function's docstring: AMFI's own category outranks
mf_scheme_master.category, mfapi's own field, which is unreliable — "Income"
alone on over half of all rows).

Promoted from scratchpad's populate_entity_relationship.py's
load_amfi_category_map(), which used this same S3-read-plus-parse logic for
the one-off historical bulk backfill of financial_entity_relationship.
Making it a reusable, cached lookup here is what lets
mf_ingestion/sync.py's live per-scheme entity sync (services/
entity_ingestion.ingest_scheme_plan's amfi_category parameter) use the same
authoritative signal going forward, not just that one historical run —
before this module existed, nothing in src/ ever supplied a real
amfi_category value, so every scheme processed by the live sync was
classified using only mfapi's unreliable category field.
"""
from __future__ import annotations

from functools import lru_cache

import boto3
import structlog

from financial_pipeline.config import settings
from financial_pipeline.mf_ingestion.amfi_nav_parser import parse_nav_text

log = structlog.get_logger()

_NAVALL_PREFIX = "bronze/amfi/nav/"


def _s3_client():
    return boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )


def _latest_navall_key(s3) -> str | None:
    resp = s3.list_objects_v2(Bucket=settings.s3_bucket, Prefix=_NAVALL_PREFIX, Delimiter="/")
    dates = sorted(p["Prefix"] for p in resp.get("CommonPrefixes", []))
    if not dates:
        return None
    return dates[-1] + "NAVAll.txt"


@lru_cache(maxsize=1)
def load_amfi_category_map() -> dict[str, str]:
    """scheme_code -> AMFI's real section-header category, from the latest
    bronze/amfi/nav/ S3 snapshot (fetched by scripts/fetch_amfi_nav.py).

    Cached for the process lifetime: this is a daily-cadence snapshot, not
    something that needs re-fetching per call — mf_ingestion/sync.py
    processes thousands of schemes per run, and hitting S3 (list + get)
    once per scheme instead of once per process would be pure overhead for
    data that doesn't change within a single sync run. Call
    load_amfi_category_map.cache_clear() if a fresh snapshot needs picking
    up within a long-lived process.

    Returns {} (not an error) if no snapshot has ever been fetched —
    fetch_amfi_nav.py is not (yet) on a schedule, so a fresh environment or
    one that predates the last manual fetch may genuinely have none.
    Callers should treat a scheme_code miss the same as "no AMFI signal
    available for this scheme" — resolve_category() already falls back to
    mf_scheme_master.category and then the scheme_name pattern match, per
    its documented signal-priority order.
    """
    s3 = _s3_client()
    key = _latest_navall_key(s3)
    if key is None:
        log.warning("amfi_category_source.no_snapshot_found", bucket=settings.s3_bucket, prefix=_NAVALL_PREFIX)
        return {}

    obj = s3.get_object(Bucket=settings.s3_bucket, Key=key)
    raw = obj["Body"].read()
    df = parse_nav_text(raw)
    log.info("amfi_category_source.loaded", key=key, rows=len(df))
    return dict(zip(df["scheme_code"], df["category"]))
