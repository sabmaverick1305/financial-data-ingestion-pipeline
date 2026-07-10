"""Reads the mfapi.in scheme-master list from S3.

Self-contained: deliberately does not reuse storage/s3.py's S3Storage
(that class is wired for the AMFI PDF pipeline's parquet/CSV artifacts).
This ingestion source is a different dataset and stays independent.
"""

from __future__ import annotations

import json
import re
from typing import Any

import boto3
import structlog

log = structlog.get_logger()

_DATE_FOLDER_RE = re.compile(r"^\d{4}-\d{2}-\d{2}/$")


def resolve_latest_scheme_master_key(
    bucket: str,
    prefix: str,
    region: str = "ap-south-1",
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
) -> str:
    """Find the most recent dated snapshot under `prefix` (e.g. 'amfi/mf_scheme_master/raw').

    Snapshots are stored as `{prefix}/{YYYY-MM-DD}/mf_scheme_master.json` — date
    strings sort lexicographically, so the greatest "directory" prefix is newest.
    Keeps a recurring daily job correct even though today's snapshot key isn't
    known ahead of time and no snapshot is guaranteed for every single day.
    """
    client = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    prefix = prefix.rstrip("/") + "/"
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    # Only real YYYY-MM-DD folders — a non-dated common prefix (e.g. a stray
    # "raw/" leftover from a one-off manual copy) would otherwise sort after
    # any date lexicographically and get picked as "latest" by mistake.
    date_prefixes = sorted(
        p["Prefix"] for p in resp.get("CommonPrefixes", []) if _DATE_FOLDER_RE.match(p["Prefix"][len(prefix):])
    )
    if not date_prefixes:
        raise FileNotFoundError(f"no dated snapshots found under s3://{bucket}/{prefix}")
    latest = date_prefixes[-1]
    key = f"{latest}mf_scheme_master.json"
    log.info("mf_ingestion.resolved_latest_key", bucket=bucket, key=key)
    return key


def read_scheme_master_json(
    bucket: str,
    key: str,
    region: str = "ap-south-1",
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch and parse the raw scheme-list JSON array from S3."""
    client = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    obj = client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    records = json.loads(body)
    if not isinstance(records, list):
        raise ValueError(f"expected a JSON array at s3://{bucket}/{key}, got {type(records).__name__}")
    log.info("mf_ingestion.s3_read", bucket=bucket, key=key, records=len(records))
    return records
