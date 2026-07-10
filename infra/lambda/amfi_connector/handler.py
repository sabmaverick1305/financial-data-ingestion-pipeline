"""AMFI NAV connector — runs inside Lambda (stdlib + boto3 only, no extra deps).

Triggered by EventBridge Scheduler. Fetches the AMFI NAV text file and
writes the raw bytes to S3 under (bronze/ medallion layer — raw, as-fetched):
    s3://<S3_BUCKET>/<S3_PREFIX>/YYYY-MM-DD/NAVAll.txt

Environment variables (set by CloudFormation / the ECS task definition):
    S3_BUCKET   – destination bucket
    S3_PREFIX   – key prefix (default: bronze/amfi/nav)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import date, timezone, datetime

import boto3

AMFI_NAV_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"
REQUEST_TIMEOUT = 30


def _fetch(url: str, timeout: int = REQUEST_TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "amfi-nav-connector/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def lambda_handler(event: dict, context: object) -> dict:
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "bronze/amfi/nav").rstrip("/")
    today = date.today().isoformat()

    print(json.dumps({"event": "fetch.started", "url": AMFI_NAV_URL}))
    raw = _fetch(AMFI_NAV_URL)
    print(json.dumps({"event": "fetch.completed", "bytes": len(raw)}))

    key = f"{prefix}/{today}/NAVAll.txt"
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=raw,
        ContentType="text/plain; charset=utf-8",
        Metadata={"source-url": AMFI_NAV_URL, "ingested-at": datetime.now(timezone.utc).isoformat()},
    )
    print(json.dumps({"event": "upload.completed", "s3_key": key, "bytes": len(raw)}))

    return {
        "statusCode": 200,
        "body": json.dumps({"s3_key": key, "bytes": len(raw), "date": today}),
    }
