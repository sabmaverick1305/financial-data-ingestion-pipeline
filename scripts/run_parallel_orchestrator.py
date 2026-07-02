#!/usr/bin/env python3
"""Drain the full document processing backlog with parallel ECS Fargate tasks.

Runs three stages in sequence:
  1. text-worker  — PyMuPDF text extraction for all new documents
  2. table-worker — Docling table extraction for native-text PDFs
  3. ocr-worker   — Docling OCR extraction for scanned PDFs
  4. chunk-worker — text chunking for all tables_extracted docs

Key design decisions vs. the original wave-based approach:
  - Each task runs with --loop so it self-drains until the queue is empty,
    then exits cleanly. No more waves — a fixed pool of N tasks consumes
    the whole backlog without the orchestrator re-launching between waves.
  - Dynamic concurrency: num_tasks = min(max_concurrency, ceil(n/BATCH_SIZE))
    so we never over-provision Fargate tasks for small queues.
  - CloudWatch custom metrics emitted after each stage so alarms can fire
    when queue depth stalls.
  - The --loop flag is passed via ECS containerOverrides so the task
    definition itself doesn't need to change.

Usage:
    python scripts/run_parallel_orchestrator.py [--dry-run]

Environment:
    Reads AWS credentials and DB URL from .env via financial_pipeline.config.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import boto3
import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from financial_pipeline.config import settings  # noqa: E402

AWS_KW = dict(
    region_name=settings.aws_region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
)

ecs = boto3.client("ecs", **AWS_KW)
cw = boto3.client("cloudwatch", **AWS_KW)

CLUSTER = "weather-agent-cluster"
SUBNETS = ["subnet-089bc3cd1c8760332"]
SGS = ["sg-0eecd839abfcf4daa"]
CW_NAMESPACE = "AMFI/Pipeline"

# BATCH_SIZE controls how many docs each task claims per iteration when running
# in --loop mode. Keep it small so claim_expires_at TTL (30 min) is not
# exceeded mid-batch for heavy Docling tasks.
BATCH_SIZE = 20

_pg = settings.postgres_url.replace("postgresql+psycopg2://", "")
_userpass, _hostdb = _pg.split("@", 1)
_user, _pass = _userpass.split(":", 1)
_hostport, _dbname = _hostdb.split("/", 1)
_host, _port = (_hostport.split(":", 1) + ["5432"])[:2]

PG_CONNECT = dict(
    host=_host,
    port=int(_port),
    dbname=_dbname,
    user=_user,
    password=_pass.replace("%40", "@"),
)


def pending_count(query: str) -> int:
    conn = psycopg2.connect(**PG_CONNECT)
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchone()[0]
    finally:
        conn.close()


def emit_metric(metric_name: str, value: float, unit: str = "Count") -> None:
    try:
        cw.put_metric_data(
            Namespace=CW_NAMESPACE,
            MetricData=[{"MetricName": metric_name, "Value": value, "Unit": unit}],
        )
    except Exception as exc:
        print(f"  [metrics] failed to emit {metric_name}: {exc}", flush=True)


def launch(task_def: str, extra_args: list[str] | None = None, dry_run: bool = False) -> str | None:
    """Launch a single Fargate task, optionally overriding the container command."""
    if dry_run:
        print(f"  [dry-run] would launch {task_def} {extra_args or ''}", flush=True)
        return f"dry-run-{task_def}"

    run_kw: dict = dict(
        cluster=CLUSTER,
        taskDefinition=task_def,
        launchType="FARGATE",
        count=1,
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": SUBNETS,
                "securityGroups": SGS,
                "assignPublicIp": "ENABLED",
            }
        },
    )
    if extra_args:
        # Retrieve the container name from the task definition so we can
        # build a valid containerOverrides entry.
        td = ecs.describe_task_definition(taskDefinition=task_def)
        container_name = td["taskDefinition"]["containerDefinitions"][0]["name"]
        existing_cmd = td["taskDefinition"]["containerDefinitions"][0].get("command", [])
        run_kw["overrides"] = {
            "containerOverrides": [
                {
                    "name": container_name,
                    "command": existing_cmd + extra_args,
                }
            ]
        }

    resp = ecs.run_task(**run_kw)
    if resp.get("failures"):
        print(f"  run_task FAILURES: {resp['failures']}", flush=True)
        return None
    arn = resp["tasks"][0]["taskArn"]
    print(f"  started {arn}", flush=True)
    return arn


def wait_all(arns: list[str], dry_run: bool = False) -> list[tuple[str, int]]:
    if dry_run:
        return [(a, 0) for a in arns]
    results: dict[str, int] = {}
    while len(results) < len(arns):
        remaining = [a for a in arns if a not in results]
        resp = ecs.describe_tasks(cluster=CLUSTER, tasks=remaining)
        for d in resp["tasks"]:
            if d["lastStatus"] == "STOPPED":
                exit_code = d["containers"][0].get("exitCode", -1)
                results[d["taskArn"]] = exit_code
                print(f"  {d['taskArn'].split('/')[-1]}: exitCode={exit_code}", flush=True)
        if len(results) < len(arns):
            time.sleep(15)
    return [(a, results[a]) for a in arns]


def drain(
    stage_name: str,
    task_def: str,
    pending_query: str,
    max_concurrency: int,
    dry_run: bool = False,
) -> bool:
    """Launch up to max_concurrency tasks in --loop mode; each self-drains.

    Returns True if the stage completed cleanly, False if tasks failed.
    """
    print(f"\n=== {stage_name} ===", flush=True)
    n = pending_count(pending_query)
    emit_metric(f"{stage_name}.queue_depth_start", n)

    if n == 0:
        print(f"{stage_name}: backlog clear.", flush=True)
        return True

    # Don't over-provision: no point launching 5 tasks for 3 documents.
    num_tasks = max(1, min(max_concurrency, math.ceil(n / BATCH_SIZE)))
    print(f"{stage_name}: {n} pending -> launching {num_tasks} task(s) in loop mode", flush=True)

    arns = [a for a in (launch(task_def, ["--loop"], dry_run) for _ in range(num_tasks)) if a]
    if not arns:
        print(f"{stage_name}: all launches failed, stopping.", flush=True)
        return False

    results = wait_all(arns, dry_run)
    bad = [c for _, c in results if c not in (0, 1)]
    if bad:
        print(f"{stage_name}: {len(bad)} task(s) exited unexpectedly ({bad}).", flush=True)
        emit_metric(f"{stage_name}.task_failures", len(bad))
        return False

    remaining = pending_count(pending_query)
    emit_metric(f"{stage_name}.queue_depth_end", remaining)
    if remaining > 0:
        print(
            f"{stage_name}: WARNING — {remaining} docs still pending after tasks exited.",
            flush=True,
        )

    print(f"{stage_name}: done.", flush=True)
    return True


def main(dry_run: bool = False) -> None:
    ok = True

    ok &= drain(
        "text-worker",
        "amfi-text-worker",
        "SELECT count(*) FROM document_metadata WHERE processing_status IN ('uploaded','classified')",
        max_concurrency=3,
        dry_run=dry_run,
    )
    ok &= drain(
        "table-worker",
        "amfi-table-worker",
        "SELECT count(*) FROM document_metadata WHERE processing_status='text_extracted' AND has_text_layer=true",
        max_concurrency=5,
        dry_run=dry_run,
    )
    ok &= drain(
        "ocr-worker",
        "amfi-ocr-worker",
        "SELECT count(*) FROM document_metadata WHERE processing_status='text_extracted' AND has_text_layer=false",
        max_concurrency=3,
        dry_run=dry_run,
    )
    ok &= drain(
        "chunk-worker",
        "amfi-chunk-worker",
        "SELECT count(*) FROM document_metadata WHERE processing_status='tables_extracted'",
        max_concurrency=3,
        dry_run=dry_run,
    )
    ok &= drain(
        "embed-worker",
        "amfi-embed-worker",
        "SELECT count(*) FROM document_metadata WHERE processing_status='processed'",
        max_concurrency=3,
        dry_run=dry_run,
    )

    if ok:
        print("\n=== ALL STAGES DRAINED ===", flush=True)
        emit_metric("pipeline.all_stages_drained", 1)
    else:
        print("\n=== PIPELINE FINISHED WITH ERRORS ===", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be launched without actually running ECS tasks",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
